import { GoogleGenAI } from "@google/genai";
import OpenAI from "openai";

// ─── API Key resolution ────────────────────────────────────────────────────────
const getApiKey = (key: string): string | undefined => {
  if (key === 'GEMINI_API_KEY') {
    // @ts-ignore
    const k = process.env.GEMINI_API_KEY || import.meta.env.VITE_GEMINI_API || import.meta.env.VITE_GEMINI_API_KEY;
    if (k) return k;
  }
  if (key === 'OPENAI_API_KEY') {
    // @ts-ignore
    const k = process.env.OPENAI_API_KEY || import.meta.env.VITE_OPENAI_API_KEY;
    if (k) return k;
  }
  try {
    // @ts-ignore
    return import.meta.env[`VITE_${key}`] || process.env[key];
  } catch {
    return undefined;
  }
};

const geminiApiKey = getApiKey('GEMINI_API_KEY');
const openaiApiKey = getApiKey('OPENAI_API_KEY');

let openaiInstance: OpenAI | null = null;
let aiInstance: GoogleGenAI | null = null;

function getGemini(): GoogleGenAI | null {
  if (aiInstance) return aiInstance;
  if (!geminiApiKey) return null;
  aiInstance = new GoogleGenAI({ apiKey: geminiApiKey });
  return aiInstance;
}

async function getOpenAI(): Promise<OpenAI | null> {
  if (openaiInstance) return openaiInstance;
  if (!openaiApiKey) return null;
  try {
    openaiInstance = new OpenAI({
      apiKey: openaiApiKey,
      dangerouslyAllowBrowser: true,
      fetch: (...args) => window.fetch(...args),
    });
    return openaiInstance;
  } catch (err) {
    console.error("OpenAI init failed:", err);
    return null;
  }
}

// ─── Public status helper ─────────────────────────────────────────────────────
export function getBackendStatus(): { whisper: boolean; gpt4o: boolean; gemini: boolean } {
  return { whisper: !!openaiApiKey, gpt4o: !!openaiApiKey, gemini: !!geminiApiKey };
}

export interface TranscriptionResult {
  transcript: string;       // 適度整理的逐字稿（去除口頭禪，保留所有內容）
  rawTranscript?: string;   // 原始未處理文字
  highlights?: { topic: string; points: string[] }[]; // 主題分組重點
  decisions?: string[];     // 本次明確決議
  actionItems?: string[];   // 行動項目（誰做什麼）
  summary?: string;         // 相容舊欄位，存放格式化後的完整分析文字
  modelInfo?: string;
}

/** 24 MB safe margin for Whisper 25 MB hard limit */
export const WHISPER_SAFE_SIZE_BYTES = 24 * 1024 * 1024;

// ─── Utility ──────────────────────────────────────────────────────────────────
function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(blob);
    reader.onloadend = () => resolve((reader.result as string).split(',')[1]);
    reader.onerror = reject;
  });
}

// ─── Step 1a: Whisper raw transcription ──────────────────────────────────────
/**
 * NOTE: whisper-1 does NOT distinguish speakers.
 * It outputs a continuous text stream. Speaker labeling is done by GPT-4o in step 1b.
 */
async function rawTranscribeWithWhisper(
  audioBlob: Blob,
  chunkIndex: number,
  contextHint?: string
): Promise<{ rawText: string; usedModel: 'whisper' | 'gemini' }> {
  const sanitizedMime = audioBlob.type.split(';')[0];
  const openai = await getOpenAI();

  if (openaiApiKey && openai) {
    try {
      const file = new File([audioBlob], `chunk_${chunkIndex}.webm`, { type: sanitizedMime });
      const res = await openai.audio.transcriptions.create({
        file,
        model: 'whisper-1',
        response_format: 'text',
        prompt: contextHint
          ? `Context: ${contextHint}. Mixed Chinese and English speech.`
          : 'Mixed Chinese and English speech.',
      });
      const rawText = typeof res === 'string' ? res : (res as any).text ?? '';
      console.log(`[Whisper] chunk ${chunkIndex} ok, ${rawText.length} chars`);
      return { rawText: rawText.trim(), usedModel: 'whisper' };
    } catch (err) {
      console.error(`[Whisper] chunk ${chunkIndex} error:`, err);
      // fall through to Gemini
    }
  }

  // Gemini fallback
  const ai = getGemini();
  if (!ai) throw new Error('沒有可用的 API 金鑰（OpenAI / Gemini 均未設定）');

  const base64 = await blobToBase64(audioBlob);
  const gemRes = await ai.models.generateContent({
    model: 'gemini-1.5-flash',
    contents: {
      parts: [
        { inlineData: { mimeType: sanitizedMime, data: base64 } },
        {
          text: `Transcribe EXACTLY what is spoken. Output ONLY the spoken words in their original language.
For Chinese use Traditional Chinese (繁體中文). If silent or unintelligible output empty string.
${contextHint ? `Context: ${contextHint}` : ''}`,
        },
      ],
    },
    config: {
      systemInstruction: 'Professional transcription engine. Output only spoken text. No commentary.',
    },
  });
  const rawText = gemRes.text?.trim() ?? '';
  console.log(`[Gemini] chunk ${chunkIndex} ok, ${rawText.length} chars`);
  return { rawText, usedModel: 'gemini' };
}

// ─── Step 1b: GPT-4o diarization ─────────────────────────────────────────────
/**
 * Takes Whisper raw text and uses GPT-4o to:
 * - Infer speaker ROLES (主持人、報告者、與會者A…) from conversational cues
 * - Insert estimated timestamps [HH:mm:ss] starting from chunkOffset
 * - Light cleanup: remove filler words (嗯、啊、就是說) but preserve all content
 */
async function diarizeWithGPT4o(
  rawText: string,
  chunkIndex: number,
  chunkDurationSeconds: number,
  contextHint?: string
): Promise<string> {
  const openai = await getOpenAI();
  if (!openaiApiKey || !openai || !rawText.trim()) return rawText;

  const offsetSecs = chunkIndex * chunkDurationSeconds;
  const hh = String(Math.floor(offsetSecs / 3600)).padStart(2, '0');
  const mm = String(Math.floor((offsetSecs % 3600) / 60)).padStart(2, '0');
  const ss = String(offsetSecs % 60).padStart(2, '0');
  const startLabel = `${hh}:${mm}:${ss}`;

  try {
    const res = await openai.chat.completions.create({
      model: 'gpt-4o',
      temperature: 0.1,
      messages: [
        {
          role: 'system',
          content: `你是一位專業會議逐字稿整理專家。

任務：將 Whisper 輸出的純文字整理成帶有「說話者角色」和「時間戳記」的逐字稿。

說話者角色判斷規則：
- 根據說話內容、語氣、職責推斷角色，優先使用以下標籤：
  主持人（主導議程、宣布開始結束、點名發言）
  報告者（報告進度、呈現資料）
  提問者（主要在問問題）
  與會者A / 與會者B / 與會者C（其他發言者，依出現順序命名）
- 同一人在整份逐字稿中必須使用同一個標籤，不可改變
- 若只有一位說話者，全部標記為「講者」
${contextHint ? `- 會議背景：${contextHint}，可用來輔助判斷角色` : ''}

整理規則：
1. 時間戳記從 ${startLabel} 開始，依對話節奏推估（格式：[HH:mm:ss]）
2. 去除口頭禪（嗯、啊、呃、就是說、那個那個）但保留所有實質內容
3. 保持原始語言，不翻譯，不改寫句意。中文使用繁體中文。
4. 若文字過短或是噪音，返回空字串
5. 只輸出格式化後的逐字稿，不加任何說明

輸出格式：
主持人: [00:00:02] 大家好，今天主要討論 Q3 預算規劃。
報告者: [00:00:08] 好的，這邊我先報告上週的執行狀況。
與會者A: [00:00:45] 請問這個數字是含稅的嗎？
報告者: [00:00:49] 對，這邊是含稅金額。`,
        },
        {
          role: 'user',
          content: rawText,
        },
      ],
    });
    const result = res.choices[0]?.message?.content?.trim() ?? rawText;
    console.log(`[GPT-4o diarize] chunk ${chunkIndex} ok`);
    return result;
  } catch (err) {
    console.error(`[GPT-4o diarize] chunk ${chunkIndex} error:`, err);
    return rawText;
  }
}

// ─── Main export: transcribeChunk ─────────────────────────────────────────────
/**
 * Full pipeline:
 *   audioBlob → Whisper (raw text) → GPT-4o (speaker labels + timestamps)
 *
 * If no OpenAI key: audioBlob → Gemini (all-in-one)
 *
 * @param audioBlob       - The audio segment to transcribe
 * @param chunkIndex      - Zero-based index (used to compute timestamp offset)
 * @param contextHint     - Optional meeting context to improve accuracy
 * @param chunkDurationSeconds - Duration of each chunk in seconds (default 180 = 3 min)
 */
export async function transcribeChunk(
  audioBlob: Blob,
  chunkIndex: number,
  contextHint?: string,
  chunkDurationSeconds: number = 180
): Promise<{ transcript: string; modelInfo: string }> {
  if (audioBlob.size < 500) {
    console.log(`[chunk ${chunkIndex}] Skipped (${audioBlob.size}B too small)`);
    return { transcript: '', modelInfo: 'skipped' };
  }
  if (audioBlob.size > WHISPER_SAFE_SIZE_BYTES) {
    console.warn(`[chunk ${chunkIndex}] WARNING: ${audioBlob.size}B > 24MB limit`);
  }

  console.log(`[chunk ${chunkIndex}] Transcribing ${(audioBlob.size/1024).toFixed(0)}KB…`);

  // Step 1a
  const { rawText, usedModel } = await rawTranscribeWithWhisper(audioBlob, chunkIndex, contextHint);

  if (!rawText.trim()) {
    return { transcript: '', modelInfo: usedModel === 'whisper' ? 'OpenAI Whisper' : 'Gemini' };
  }

  // Step 1b — only when Whisper was used (Gemini handles diarization itself)
  if (usedModel === 'whisper') {
    const diarized = await diarizeWithGPT4o(rawText, chunkIndex, chunkDurationSeconds, contextHint);
    return { transcript: diarized, modelInfo: 'OpenAI Whisper + GPT-4o' };
  }

  return { transcript: rawText, modelInfo: 'Gemini' };
}

// ─── Merge chunks ─────────────────────────────────────────────────────────────
/**
 * Concatenates chunk transcripts.
 * Timestamps are already absolute (inserted by GPT-4o diarize step), no offset math needed.
 */
export function mergeChunkTranscripts(chunkTranscripts: string[]): string {
  return chunkTranscripts.filter(t => t?.trim()).join('\n');
}

// ─── Step 2: Analyze (polish + summary + action items) ────────────────────────
export async function analyzeTranscript(
  transcript: string,
  contextHint?: string
): Promise<TranscriptionResult> {
  const openai = await getOpenAI();

  const MAX_TRANSCRIPT_CHARS = 80000;
  const trimmedTranscript = transcript.length > MAX_TRANSCRIPT_CHARS
    ? transcript.slice(0, MAX_TRANSCRIPT_CHARS) + '\n\n[...逐字稿過長，已截取前段進行分析]'
    : transcript;

  const systemPrompt = `你是一位專業的會議記錄整理師。
${contextHint ? `本次會議背景：${contextHint}。` : ''}

你的工作原則：
1. 【忠實原文】所有輸出內容必須直接來自逐字稿，不推測、不創造、不補充沒說過的話
2. 【適度整理逐字稿】去除口頭禪（嗯、啊、呃、就是說）和明顯重複，但保留每個人說過的所有實質內容與原始措辭
3. 【主題分組】將對話依照討論主題自然分組，每組列出該主題下的具體重點（直接引用或接近原文）
4. 【決議與行動】只記錄明確說出的決定和承諾，不推斷`;

  const userPrompt = `請整理以下會議逐字稿：

${trimmedTranscript}

請以 JSON 格式返回（不含 markdown 代碼框）：
{
  "transcript": "適度整理後的逐字稿（去除口頭禪，保留說話者角色標籤和時間戳記，保留所有實質內容）",
  "highlights": [
    {
      "topic": "討論主題名稱（從對話中提取，不要自己命名）",
      "points": [
        "重點一（接近原文，說明是誰說的）",
        "重點二",
        "重點三"
      ]
    }
  ],
  "decisions": [
    "明確決議一（逐字稿中有明確說出的決定）"
  ],
  "actionItems": [
    "誰 → 做什麼（截止日期，若有提及）"
  ]
}

注意：
- highlights 的 points 直接從逐字稿擷取，不要改寫成摘要句
- decisions 只填有明確說「決定」、「確定」、「就這樣」的內容
- 若沒有明確決議或行動項目，對應陣列留空 []`;

  if (openaiApiKey && openai) {
    try {
      const res = await openai.chat.completions.create({
        model: 'gpt-4o',
        temperature: 0.1,  // 低溫度確保忠實，不要自由發揮
        max_tokens: 4096,
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: userPrompt },
        ],
        response_format: { type: 'json_object' },
      });

      const raw = res.choices[0].message.content ?? '{}';
      let parsed: any = {};
      try {
        parsed = JSON.parse(raw);
      } catch {
        console.warn('[GPT-4o analyze] JSON parse failed, using fallback');
        parsed = {
          transcript: trimmedTranscript,
          highlights: [{ topic: '（解析失敗）', points: ['請重新生成'] }],
          decisions: [],
          actionItems: [],
        };
      }

      // Build summary field for backward compatibility with UI rendering
      const summaryLines: string[] = [];
      if (parsed.highlights?.length) {
        for (const h of parsed.highlights) {
          summaryLines.push(`## ${h.topic}`);
          for (const p of h.points ?? []) summaryLines.push(`- ${p}`);
          summaryLines.push('');
        }
      }
      if (parsed.decisions?.length) {
        summaryLines.push('## 決議事項');
        for (const d of parsed.decisions) summaryLines.push(`- ${d}`);
      }
      parsed.summary = summaryLines.join('\n').trim();

      return { ...parsed, rawTranscript: transcript, modelInfo: 'GPT-4o' };
    } catch (err) {
      console.error('[GPT-4o analyze] error:', err);
    }
  }

  // Gemini fallback
  const ai = getGemini();
  if (!ai) throw new Error('沒有可用的 API 金鑰（OpenAI / Gemini 均未設定）');

  const res = await ai.models.generateContent({
    model: 'gemini-1.5-flash',
    contents: `${systemPrompt}\n\n${userPrompt}`,
    config: { responseMimeType: 'application/json' },
  });

  let parsed: any = {};
  try {
    parsed = JSON.parse(res.text ?? '{}');
  } catch {
    parsed = {
      transcript: trimmedTranscript,
      highlights: [{ topic: '（解析失敗）', points: ['請重新生成'] }],
      decisions: [],
      actionItems: [],
    };
  }

  const summaryLines: string[] = [];
  if (parsed.highlights?.length) {
    for (const h of parsed.highlights) {
      summaryLines.push(`## ${h.topic}`);
      for (const p of h.points ?? []) summaryLines.push(`- ${p}`);
      summaryLines.push('');
    }
  }
  if (parsed.decisions?.length) {
    summaryLines.push('## 決議事項');
    for (const d of parsed.decisions) summaryLines.push(`- ${d}`);
  }
  parsed.summary = summaryLines.join('\n').trim();

  return { ...parsed, rawTranscript: transcript, modelInfo: 'Gemini' };
}
