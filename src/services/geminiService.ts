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
  // Must match extension to actual codec or Whisper will fail to decode
  const ext = (sanitizedMime.includes('mp4') || sanitizedMime.includes('aac')) ? 'mp4'
    : sanitizedMime.includes('ogg') ? 'ogg'
    : 'webm';
  const openai = await getOpenAI();

  if (openaiApiKey && openai) {
    try {
      const file = new File([audioBlob], `chunk_${chunkIndex}.${ext}`, { type: sanitizedMime });
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


// ─── Step 2: Analyze (highlights + decisions + action items) ──────────────────
export async function analyzeTranscript(
  transcript: string,
  contextHint?: string
): Promise<TranscriptionResult> {
  const openai = await getOpenAI();

  // Keep input under 60k chars — leaves room for 8k token output
  const MAX_INPUT_CHARS = 60000;
  const trimmedTranscript = transcript.length > MAX_INPUT_CHARS
    ? transcript.slice(0, MAX_INPUT_CHARS) + '\n\n[...逐字稿過長，已截取前段]'
    : transcript;

  const systemPrompt = `你是一位專業的會議記錄整理師。${contextHint ? `\n本次會議背景：${contextHint}。` : ''}

工作原則：
1. 所有內容必須直接來自逐字稿，不推測、不創造
2. 主題分組：依討論主題自然分組，每組條列重點（接近原文措辭）
3. 決議：只記錄明確說出「決定、確定、就這樣」的內容
4. 行動項目：記錄明確的分工承諾
5. 回傳的 JSON 必須完整正確`;

  // ⚠️ 不要求 GPT-4o 在 JSON 裡重新輸出完整逐字稿
  // 那樣會把大部分 token 用完，導致 highlights 被截斷
  const userPrompt = `分析以下會議逐字稿，只輸出重點彙整、決議和行動項目：

${trimmedTranscript}

以 JSON 格式返回（不含 markdown 代碼框）：
{
  "highlights": [
    {
      "topic": "討論主題（從對話中提取）",
      "points": ["重點一（說明是誰說的，接近原文）", "重點二"]
    }
  ],
  "decisions": ["明確決議"],
  "actionItems": ["誰 → 做什麼（截止日期）"]
}

規則：points 盡量引用原文；沒有決議或行動項目就設為空陣列 []`;

  if (openaiApiKey && openai) {
    try {
      const res = await openai.chat.completions.create({
        model: 'gpt-4o',
        temperature: 0.1,
        max_tokens: 8192,
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: userPrompt },
        ],
        response_format: { type: 'json_object' },
      });

      const choice = res.choices[0];
      if (choice.finish_reason === 'length') {
        console.warn('[GPT-4o analyze] finish_reason=length — still truncated at 8192');
      }

      const raw = choice.message.content ?? '{}';
      let parsed: any = {};
      try {
        parsed = JSON.parse(raw);
      } catch (parseErr) {
        console.warn('[GPT-4o analyze] JSON parse failed:', parseErr);
        // Try partial recovery: close any open structure
        try {
          const fixed = raw.replace(/,?\s*$/, '') + ']}';
          parsed = JSON.parse(fixed);
        } catch {
          parsed = {};
        }
      }

      return buildResult(transcript, parsed, 'GPT-4o');
    } catch (err) {
      console.error('[GPT-4o analyze] error:', err);
    }
  }

  // ── Gemini fallback ───────────────────────────────────────────────────────
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
    console.warn('[Gemini analyze] JSON parse failed');
  }

  return buildResult(transcript, parsed, 'Gemini');
}

// ─── Helper: assemble TranscriptionResult from parsed JSON ───────────────────
function buildResult(
  originalTranscript: string,
  parsed: any,
  modelInfo: string
): TranscriptionResult {
  const highlights: { topic: string; points: string[] }[] = parsed.highlights ?? [];
  const decisions: string[] = parsed.decisions ?? [];
  const actionItems: string[] = parsed.actionItems ?? [];

  // Build summary string for backward compat
  const lines: string[] = [];
  for (const h of highlights) {
    lines.push(`## ${h.topic}`);
    for (const p of h.points ?? []) lines.push(`- ${p}`);
    lines.push('');
  }
  if (decisions.length) {
    lines.push('## 決議事項');
    for (const d of decisions) lines.push(`- ${d}`);
  }

  return {
    transcript: originalTranscript,
    rawTranscript: originalTranscript,
    highlights,
    decisions,
    actionItems,
    summary: lines.join('\n').trim(),
    modelInfo,
  };
}
