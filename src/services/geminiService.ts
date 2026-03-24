import { GoogleGenAI } from "@google/genai";
import OpenAI from "openai";

// Environment variable handling for both AI Studio and Vercel/Vite environments
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
  } catch (e) {
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
      fetch: (...args) => window.fetch(...args)
    });
    return openaiInstance;
  } catch (error) {
    console.error("Failed to initialize OpenAI:", error);
    return null;
  }
}

export interface TranscriptionResult {
  transcript: string;
  rawTranscript?: string;
  summary?: string;
  actionItems?: string[];
  modelInfo?: string;
}

/**
 * Whisper API hard limit: 25MB per request.
 * This function checks if a blob exceeds the safe limit.
 */
export const WHISPER_SAFE_SIZE_BYTES = 24 * 1024 * 1024; // 24MB safety margin

/**
 * Transcribe a single audio chunk (Blob → base64 internally).
 * Suitable for both full recordings and chunked segments.
 */
export async function transcribeChunk(
  audioBlob: Blob,
  chunkIndex: number,
  contextHint?: string
): Promise<{ transcript: string; modelInfo: string }> {
  if (audioBlob.size < 100) {
    return { transcript: '', modelInfo: 'skipped (too small)' };
  }

  // Convert blob to base64
  const base64Audio = await new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(audioBlob);
    reader.onloadend = () => resolve((reader.result as string).split(',')[1]);
    reader.onerror = reject;
  });

  const sanitizedMimeType = audioBlob.type.split(';')[0];

  // --- OpenAI Whisper path ---
  const openai = await getOpenAI();
  if (openaiApiKey && openai) {
    try {
      const file = new File([audioBlob], `chunk_${chunkIndex}.webm`, { type: sanitizedMimeType });
      const transcription = await openai.audio.transcriptions.create({
        file,
        model: 'whisper-1',
        prompt: `Segment ${chunkIndex + 1}. Transcription of a conversation (Chinese, English, etc.). ${contextHint ? `Context: ${contextHint}.` : ''}
Format: Speaker: [HH:mm:ss] Content
Example:
A: [00:00:02] Hello
B: [00:00:04] 你好

Rules:
1. Keep original language. Do NOT translate.
2. Keep numerical timestamps.
3. If silent, return empty string. Never hallucinate.`
      });
      return { transcript: transcription.text, modelInfo: 'OpenAI Whisper' };
    } catch (error) {
      console.error(`Whisper chunk ${chunkIndex} failed:`, error);
      // Fall through to Gemini
    }
  }

  // --- Gemini fallback ---
  const ai = getGemini();
  if (!ai) throw new Error('API 金鑰缺失');

  const response = await ai.models.generateContent({
    model: 'gemini-3-flash-preview',
    contents: {
      parts: [
        { inlineData: { mimeType: sanitizedMimeType, data: base64Audio } },
        {
          text: `TRANSCRIPTION TASK (Segment ${chunkIndex + 1}):
1. Transcribe the audio. Distinguish speakers (A, B, C...).
2. Include timestamps [HH:mm:ss]. Keep original language.
3. Use Traditional Chinese (繁體中文) for Chinese speech.
4. If silent or unintelligible, return empty string.
5. DO NOT hallucinate or use templates.`
        }
      ]
    },
    config: {
      systemInstruction: 'You are a professional audio transcription engine. Output only what is heard. For Chinese, use Traditional Chinese. If no clear speech, return empty string.'
    }
  });

  return { transcript: response.text || '', modelInfo: 'Gemini 3 Flash' };
}

/**
 * Merge multiple chunk transcripts into a single coherent transcript.
 * Adjusts timestamps so they are cumulative across chunks.
 */
export function mergeChunkTranscripts(
  chunkTranscripts: string[],
  chunkDurationSeconds: number
): string {
  return chunkTranscripts
    .map((transcript, i) => {
      if (!transcript.trim()) return '';
      const offsetSeconds = i * chunkDurationSeconds;
      // Shift all [HH:mm:ss] timestamps by offset
      return transcript.replace(/\[(\d{2}):(\d{2}):(\d{2})\]/g, (_match, hh, mm, ss) => {
        const totalSecs = parseInt(hh) * 3600 + parseInt(mm) * 60 + parseInt(ss) + offsetSeconds;
        const newH = Math.floor(totalSecs / 3600).toString().padStart(2, '0');
        const newM = Math.floor((totalSecs % 3600) / 60).toString().padStart(2, '0');
        const newS = (totalSecs % 60).toString().padStart(2, '0');
        return `[${newH}:${newM}:${newS}]`;
      });
    })
    .filter(Boolean)
    .join('\n');
}

/**
 * Step 2: Deep analysis and polishing based on transcript
 */
export async function analyzeTranscript(transcript: string, contextHint?: string): Promise<TranscriptionResult> {
  const openai = await getOpenAI();
  if (openaiApiKey && openai) {
    try {
      const response = await openai.chat.completions.create({
        model: 'gpt-4o',
        messages: [
          {
            role: 'system',
            content: `您是一位專業且極具洞察力的會議秘書。${contextHint ? `本次對話背景：${contextHint}。` : ''}
            
**核心原則：**
1. **嚴禁虛構內容**：僅根據提供的逐字稿進行分析。如果逐字稿不足以生成摘要，請在 summary 中說明「無法從音訊中提取足夠資訊進行分析」。
2. **語言保持**：修飾後的逐字稿保持原始語言，不要翻譯。分析內容請使用繁體中文。`
          },
          {
            role: 'user',
            content: `
原始逐字稿：
"${transcript}"

請執行以下任務：
1. 修飾逐字稿：去除贅字，轉化為流暢專業的「精華逐字稿」。
2. **保留原始的數字時間戳記（如 [00:00:05]）與說話者，絕對不要替換成文字佔位符。**
3. 深度分析：根據對話實際性質生成總結。嚴禁虛構。
4. 行動項目：列出具體分工（若有）。

請以 JSON 格式返回：
{
  "transcript": "修飾後的內容...",
  "summary": "真實且具洞察力的總結（Markdown）...",
  "actionItems": []
}`
          }
        ],
        response_format: { type: 'json_object' }
      });

      const content = response.choices[0].message.content || '';
      const result = JSON.parse(content.replace(/```json\n?|```/g, '').trim());
      return { ...result, rawTranscript: transcript, modelInfo: 'GPT-4o' };
    } catch (error) {
      console.error('GPT-4o 分析失敗:', error);
    }
  }

  // Gemini fallback
  const ai = getGemini();
  if (!ai) throw new Error('Gemini API key is missing');

  const response = await ai.models.generateContent({
    model: 'gemini-3-flash-preview',
    contents: `
請根據以下逐字稿生成一份真實且具深度的會議紀錄：
"${transcript}"

${contextHint ? `本次對話背景資訊：${contextHint}。` : ''}

**嚴格指令：**
1. 嚴禁虛構內容。如逐字稿不足，在 summary 中說明「無法從音訊中提取足夠資訊」。
2. 保留說話者與原始數字時間戳記（如 [00:01:23]）。保持原始語言。
3. 生成總結：使用 Markdown 格式，捕捉真實細節。
4. 行動項目：列出具體項目（若有）。

以 JSON 格式返回：
{
  "transcript": "修飾後的專業精華逐字稿...",
  "summary": "真實且具洞察力的對話總結...",
  "actionItems": ["項目 1", "項目 2"]
}
分析內容請使用繁體中文。`,
    config: { responseMimeType: 'application/json' }
  });

  const result = JSON.parse(response.text || '{}');
  return { ...result, rawTranscript: transcript, modelInfo: 'Gemini 3 Flash' };
}

export async function summarizeTranscript(transcript: string, contextHint?: string): Promise<Partial<TranscriptionResult>> {
  const openai = await getOpenAI();
  if (openaiApiKey && openai) {
    try {
      const response = await openai.chat.completions.create({
        model: 'gpt-4o',
        messages: [
          {
            role: 'system',
            content: `您是一位專業且具備深度的 AI 會議助理。${contextHint ? `本次對話背景：${contextHint}。` : ''}請根據逐字稿生成真實、自然且具備洞察力的會議總結。`
          },
          {
            role: 'user',
            content: `分析以下會議逐字稿：
"${transcript}"

以 JSON 格式返回：
{
  "summary": "真實且具洞察力的對話總結（Markdown 格式）...",
  "actionItems": ["具體且可執行的行動項目 1", "項目 2"]
}
請使用繁體中文，不要虛構內容。`
          }
        ],
        response_format: { type: 'json_object' }
      });
      const result = JSON.parse((response.choices[0].message.content || '').replace(/```json\n?|```/g, '').trim());
      return { ...result, modelInfo: 'GPT-4o' };
    } catch (error) {
      console.error('OpenAI 摘要失敗:', error);
    }
  }

  const ai = getGemini();
  if (!ai) throw new Error('Gemini API key is missing');

  const response = await ai.models.generateContent({
    model: 'gemini-3-flash-preview',
    contents: `分析以下會議逐字稿，生成真實且具深度的會議紀錄：
"${transcript}"
${contextHint ? `已知背景資訊：${contextHint}。` : ''}

以 JSON 格式返回：
{
  "summary": "真實且具洞察力的對話總結（Markdown 格式）...",
  "actionItems": ["具體項目 1", "項目 2"]
}
請使用繁體中文，不要強行套用不相關的模板。`,
    config: { responseMimeType: 'application/json' }
  });

  const result = JSON.parse((response.text || '').replace(/```json\n?|```/g, '').trim());
  return { ...result, modelInfo: 'Gemini 3 Flash' };
}
