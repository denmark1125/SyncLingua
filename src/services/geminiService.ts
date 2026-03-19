import { GoogleGenAI, GenerateContentResponse } from "@google/genai";
import OpenAI from "openai";

// Environment variable handling for both AI Studio and Vercel/Vite environments
const getApiKey = (key: string): string | undefined => {
  // Check for Gemini specifically
  if (key === 'GEMINI_API_KEY') {
    // @ts-ignore
    const k = process.env.GEMINI_API_KEY || import.meta.env.VITE_GEMINI_API || import.meta.env.VITE_GEMINI_API_KEY || import.meta.env.VITE_GEMINI_API;
    if (k) return k;
  }

  // Check for OpenAI specifically
  if (key === 'OPENAI_API_KEY') {
    // @ts-ignore
    const k = process.env.OPENAI_API_KEY || import.meta.env.VITE_OPENAI_API_KEY;
    if (k) return k;
  }

  // Fallback for other keys
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
    // Lazy initialization to avoid global side effects on load
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
 * Step 1: High-quality transcription only
 */
export async function transcribeAudio(audioBase64: string, mimeType: string, contextHint?: string): Promise<{ transcript: string, modelInfo: string }> {
  // Try OpenAI Whisper first
  const openai = await getOpenAI();
  if (openaiApiKey && openai) {
    try {
      const byteCharacters = atob(audioBase64);
      const byteNumbers = new Array(byteCharacters.length);
      for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
      }
      const byteArray = new Uint8Array(byteNumbers);
      const blob = new Blob([byteArray], { type: mimeType });
      const file = new File([blob], "recording.webm", { type: mimeType });

      const transcription = await openai.audio.transcriptions.create({
        file: file,
        model: "whisper-1",
        language: "zh", // Hint for Chinese
        prompt: `這是一段會議錄音。${contextHint ? `關鍵字與背景：${contextHint}。` : ""}請完整、準確地轉錄所有內容，特別注意專有名詞的正確性。`
      });

      return { 
        transcript: transcription.text, 
        modelInfo: "OpenAI Whisper" 
      };
    } catch (error) {
      console.error("Whisper 轉錄失敗:", error);
    }
  }

  // Fallback to Gemini
  const ai = getGemini();
  if (!ai) throw new Error("API 金鑰缺失");
  
  const model = "gemini-3-flash-preview";
  const prompt = `請完整且準確地轉錄這段音訊的所有對話內容。${contextHint ? `已知背景資訊：${contextHint}。請確保轉錄中的專有名詞與此資訊一致。` : ""}請直接輸出逐字稿，不要做摘要或修飾。使用繁體中文。`;
  
  const audioPart = {
    inlineData: { mimeType, data: audioBase64 },
  };

  const response = await ai.models.generateContent({
    model,
    contents: { parts: [audioPart, { text: prompt }] },
  });

  const text = response.text;
  if (!text) throw new Error("Gemini 轉錄無回應");

  return { 
    transcript: text, 
    modelInfo: "Gemini 3 Flash" 
  };
}

/**
 * Step 2: Deep analysis and polishing based on transcript
 */
export async function analyzeTranscript(transcript: string, contextHint?: string): Promise<TranscriptionResult> {
  const openai = await getOpenAI();
  if (openaiApiKey && openai) {
    try {
      const response = await openai.chat.completions.create({
        model: "gpt-4o",
        messages: [
          {
            role: "system",
            content: `您是一位頂級的 AI 會議助理。${contextHint ? `本次會議背景：${contextHint}。` : ""}請根據提供的原始逐字稿，生成一份極其詳盡、具備敘事感且結構嚴謹的『大師級會議紀錄』。請特別注意識別並保留正確的專有名詞（如公司名、人名、產品名）。`
          },
          {
            role: "user",
            content: `
              原始逐字稿：
              "${transcript}"
              
              請執行以下任務：
              1. 修飾逐字稿：去除贅字，將其轉化為流暢、專業的『精華逐字稿』。
              2. 深度分析：生成包含會議總覽、核心決議、品牌定位、人物塑造、關鍵故事、市場分析、行業洞察及個人生活細節的詳盡總結。
              3. 行動項目：列出具體的分工與後續步驟。
              
              請以 JSON 格式返回：
              {
                "transcript": "修飾後的專業精華逐字稿...",
                "summary": "極其詳盡且具敘事感的會議總結（需包含上述所有結構，文字需優美）...",
                "actionItems": ["具體項目 1", "具體項目 2", ...]
              }
              
              請使用繁體中文。
            `
          }
        ],
        response_format: { type: "json_object" }
      });

      const content = response.choices[0].message.content;
      if (!content) throw new Error("GPT-4o 分析無回應");
      
      const result = JSON.parse(content);
      return {
        ...result,
        rawTranscript: transcript,
        modelInfo: "GPT-4o"
      };
    } catch (error) {
      console.error("GPT-4o 分析失敗:", error);
    }
  }

  // Fallback to Gemini
  const ai = getGemini();
  if (!ai) throw new Error("Gemini API key is missing");
  const model = "gemini-3-flash-preview";
  
  const prompt = `
    請根據以下逐字稿生成一份『大師級會議紀錄』：
    "${transcript}"
    
    ${contextHint ? `本次會議背景資訊：${contextHint}。請確保分析中提及的專有名詞與此資訊一致。` : ""}
    
    任務：
    1. 修飾逐字稿為專業流暢的內容。
    2. 生成包含總覽、決議、品牌/人物 IP 定位、關鍵故事、市場/行業洞察、個人細節的詳盡總結。
    3. 列出具體行動項目。
    
    請以 JSON 格式返回：
    {
      "transcript": "修飾後的專業精華逐字稿...",
      "summary": "極其詳盡且具敘事感的會議總結...",
      "actionItems": ["項目 1", "項目 2", ...]
    }
    使用繁體中文。
  `;

  const response = await ai.models.generateContent({
    model,
    contents: prompt,
    config: { responseMimeType: "application/json" },
  });

  const text = response.text;
  if (!text) throw new Error("Gemini 分析無回應");
  
  const result = JSON.parse(text);
  return {
    ...result,
    rawTranscript: transcript,
    modelInfo: "Gemini 3 Flash"
  };
}

export async function processMeetingAudio(audioBase64: string, mimeType: string): Promise<TranscriptionResult> {
  // For backward compatibility or one-click flow, we still keep this but it now calls the two steps
  const { transcript, modelInfo: tModel } = await transcribeAudio(audioBase64, mimeType);
  const analysis = await analyzeTranscript(transcript);
  return {
    ...analysis,
    rawTranscript: transcript,
    modelInfo: `${tModel} + ${analysis.modelInfo}`
  };
}

export async function summarizeTranscript(transcript: string, contextHint?: string): Promise<Partial<TranscriptionResult>> {
  const openai = await getOpenAI();
  if (openaiApiKey && openai) {
    try {
      const response = await openai.chat.completions.create({
        model: "gpt-4o",
        messages: [
          {
            role: "system",
            content: `您是一位頂級的 AI 會議助理。${contextHint ? `本次會議背景：${contextHint}。` : ""}請根據逐字稿生成極其詳盡、具備敘事感且結構嚴謹的會議總結，包含背景、決議、品牌定位、人物塑造、關鍵故事、市場分析及個人生活細節。`
          },
          {
            role: "user",
            content: `
              分析以下會議逐字稿，並生成一份如『大師級紀錄』般詳盡的總結：
              "${transcript}"
              
              請以 JSON 格式返回：
              {
                "summary": "極其詳盡且具敘事感的會議總結（包含總覽、決議、定位、IP塑造、關鍵故事、市場洞察等結構）...",
                "actionItems": ["具體且可執行的行動項目 1", "具體且可執行的行動項目 2", ...]
              }
              
              請使用繁體中文，並務必捕捉所有具體細節。
            `
          }
        ],
        response_format: { type: "json_object" }
      });
      const content = response.choices[0].message.content;
      if (content) {
        const result = JSON.parse(content);
        return {
          ...result,
          modelInfo: "GPT-4o"
        };
      }
    } catch (error) {
      console.error("OpenAI 摘要失敗:", error);
    }
  }

  const ai = getGemini();
  if (!ai) throw new Error("Gemini API key is missing");

  const model = "gemini-3-flash-preview";
  
  const prompt = `
    分析以下會議逐字稿，並生成一份極其詳盡、具備敘事感且結構嚴謹的『大師級會議紀錄』：
    "${transcript}"
    
    ${contextHint ? `已知背景資訊：${contextHint}。請確保總結中的專有名詞與此資訊一致。` : ""}
    
    請務必包含以下結構（若內容中有提及）：
    1. 會議總覽
    2. 核心決議與後續步驟
    3. 品牌/專案定位
    4. 個人 IP/人物塑造
    5. 關鍵故事與內容素材（挖掘生動細節）
    6. 市場與客群分析
    7. 行業洞察
    8. 個人生活與興趣
    
    請以 JSON 格式返回：
    {
      "summary": "極其詳盡且具敘事感的會議總結（包含上述所有結構）...",
      "actionItems": ["具體且可執行的行動項目 1", "具體且可執行的行動項目 2", ...]
    }
    
    請使用繁體中文，並捕捉具體的數字、人名與細節。
  `;

  const response = await ai.models.generateContent({
    model,
    contents: prompt,
    config: {
      responseMimeType: "application/json",
    },
  });

  const text = response.text;
  if (!text) throw new Error("AI 沒有回應");
  
  const result = JSON.parse(text);
  return {
    ...result,
    modelInfo: "Gemini 3 Flash"
  };
}
