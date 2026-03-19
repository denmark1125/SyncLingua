import { GoogleGenAI, GenerateContentResponse } from "@google/genai";
import OpenAI from "openai";

// Environment variable handling for both AI Studio and Vercel/Vite environments
const getApiKey = (key: string): string | undefined => {
  // @ts-ignore - Vite environment variables
  const viteKey = import.meta.env[`VITE_${key}`];
  if (viteKey) return viteKey;
  
  // Handle variations like VITE_GEMINI_API (from user screenshot)
  if (key === 'GEMINI_API_KEY') {
    // @ts-ignore
    const altViteKey = import.meta.env.VITE_GEMINI_API;
    if (altViteKey) return altViteKey;
  }
  
  // Fallback to process.env for AI Studio environment
  try {
    return process.env[key];
  } catch (e) {
    return undefined;
  }
};

const geminiApiKey = getApiKey('GEMINI_API_KEY');
const openaiApiKey = getApiKey('OPENAI_API_KEY');

const ai = geminiApiKey ? new GoogleGenAI({ apiKey: geminiApiKey }) : null;
const openai = openaiApiKey ? new OpenAI({ apiKey: openaiApiKey, dangerouslyAllowBrowser: true }) : null;

export interface TranscriptionResult {
  transcript: string;
  summary: string;
  actionItems: string[];
  modelInfo?: string;
}

export async function processMeetingAudio(audioBase64: string, mimeType: string): Promise<TranscriptionResult> {
  // Check if OpenAI is configured
  if (openaiApiKey && openai) {
    try {
      // 1. Transcription using Whisper
      // Convert base64 to Buffer then to File-like object
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
      });

      const rawTranscript = transcription.text;

      // 2. Polishing and Analysis using GPT-4o
      const response = await openai.chat.completions.create({
        model: "gpt-4o",
        messages: [
          {
            role: "system",
            content: "您是一位頂級的 AI 會議助理，專精於撰寫極其詳細、具備敘事感且結構嚴謹的『大師級會議紀錄』。您的目標是捕捉會議中的每一個生動細節、具體數字、感人故事以及專業洞察。"
          },
          {
            role: "user",
            content: `
              請根據以下原始逐字稿生成一份詳盡、專業且具備深度的會議紀錄。
              
              原始逐字稿：
              "${rawTranscript}"
              
              請務必包含以下結構（若逐字稿中有提及）：
              1. 會議總覽：背景、目標與核心探討範圍。
              2. 核心決議與後續步驟：具體的分工、期限與行動方案。
              3. 品牌/專案定位：核心價值、市場區隔、經營理念。
              4. 個人 IP/人物塑造：人物設定、多重身份、性格特質、穿搭風格。
              5. 關鍵故事與內容素材：挖掘具傳奇色彩、感人或具備行銷價值的具體細節（例如：創業故事、貴人相助、特殊神蹟、專案經驗）。
              6. 市場與客群分析：主要客群、目標市場策略。
              7. 行業洞察：產業現況、專業分類、趨勢分析。
              8. 個人生活與興趣：休閒娛樂、收藏、感情觀等感性細節。
              
              請以 JSON 格式返回：
              {
                "transcript": "修飾後的專業精華逐字稿（保留專業語氣，去除贅字）...",
                "summary": "極其詳盡的會議總結（需包含上述所有結構，文字需優美且具備敘事感）...",
                "actionItems": ["具體的行動項目 1", "具體的行動項目 2", ...]
              }
              
              請確保所有內容使用繁體中文。
            `
          }
        ],
        response_format: { type: "json_object" }
      });

      const content = response.choices[0].message.content;
      if (!content) throw new Error("GPT-4o 沒有回應");
      
      const result = JSON.parse(content) as TranscriptionResult;
      return {
        ...result,
        modelInfo: "OpenAI (Whisper + GPT-4o)"
      };
    } catch (error) {
      console.error("OpenAI 處理失敗，切換回 Gemini:", error);
      // Fallback to Gemini if OpenAI fails
    }
  }

  // Fallback or Primary: Gemini 3 Flash
  if (!ai) {
    throw new Error("請在設定中配置 Gemini 或 OpenAI API 金鑰。");
  }

  const model = "gemini-3-flash-preview";
  
  const prompt = `
    您是一位頂級的 AI 會議助理，專精於撰寫極其詳細、具備敘事感且結構嚴謹的『大師級會議紀錄』。
    1. 準確地轉錄提供的音訊內容，並將其修飾成專業、流暢且結構化的『精華逐字稿』。
    2. 提供一份詳盡、專業且具備深度的會議總結。
    
    請務必包含以下結構（若內容中有提及）：
    - 會議總覽：背景、目標與核心探討範圍。
    - 核心決議與後續步驟：具體的分工與行動方案。
    - 品牌/專案定位：核心價值、市場區隔、經營理念。
    - 個人 IP/人物塑造：人物設定、多重身份、性格特質、風格。
    - 關鍵故事與內容素材：挖掘具傳奇色彩、感人或具備行銷價值的具體細節（例如：創業故事、貴人相助、神蹟、特殊專案）。
    - 市場與客群分析：主要客群、目標市場策略。
    - 行業洞察：產業現況、專業知識。
    - 個人生活與興趣：人性化的感性細節。
    
    請以 JSON 格式返回結果：
    {
      "transcript": "修飾後的專業精華逐字稿...",
      "summary": "極其詳盡且具敘事感的會議總結（需包含上述結構內容）...",
      "actionItems": ["具體且可執行的行動項目 1", "具體且可執行的行動項目 2", ...]
    }
    
    請確保所有內容都使用繁體中文，並捕捉具體的數字、人名與生動細節。
  `;

  const audioPart = {
    inlineData: {
      mimeType,
      data: audioBase64,
    },
  };

  const response: GenerateContentResponse = await ai.models.generateContent({
    model,
    contents: { parts: [audioPart, { text: prompt }] },
    config: {
      responseMimeType: "application/json",
    },
  });

  const text = response.text;
  if (!text) {
    throw new Error("AI 沒有回應");
  }

  try {
    const result = JSON.parse(text) as TranscriptionResult;
    return {
      ...result,
      modelInfo: "Gemini 3 Flash"
    };
  } catch (e) {
    console.error("解析 AI 回應失敗:", text);
    throw new Error("AI 回應格式錯誤");
  }
}

export async function summarizeTranscript(transcript: string): Promise<Partial<TranscriptionResult>> {
  if (openaiApiKey && openai) {
    try {
      const response = await openai.chat.completions.create({
        model: "gpt-4o",
        messages: [
          {
            role: "system",
            content: "您是一位頂級的 AI 會議助理。請根據逐字稿生成極其詳盡、具備敘事感且結構嚴謹的會議總結，包含背景、決議、品牌定位、人物塑造、關鍵故事、市場分析及個人生活細節。"
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

  if (!ai) throw new Error("Gemini API key is missing");

  const model = "gemini-3-flash-preview";
  
  const prompt = `
    分析以下會議逐字稿，並生成一份極其詳盡、具備敘事感且結構嚴謹的『大師級會議紀錄』：
    "${transcript}"
    
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
