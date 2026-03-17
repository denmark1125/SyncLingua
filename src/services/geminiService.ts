import { GoogleGenAI, GenerateContentResponse } from "@google/genai";
import OpenAI from "openai";

const geminiApiKey = process.env.GEMINI_API_KEY;
const openaiApiKey = process.env.OPENAI_API_KEY;

const ai = geminiApiKey ? new GoogleGenAI({ apiKey: geminiApiKey }) : null;
const openai = openaiApiKey ? new OpenAI({ apiKey: openaiApiKey, dangerouslyAllowBrowser: true }) : null;

export interface TranscriptionResult {
  transcript: string;
  summary: string;
  actionItems: string[];
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
            content: "您是一位頂級的 AI 會議助理。您的任務是將原始的會議逐字稿修飾成專業、流暢且易於閱讀的『精華紀錄』。請保留所有關鍵細節，但去除贅字、口語修正，並使其結構化。"
          },
          {
            role: "user",
            content: `
              請根據以下原始逐字稿生成專業的會議紀錄：
              "${rawTranscript}"
              
              請以 JSON 格式返回：
              {
                "transcript": "修飾後的專業精華逐字稿...",
                "summary": "深入的會議摘要...",
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
      
      return JSON.parse(content) as TranscriptionResult;
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
    您是一位頂級的 AI 會議助理。
    1. 準確地轉錄提供的音訊內容，並將其修飾成專業、流暢且結構化的『精華逐字稿』。
    2. 提供會議的深入摘要。
    3. 提取清晰、具體且可執行的行動項目列表。
    
    請以 JSON 格式返回結果：
    {
      "transcript": "修飾後的專業精華逐字稿...",
      "summary": "深入的會議摘要...",
      "actionItems": ["行動項目 1", "行動項目 2", ...]
    }
    
    請確保所有內容（逐字稿、摘要、行動項目）都使用繁體中文。
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
    return JSON.parse(text) as TranscriptionResult;
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
            content: "您是一位頂級的 AI 會議助理。請根據逐字稿生成深入的摘要和行動項目。"
          },
          {
            role: "user",
            content: `
              分析以下會議逐字稿：
              "${transcript}"
              
              請以 JSON 格式返回：
              {
                "summary": "...",
                "actionItems": ["...", "..."]
              }
              
              請使用繁體中文。
            `
          }
        ],
        response_format: { type: "json_object" }
      });
      const content = response.choices[0].message.content;
      if (content) return JSON.parse(content);
    } catch (error) {
      console.error("OpenAI 摘要失敗:", error);
    }
  }

  if (!ai) throw new Error("Gemini API key is missing");

  const model = "gemini-3-flash-preview";
  
  const prompt = `
    分析以下會議逐字稿：
    "${transcript}"
    
    請提供：
    1. 簡明摘要。
    2. 行動項目列表。
    
    請以 JSON 格式返回：
    {
      "summary": "...",
      "actionItems": ["...", "..."]
    }
    
    請使用繁體中文。
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
  
  return JSON.parse(text);
}
