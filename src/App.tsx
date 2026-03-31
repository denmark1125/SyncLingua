/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 *
 * Mobile-first redesign:
 * - Phone: bottom tab bar (錄音 / 記錄 / 我的) + fullscreen recording view + bottom-sheet detail
 * - Desktop: original sidebar + main panel layout preserved
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  auth, loginWithGoogle, logout, db, collection, doc, setDoc,
  onSnapshot, query, where, orderBy, addDoc, updateDoc, deleteDoc,
  Timestamp, onAuthStateChanged, User, OperationType, handleFirestoreError
} from './firebase';
import {
  Mic, StopCircle, FileText, ListChecks, History, Trash2, LogOut,
  CheckCircle2, Clock, User as UserIcon, Search, X, Loader2, Layers,
  ChevronDown, Sparkles, BookOpen, Copy, Download, Check
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';
import { format } from 'date-fns';
import ReactMarkdown from 'react-markdown';
import {
  transcribeChunk, mergeChunkTranscripts, analyzeTranscript, WHISPER_SAFE_SIZE_BYTES
} from './services/geminiService';

function cn(...inputs: ClassValue[]) { return twMerge(clsx(inputs)); }

const CHUNK_INTERVAL_MS = 3 * 60 * 1000;

interface Meeting {
  id: string; userId: string; title: string; date: Timestamp;
  duration?: number; transcript?: string; rawTranscript?: string;
  summary?: string;
  highlights?: { topic: string; points: string[] }[];
  decisions?: string[];
  actionItems?: string[]; modelInfo?: string;
  contextHint?: string; status: 'recording' | 'processing' | 'completed' | 'error';
}

// ─── Logo ─────────────────────────────────────────────────────────────────────
const Logo = ({ className }: { className?: string }) => (
  <svg viewBox="0 0 100 100" className={cn('w-12 h-12', className)} fill="none" xmlns="http://www.w3.org/2000/svg">
    <circle cx="50" cy="50" r="45" className="stroke-wood/10" strokeWidth="0.5" />
    <circle cx="50" cy="50" r="35" className="stroke-wood/20" strokeWidth="0.5" />
    <circle cx="50" cy="50" r="25" className="stroke-wood/30" strokeWidth="0.5" />
    <motion.path d="M30 50 Q 40 20, 50 50 T 70 50" className="stroke-forest" strokeWidth="2" strokeLinecap="round"
      animate={{ d: ['M30 50 Q 40 20, 50 50 T 70 50','M30 50 Q 40 80, 50 50 T 70 50','M30 50 Q 40 20, 50 50 T 70 50'] }}
      transition={{ repeat: Infinity, duration: 4, ease: 'easeInOut' }} />
    <motion.path d="M35 50 Q 45 35, 50 50 T 65 50" className="stroke-terracotta" strokeWidth="1.5" strokeLinecap="round"
      animate={{ d: ['M35 50 Q 45 35, 50 50 T 65 50','M35 50 Q 45 65, 50 50 T 65 50','M35 50 Q 45 35, 50 50 T 65 50'] }}
      transition={{ repeat: Infinity, duration: 3, ease: 'easeInOut', delay: 0.5 }} />
    <motion.circle cx="50" cy="50" r="4" className="fill-sage"
      animate={{ scale: [1,1.5,1], opacity: [0.5,1,0.5] }} transition={{ repeat: Infinity, duration: 2 }} />
  </svg>
);

// ─── Animated waveform ────────────────────────────────────────────────────────
const Waveform = ({ volume, active }: { volume: number; active: boolean }) => (
  <div className="flex items-center justify-center gap-[3px] h-14 w-full">
    {Array.from({ length: 24 }).map((_, i) => {
      const shape = Math.sin((i / 24) * Math.PI) * 0.7 + 0.15;
      return (
        <motion.div key={i} className="w-[3px] rounded-full bg-white/70"
          animate={{ height: active ? [Math.max(4, volume * 0.3 * shape), Math.max(4, volume * 0.5 * shape), Math.max(4, volume * 0.3 * shape)] : [4] }}
          transition={{ repeat: Infinity, duration: 0.35 + i * 0.025, ease: 'easeInOut' }}
          style={{ minHeight: 4, maxHeight: 52 }}
        />
      );
    })}
  </div>
);

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN APP
// ═══════════════════════════════════════════════════════════════════════════════
export default function App() {
  const [user, setUser] = useState<User | null>(null);
  const [isAuthReady, setIsAuthReady] = useState(false);
  const [meetings, setMeetings] = useState<Meeting[]>([]);
  const [selectedMeeting, setSelectedMeeting] = useState<Meeting | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [contextHint, setContextHint] = useState('');
  const [audioVolume, setAudioVolume] = useState(0);
  const [activeTab, setActiveTab] = useState<'record'|'history'|'profile'>('record');
  const [showContextInput, setShowContextInput] = useState(false);

  // Chunked recording
  const [liveTranscript, setLiveTranscript] = useState('');
  const [chunksProcessed, setChunksProcessed] = useState(0);
  const [isTranscribingChunk, setIsTranscribingChunk] = useState(false);
  const currentMeetingIdRef = useRef<string>('');
  const chunkTranscriptsRef = useRef<string[]>([]);
  const transcribeModelRef = useRef<string>('');

  // MediaRecorder
  const streamRef = useRef<MediaStream | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const chunkTimerRef = useRef<NodeJS.Timeout | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const mimeTypeRef = useRef<string>('audio/webm');
  const chunkIndexRef = useRef(0);
  const isRecordingRef = useRef(false);
  useEffect(() => { isRecordingRef.current = isRecording; }, [isRecording]);
  /**
   * isFinalRef is set to true SYNCHRONOUSLY inside stopRecording() BEFORE
   * mediaRecorder.stop() is called, so the onstop closure always reads the
   * correct value — no React batch-update timing issue.
   */
  const isFinalRef = useRef(false);

  // Auth
  useEffect(() => {
    const unsub = onAuthStateChanged(auth, (u) => {
      setUser(u); setIsAuthReady(true);
      if (u) {
        setDoc(doc(db, 'users', u.uid), {
          uid: u.uid, email: u.email, displayName: u.displayName,
          photoURL: u.photoURL, createdAt: Timestamp.now()
        }, { merge: true }).catch(e => handleFirestoreError(e, OperationType.WRITE, `users/${u.uid}`));
      }
    });
    return unsub;
  }, []);

  // Meetings listener
  useEffect(() => {
    if (!user) { setMeetings([]); return; }
    const q = query(collection(db,'meetings'), where('userId','==',user.uid), orderBy('date','desc'));
    const unsub = onSnapshot(q, (s) => setMeetings(s.docs.map(d => ({ id: d.id, ...d.data() })) as Meeting[]),
      (e) => handleFirestoreError(e, OperationType.LIST, 'meetings'));
    return unsub;
  }, [user]);

  // Browser tab title
  useEffect(() => {
    if (isRecording) {
      const mins = String(Math.floor(recordingTime / 60)).padStart(2, '0');
      const secs = String(recordingTime % 60).padStart(2, '0');
      document.title = `⏺ ${mins}:${secs} 錄音中 — SyncLingua`;
    } else if (isProcessing) {
      document.title = '⚙️ 處理中 — SyncLingua';
    } else {
      document.title = 'SyncLingua — AI 會議錄音';
    }
  }, [isRecording, isProcessing, recordingTime]);

  // Timer
  useEffect(() => {
    if (isRecording) { timerRef.current = setInterval(() => setRecordingTime(p => p + 1), 1000); }
    else { if (timerRef.current) clearInterval(timerRef.current); setRecordingTime(0); }
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [isRecording]);

  // Flush chunk
  const flushChunk = useCallback(async (chunks: Blob[], idx: number, mime: string, isFinal: boolean) => {
    if (!chunks.length) return;
    const blob = new Blob(chunks, { type: mime });
    if (blob.size > WHISPER_SAFE_SIZE_BYTES) console.warn(`Chunk ${idx} > 24MB`);
    setIsTranscribingChunk(true);
    try {
      const { transcript, modelInfo } = await transcribeChunk(blob, idx, contextHint, CHUNK_INTERVAL_MS / 1000);
      transcribeModelRef.current = modelInfo;
      chunkTranscriptsRef.current[idx] = transcript;
      const full = mergeChunkTranscripts(chunkTranscriptsRef.current);
      setLiveTranscript(full);
      setChunksProcessed(p => p + 1);
      if (currentMeetingIdRef.current) {
        await updateDoc(doc(db, 'meetings', currentMeetingIdRef.current), {
          rawTranscript: full, transcript: full, modelInfo, status: isFinal ? 'completed' : 'recording'
        });
      }
      if (isFinal) {
        setSelectedMeeting(p => p ? { ...p, rawTranscript: full, transcript: full, modelInfo, status: 'completed' } : null);
        setIsTranscribingChunk(false); setIsProcessing(false);
      }
    } catch (e) {
      console.error(`Chunk ${idx} error:`, e);
      if (isFinal) {
        if (currentMeetingIdRef.current) await updateDoc(doc(db,'meetings',currentMeetingIdRef.current), { status: 'error', transcript: '轉錄發生錯誤，請稍後再試。' });
        setIsTranscribingChunk(false); setIsProcessing(false);
      }
    }
  }, [contextHint]);

  const startRecording = async () => {
    try {
      chunkTranscriptsRef.current = []; chunkIndexRef.current = 0;
      setLiveTranscript(''); setChunksProcessed(0);
      transcribeModelRef.current = ''; currentMeetingIdRef.current = '';
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
      streamRef.current = stream;
      const ac = new (window.AudioContext || (window as any).webkitAudioContext)();
      const src = ac.createMediaStreamSource(stream);
      const an = ac.createAnalyser(); an.fftSize = 256; src.connect(an);
      audioContextRef.current = ac; analyserRef.current = an;
      const buf = new Uint8Array(an.frequencyBinCount);
      const tick = () => { if (!analyserRef.current) return; analyserRef.current.getByteFrequencyData(buf); setAudioVolume(buf.reduce((a,b)=>a+b,0)/buf.length); animationFrameRef.current = requestAnimationFrame(tick); };
      tick();
      mimeTypeRef.current = MediaRecorder.isTypeSupported('audio/webm;codecs=opus') ? 'audio/webm;codecs=opus' : 'audio/webm';
      setIsRecording(true);
      if (user) {
        const data: Omit<Meeting,'id'> = { userId: user.uid, title: `${format(new Date(),'yyyy年MM月dd日 HH:mm')} 的錄音`, date: Timestamp.now(), contextHint, status: 'recording' };
        const ref = await addDoc(collection(db,'meetings'), data);
        currentMeetingIdRef.current = ref.id;
        setSelectedMeeting({ ...data, id: ref.id });
      }
      isFinalRef.current = false; // reset at start of every session
      const mime = mimeTypeRef.current;
      const launch = () => {
        const idx = chunkIndexRef.current;
        const rec = new MediaRecorder(stream, { mimeType: mime });
        mediaRecorderRef.current = rec;
        const local: Blob[] = [];
        rec.ondataavailable = e => { if (e.data.size > 0) local.push(e.data); };
        rec.onstop = () => {
          // isFinalRef.current is set SYNCHRONOUSLY inside stopRecording() before
          // rec.stop() is called, so this always reads the correct intent.
          flushChunk(local, idx, mime, isFinalRef.current);
        };
        rec.start(1000);
        chunkTimerRef.current = setTimeout(() => {
          if (mediaRecorderRef.current?.state === 'recording') {
            chunkIndexRef.current += 1;
            // isFinalRef stays false for auto-rotation chunks
            rec.stop();
            setTimeout(launch, 150); // small gap to let onstop fire first
          }
        }, CHUNK_INTERVAL_MS);
      };
      launch();
    } catch (e) { console.error(e); alert('無法存取麥克風，請檢查權限。'); }
  };

  const stopRecording = () => {
    if (!isRecording) return;
    // Cancel auto-rotation timer
    if (chunkTimerRef.current) { clearTimeout(chunkTimerRef.current); chunkTimerRef.current = null; }
    // Stop volume monitoring
    if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
    if (audioContextRef.current) audioContextRef.current.close();
    setAudioVolume(0);
    // CRITICAL: set isFinalRef BEFORE calling rec.stop(), so that the onstop
    // closure which fires synchronously (or near-synchronously) sees isFinal=true.
    isFinalRef.current = true;
    setIsRecording(false);
    isRecordingRef.current = false;
    setIsProcessing(true);
    // Trigger final flush
    if (mediaRecorderRef.current?.state === 'recording') mediaRecorderRef.current.stop();
    // Release mic
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;
  };

  const [copiedId, setCopiedId] = useState<string | null>(null);

  const copyTranscript = async (text: string, id: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedId(id);
      setTimeout(() => setCopiedId(null), 2000);
    } catch {
      // Fallback for older browsers
      const el = document.createElement('textarea');
      el.value = text;
      document.body.appendChild(el);
      el.select();
      document.execCommand('copy');
      document.body.removeChild(el);
      setCopiedId(id);
      setTimeout(() => setCopiedId(null), 2000);
    }
  };

  const downloadTranscript = (text: string, title: string) => {
    const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${title.replace(/[/\\?%*:|"<>]/g, '-')}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleManualAnalysis = async () => {
    const tx = selectedMeeting?.rawTranscript || liveTranscript;
    if (!selectedMeeting || !tx) return;
    setIsProcessing(true);
    try {
      const r = await analyzeTranscript(tx, selectedMeeting.contextHint);
      await updateDoc(doc(db,'meetings',selectedMeeting.id), {
        transcript: r.transcript,
        summary: r.summary,
        highlights: r.highlights ?? [],
        decisions: r.decisions ?? [],
        actionItems: r.actionItems ?? [],
        modelInfo: r.modelInfo
      });
      setSelectedMeeting(p => p ? {
        ...p,
        transcript: r.transcript,
        summary: r.summary,
        highlights: r.highlights,
        decisions: r.decisions,
        actionItems: r.actionItems,
        modelInfo: r.modelInfo
      } : null);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      alert(`AI 分析失敗：${msg.substring(0, 120)}`);
    } finally {
      setIsProcessing(false);
    }
  };

  const deleteMeeting = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!window.confirm('確定要刪除此會議記錄嗎？')) return;
    try { await deleteDoc(doc(db,'meetings',id)); if (selectedMeeting?.id === id) setSelectedMeeting(null); }
    catch (e) { handleFirestoreError(e, OperationType.DELETE, `meetings/${id}`); }
  };

  const filtered = meetings.filter(m =>
    m.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
    m.summary?.toLowerCase().includes(searchQuery.toLowerCase())
  );

  // ── Loading ──────────────────────────────────────────────────────────────────
  if (!isAuthReady) return (
    <div className="min-h-screen bg-paper flex items-center justify-center">
      <motion.div animate={{ scale:[1,1.1,1] }} transition={{ repeat: Infinity, duration: 2 }} className="w-12 h-12 bg-forest rounded-full" />
    </div>
  );

  // ── Login ────────────────────────────────────────────────────────────────────
  if (!user) return (
    <div className="min-h-screen bg-paper flex flex-col items-center justify-center p-6 overflow-hidden">
      <div className="absolute inset-0 opacity-20 pointer-events-none">
        <div className="absolute top-[-10%] left-[-10%] w-[50%] h-[50%] bg-sage blur-[120px] rounded-full" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[50%] h-[50%] bg-terracotta blur-[120px] rounded-full" />
      </div>
      <motion.div initial={{ opacity: 0, y: 40 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8, ease: [0.16,1,0.3,1] }}
        className="w-full max-w-sm text-center relative z-10"
      >
        <Logo className="w-20 h-20 mx-auto mb-8" />
        <h1 className="text-6xl font-serif font-bold text-forest mb-5 tracking-tighter leading-[0.9]">
          Sync<br /><span className="text-terracotta italic">Lingua</span>
        </h1>
        <p className="text-wood/70 mb-10 text-base font-serif italic leading-relaxed">
          捕捉對話中的詩意與深度。<br />將每次會議轉化為專業紀錄。
        </p>
        <button onClick={loginWithGoogle}
          className="w-full py-4 bg-forest text-white rounded-2xl font-bold text-base flex items-center justify-center gap-3 shadow-xl shadow-forest/20 active:scale-95 transition-all"
        >
          <UserIcon className="w-5 h-5" />使用 Google 帳號開始
        </button>
        <div className="mt-6 text-[10px] uppercase tracking-[0.3em] text-sage font-bold">AI-Powered · 即時轉錄 · 無限錄音</div>
      </motion.div>
    </div>
  );

  // ══════════════════════════════════════════════════════════════════════════════
  // MAIN APP
  // ══════════════════════════════════════════════════════════════════════════════

  // ── Shared MeetingDetail bottom-sheet content ────────────────────────────────
  const MeetingDetailContent = ({ m }: { m: Meeting }) => (
    <div className="px-5 pt-3 pb-10 space-y-4">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div className="flex-1 pr-3">
          <div className="text-[10px] font-bold uppercase tracking-widest text-sage mb-1">
            {format(m.date.toDate(), 'yyyy年MM月dd日 HH:mm')}
          </div>
          <h3 className="font-serif font-bold text-xl text-forest leading-tight">{m.title}</h3>
          {m.modelInfo && (
            <div className="mt-2 inline-flex items-center gap-1.5 px-3 py-1 bg-sage/10 rounded-full">
              <div className="w-1.5 h-1.5 rounded-full bg-sage" />
              <span className="text-[10px] font-bold uppercase tracking-wider text-sage">{m.modelInfo}</span>
            </div>
          )}
        </div>
        <button onClick={() => setSelectedMeeting(null)}
          className="p-2.5 bg-white rounded-xl border border-wood/10 active:bg-paper shrink-0">
          <X className="w-4 h-4 text-wood" />
        </button>
      </div>

      {/* Live recording notice */}
      {isRecording && m.id === currentMeetingIdRef.current && (
        <div className="p-4 bg-terracotta/5 border border-terracotta/15 rounded-2xl">
          <div className="flex items-center gap-2 mb-2">
            <motion.div className="w-2 h-2 rounded-full bg-terracotta" animate={{ scale:[1,1.5,1] }} transition={{ repeat: Infinity, duration: 1 }} />
            <span className="text-xs font-bold uppercase tracking-widest text-terracotta">錄音中</span>
            {isTranscribingChunk && <Loader2 className="w-3.5 h-3.5 text-wood/50 animate-spin ml-auto" />}
          </div>
          {liveTranscript
            ? <p className="text-xs text-forest/70 font-serif leading-relaxed line-clamp-4">{liveTranscript}</p>
            : <p className="text-xs text-wood/40 font-serif italic">錄音中，逐字稿轉錄完成後將顯示於此...</p>}
        </div>
      )}

      {/* Generate button */}
      {(m.transcript || liveTranscript) && !m.summary && !isRecording && (
        <button onClick={handleManualAnalysis} disabled={isProcessing}
          className={cn('w-full py-4 rounded-2xl font-bold text-sm flex items-center justify-center gap-2.5 bg-forest text-white shadow-lg shadow-forest/15 active:scale-[0.98] transition-all', isProcessing && 'opacity-50 cursor-not-allowed')}
        >
          {isProcessing ? <><Loader2 className="w-4 h-4 animate-spin" />生成中...</> : <><Sparkles className="w-4 h-4" />一鍵生成會議紀錄</>}
        </button>
      )}

      {/* Summary */}
      {/* Highlights — topic grouped key points */}
      {m.highlights && m.highlights.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <FileText className="w-4 h-4 text-terracotta" />
            <span className="text-[10px] font-bold uppercase tracking-widest text-terracotta">重點彙整</span>
          </div>
          {m.highlights.map((h, i) => (
            <div key={i} className="bg-white rounded-2xl border border-wood/10 overflow-hidden">
              <div className="px-4 py-2.5 bg-terracotta/5 border-b border-wood/5">
                <span className="text-xs font-bold text-forest">{h.topic}</span>
              </div>
              <div className="px-4 py-3 space-y-2">
                {h.points.map((p, j) => (
                  <div key={j} className="flex items-start gap-2.5">
                    <div className="w-1 h-1 rounded-full bg-wood/40 mt-1.5 shrink-0" />
                    <span className="text-sm font-serif text-forest/80 leading-relaxed">{p}</span>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Decisions */}
      {m.decisions && m.decisions.length > 0 && (
        <div>
          <div className="flex items-center gap-2 mb-2.5">
            <CheckCircle2 className="w-4 h-4 text-sage" />
            <span className="text-[10px] font-bold uppercase tracking-widest text-sage">決議事項</span>
          </div>
          <div className="space-y-2">
            {m.decisions.map((d, i) => (
              <div key={i} className="flex items-start gap-3 p-3.5 bg-sage/8 rounded-xl border border-sage/15">
                <CheckCircle2 className="w-4 h-4 text-sage mt-0.5 shrink-0" />
                <span className="font-serif text-forest/80 text-sm">{d}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Fallback: old summary format */}
      {(!m.highlights || m.highlights.length === 0) && m.summary && (
        <div>
          <div className="flex items-center gap-2 mb-2.5">
            <FileText className="w-4 h-4 text-terracotta" />
            <span className="text-[10px] font-bold uppercase tracking-widest text-terracotta">會議總結</span>
          </div>
          <div className="bg-white p-4 rounded-2xl border border-wood/10">
            <div className="markdown-body text-sm"><ReactMarkdown>{m.summary}</ReactMarkdown></div>
          </div>
        </div>
      )}

      {/* Action items */}
      {m.actionItems && m.actionItems.length > 0 && (
        <div>
          <div className="flex items-center gap-2 mb-2.5">
            <ListChecks className="w-4 h-4 text-forest" />
            <span className="text-[10px] font-bold uppercase tracking-widest text-forest">行動項目</span>
          </div>
          <div className="space-y-2">
            {m.actionItems.map((item, i) => (
              <div key={i} className="flex items-start gap-3 p-3.5 bg-white rounded-xl border border-wood/10">
                <div className="w-4 h-4 rounded border-2 border-wood/30 mt-0.5 shrink-0" />
                <span className="font-serif text-forest/80 text-sm">{item}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Transcript */}
      {m.transcript && (
        <div>
          <div className="flex items-center justify-between mb-2.5">
            <div className="flex items-center gap-2">
              <History className="w-4 h-4 text-wood" />
              <span className="text-[10px] font-bold uppercase tracking-widest text-wood">
                {m.summary ? '精華逐字稿' : '完整逐字稿'}
              </span>
            </div>
            <div className="flex items-center gap-1.5">
              <button
                onClick={() => copyTranscript(m.transcript!, m.id + '-mobile')}
                className="flex items-center gap-1 px-3 py-1.5 rounded-xl bg-white border border-wood/10 text-wood/70 active:bg-paper transition-all"
              >
                {copiedId === m.id + '-mobile'
                  ? <><Check className="w-3.5 h-3.5 text-sage" /><span className="text-[10px] font-bold">已複製</span></>
                  : <><Copy className="w-3.5 h-3.5" /><span className="text-[10px] font-bold">複製</span></>
                }
              </button>
              <button
                onClick={() => downloadTranscript(m.transcript!, m.title)}
                className="flex items-center gap-1 px-3 py-1.5 rounded-xl bg-white border border-wood/10 text-wood/70 active:bg-paper transition-all"
              >
                <Download className="w-3.5 h-3.5" /><span className="text-[10px] font-bold">.txt</span>
              </button>
            </div>
          </div>
          <div className="bg-white p-4 rounded-2xl border border-wood/10 max-h-64 overflow-y-auto">
            <p className="whitespace-pre-wrap text-forest/80 text-sm font-serif leading-relaxed">{m.transcript}</p>
          </div>
        </div>
      )}

      {/* Delete */}
      <button onClick={(e) => { deleteMeeting(m.id, e); setSelectedMeeting(null); }}
        className="w-full py-3.5 rounded-2xl border-2 border-terracotta/20 text-terracotta/80 font-bold text-sm flex items-center justify-center gap-2 active:bg-terracotta/5 transition-all"
      >
        <Trash2 className="w-4 h-4" />刪除此紀錄
      </button>
    </div>
  );

  return (
    <div className="min-h-screen bg-paper font-sans text-forest">

      {/* ════════════════════════════════════════════════════════════════════════
          DESKTOP  (md+)
      ════════════════════════════════════════════════════════════════════════ */}
      <div className="hidden md:flex min-h-screen">

        {/* Sidebar */}
        <aside className="w-80 bg-white border-r border-wood/5 flex flex-col h-screen sticky top-0 z-20">
          <div className="p-8 flex items-center gap-4">
            <Logo className="w-10 h-10" />
            <div>
              <div className="font-serif font-bold text-2xl tracking-tighter text-forest leading-none">SyncLingua</div>
              <div className="text-[8px] uppercase tracking-[0.4em] text-sage font-bold mt-1">Intelligence</div>
            </div>
          </div>
          <div className="px-6 py-4 space-y-4">
            <div className="space-y-2">
              <label className="text-[10px] uppercase tracking-widest text-sage font-bold px-1">會議背景提示 (選填)</label>
              <input type="text" placeholder="例如：LIZ學堂、財經..." value={contextHint} onChange={e => setContextHint(e.target.value)} disabled={isRecording}
                className="w-full px-4 py-3 bg-paper/30 border border-transparent focus:border-wood/10 rounded-xl text-xs focus:ring-0 transition-all placeholder:text-sage/40 disabled:opacity-50" />
            </div>
            <button onClick={isRecording ? stopRecording : startRecording} disabled={isProcessing && !isRecording}
              className={cn('w-full py-5 rounded-2xl font-medium flex flex-col items-center justify-center gap-2 transition-all relative overflow-hidden group',
                isRecording ? 'bg-terracotta text-white shadow-xl shadow-terracotta/20' : 'bg-forest text-white shadow-xl shadow-forest/20 hover:bg-forest/90',
                (isProcessing && !isRecording) && 'opacity-50 cursor-not-allowed')}
            >
              {isRecording
                ? <motion.div animate={{ scale:[1,1.2,1] }} transition={{ repeat: Infinity, duration: 1.5 }}><StopCircle className="w-6 h-6" /></motion.div>
                : <Mic className="w-6 h-6" />}
              <span className="text-xs uppercase tracking-widest font-bold">
                {isRecording ? `錄音中 ${format(recordingTime*1000,'mm:ss')}` : '啟動新會議錄製'}
              </span>
              {isRecording && chunksProcessed > 0 && (
                <span className="text-[9px] text-white/60 flex items-center gap-1">
                  <Layers className="w-3 h-3" />已完成 {chunksProcessed} 段
                  {isTranscribingChunk && <Loader2 className="w-3 h-3 animate-spin ml-1" />}
                </span>
              )}
            </button>
          </div>
          <div className="flex-1 overflow-y-auto px-4 py-4 space-y-3">
            <div className="relative">
              <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-sage" />
              <input type="text" placeholder="搜尋紀錄..." value={searchQuery} onChange={e => setSearchQuery(e.target.value)}
                className="w-full pl-12 pr-4 py-3 bg-paper/30 border border-transparent focus:border-wood/10 rounded-xl text-sm focus:ring-0 transition-all placeholder:text-sage/50" />
            </div>
            <div className="text-[9px] font-bold text-sage/60 uppercase tracking-[0.3em] px-2">Archive</div>
            {filtered.length === 0
              ? <div className="py-12 text-center text-sage/40 text-sm italic font-serif">尚無會議紀錄</div>
              : filtered.map(m => (
                <div key={m.id} onClick={() => setSelectedMeeting(m)} role="button" tabIndex={0}
                  onKeyDown={e => { if (e.key==='Enter') setSelectedMeeting(m); }}
                  className={cn('p-4 rounded-xl transition-all group flex items-start gap-3 border border-transparent cursor-pointer',
                    selectedMeeting?.id===m.id ? 'bg-paper border-wood/10 shadow-sm' : 'hover:bg-paper/40')}
                >
                  <div className={cn('mt-2 w-1.5 h-1.5 rounded-full shrink-0',
                    m.status==='completed'?'bg-forest': m.status==='recording'?'bg-sage animate-pulse': m.status==='error'?'bg-red-400':'bg-wood/20')} />
                  <div className="flex-1 min-w-0">
                    <div className="font-serif font-bold truncate text-base text-forest/90">{m.title}</div>
                    <div className="text-[10px] text-wood/60 mt-1 flex items-center gap-1 uppercase tracking-wider font-bold">
                      <Clock className="w-3 h-3" />{format(m.date.toDate(),'yyyy.MM.dd')}
                    </div>
                  </div>
                  <button onClick={e => deleteMeeting(m.id,e)}
                    className="opacity-0 group-hover:opacity-100 p-1.5 hover:bg-terracotta/10 hover:text-terracotta rounded-lg transition-all">
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              ))
            }
          </div>
          <div className="p-6 border-t border-wood/5">
            <div className="flex items-center gap-4">
              <img src={user.photoURL||`https://ui-avatars.com/api/?name=${user.displayName}`} alt="" referrerPolicy="no-referrer"
                className="w-10 h-10 rounded-full border-2 border-white shadow-md" />
              <div className="flex-1 min-w-0">
                <div className="text-sm font-bold text-forest truncate">{user.displayName}</div>
                <div className="text-[9px] text-wood/60 truncate uppercase tracking-widest font-bold">{user.email}</div>
              </div>
              <button onClick={() => logout()} className="p-2 hover:bg-terracotta/10 hover:text-terracotta rounded-xl transition-all">
                <LogOut className="w-4 h-4" />
              </button>
            </div>
          </div>
        </aside>

        {/* Desktop main */}
        <main className="flex-1 relative overflow-hidden">
          <AnimatePresence mode="wait">
            {selectedMeeting ? (
              <motion.div key={selectedMeeting.id}
                initial={{ opacity:0, x:30 }} animate={{ opacity:1, x:0 }} exit={{ opacity:0, x:-30 }}
                transition={{ duration:0.4, ease:[0.16,1,0.3,1] }}
                className="absolute inset-0 overflow-y-auto"
              >
                <div className="max-w-4xl mx-auto p-12 lg:p-20 pb-40">
                  <div className="flex items-start justify-between mb-16">
                    <div className="flex-1">
                      <div className="text-[10px] font-bold uppercase tracking-[0.3em] text-sage mb-3">
                        {format(selectedMeeting.date.toDate(), 'yyyy年MM月dd日 · EEEE')}
                      </div>
                      <h2 className="text-5xl font-serif font-bold text-forest tracking-tighter">{selectedMeeting.title}</h2>
                      {selectedMeeting.modelInfo && (
                        <div className="mt-4 inline-flex items-center gap-2 px-4 py-2 bg-sage/10 rounded-full">
                          <div className="w-1.5 h-1.5 rounded-full bg-sage" />
                          <span className="text-[10px] font-bold uppercase tracking-widest text-sage">{selectedMeeting.modelInfo}</span>
                        </div>
                      )}
                    </div>
                    <button onClick={() => setSelectedMeeting(null)} className="p-3 hover:bg-wood/5 rounded-2xl transition-all ml-6 shrink-0">
                      <X className="w-5 h-5 text-wood" />
                    </button>
                  </div>
                  {isRecording && selectedMeeting.id === currentMeetingIdRef.current && (
                    <motion.div initial={{ opacity:0, y:10 }} animate={{ opacity:1, y:0 }}
                      className="mb-10 p-8 bg-terracotta/5 border border-terracotta/20 rounded-[32px]"
                    >
                      <div className="flex items-center gap-3 mb-4">
                        <motion.div className="w-2 h-2 rounded-full bg-terracotta" animate={{ scale:[1,1.5,1] }} transition={{ repeat: Infinity, duration: 1 }} />
                        <span className="text-xs font-bold uppercase tracking-widest text-terracotta">錄音中</span>
                        {isTranscribingChunk && <span className="flex items-center gap-1 text-[10px] text-wood/60 ml-2"><Loader2 className="w-3 h-3 animate-spin" />轉錄中...</span>}
                      </div>
                      {liveTranscript
                        ? <div className="whitespace-pre-wrap text-forest/70 text-sm font-serif leading-relaxed max-h-60 overflow-y-auto">{liveTranscript}</div>
                        : <div className="text-wood/40 text-sm italic font-serif">錄音完成後，逐字稿將顯示於此...</div>}
                    </motion.div>
                  )}
                  {(selectedMeeting.transcript||liveTranscript) && !selectedMeeting.summary && !isRecording && (
                    <motion.div initial={{ opacity:0, y:10 }} animate={{ opacity:1, y:0 }} className="mb-10">
                      <button onClick={handleManualAnalysis} disabled={isProcessing}
                        className={cn('w-full py-6 rounded-[24px] font-bold text-lg flex items-center justify-center gap-3 bg-forest text-white shadow-xl shadow-forest/20 hover:scale-[1.01] active:scale-[0.99] transition-all', isProcessing&&'opacity-50 cursor-not-allowed')}
                      >
                        {isProcessing ? <><Loader2 className="w-5 h-5 animate-spin" />正在生成...</> : <><Sparkles className="w-5 h-5" />一鍵生成會議紀錄</>}
                      </button>
                    </motion.div>
                  )}
                  {/* Meeting analysis results */}
                  {(selectedMeeting.highlights?.length || selectedMeeting.summary) && (
                    <div className="space-y-12 mb-12">

                      {/* Highlights — topic grouped */}
                      {selectedMeeting.highlights && selectedMeeting.highlights.length > 0 && (
                        <section>
                          <div className="flex items-center gap-4 mb-8">
                            <FileText className="w-5 h-5 text-terracotta" />
                            <h4 className="font-bold uppercase text-[10px] tracking-[0.3em] text-terracotta">重點彙整</h4>
                          </div>
                          <div className="space-y-5">
                            {selectedMeeting.highlights.map((h, i) => (
                              <div key={i} className="bg-white rounded-[28px] border border-wood/10 shadow-sm overflow-hidden">
                                <div className="px-8 py-4 bg-terracotta/5 border-b border-wood/5">
                                  <span className="font-serif font-bold text-base text-forest">{h.topic}</span>
                                </div>
                                <div className="px-8 py-5 space-y-3">
                                  {h.points.map((p, j) => (
                                    <div key={j} className="flex items-start gap-4">
                                      <div className="w-1.5 h-1.5 rounded-full bg-terracotta/50 mt-2.5 shrink-0" />
                                      <span className="font-serif text-forest/80 leading-relaxed">{p}</span>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            ))}
                          </div>
                        </section>
                      )}

                      {/* Decisions */}
                      {selectedMeeting.decisions && selectedMeeting.decisions.length > 0 && (
                        <section>
                          <div className="flex items-center gap-4 mb-8">
                            <CheckCircle2 className="w-5 h-5 text-sage" />
                            <h4 className="font-bold uppercase text-[10px] tracking-[0.3em] text-sage">決議事項</h4>
                          </div>
                          <div className="space-y-3">
                            {selectedMeeting.decisions.map((d, i) => (
                              <div key={i} className="flex items-start gap-4 p-6 bg-sage/5 rounded-2xl border border-sage/15">
                                <CheckCircle2 className="w-5 h-5 text-sage mt-0.5 shrink-0" />
                                <span className="font-serif text-forest/80">{d}</span>
                              </div>
                            ))}
                          </div>
                        </section>
                      )}

                      {/* Action items */}
                      {selectedMeeting.actionItems && selectedMeeting.actionItems.length > 0 && (
                        <section>
                          <div className="flex items-center gap-4 mb-8">
                            <ListChecks className="w-5 h-5 text-forest" />
                            <h4 className="font-bold uppercase text-[10px] tracking-[0.3em] text-forest">行動項目</h4>
                          </div>
                          <div className="space-y-3">
                            {selectedMeeting.actionItems.map((item, i) => (
                              <div key={i} className="flex items-start gap-4 p-6 bg-white rounded-2xl border border-wood/10">
                                <div className="w-5 h-5 rounded border-2 border-wood/30 mt-0.5 shrink-0" />
                                <span className="font-serif text-forest/80">{item}</span>
                              </div>
                            ))}
                          </div>
                        </section>
                      )}

                      {/* Fallback: old summary format for existing records */}
                      {(!selectedMeeting.highlights || selectedMeeting.highlights.length === 0) && selectedMeeting.summary && (
                        <section>
                          <div className="flex items-center gap-4 mb-8"><FileText className="w-5 h-5 text-terracotta" /><h4 className="font-bold uppercase text-[10px] tracking-[0.3em] text-terracotta">會議總結</h4></div>
                          <div className="bg-white p-12 rounded-[40px] border border-wood/10 shadow-sm"><div className="markdown-body"><ReactMarkdown>{selectedMeeting.summary}</ReactMarkdown></div></div>
                        </section>
                      )}
                    </div>
                  )}
                  {selectedMeeting.transcript && (
                    <section>
                      <div className="flex items-center justify-between mb-8">
                        <div className="flex items-center gap-4"><History className="w-5 h-5 text-wood" /><h4 className="font-bold uppercase text-[10px] tracking-[0.3em] text-wood">{selectedMeeting.summary?'精華逐字稿':'完整逐字稿'}</h4></div>
                        <div className="flex items-center gap-2">
                          <button
                            onClick={() => copyTranscript(selectedMeeting.transcript!, selectedMeeting.id + '-desktop')}
                            className="flex items-center gap-1.5 px-4 py-2 rounded-xl border border-wood/10 bg-white hover:bg-paper text-wood/70 hover:text-forest text-xs font-bold uppercase tracking-wider transition-all"
                          >
                            {copiedId === selectedMeeting.id + '-desktop' ? <><Check className="w-3.5 h-3.5 text-sage" />已複製</> : <><Copy className="w-3.5 h-3.5" />複製</>}
                          </button>
                          <button
                            onClick={() => downloadTranscript(selectedMeeting.transcript!, selectedMeeting.title)}
                            className="flex items-center gap-1.5 px-4 py-2 rounded-xl border border-wood/10 bg-white hover:bg-paper text-wood/70 hover:text-forest text-xs font-bold uppercase tracking-wider transition-all"
                          >
                            <Download className="w-3.5 h-3.5" />下載 .txt
                          </button>
                        </div>
                      </div>
                      <div className="bg-white p-12 rounded-[40px] border border-wood/10"><div className="whitespace-pre-wrap text-forest/80 leading-relaxed text-base font-serif">{selectedMeeting.transcript}</div></div>
                    </section>
                  )}
                </div>
              </motion.div>
            ) : (
              <motion.div key="empty" initial={{ opacity:0 }} animate={{ opacity:1 }}
                className="flex flex-col items-center justify-center min-h-full p-20 text-center"
              >
                <Logo className="w-24 h-24 mx-auto mb-8" />
                <h2 className="text-6xl font-serif font-bold tracking-tighter mb-4 text-forest">Sync<span className="text-terracotta italic">Lingua</span></h2>
                <p className="text-wood/60 text-lg font-serif italic">從左側啟動錄音，或選取一筆會議紀錄。</p>
              </motion.div>
            )}
          </AnimatePresence>
          <AnimatePresence>
            {isProcessing && !isRecording && (
              <motion.div initial={{ opacity:0 }} animate={{ opacity:1 }} exit={{ opacity:0 }}
                className="fixed inset-0 bg-forest/40 backdrop-blur-xl z-50 flex items-center justify-center p-6"
              >
                <div className="bg-white p-20 rounded-[64px] shadow-2xl max-w-xl w-full text-center border border-wood/10 relative overflow-hidden">
                  <div className="absolute top-0 left-0 w-full h-2 bg-paper">
                    <motion.div initial={{ x:'-100%' }} animate={{ x:'100%' }} transition={{ repeat: Infinity, duration: 2, ease:'easeInOut' }} className="w-full h-full bg-terracotta" />
                  </div>
                  <Logo className="w-20 h-20 mx-auto mb-10" />
                  <h3 className="text-5xl font-serif font-bold mb-6 text-forest tracking-tight">正在編織紀錄...</h3>
                  <p className="text-wood/80 font-serif italic text-xl">正在完成最後一段轉錄，請稍候。</p>
                  <div className="mt-12 flex justify-center gap-2">
                    {[0,1,2].map(i => <motion.div key={i} className="w-2 h-2 rounded-full bg-sage" animate={{ scale:[1,1.5,1], opacity:[0.3,1,0.3] }} transition={{ repeat: Infinity, duration:1, delay:i*0.2 }} />)}
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </main>
      </div>

      {/* ════════════════════════════════════════════════════════════════════════
          MOBILE  (below md)
      ════════════════════════════════════════════════════════════════════════ */}
      <div className="flex flex-col md:hidden min-h-[100dvh]">

        {/* Top bar */}
        <header className="sticky top-0 z-30 bg-paper/92 backdrop-blur-md border-b border-wood/5 px-5 safe-top" style={{ paddingTop: 'env(safe-area-inset-top)' }}>
          <div className="flex items-center justify-between py-3">
            <div className="flex items-center gap-2.5">
              <Logo className="w-7 h-7" />
              <span className="font-serif font-bold text-xl tracking-tighter text-forest">SyncLingua</span>
            </div>
            {isRecording && (
              <motion.div animate={{ opacity:[1,0.4,1] }} transition={{ repeat: Infinity, duration: 1.2 }}
                className="flex items-center gap-1.5 px-3 py-1 bg-terracotta/10 rounded-full"
              >
                <div className="w-2 h-2 rounded-full bg-terracotta" />
                <span className="text-[11px] font-bold text-terracotta uppercase tracking-wider">{format(recordingTime*1000,'mm:ss')}</span>
              </motion.div>
            )}
          </div>
        </header>

        {/* Tab content */}
        <div className="flex-1 overflow-hidden relative">
          <AnimatePresence mode="wait">

            {/* ─── RECORD tab ─── */}
            {activeTab === 'record' && (
              <motion.div key="record" initial={{ opacity:0 }} animate={{ opacity:1 }} exit={{ opacity:0 }}
                transition={{ duration:0.2 }} className="absolute inset-0 overflow-y-auto pb-24"
              >
                {isRecording ? (
                  /* FULLSCREEN RECORDING */
                  <div className="min-h-full flex flex-col bg-forest relative overflow-hidden">
                    <div className="absolute inset-0 pointer-events-none">
                      <div className="absolute -top-20 -right-20 w-80 h-80 bg-sage/20 blur-[100px] rounded-full" />
                      <div className="absolute bottom-10 -left-20 w-80 h-80 bg-terracotta/15 blur-[100px] rounded-full" />
                    </div>
                    <div className="relative z-10 flex flex-col items-center justify-center flex-1 px-8 gap-7 py-10">
                      {/* Timer */}
                      <div className="text-center">
                        <div className="text-7xl font-serif font-bold text-white tracking-tighter tabular-nums leading-none">
                          {format(recordingTime*1000,'mm:ss')}
                        </div>
                        <div className="text-white/40 text-xs uppercase tracking-[0.3em] mt-3 font-bold">正在錄音</div>
                      </div>
                      {/* Waveform */}
                      <Waveform volume={audioVolume} active={isRecording} />
                      {/* Status badges */}
                      <div className="flex flex-col items-center gap-3">
                        {contextHint && (
                          <div className="px-4 py-2 bg-white/10 rounded-full">
                            <span className="text-white/60 text-xs font-serif italic">{contextHint}</span>
                          </div>
                        )}
                        {chunksProcessed > 0 && (
                          <motion.div initial={{ opacity:0, scale:0.9 }} animate={{ opacity:1, scale:1 }}
                            className="flex items-center gap-2 px-4 py-2 bg-white/10 rounded-full"
                          >
                            <Layers className="w-3.5 h-3.5 text-white/60" />
                            <span className="text-white/60 text-xs font-bold">已完成 {chunksProcessed} 段轉錄</span>
                            {isTranscribingChunk && <Loader2 className="w-3.5 h-3.5 text-white/50 animate-spin" />}
                          </motion.div>
                        )}
                      </div>
                      {/* Live transcript peek */}
                      {liveTranscript && (
                        <div className="w-full bg-white/8 rounded-2xl p-4 max-h-28 overflow-hidden relative">
                          <div className="text-[10px] font-bold uppercase tracking-widest text-white/40 mb-1.5">即時逐字稿</div>
                          <p className="text-white/60 text-xs font-serif leading-relaxed line-clamp-3">{liveTranscript}</p>
                          <div className="absolute bottom-0 inset-x-0 h-8 bg-gradient-to-t from-forest/80 to-transparent rounded-b-2xl" />
                        </div>
                      )}
                      {/* STOP button — large, thumb-friendly */}
                      <button onClick={stopRecording}
                        className="w-28 h-28 bg-white rounded-full flex flex-col items-center justify-center shadow-2xl shadow-black/30 active:scale-90 transition-transform mt-2"
                      >
                        <div className="w-9 h-9 bg-terracotta rounded-xl" />
                        <span className="text-[11px] text-terracotta font-bold uppercase tracking-wider mt-2">停止</span>
                      </button>
                    </div>
                  </div>
                ) : (
                  /* IDLE */
                  <div className="flex flex-col items-center px-5 pt-8 gap-6">
                    <div className="text-center">
                      <h2 className="text-3xl font-serif font-bold tracking-tighter text-forest">準備錄音</h2>
                      <p className="text-wood/60 text-sm font-serif italic mt-1.5">按下按鈕即可開始錄音</p>
                    </div>

                    {/* Context hint accordion */}
                    <div className="w-full max-w-sm">
                      <button onClick={() => setShowContextInput(!showContextInput)}
                        className="w-full flex items-center justify-between px-5 py-3.5 bg-white rounded-2xl border border-wood/10 shadow-sm"
                      >
                        <div className="flex-1 min-w-0 text-left">
                          <div className="text-[10px] font-bold uppercase tracking-widest text-sage">會議背景（選填）</div>
                          <div className={cn('text-sm mt-0.5 truncate', contextHint ? 'text-forest font-serif italic' : 'text-sage/40')}>
                            {contextHint || '輸入以提升轉錄精準度...'}
                          </div>
                        </div>
                        <ChevronDown className={cn('w-4 h-4 text-sage/50 transition-transform ml-2 shrink-0', showContextInput && 'rotate-180')} />
                      </button>
                      <AnimatePresence>
                        {showContextInput && (
                          <motion.div initial={{ height:0, opacity:0 }} animate={{ height:'auto', opacity:1 }}
                            exit={{ height:0, opacity:0 }} transition={{ duration:0.2 }} className="overflow-hidden mt-2"
                          >
                            <input type="text" autoFocus placeholder="例如：產品週會、客戶訪談..."
                              value={contextHint} onChange={e => setContextHint(e.target.value)}
                              className="w-full px-5 py-3.5 bg-white rounded-2xl border border-wood/10 text-sm focus:ring-0 focus:border-sage/30 placeholder:text-sage/40 font-serif"
                            />
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </div>

                    {/* Big record button */}
                    <button onClick={startRecording} disabled={isProcessing}
                      className={cn('w-40 h-40 rounded-full flex flex-col items-center justify-center shadow-2xl shadow-forest/25 bg-forest text-white active:scale-95 transition-transform', isProcessing && 'opacity-50 cursor-not-allowed')}
                    >
                      <Mic className="w-14 h-14 mb-1" />
                      <span className="text-xs font-bold uppercase tracking-widest">開始錄音</span>
                    </button>

                    {/* Feature list */}
                    <div className="w-full max-w-sm space-y-2.5">
                      {[
                        { icon: Layers, title: '智慧分段', desc: '自動切段上傳，支援長時間錄音' },
                        { icon: FileText, title: '即時轉錄', desc: 'Whisper 邊錄邊轉' },
                        { icon: Sparkles, title: '一鍵紀錄', desc: 'GPT-4o 生成摘要與行動項目' },
                      ].map((f,i) => (
                        <div key={i} className="flex items-center gap-4 p-4 bg-white rounded-2xl border border-wood/5 shadow-sm">
                          <div className="w-10 h-10 rounded-xl bg-paper flex items-center justify-center shrink-0">
                            <f.icon className="w-5 h-5 text-forest" />
                          </div>
                          <div>
                            <div className="font-serif font-bold text-sm text-forest">{f.title}</div>
                            <div className="text-wood/60 text-xs font-serif italic">{f.desc}</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </motion.div>
            )}

            {/* ─── HISTORY tab ─── */}
            {activeTab === 'history' && (
              <motion.div key="history" initial={{ opacity:0 }} animate={{ opacity:1 }} exit={{ opacity:0 }}
                transition={{ duration:0.2 }} className="absolute inset-0 overflow-y-auto pb-24"
              >
                <div className="px-5 pt-5 space-y-3">
                  <h2 className="font-serif font-bold text-2xl text-forest">會議記錄</h2>
                  <div className="relative">
                    <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-sage" />
                    <input type="text" placeholder="搜尋紀錄..." value={searchQuery} onChange={e => setSearchQuery(e.target.value)}
                      className="w-full pl-11 pr-4 py-3 bg-white rounded-xl border border-wood/5 shadow-sm text-sm focus:ring-0 focus:border-sage/20 placeholder:text-sage/50" />
                  </div>
                  {filtered.length === 0 ? (
                    <div className="py-20 text-center">
                      <BookOpen className="w-12 h-12 text-sage/25 mx-auto mb-4" />
                      <div className="text-sage/50 text-sm font-serif italic">尚無會議紀錄</div>
                    </div>
                  ) : filtered.map(m => (
                    <button key={m.id} onClick={() => setSelectedMeeting(m)}
                      className="w-full text-left p-4 bg-white rounded-2xl border border-wood/5 shadow-sm active:bg-paper/60 transition-all"
                    >
                      <div className="flex items-start gap-3">
                        <div className={cn('mt-1.5 w-2 h-2 rounded-full shrink-0',
                          m.status==='completed'?'bg-forest': m.status==='recording'?'bg-sage animate-pulse': m.status==='error'?'bg-red-400':'bg-wood/20')} />
                        <div className="flex-1 min-w-0">
                          <div className="font-serif font-bold text-base text-forest truncate">{m.title}</div>
                          <div className="text-[11px] text-wood/60 mt-1 flex items-center gap-1.5">
                            <Clock className="w-3 h-3" />{format(m.date.toDate(),'yyyy年MM月dd日 HH:mm')}
                          </div>
                          {typeof m.summary === 'string' && m.summary && (
                            <p className="text-xs text-wood/50 mt-2 font-serif italic line-clamp-2">
                              {m.summary.replace(/[#*`]/g,'').substring(0,80)}...
                            </p>
                          )}
                          {m.contextHint && (
                            <span className="inline-block mt-2 px-2.5 py-0.5 bg-sage/10 rounded-full text-[10px] text-sage font-bold uppercase tracking-wider">
                              {m.contextHint}
                            </span>
                          )}
                        </div>
                      </div>
                    </button>
                  ))}
                </div>
              </motion.div>
            )}

            {/* ─── PROFILE tab ─── */}
            {activeTab === 'profile' && (
              <motion.div key="profile" initial={{ opacity:0 }} animate={{ opacity:1 }} exit={{ opacity:0 }}
                transition={{ duration:0.2 }} className="absolute inset-0 overflow-y-auto pb-24"
              >
                <div className="px-5 pt-5 space-y-4">
                  <h2 className="font-serif font-bold text-2xl text-forest">我的帳號</h2>
                  <div className="bg-white rounded-2xl border border-wood/5 shadow-sm p-5">
                    <div className="flex items-center gap-4">
                      <img src={user.photoURL||`https://ui-avatars.com/api/?name=${user.displayName}`} alt="" referrerPolicy="no-referrer"
                        className="w-14 h-14 rounded-full border-2 border-white shadow-md" />
                      <div>
                        <div className="font-bold text-forest text-base">{user.displayName}</div>
                        <div className="text-wood/60 text-xs mt-0.5">{user.email}</div>
                      </div>
                    </div>
                  </div>
                  <div className="bg-white rounded-2xl border border-wood/5 shadow-sm overflow-hidden">
                    <div className="px-5 py-3.5 border-b border-wood/5">
                      <div className="text-[10px] font-bold uppercase tracking-widest text-sage">使用統計</div>
                    </div>
                    <div className="grid grid-cols-2 divide-x divide-wood/5">
                      <div className="px-5 py-5 text-center">
                        <div className="text-4xl font-serif font-bold text-forest">{meetings.length}</div>
                        <div className="text-xs text-wood/60 mt-1">會議紀錄</div>
                      </div>
                      <div className="px-5 py-5 text-center">
                        <div className="text-4xl font-serif font-bold text-forest">{meetings.filter(m=>m.summary).length}</div>
                        <div className="text-xs text-wood/60 mt-1">已生成摘要</div>
                      </div>
                    </div>
                  </div>
                  <button onClick={() => logout()}
                    className="w-full py-4 rounded-2xl border-2 border-terracotta/20 text-terracotta font-bold text-sm flex items-center justify-center gap-2 active:bg-terracotta/5 transition-all"
                  >
                    <LogOut className="w-4 h-4" />登出
                  </button>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* ── Bottom navigation ── */}
        <nav className="fixed bottom-0 left-0 right-0 z-40 bg-white/95 backdrop-blur-md border-t border-wood/5"
          style={{ paddingBottom: 'env(safe-area-inset-bottom)' }}>
          <div className="flex items-center justify-around px-2 pt-2 pb-2">
            {([
              { key:'record', label:'錄音', icon: <Mic className="w-5 h-5" /> },
              { key:'history', label:'記錄', icon: <BookOpen className="w-5 h-5" />, badge: meetings.length },
              { key:'profile', label:'我的', icon: (
                <img src={user.photoURL||`https://ui-avatars.com/api/?name=${user.displayName}`} alt="" referrerPolicy="no-referrer"
                  className={cn('w-5 h-5 rounded-full border-2 transition-all', activeTab==='profile'?'border-forest':'border-transparent')} />
              )},
            ] as const).map(tab => (
              <button key={tab.key} onClick={() => setActiveTab(tab.key as any)}
                className={cn('flex flex-col items-center gap-1 flex-1 py-2 rounded-xl transition-all relative', activeTab===tab.key?'text-forest':'text-sage/50')}
              >
                {tab.icon}
                <span className="text-[10px] font-bold uppercase tracking-wider">{tab.label}</span>
                {'badge' in tab && tab.badge > 0 && (
                  <span className="absolute top-1.5 right-4 min-w-[16px] h-4 bg-terracotta rounded-full text-[9px] text-white font-bold flex items-center justify-center px-1">
                    {tab.badge > 99 ? '99' : tab.badge}
                  </span>
                )}
              </button>
            ))}
          </div>
        </nav>

        {/* ── Meeting detail bottom sheet ── */}
        <AnimatePresence>
          {selectedMeeting && (
            <>
              <motion.div initial={{ opacity:0 }} animate={{ opacity:1 }} exit={{ opacity:0 }}
                className="fixed inset-0 z-50 bg-forest/30 backdrop-blur-sm"
                onClick={() => setSelectedMeeting(null)} />
              <motion.div
                initial={{ y:'100%' }} animate={{ y:0 }} exit={{ y:'100%' }}
                transition={{ type:'spring', damping:32, stiffness:320 }}
                className="fixed bottom-0 left-0 right-0 z-50 bg-paper rounded-t-[28px] shadow-2xl overflow-hidden"
                style={{ maxHeight:'92dvh' }}
              >
                <div className="flex justify-center pt-3 pb-0.5">
                  <div className="w-10 h-1 bg-wood/20 rounded-full" />
                </div>
                <div className="overflow-y-auto" style={{ maxHeight:'calc(92dvh - 20px)' }}>
                  <MeetingDetailContent m={selectedMeeting} />
                </div>
              </motion.div>
            </>
          )}
        </AnimatePresence>

        {/* Mobile processing overlay */}
        <AnimatePresence>
          {isProcessing && !isRecording && (
            <motion.div initial={{ opacity:0 }} animate={{ opacity:1 }} exit={{ opacity:0 }}
              className="fixed inset-0 z-[60] bg-forest/60 backdrop-blur-xl flex items-end justify-center pb-10 px-6"
            >
              <motion.div initial={{ y:60, opacity:0 }} animate={{ y:0, opacity:1 }}
                className="bg-white rounded-3xl p-8 w-full max-w-sm text-center shadow-2xl"
              >
                <motion.div animate={{ rotate:360 }} transition={{ repeat: Infinity, duration: 2.5, ease:'linear' }}
                  className="w-12 h-12 border-4 border-forest/10 border-t-forest rounded-full mx-auto mb-5" />
                <div className="font-serif font-bold text-xl text-forest mb-2">正在編織紀錄</div>
                <div className="text-wood/60 text-sm font-serif italic">正在完成最後一段轉錄...</div>
                <div className="mt-5 flex justify-center gap-1.5">
                  {[0,1,2].map(i => (
                    <motion.div key={i} className="w-1.5 h-1.5 rounded-full bg-sage"
                      animate={{ scale:[1,1.5,1], opacity:[0.3,1,0.3] }}
                      transition={{ repeat: Infinity, duration:1, delay:i*0.2 }} />
                  ))}
                </div>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <div id="error-portal" />
    </div>
  );
}
