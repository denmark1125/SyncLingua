/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 *
 * KEY CHANGES vs original:
 * - Chunked recording: every CHUNK_INTERVAL_MS (3 min) the recorder auto-restarts,
 *   the previous chunk is immediately sent to Whisper/Gemini for transcription,
 *   and the result is appended to liveTranscript shown in real time.
 * - stopRecording() drains the final (possibly short) chunk before finishing.
 * - handleAudioProcessing() is now only called with a SINGLE chunk at a time.
 * - "一鍵生成會議紀錄" (handleManualAnalysis) reads the accumulated full transcript.
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  auth,
  loginWithGoogle,
  logout,
  db,
  collection,
  doc,
  setDoc,
  onSnapshot,
  query,
  where,
  orderBy,
  addDoc,
  updateDoc,
  deleteDoc,
  Timestamp,
  onAuthStateChanged,
  User,
  OperationType,
  handleFirestoreError
} from './firebase';
import {
  Mic,
  StopCircle,
  FileText,
  ListChecks,
  History,
  Plus,
  Trash2,
  LogOut,
  CheckCircle2,
  Clock,
  AlertCircle,
  ChevronRight,
  User as UserIcon,
  Search,
  MoreVertical,
  X,
  Loader2,
  Layers
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';
import { format } from 'date-fns';
import ReactMarkdown from 'react-markdown';
import {
  transcribeChunk,
  mergeChunkTranscripts,
  analyzeTranscript,
  WHISPER_SAFE_SIZE_BYTES
} from './services/geminiService';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// ─── Constants ────────────────────────────────────────────────────────────────
/** Auto-split recording every N ms. 3 min = safe under Whisper 25MB limit for most setups. */
const CHUNK_INTERVAL_MS = 3 * 60 * 1000;

// ─── Types ────────────────────────────────────────────────────────────────────
interface Meeting {
  id: string;
  userId: string;
  title: string;
  date: Timestamp;
  duration?: number;
  transcript?: string;
  rawTranscript?: string;
  summary?: string;
  actionItems?: string[];
  modelInfo?: string;
  contextHint?: string;
  status: 'recording' | 'processing' | 'completed' | 'error';
}

// ─── Logo ─────────────────────────────────────────────────────────────────────
const Logo = ({ className }: { className?: string }) => (
  <svg viewBox="0 0 100 100" className={cn('w-12 h-12', className)} fill="none" xmlns="http://www.w3.org/2000/svg">
    <circle cx="50" cy="50" r="45" className="stroke-wood/10" strokeWidth="0.5" />
    <circle cx="50" cy="50" r="35" className="stroke-wood/20" strokeWidth="0.5" />
    <circle cx="50" cy="50" r="25" className="stroke-wood/30" strokeWidth="0.5" />
    <motion.path
      d="M30 50 Q 40 20, 50 50 T 70 50"
      className="stroke-forest"
      strokeWidth="2"
      strokeLinecap="round"
      animate={{ d: ['M30 50 Q 40 20, 50 50 T 70 50', 'M30 50 Q 40 80, 50 50 T 70 50', 'M30 50 Q 40 20, 50 50 T 70 50'] }}
      transition={{ repeat: Infinity, duration: 4, ease: 'easeInOut' }}
    />
    <motion.path
      d="M35 50 Q 45 35, 50 50 T 65 50"
      className="stroke-terracotta"
      strokeWidth="1.5"
      strokeLinecap="round"
      animate={{ d: ['M35 50 Q 45 35, 50 50 T 65 50', 'M35 50 Q 45 65, 50 50 T 65 50', 'M35 50 Q 45 35, 50 50 T 65 50'] }}
      transition={{ repeat: Infinity, duration: 3, ease: 'easeInOut', delay: 0.5 }}
    />
    <motion.circle
      cx="50" cy="50" r="4"
      className="fill-sage"
      animate={{ scale: [1, 1.5, 1], opacity: [0.5, 1, 0.5] }}
      transition={{ repeat: Infinity, duration: 2 }}
    />
  </svg>
);

// ─── Main App ─────────────────────────────────────────────────────────────────
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

  // ── Chunked recording state ──────────────────────────────────────────────
  /** Accumulated transcript from all completed chunks (real-time display) */
  const [liveTranscript, setLiveTranscript] = useState('');
  /** Number of chunks processed so far */
  const [chunksProcessed, setChunksProcessed] = useState(0);
  /** Whether a chunk is currently being transcribed (background) */
  const [isTranscribingChunk, setIsTranscribingChunk] = useState(false);
  /** Firestore meeting ID for the ongoing session */
  const currentMeetingIdRef = useRef<string>('');
  /** All raw chunk transcripts in order */
  const chunkTranscriptsRef = useRef<string[]>([]);
  /** Model used (to show in UI) */
  const transcribeModelRef = useRef<string>('');

  // ── MediaRecorder refs ───────────────────────────────────────────────────
  const streamRef = useRef<MediaStream | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const chunkTimerRef = useRef<NodeJS.Timeout | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const mimeTypeRef = useRef<string>('audio/webm');
  /** Track current chunk index for timestamp offset calculation */
  const chunkIndexRef = useRef(0);

  // ── Auth ─────────────────────────────────────────────────────────────────
  useEffect(() => {
    const unsub = onAuthStateChanged(auth, (currentUser) => {
      setUser(currentUser);
      setIsAuthReady(true);
      if (currentUser) {
        const userRef = doc(db, 'users', currentUser.uid);
        setDoc(userRef, {
          uid: currentUser.uid,
          email: currentUser.email,
          displayName: currentUser.displayName,
          photoURL: currentUser.photoURL,
          createdAt: Timestamp.now()
        }, { merge: true }).catch(err => handleFirestoreError(err, OperationType.WRITE, `users/${currentUser.uid}`));
      }
    });
    return unsub;
  }, []);

  // ── Meetings listener ────────────────────────────────────────────────────
  useEffect(() => {
    if (!user) { setMeetings([]); return; }
    const q = query(
      collection(db, 'meetings'),
      where('userId', '==', user.uid),
      orderBy('date', 'desc')
    );
    const unsub = onSnapshot(q, (snap) => {
      setMeetings(snap.docs.map(d => ({ id: d.id, ...d.data() })) as Meeting[]);
    }, (err) => handleFirestoreError(err, OperationType.LIST, 'meetings'));
    return unsub;
  }, [user]);

  // ── Recording timer ──────────────────────────────────────────────────────
  useEffect(() => {
    if (isRecording) {
      timerRef.current = setInterval(() => setRecordingTime(prev => prev + 1), 1000);
    } else {
      if (timerRef.current) clearInterval(timerRef.current);
      setRecordingTime(0);
    }
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [isRecording]);

  // ── Core: flush one MediaRecorder cycle as a chunk ───────────────────────
  /**
   * Called when a MediaRecorder stops (either by auto-rotation or final stop).
   * Converts collected audio data to a Blob, sends to Whisper/Gemini,
   * and appends result to liveTranscript.
   */
  const flushChunk = useCallback(async (chunks: Blob[], chunkIdx: number, mimeType: string, isFinal: boolean) => {
    if (chunks.length === 0) return;

    const audioBlob = new Blob(chunks, { type: mimeType });
    console.log(`Flushing chunk ${chunkIdx}: ${audioBlob.size} bytes, final=${isFinal}`);

    // Safety: if somehow > safe size, log a warning (shouldn't happen with 3-min chunks)
    if (audioBlob.size > WHISPER_SAFE_SIZE_BYTES) {
      console.warn(`Chunk ${chunkIdx} exceeds 24MB (${audioBlob.size} bytes). Transcription may fail.`);
    }

    setIsTranscribingChunk(true);
    try {
      const { transcript, modelInfo } = await transcribeChunk(audioBlob, chunkIdx, contextHint);
      transcribeModelRef.current = modelInfo;

      // Store raw chunk transcript
      chunkTranscriptsRef.current[chunkIdx] = transcript;

      // Build cumulative transcript with adjusted timestamps
      const fullTranscript = mergeChunkTranscripts(
        chunkTranscriptsRef.current,
        CHUNK_INTERVAL_MS / 1000
      );

      setLiveTranscript(fullTranscript);
      setChunksProcessed(prev => prev + 1);

      // Persist to Firestore incrementally
      if (currentMeetingIdRef.current) {
        await updateDoc(doc(db, 'meetings', currentMeetingIdRef.current), {
          rawTranscript: fullTranscript,
          transcript: fullTranscript,
          modelInfo,
          status: isFinal ? 'completed' : 'recording'
        });
      }

      if (isFinal) {
        // Update selectedMeeting so the detail view refreshes
        setSelectedMeeting(prev => prev
          ? { ...prev, rawTranscript: fullTranscript, transcript: fullTranscript, modelInfo, status: 'completed' }
          : null
        );
        setIsTranscribingChunk(false);
        setIsProcessing(false);
      }
    } catch (err) {
      console.error(`Chunk ${chunkIdx} transcription error:`, err);
      if (isFinal) {
        if (currentMeetingIdRef.current) {
          await updateDoc(doc(db, 'meetings', currentMeetingIdRef.current), {
            status: 'error',
            transcript: '轉錄過程中發生錯誤，請檢查網路連線或稍後再試。'
          });
        }
        setIsTranscribingChunk(false);
        setIsProcessing(false);
      }
    }
  }, [contextHint]);

  // ── Start a fresh MediaRecorder segment ─────────────────────────────────
  const startNewSegment = useCallback(() => {
    if (!streamRef.current) return;

    const mimeType = mimeTypeRef.current;
    const recorder = new MediaRecorder(streamRef.current, { mimeType });
    mediaRecorderRef.current = recorder;
    audioChunksRef.current = [];

    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) audioChunksRef.current.push(e.data);
    };

    recorder.onstop = async () => {
      // This fires EITHER from chunkTimer rotation OR from final stopRecording()
      // We determine which by checking isRecording state via a ref isn't reliable after
      // async; instead we pass intent via a closure variable set before calling .stop()
    };

    recorder.start(1000); // collect every second
  }, []);

  // ── Start recording ──────────────────────────────────────────────────────
  const startRecording = async () => {
    try {
      // Reset state
      chunkTranscriptsRef.current = [];
      chunkIndexRef.current = 0;
      setLiveTranscript('');
      setChunksProcessed(0);
      transcribeModelRef.current = '';
      currentMeetingIdRef.current = '';

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        }
      });
      streamRef.current = stream;

      // Volume monitoring
      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      const source = audioContext.createMediaStreamSource(stream);
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);
      audioContextRef.current = audioContext;
      analyserRef.current = analyser;

      const bufferLength = analyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);
      const updateVolume = () => {
        if (!analyserRef.current) return;
        analyserRef.current.getByteFrequencyData(dataArray);
        const avg = dataArray.reduce((a, b) => a + b, 0) / bufferLength;
        setAudioVolume(avg);
        animationFrameRef.current = requestAnimationFrame(updateVolume);
      };
      updateVolume();

      mimeTypeRef.current = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : 'audio/webm';

      setIsRecording(true);

      // Create Firestore record immediately
      if (user) {
        const meetingData: Omit<Meeting, 'id'> = {
          userId: user.uid,
          title: `${format(new Date(), 'yyyy年MM月dd日 HH:mm')} 的錄音`,
          date: Timestamp.now(),
          contextHint,
          status: 'recording'
        };
        const docRef = await addDoc(collection(db, 'meetings'), meetingData);
        currentMeetingIdRef.current = docRef.id;
        setSelectedMeeting({ ...meetingData, id: docRef.id });
      }

      // ── Start first segment ──────────────────────────────────────────────
      const mimeType = mimeTypeRef.current;

      const launchSegment = () => {
        const idx = chunkIndexRef.current;
        const recorder = new MediaRecorder(stream, { mimeType });
        mediaRecorderRef.current = recorder;
        const localChunks: Blob[] = [];

        recorder.ondataavailable = (e) => {
          if (e.data.size > 0) localChunks.push(e.data);
        };

        recorder.onstop = () => {
          // Check if we should keep going (rotation) or wrap up (final stop)
          // We communicate intent via chunkIndexRef:
          // if isRecordingRef.current is still true → rotation, else → final
          const isFinalChunk = !isRecordingRef.current;
          flushChunk(localChunks, idx, mimeType, isFinalChunk);
        };

        recorder.start(1000);

        // Schedule auto-rotation
        chunkTimerRef.current = setTimeout(() => {
          if (mediaRecorderRef.current?.state === 'recording') {
            chunkIndexRef.current += 1;
            recorder.stop();
            // After onstop fires and flushChunk begins (async),
            // start the next segment immediately
            setTimeout(launchSegment, 100);
          }
        }, CHUNK_INTERVAL_MS);
      };

      launchSegment();

    } catch (error) {
      console.error('Error starting recording:', error);
      alert('無法存取麥克風。請檢查權限設定。');
    }
  };

  // ── isRecording ref (to communicate with onstop closure) ────────────────
  const isRecordingRef = useRef(false);
  useEffect(() => {
    isRecordingRef.current = isRecording;
  }, [isRecording]);

  // ── Stop recording ───────────────────────────────────────────────────────
  const stopRecording = () => {
    if (!isRecording) return;

    // Clear auto-rotation timer first
    if (chunkTimerRef.current) {
      clearTimeout(chunkTimerRef.current);
      chunkTimerRef.current = null;
    }

    // Stop volume monitoring
    if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
    if (audioContextRef.current) audioContextRef.current.close();
    setAudioVolume(0);

    // Signal final stop BEFORE calling recorder.stop() so the onstop closure sees it
    setIsRecording(false);
    isRecordingRef.current = false;
    setIsProcessing(true);

    // Stop the current segment → triggers onstop → flushChunk(isFinal=true)
    if (mediaRecorderRef.current?.state === 'recording') {
      mediaRecorderRef.current.stop();
    }

    // Stop all mic tracks
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;
  };

  // ── Manual analysis (generate meeting minutes) ───────────────────────────
  const handleManualAnalysis = async () => {
    const transcript = selectedMeeting?.rawTranscript || liveTranscript;
    if (!selectedMeeting || !transcript) return;

    setIsProcessing(true);
    try {
      const result = await analyzeTranscript(transcript, selectedMeeting.contextHint);

      await updateDoc(doc(db, 'meetings', selectedMeeting.id), {
        transcript: result.transcript,
        summary: result.summary,
        actionItems: result.actionItems,
        modelInfo: result.modelInfo,
      });

      setSelectedMeeting(prev => prev ? {
        ...prev,
        transcript: result.transcript,
        summary: result.summary,
        actionItems: result.actionItems,
        modelInfo: result.modelInfo,
      } : null);
    } catch (error) {
      console.error('Analysis error:', error);
      alert('AI 分析失敗，請稍後再試。');
    } finally {
      setIsProcessing(false);
    }
  };

  const deleteMeeting = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (window.confirm('您確定要刪除此會議記錄嗎？')) {
      try {
        await deleteDoc(doc(db, 'meetings', id));
        if (selectedMeeting?.id === id) setSelectedMeeting(null);
      } catch (error) {
        handleFirestoreError(error, OperationType.DELETE, `meetings/${id}`);
      }
    }
  };

  const filteredMeetings = meetings.filter(m =>
    m.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
    m.summary?.toLowerCase().includes(searchQuery.toLowerCase())
  );

  // ── Auth loading ─────────────────────────────────────────────────────────
  if (!isAuthReady) {
    return (
      <div className="min-h-screen bg-paper flex items-center justify-center">
        <motion.div
          animate={{ scale: [1, 1.1, 1] }}
          transition={{ repeat: Infinity, duration: 2 }}
          className="w-12 h-12 bg-forest rounded-full"
        />
      </div>
    );
  }

  // ── Login page ───────────────────────────────────────────────────────────
  if (!user) {
    return (
      <div className="min-h-screen bg-paper flex flex-col items-center justify-center p-6 font-sans overflow-hidden">
        <div className="absolute inset-0 opacity-20 pointer-events-none">
          <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-sage blur-[120px] rounded-full" />
          <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-terracotta blur-[120px] rounded-full" />
        </div>
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, ease: [0.16, 1, 0.3, 1] }}
          className="max-w-2xl w-full text-center relative z-10"
        >
          <Logo className="mx-auto mb-12 w-24 h-24" />
          <h1 className="text-8xl font-serif font-bold text-forest mb-8 tracking-tighter leading-[0.85]">
            Sync<br /><span className="text-terracotta italic">Lingua</span>
          </h1>
          <p className="text-wood/80 mb-16 text-xl leading-relaxed font-serif italic max-w-lg mx-auto">
            捕捉對話中的詩意與深度。將每一次會議轉化為具備敘事感的專業紀錄。
          </p>
          <div className="flex flex-col items-center gap-6">
            <button
              onClick={loginWithGoogle}
              className="group relative px-12 py-5 bg-forest text-white rounded-full font-medium transition-all flex items-center gap-4 overflow-hidden shadow-2xl shadow-forest/20 hover:scale-105 active:scale-95"
            >
              <div className="absolute inset-0 bg-white/10 translate-y-full group-hover:translate-y-0 transition-transform duration-500" />
              <UserIcon className="w-5 h-5 relative z-10" />
              <span className="relative z-10 text-lg">使用 Google 帳號開啟旅程</span>
            </button>
            <div className="text-[10px] uppercase tracking-[0.3em] text-sage font-bold">
              AI-Powered Narrative Intelligence
            </div>
          </div>
        </motion.div>
      </div>
    );
  }

  // ── Main app ─────────────────────────────────────────────────────────────
  return (
    <div className="min-h-screen bg-paper flex font-sans text-forest">

      {/* ── Sidebar ── */}
      <aside className="w-80 bg-white border-r border-wood/5 flex flex-col h-screen sticky top-0 z-20">
        <div className="p-8 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Logo className="w-10 h-10" />
            <div className="flex flex-col">
              <span className="font-serif font-bold text-2xl tracking-tighter text-forest leading-none">SyncLingua</span>
              <span className="text-[8px] uppercase tracking-[0.4em] text-sage font-bold mt-1">Intelligence</span>
            </div>
          </div>
        </div>

        <div className="px-6 py-4 space-y-4">
          <div className="space-y-2">
            <label className="text-[10px] uppercase tracking-widest text-sage font-bold px-1">會議背景提示 (選填)</label>
            <input
              type="text"
              placeholder="例如：LIZ學堂、財經、職場防備..."
              value={contextHint}
              onChange={(e) => setContextHint(e.target.value)}
              disabled={isRecording}
              className="w-full px-4 py-3 bg-paper/30 border border-transparent focus:border-wood/10 rounded-xl text-xs focus:ring-0 transition-all placeholder:text-sage/40 disabled:opacity-50"
            />
          </div>

          <button
            onClick={isRecording ? stopRecording : startRecording}
            disabled={isProcessing && !isRecording}
            className={cn(
              'w-full py-5 rounded-2xl font-medium flex flex-col items-center justify-center gap-2 transition-all relative overflow-hidden group',
              isRecording
                ? 'bg-terracotta text-white shadow-xl shadow-terracotta/20'
                : 'bg-forest text-white shadow-xl shadow-forest/20 hover:bg-forest/90',
              (isProcessing && !isRecording) && 'opacity-50 cursor-not-allowed'
            )}
          >
            <div className="absolute inset-0 bg-white/5 opacity-0 group-hover:opacity-100 transition-opacity" />
            {isRecording ? (
              <motion.div animate={{ scale: [1, 1.2, 1] }} transition={{ repeat: Infinity, duration: 1.5 }}>
                <StopCircle className="w-6 h-6" />
              </motion.div>
            ) : <Mic className="w-6 h-6" />}
            <span className="text-xs uppercase tracking-widest font-bold">
              {isRecording ? `錄音中 ${format(recordingTime * 1000, 'mm:ss')}` : '啟動新會議錄製'}
            </span>
            {/* Chunk indicator during recording */}
            {isRecording && chunksProcessed > 0 && (
              <span className="text-[9px] text-white/60 flex items-center gap-1">
                <Layers className="w-3 h-3" />
                已完成 {chunksProcessed} 段轉錄
                {isTranscribingChunk && <Loader2 className="w-3 h-3 animate-spin ml-1" />}
              </span>
            )}
          </button>
        </div>

        {/* Meeting list */}
        <div className="flex-1 overflow-y-auto px-4 py-6 space-y-8">
          <div className="px-2">
            <div className="relative group">
              <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-sage group-focus-within:text-forest transition-colors" />
              <input
                type="text"
                placeholder="搜尋紀錄..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-12 pr-4 py-3 bg-paper/30 border border-transparent focus:border-wood/10 rounded-xl text-sm focus:ring-0 transition-all placeholder:text-sage/50"
              />
            </div>
          </div>

          <div className="space-y-2">
            <div className="px-4 text-[9px] font-bold text-sage/60 uppercase tracking-[0.3em] mb-4">
              Archive / 會議存檔
            </div>
            {filteredMeetings.length === 0 ? (
              <div className="px-4 py-12 text-center text-sage/40 text-sm italic font-serif">尚無會議紀錄</div>
            ) : (
              <div className="space-y-1">
                {filteredMeetings.map((meeting) => (
                  <div
                    key={meeting.id}
                    onClick={() => setSelectedMeeting(meeting)}
                    className={cn(
                      'w-full text-left p-4 rounded-xl transition-all group flex items-start gap-4 border border-transparent cursor-pointer',
                      selectedMeeting?.id === meeting.id
                        ? 'bg-paper border-wood/10 shadow-sm'
                        : 'hover:bg-paper/40'
                    )}
                    role="button"
                    tabIndex={0}
                    onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setSelectedMeeting(meeting); }}
                  >
                    <div className={cn(
                      'mt-2 w-1.5 h-1.5 rounded-full shrink-0',
                      meeting.status === 'completed' ? 'bg-forest' :
                        meeting.status === 'processing' ? 'bg-terracotta animate-pulse' :
                          meeting.status === 'recording' ? 'bg-sage animate-pulse' :
                            meeting.status === 'error' ? 'bg-red-400' : 'bg-wood/20'
                    )} />
                    <div className="flex-1 min-w-0">
                      <div className="font-serif font-bold truncate text-lg text-forest/90 leading-tight">{meeting.title}</div>
                      <div className="text-[10px] text-wood/60 mt-1.5 flex items-center gap-2 uppercase tracking-wider font-bold">
                        <Clock className="w-3 h-3" />
                        {format(meeting.date.toDate(), 'yyyy.MM.dd')}
                      </div>
                    </div>
                    <button
                      onClick={(e) => deleteMeeting(meeting.id, e)}
                      className="opacity-0 group-hover:opacity-100 p-1.5 hover:bg-terracotta/10 hover:text-terracotta rounded-lg transition-all"
                      aria-label="刪除會議"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* User info */}
        <div className="p-6 border-t border-wood/5 bg-paper/20">
          <div className="flex items-center gap-4">
            <div className="relative">
              <img
                src={user.photoURL || `https://ui-avatars.com/api/?name=${user.displayName}`}
                alt={user.displayName || 'User'}
                className="w-11 h-11 rounded-full border-2 border-white shadow-md"
                referrerPolicy="no-referrer"
              />
              <div className="absolute bottom-0 right-0 w-3 h-3 bg-emerald-500 border-2 border-white rounded-full" />
            </div>
            <div className="flex-1 min-w-0">
              <div className="text-sm font-bold text-forest truncate">{user.displayName}</div>
              <div className="text-[9px] text-wood/60 truncate uppercase tracking-widest font-bold">{user.email}</div>
            </div>
            <button
              onClick={() => logout()}
              className="p-2 hover:bg-terracotta/10 hover:text-terracotta rounded-xl transition-all"
              aria-label="登出"
            >
              <LogOut className="w-4 h-4" />
            </button>
          </div>
        </div>
      </aside>

      {/* ── Main content ── */}
      <main className="flex-1 relative overflow-hidden">
        <AnimatePresence mode="wait">

          {/* ── Meeting detail / live view ── */}
          {selectedMeeting ? (
            <motion.div
              key={selectedMeeting.id}
              initial={{ opacity: 0, x: 30 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -30 }}
              transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
              className="absolute inset-0 overflow-y-auto"
            >
              <div className="max-w-4xl mx-auto p-12 lg:p-20 pb-40">

                {/* Header */}
                <div className="flex items-start justify-between mb-16">
                  <div className="flex-1">
                    <div className="text-[10px] font-bold uppercase tracking-[0.3em] text-sage mb-3">
                      {format(selectedMeeting.date.toDate(), 'yyyy年MM月dd日 · EEEE')}
                    </div>
                    <h2 className="text-5xl font-serif font-bold text-forest leading-tight tracking-tighter">
                      {selectedMeeting.title}
                    </h2>
                    {selectedMeeting.modelInfo && (
                      <div className="mt-4 inline-flex items-center gap-2 px-4 py-2 bg-sage/10 rounded-full">
                        <div className="w-1.5 h-1.5 rounded-full bg-sage" />
                        <span className="text-[10px] font-bold uppercase tracking-widest text-sage">
                          {selectedMeeting.modelInfo}
                        </span>
                      </div>
                    )}
                  </div>
                  <button
                    onClick={() => setSelectedMeeting(null)}
                    className="p-3 hover:bg-wood/5 rounded-2xl transition-all ml-6 shrink-0"
                  >
                    <X className="w-5 h-5 text-wood" />
                  </button>
                </div>

                {/* ── Live transcription progress during recording ── */}
                {isRecording && selectedMeeting.id === currentMeetingIdRef.current && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="mb-10 p-8 bg-terracotta/5 border border-terracotta/20 rounded-[32px]"
                  >
                    <div className="flex items-center gap-3 mb-4">
                      <motion.div
                        className="w-2 h-2 rounded-full bg-terracotta"
                        animate={{ scale: [1, 1.5, 1], opacity: [1, 0.5, 1] }}
                        transition={{ repeat: Infinity, duration: 1 }}
                      />
                      <span className="text-xs font-bold uppercase tracking-widest text-terracotta">
                        錄音中 — 每 3 分鐘自動轉錄一段
                      </span>
                      {isTranscribingChunk && (
                        <span className="flex items-center gap-1 text-[10px] text-wood/60">
                          <Loader2 className="w-3 h-3 animate-spin" />
                          正在轉錄中...
                        </span>
                      )}
                    </div>
                    {liveTranscript ? (
                      <div className="whitespace-pre-wrap text-forest/70 text-sm font-serif leading-relaxed max-h-60 overflow-y-auto">
                        {liveTranscript}
                      </div>
                    ) : (
                      <div className="text-wood/40 text-sm italic font-serif">
                        首段 3 分鐘完成後，逐字稿將即時顯示於此...
                      </div>
                    )}
                  </motion.div>
                )}

                {/* ── Generate meeting minutes button ── */}
                {(selectedMeeting.transcript || liveTranscript) && !selectedMeeting.summary && !isRecording && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="mb-10"
                  >
                    <button
                      onClick={handleManualAnalysis}
                      disabled={isProcessing}
                      className={cn(
                        'w-full py-6 rounded-[24px] font-bold text-lg flex items-center justify-center gap-3 transition-all',
                        'bg-forest text-white shadow-xl shadow-forest/20 hover:scale-[1.01] active:scale-[0.99]',
                        isProcessing && 'opacity-50 cursor-not-allowed'
                      )}
                    >
                      {isProcessing ? (
                        <>
                          <Loader2 className="w-5 h-5 animate-spin" />
                          <span>正在生成會議紀錄...</span>
                        </>
                      ) : (
                        <>
                          <ListChecks className="w-5 h-5" />
                          <span>一鍵生成會議紀錄</span>
                        </>
                      )}
                    </button>
                    <p className="text-center text-wood/50 text-xs mt-3 font-serif italic">
                      GPT-4o 將自動整理精華逐字稿、總結與行動項目
                    </p>
                  </motion.div>
                )}

                {/* ── Meeting minutes (summary) ── */}
                {selectedMeeting.summary && (
                  <div className="space-y-12">
                    <section>
                      <div className="flex items-center gap-4 mb-8">
                        <FileText className="w-5 h-5 text-terracotta" />
                        <h4 className="font-bold uppercase text-[10px] tracking-[0.3em] text-terracotta">會議總結</h4>
                      </div>
                      <div className="bg-white p-12 rounded-[40px] border border-wood/10 shadow-sm">
                        <div className="markdown-body">
                          <ReactMarkdown>{selectedMeeting.summary}</ReactMarkdown>
                        </div>
                      </div>
                    </section>

                    {selectedMeeting.actionItems && selectedMeeting.actionItems.length > 0 && (
                      <section>
                        <div className="flex items-center gap-4 mb-8">
                          <ListChecks className="w-5 h-5 text-forest" />
                          <h4 className="font-bold uppercase text-[10px] tracking-[0.3em] text-forest">行動項目</h4>
                        </div>
                        <div className="space-y-3">
                          {selectedMeeting.actionItems.map((item, i) => (
                            <div key={i} className="flex items-start gap-4 p-6 bg-white rounded-2xl border border-wood/10">
                              <CheckCircle2 className="w-5 h-5 text-sage mt-0.5 shrink-0" />
                              <span className="font-serif text-forest/80">{item}</span>
                            </div>
                          ))}
                        </div>
                      </section>
                    )}
                  </div>
                )}

                {/* ── Polished transcript ── */}
                {selectedMeeting.transcript && (
                  <section className="mt-12">
                    <div className="flex items-center gap-4 mb-8">
                      <History className="w-5 h-5 text-wood" />
                      <h4 className="font-bold uppercase text-[10px] tracking-[0.3em] text-wood">
                        {selectedMeeting.summary ? '精華逐字稿' : '完整逐字稿'}
                      </h4>
                    </div>
                    <div className="bg-white p-12 rounded-[40px] border border-wood/10">
                      <div className="whitespace-pre-wrap text-forest/80 leading-relaxed text-base font-serif">
                        {selectedMeeting.transcript}
                      </div>
                    </div>
                  </section>
                )}

                {/* ── Raw transcript backup ── */}
                {selectedMeeting.rawTranscript && selectedMeeting.rawTranscript !== selectedMeeting.transcript && (
                  <section className="mt-12 relative">
                    <div className="absolute -right-8 top-0 writing-vertical text-[9px] font-bold uppercase tracking-[0.4em] text-sage/40 h-full flex items-center">
                      Raw Data / 原始
                    </div>
                    <div className="bg-sage/5 p-12 rounded-[40px] border border-wood/10">
                      <div className="flex items-center gap-4 mb-8">
                        <History className="w-5 h-5 text-sage" />
                        <h4 className="font-bold uppercase text-[10px] tracking-[0.3em] text-sage">原始逐字稿備份</h4>
                      </div>
                      <div className="whitespace-pre-wrap text-wood/60 leading-relaxed text-base italic font-serif">
                        {selectedMeeting.rawTranscript}
                      </div>
                    </div>
                  </section>
                )}
              </div>
            </motion.div>

          ) : (
            /* ── Empty / home state ── */
            <motion.div
              key="empty"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex-1 flex flex-col items-center justify-start p-12 lg:p-20 text-center relative overflow-y-auto min-h-full pb-32"
            >
              <div className="absolute inset-0 opacity-10 pointer-events-none">
                <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-sage blur-[150px] rounded-full" />
                <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-terracotta blur-[150px] rounded-full" />
              </div>

              <div className="relative z-10 w-full max-w-4xl">
                <div className="mb-12">
                  <Logo className="w-32 h-32 mx-auto mb-8" />
                  <h2 className="text-7xl font-serif font-bold tracking-tighter mb-4 text-forest leading-none">
                    Sync<span className="text-terracotta italic">Lingua</span>
                  </h2>
                  <p className="text-wood/70 max-w-xl text-xl leading-relaxed font-serif italic mx-auto">
                    捕捉對話中的靈魂。
                  </p>
                </div>

                <div className="bg-white/80 backdrop-blur-md p-12 rounded-[64px] border border-wood/10 shadow-2xl shadow-forest/5 mb-16 relative group overflow-hidden">
                  <div className="absolute inset-0 bg-gradient-to-br from-sage/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-700" />
                  <div className="relative z-10 space-y-10">
                    <div className="space-y-4">
                      <h3 className="text-3xl font-serif font-bold text-forest">準備好開始了嗎？</h3>
                      <p className="text-wood/60 font-serif italic">在下方輸入會議背景，讓 AI 轉錄與分析更精準。</p>
                    </div>
                    <div className="max-w-md mx-auto space-y-6">
                      <input
                        type="text"
                        placeholder="例如：產品週會、客戶訪談、創意發想..."
                        value={contextHint}
                        onChange={(e) => setContextHint(e.target.value)}
                        className="w-full px-8 py-5 bg-paper/50 border-2 border-transparent focus:border-sage/20 rounded-3xl text-lg focus:ring-0 transition-all placeholder:text-sage/30 text-center font-serif italic shadow-inner"
                      />
                      <button
                        onClick={isRecording ? stopRecording : startRecording}
                        disabled={isProcessing && !isRecording}
                        className={cn(
                          'w-full py-8 rounded-[32px] font-bold text-2xl flex flex-col items-center justify-center gap-4 transition-all relative overflow-hidden group shadow-2xl',
                          isRecording
                            ? 'bg-terracotta text-white shadow-terracotta/30'
                            : 'bg-forest text-white shadow-forest/30 hover:scale-[1.02] active:scale-[0.98]',
                          (isProcessing && !isRecording) && 'opacity-50 cursor-not-allowed'
                        )}
                      >
                        <div className="absolute inset-0 bg-white/10 opacity-0 group-hover:opacity-100 transition-opacity" />
                        {isRecording ? (
                          <div className="flex flex-col items-center gap-4">
                            <motion.div
                              animate={{ scale: [1, 1.2, 1], opacity: [1, 0.8, 1] }}
                              transition={{ repeat: Infinity, duration: 1.5 }}
                              className="w-16 h-16 bg-white/20 rounded-full flex items-center justify-center"
                            >
                              <div className="w-6 h-6 bg-white rounded-sm" />
                            </motion.div>
                            <div className="w-32 h-1 bg-white/20 rounded-full overflow-hidden mt-2">
                              <motion.div
                                className="h-full bg-white"
                                animate={{ width: `${Math.min(audioVolume * 2, 100)}%` }}
                                transition={{ type: 'spring', bounce: 0, duration: 0.1 }}
                              />
                            </div>
                            <span className="tracking-[0.2em] uppercase text-sm">
                              正在錄音 {format(recordingTime * 1000, 'mm:ss')}
                            </span>
                            {chunksProcessed > 0 && (
                              <motion.span
                                initial={{ opacity: 0, y: 4 }}
                                animate={{ opacity: 1, y: 0 }}
                                className="text-[10px] text-white/70 font-serif italic flex items-center gap-1"
                              >
                                <Layers className="w-3 h-3" />
                                已完成 {chunksProcessed} 段分段轉錄
                                {isTranscribingChunk && <Loader2 className="w-3 h-3 animate-spin ml-1" />}
                              </motion.span>
                            )}
                          </div>
                        ) : (
                          <div className="flex flex-col items-center gap-4">
                            <div className="w-16 h-16 bg-white/20 rounded-full flex items-center justify-center group-hover:bg-white/30 transition-colors">
                              <Mic className="w-8 h-8" />
                            </div>
                            <span className="tracking-[0.2em] uppercase text-sm">啟動新會議錄製</span>
                          </div>
                        )}
                      </button>
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-8 w-full">
                  {[
                    { icon: Mic, title: '自動分段', desc: '每 3 分鐘自動切段上傳，突破 Whisper 25MB 限制，支援無限長度錄音。' },
                    { icon: FileText, title: '即時轉錄', desc: 'Whisper 模型邊錄邊轉，完成時逐字稿已全數就緒。' },
                    { icon: ListChecks, title: '一鍵紀錄', desc: '錄音結束後，GPT-4o 一鍵生成精華逐字稿、總結與行動項目。' }
                  ].map((feature, i) => (
                    <div key={i} className="bg-white/40 backdrop-blur-sm p-8 rounded-[40px] border border-wood/5 text-left hover:bg-white/60 transition-all group">
                      <div className="w-12 h-12 rounded-2xl bg-paper flex items-center justify-center mb-6 group-hover:bg-sage/10 transition-all">
                        <feature.icon className="w-6 h-6 text-forest" />
                      </div>
                      <h4 className="font-serif font-bold text-xl mb-2 text-forest">{feature.title}</h4>
                      <p className="text-wood/70 text-sm leading-relaxed font-serif italic">{feature.desc}</p>
                    </div>
                  ))}
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* ── Processing overlay (final drain only) ── */}
        <AnimatePresence>
          {isProcessing && !isRecording && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 bg-forest/40 backdrop-blur-xl z-50 flex items-center justify-center p-6"
            >
              <div className="bg-white p-20 rounded-[64px] shadow-2xl max-w-xl w-full text-center border border-wood/10 relative overflow-hidden">
                <div className="absolute top-0 left-0 w-full h-2 bg-paper">
                  <motion.div
                    initial={{ x: '-100%' }}
                    animate={{ x: '100%' }}
                    transition={{ repeat: Infinity, duration: 2, ease: 'easeInOut' }}
                    className="w-full h-full bg-terracotta"
                  />
                </div>
                <motion.div
                  animate={{ scale: [1, 1.05, 1], rotate: [0, 5, -5, 0] }}
                  transition={{ repeat: Infinity, duration: 4 }}
                  className="w-32 h-32 bg-paper rounded-[40px] flex items-center justify-center mx-auto mb-12 shadow-inner"
                >
                  <Logo className="w-20 h-20" />
                </motion.div>
                <h3 className="text-5xl font-serif font-bold mb-6 text-forest tracking-tight">正在編織紀錄...</h3>
                <p className="text-wood/80 font-serif italic text-xl leading-relaxed">
                  正在完成最後一段轉錄，請稍候。
                </p>
                <div className="mt-12 flex justify-center gap-2">
                  {[0, 1, 2].map(i => (
                    <motion.div
                      key={i}
                      animate={{ scale: [1, 1.5, 1], opacity: [0.3, 1, 0.3] }}
                      transition={{ repeat: Infinity, duration: 1, delay: i * 0.2 }}
                      className="w-2 h-2 rounded-full bg-sage"
                    />
                  ))}
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      <div id="error-portal" />
    </div>
  );
}
