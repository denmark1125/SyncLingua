/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, useRef } from 'react';
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
  X
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';
import { format } from 'date-fns';
import ReactMarkdown from 'react-markdown';
import { transcribeAudio, analyzeTranscript } from './services/geminiService';

// Utility for Tailwind classes
function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// Types
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

// Logo Component - Abstract & Artistic
const Logo = ({ className }: { className?: string }) => (
  <svg viewBox="0 0 100 100" className={cn("w-12 h-12", className)} fill="none" xmlns="http://www.w3.org/2000/svg">
    {/* Abstract sound wave / record concept */}
    <circle cx="50" cy="50" r="45" className="stroke-wood/10" strokeWidth="0.5" />
    <circle cx="50" cy="50" r="35" className="stroke-wood/20" strokeWidth="0.5" />
    <circle cx="50" cy="50" r="25" className="stroke-wood/30" strokeWidth="0.5" />
    
    {/* Dynamic elements */}
    <motion.path 
      d="M30 50 Q 40 20, 50 50 T 70 50" 
      className="stroke-forest" 
      strokeWidth="2" 
      strokeLinecap="round"
      animate={{ d: ["M30 50 Q 40 20, 50 50 T 70 50", "M30 50 Q 40 80, 50 50 T 70 50", "M30 50 Q 40 20, 50 50 T 70 50"] }}
      transition={{ repeat: Infinity, duration: 4, ease: "easeInOut" }}
    />
    <motion.path 
      d="M35 50 Q 45 35, 50 50 T 65 50" 
      className="stroke-terracotta" 
      strokeWidth="1.5" 
      strokeLinecap="round"
      animate={{ d: ["M35 50 Q 45 35, 50 50 T 65 50", "M35 50 Q 45 65, 50 50 T 65 50", "M35 50 Q 45 35, 50 50 T 65 50"] }}
      transition={{ repeat: Infinity, duration: 3, ease: "easeInOut", delay: 0.5 }}
    />
    <motion.circle 
      cx="50" cy="50" r="4" 
      className="fill-sage"
      animate={{ scale: [1, 1.5, 1], opacity: [0.5, 1, 0.5] }}
      transition={{ repeat: Infinity, duration: 2 }}
    />
  </svg>
);

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
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  // Auth Listener
  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
      setUser(currentUser);
      setIsAuthReady(true);
      if (currentUser) {
        // Sync user profile
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
    return () => unsubscribe();
  }, []);

  // Meetings Listener
  useEffect(() => {
    if (!user) {
      setMeetings([]);
      return;
    }

    const q = query(
      collection(db, 'meetings'),
      where('userId', '==', user.uid),
      orderBy('date', 'desc')
    );

    const unsubscribe = onSnapshot(q, (snapshot) => {
      const meetingsData = snapshot.docs.map(doc => ({
        id: doc.id,
        ...doc.data()
      })) as Meeting[];
      setMeetings(meetingsData);
    }, (err) => handleFirestoreError(err, OperationType.LIST, 'meetings'));

    return () => unsubscribe();
  }, [user]);

  // Recording Timer
  useEffect(() => {
    if (isRecording) {
      timerRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);
    } else {
      if (timerRef.current) clearInterval(timerRef.current);
      setRecordingTime(0);
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [isRecording]);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        await handleAudioProcessing(audioBlob, contextHint);
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (error) {
      console.error('Error starting recording:', error);
      alert('無法存取麥克風。請檢查權限設定。');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
      setIsRecording(false);
    }
  };

  const handleAudioProcessing = async (audioBlob: Blob, hint?: string) => {
    if (!user) return;
    
    setIsProcessing(true);
    
    // Create initial meeting record
    const meetingData: Omit<Meeting, 'id'> = {
      userId: user.uid,
      title: `${format(new Date(), 'yyyy年MM月dd日 HH:mm')} 的會議`,
      date: Timestamp.now(),
      duration: recordingTime,
      contextHint: hint,
      status: 'processing'
    };

    let meetingId = '';
    try {
      const docRef = await addDoc(collection(db, 'meetings'), meetingData);
      meetingId = docRef.id;

      // Convert blob to base64
      const reader = new FileReader();
      reader.readAsDataURL(audioBlob);
      reader.onloadend = async () => {
        const base64Audio = (reader.result as string).split(',')[1];
        
        try {
          const { transcript, modelInfo } = await transcribeAudio(base64Audio, 'audio/webm', hint);
          
          await updateDoc(doc(db, 'meetings', meetingId), {
            rawTranscript: transcript,
            transcript: transcript, // Initially set transcript to raw
            modelInfo: modelInfo,
            status: 'completed'
          });
        } catch (error) {
          console.error('Processing error:', error);
          await updateDoc(doc(db, 'meetings', meetingId), {
            status: 'error'
          });
        } finally {
          setIsProcessing(false);
        }
      };
    } catch (error) {
      handleFirestoreError(error, OperationType.CREATE, 'meetings');
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

  const handleManualAnalysis = async () => {
    if (!selectedMeeting || !selectedMeeting.rawTranscript) return;
    
    setIsProcessing(true);
    try {
      const result = await analyzeTranscript(selectedMeeting.rawTranscript, selectedMeeting.contextHint);
      
      await updateDoc(doc(db, 'meetings', selectedMeeting.id), {
        transcript: result.transcript,
        summary: result.summary,
        actionItems: result.actionItems,
        modelInfo: result.modelInfo,
      });
      
      // Update local state for immediate feedback
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

  const filteredMeetings = meetings.filter(m => 
    m.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
    m.summary?.toLowerCase().includes(searchQuery.toLowerCase())
  );

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

  return (
    <div className="min-h-screen bg-paper flex font-sans text-forest">
      {/* Sidebar */}
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
              className="w-full px-4 py-3 bg-paper/30 border border-transparent focus:border-wood/10 rounded-xl text-xs focus:ring-0 transition-all placeholder:text-sage/40"
            />
          </div>

          <button 
            onClick={isRecording ? stopRecording : startRecording}
            disabled={isProcessing}
            className={cn(
              "w-full py-5 rounded-2xl font-medium flex flex-col items-center justify-center gap-2 transition-all relative overflow-hidden group",
              isRecording 
                ? "bg-terracotta text-white shadow-xl shadow-terracotta/20" 
                : "bg-forest text-white shadow-xl shadow-forest/20 hover:bg-forest/90",
              isProcessing && "opacity-50 cursor-not-allowed"
            )}
          >
            <div className="absolute inset-0 bg-white/5 opacity-0 group-hover:opacity-100 transition-opacity" />
            {isRecording ? (
              <motion.div 
                animate={{ scale: [1, 1.2, 1] }}
                transition={{ repeat: Infinity, duration: 1.5 }}
              >
                <StopCircle className="w-6 h-6" />
              </motion.div>
            ) : <Mic className="w-6 h-6" />}
            <span className="text-xs uppercase tracking-widest font-bold">
              {isRecording ? `錄音中 ${format(recordingTime * 1000, 'mm:ss')}` : '啟動新會議錄製'}
            </span>
          </button>
        </div>

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
              <div className="px-4 py-12 text-center text-sage/40 text-sm italic font-serif">
                尚無會議紀錄
              </div>
            ) : (
              <div className="space-y-1">
                {filteredMeetings.map((meeting) => (
                  <div
                    key={meeting.id}
                    onClick={() => setSelectedMeeting(meeting)}
                    className={cn(
                      "w-full text-left p-4 rounded-xl transition-all group flex items-start gap-4 border border-transparent cursor-pointer",
                      selectedMeeting?.id === meeting.id 
                        ? "bg-paper border-wood/10 shadow-sm" 
                        : "hover:bg-paper/40"
                    )}
                    role="button"
                    tabIndex={0}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        setSelectedMeeting(meeting);
                      }
                    }}
                  >
                    <div className={cn(
                      "mt-2 w-1.5 h-1.5 rounded-full shrink-0",
                      meeting.status === 'completed' ? "bg-forest" :
                      meeting.status === 'processing' ? "bg-terracotta animate-pulse" :
                      meeting.status === 'error' ? "bg-red-400" : "bg-wood/20"
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
              onClick={logout}
              className="p-2.5 hover:bg-terracotta/10 rounded-xl transition-all text-wood hover:text-terracotta"
            >
              <LogOut className="w-4 h-4" />
            </button>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col h-screen overflow-hidden">
        <AnimatePresence mode="wait">
          {selectedMeeting ? (
            <motion.div 
              key={selectedMeeting.id}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="flex-1 flex flex-col overflow-hidden"
            >
              {/* Header */}
              <header className="p-10 bg-white border-b border-wood/5 flex items-center justify-between relative overflow-hidden">
                <div className="absolute top-0 right-0 w-64 h-64 bg-sage/5 blur-[100px] rounded-full -mr-32 -mt-32" />
                
                <div className="relative z-10">
                  <div className="flex items-center gap-3 mb-4">
                    <span className={cn(
                      "px-4 py-1.5 rounded-full text-[9px] font-bold uppercase tracking-[0.2em] shadow-sm",
                      selectedMeeting.status === 'completed' ? "bg-forest text-white" :
                      selectedMeeting.status === 'processing' ? "bg-terracotta text-white animate-pulse" :
                      "bg-red-500 text-white"
                    )}>
                      {selectedMeeting.status}
                    </span>
                    <div className="h-px w-12 bg-wood/20" />
                    <span className="text-[9px] font-bold uppercase tracking-[0.2em] text-sage">Session ID: {selectedMeeting.id.slice(0, 8)}</span>
                  </div>
                  
                  <h2 className="text-6xl font-serif font-bold tracking-tighter text-forest leading-tight">{selectedMeeting.title}</h2>
                  
                  <div className="flex items-center gap-8 mt-6 text-wood/70 text-xs font-serif italic">
                    <span className="flex items-center gap-3">
                      <div className="w-1.5 h-1.5 rounded-full bg-sage" />
                      {format(selectedMeeting.date.toDate(), 'yyyy年MM月dd日 • HH:mm')}
                    </span>
                    <span className="flex items-center gap-3">
                      <div className="w-1.5 h-1.5 rounded-full bg-sage" />
                      {selectedMeeting.duration ? format(selectedMeeting.duration * 1000, 'mm:ss') : '--:--'}
                    </span>
                    {selectedMeeting.modelInfo && (
                      <span className="flex items-center gap-3 text-terracotta">
                        <div className="w-1.5 h-1.5 rounded-full bg-terracotta" />
                        Intelligence: {selectedMeeting.modelInfo}
                      </span>
                    )}
                  </div>
                </div>
                
                <button 
                  onClick={() => setSelectedMeeting(null)}
                  className="p-4 hover:bg-paper rounded-2xl transition-all text-wood/40 hover:text-forest relative z-10"
                >
                  <X className="w-8 h-8" />
                </button>
              </header>

              {/* Content Grid */}
              <div className="flex-1 overflow-y-auto p-10 lg:p-16 bg-paper/30">
                {selectedMeeting.status === 'processing' ? (
                  <div className="h-full flex flex-col items-center justify-center text-center">
                    <div className="relative mb-12">
                      <motion.div 
                        animate={{ rotate: 360 }}
                        transition={{ repeat: Infinity, duration: 8, ease: "linear" }}
                        className="w-32 h-32 border border-wood/10 rounded-full"
                      />
                      <motion.div 
                        animate={{ rotate: -360 }}
                        transition={{ repeat: Infinity, duration: 12, ease: "linear" }}
                        className="absolute inset-0 w-32 h-32 border-t-2 border-forest rounded-full"
                      />
                      <div className="absolute inset-0 flex items-center justify-center">
                        <Logo className="w-12 h-12" />
                      </div>
                    </div>
                    <h3 className="text-4xl font-serif font-bold mb-4 text-forest">正在解構對話...</h3>
                    <p className="text-wood/80 max-w-sm font-serif italic text-lg leading-relaxed">
                      我們的 AI 正在捕捉每一個細微的音節與情感，這通常需要一分鐘左右的時間。
                    </p>
                  </div>
                ) : selectedMeeting.status === 'error' ? (
                  <div className="h-full flex flex-col items-center justify-center text-center">
                    <div className="w-24 h-24 bg-red-50 rounded-full flex items-center justify-center mb-8">
                      <AlertCircle className="w-12 h-12 text-red-500" />
                    </div>
                    <h3 className="text-4xl font-serif font-bold mb-4 text-forest">處理中斷</h3>
                    <p className="text-wood/80 max-w-sm font-serif italic text-lg leading-relaxed">
                      在轉譯此會議時遇到了一些技術障礙。請嘗試重新錄製或檢查您的網路連接。
                    </p>
                  </div>
                ) : (
                  <div className="max-w-7xl mx-auto flex flex-col lg:flex-row gap-16">
                    {/* Left Rail - Summary & Action Items */}
                    <div className="lg:w-1/3 space-y-16">
                      <section className="relative">
                        <div className="absolute -left-8 top-0 writing-vertical text-[9px] font-bold uppercase tracking-[0.4em] text-sage/40 h-full flex items-center">
                          Summary / 摘要
                        </div>
                        <div className="bg-white p-10 rounded-[40px] shadow-2xl shadow-forest/5 border border-wood/5 relative overflow-hidden">
                          <div className="absolute top-0 right-0 p-6">
                            <FileText className="w-6 h-6 text-terracotta/20" />
                          </div>
                          <div className="markdown-body">
                            <ReactMarkdown>{selectedMeeting.summary || '尚無摘要。'}</ReactMarkdown>
                          </div>
                        </div>
                      </section>

                      <section className="relative">
                        <div className="absolute -left-8 top-0 writing-vertical text-[9px] font-bold uppercase tracking-[0.4em] text-sage/40 h-full flex items-center">
                          Actions / 行動
                        </div>
                        <div className="bg-white p-10 rounded-[40px] shadow-2xl shadow-forest/5 border border-wood/5">
                          <ul className="space-y-6">
                            {selectedMeeting.actionItems?.map((item, i) => (
                              <li key={i} className="flex items-start gap-5 group">
                                <div className="w-6 h-6 rounded-full bg-sage/5 flex items-center justify-center shrink-0 mt-1 group-hover:bg-sage/20 transition-colors">
                                  <div className="w-1.5 h-1.5 rounded-full bg-sage" />
                                </div>
                                <span className="text-forest/80 font-serif text-lg leading-relaxed">{item}</span>
                              </li>
                            )) || <li className="text-sage/40 italic text-sm font-serif">未發現行動項目。</li>}
                          </ul>
                        </div>
                      </section>
                    </div>

                    {/* Right Content - Transcript */}
                    <div className="lg:w-2/3 space-y-16">
                      <section className="relative">
                        <div className="absolute -right-8 top-0 writing-vertical text-[9px] font-bold uppercase tracking-[0.4em] text-sage/40 h-full flex items-center">
                          Transcript / 紀錄
                        </div>
                        <div className="bg-white p-12 lg:p-16 rounded-[48px] shadow-2xl shadow-forest/5 border border-wood/5">
                          <div className="flex items-center justify-between mb-12">
                            <div className="flex items-center gap-4">
                              <div className="w-10 h-10 rounded-2xl bg-paper flex items-center justify-center">
                                <Mic className="w-5 h-5 text-forest" />
                              </div>
                              <h4 className="font-bold uppercase text-[10px] tracking-[0.3em] text-forest">
                                {selectedMeeting.summary ? '精華會議紀錄' : '原始逐字稿'}
                              </h4>
                            </div>
                            {!selectedMeeting.summary && (
                              <button 
                                onClick={handleManualAnalysis}
                                disabled={isProcessing}
                                className="group relative px-10 py-4 bg-forest text-white rounded-full text-sm font-bold transition-all overflow-hidden shadow-2xl shadow-forest/20 hover:scale-105 active:scale-95"
                              >
                                <div className="absolute inset-0 bg-white/10 translate-y-full group-hover:translate-y-0 transition-transform duration-500" />
                                <span className="relative z-10 flex items-center gap-3">
                                  <Plus className="w-4 h-4" />
                                  一鍵生成 AI 深度分析
                                </span>
                              </button>
                            )}
                          </div>
                          <div className="whitespace-pre-wrap text-forest/90 leading-[2] font-serif text-2xl tracking-tight">
                            {selectedMeeting.transcript || '紀錄內容為空。'}
                          </div>
                        </div>
                      </section>

                      {selectedMeeting.summary && selectedMeeting.rawTranscript && (
                        <section className="relative">
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
                  </div>
                )}
              </div>
            </motion.div>
          ) : (
            <motion.div 
              key="empty"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex-1 flex flex-col items-center justify-center p-12 text-center relative overflow-hidden"
            >
              <div className="absolute inset-0 opacity-10 pointer-events-none">
                <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-sage blur-[150px] rounded-full" />
                <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-terracotta blur-[150px] rounded-full" />
              </div>

              <div className="relative z-10">
                <div className="mb-16">
                  <Logo className="w-48 h-48 mx-auto" />
                </div>
                <h2 className="text-8xl font-serif font-bold tracking-tighter mb-8 text-forest leading-none">
                  Sync<span className="text-terracotta italic">Lingua</span>
                </h2>
                <p className="text-wood/70 max-w-xl text-2xl leading-relaxed font-serif italic mx-auto">
                  捕捉對話中的靈魂。從側邊欄選擇一個會議，或啟動錄音來記錄您的下一個重要時刻。
                </p>
                
                <div className="grid grid-cols-1 md:grid-cols-3 gap-12 mt-24 w-full max-w-6xl">
                  {[
                    { icon: Mic, title: '捕捉聲音', desc: '以最高品質錄製您的每一場對話，不遺漏任何細節。' },
                    { icon: FileText, title: '精準轉錄', desc: 'Whisper 模型確保逐字稿的完整與準確，捕捉語氣。' },
                    { icon: ListChecks, title: '深度洞察', desc: 'GPT-4o 為您提取具備敘事感的會議精華與行動。' }
                  ].map((feature, i) => (
                    <div key={i} className="bg-white/50 backdrop-blur-sm p-12 rounded-[48px] border border-wood/5 shadow-xl shadow-forest/5 text-left hover:bg-white transition-all group">
                      <div className="w-16 h-16 rounded-3xl bg-paper flex items-center justify-center mb-8 group-hover:bg-sage/10 transition-all">
                        <feature.icon className="w-8 h-8 text-forest" />
                      </div>
                      <h4 className="font-serif font-bold text-2xl mb-4 text-forest">{feature.title}</h4>
                      <p className="text-wood/80 text-lg leading-relaxed font-serif italic">{feature.desc}</p>
                    </div>
                  ))}
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Global Processing Overlay */}
        <AnimatePresence>
          {isProcessing && (
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
                    transition={{ repeat: Infinity, duration: 2, ease: "easeInOut" }}
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
                  我們的 AI 正在將原始音訊轉化為精緻的文字紀錄。這是一個充滿藝術感的過程，請稍候。
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

      {/* Error Boundary Display */}
      <div id="error-portal" />
    </div>
  );
}

