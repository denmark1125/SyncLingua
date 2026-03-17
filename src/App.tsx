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
import { processMeetingAudio } from './services/geminiService';

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
  summary?: string;
  actionItems?: string[];
  status: 'recording' | 'processing' | 'completed' | 'error';
}

export default function App() {
  const [user, setUser] = useState<User | null>(null);
  const [isAuthReady, setIsAuthReady] = useState(false);
  const [meetings, setMeetings] = useState<Meeting[]>([]);
  const [selectedMeeting, setSelectedMeeting] = useState<Meeting | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  
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
        await handleAudioProcessing(audioBlob);
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

  const handleAudioProcessing = async (audioBlob: Blob) => {
    if (!user) return;
    
    setIsProcessing(true);
    
    // Create initial meeting record
    const meetingData: Omit<Meeting, 'id'> = {
      userId: user.uid,
      title: `${format(new Date(), 'yyyy年MM月dd日 HH:mm')} 的會議`,
      date: Timestamp.now(),
      duration: recordingTime,
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
          const result = await processMeetingAudio(base64Audio, 'audio/webm');
          
          await updateDoc(doc(db, 'meetings', meetingId), {
            transcript: result.transcript,
            summary: result.summary,
            actionItems: result.actionItems,
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

  const filteredMeetings = meetings.filter(m => 
    m.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
    m.summary?.toLowerCase().includes(searchQuery.toLowerCase())
  );

  if (!isAuthReady) {
    return (
      <div className="min-h-screen bg-stone-50 flex items-center justify-center">
        <motion.div 
          animate={{ scale: [1, 1.1, 1] }}
          transition={{ repeat: Infinity, duration: 2 }}
          className="w-12 h-12 bg-emerald-600 rounded-full"
        />
      </div>
    );
  }

  if (!user) {
    return (
      <div className="min-h-screen bg-[#E4E3E0] flex flex-col items-center justify-center p-6 font-sans">
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="max-w-md w-full bg-white p-12 rounded-3xl shadow-xl border border-black/5 text-center"
        >
          <div className="w-20 h-20 bg-indigo-600 rounded-2xl flex items-center justify-center mx-auto mb-8 shadow-lg shadow-indigo-600/20">
            <Mic className="text-white w-10 h-10" />
          </div>
          <h1 className="text-4xl font-bold text-stone-900 mb-4 tracking-tight">SyncLingua</h1>
          <p className="text-stone-500 mb-10 leading-relaxed">
            AI 驅動的會議記錄與摘要工具。輕鬆錄音、轉錄並提取行動項目，讓溝通更高效。
          </p>
          <button 
            onClick={loginWithGoogle}
            className="w-full bg-stone-900 text-white py-4 rounded-2xl font-medium hover:bg-stone-800 transition-all flex items-center justify-center gap-3 shadow-lg shadow-black/10"
          >
            <UserIcon className="w-5 h-5" />
            使用 Google 帳號登入
          </button>
        </motion.div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#F5F5F0] flex font-sans text-stone-900">
      {/* Sidebar */}
      <aside className="w-80 bg-white border-r border-stone-200 flex flex-col h-screen sticky top-0">
        <div className="p-6 border-bottom border-stone-100 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-indigo-600 rounded-xl flex items-center justify-center shadow-lg shadow-indigo-600/10">
              <Mic className="text-white w-5 h-5" />
            </div>
            <span className="font-bold text-xl tracking-tight">SyncLingua</span>
          </div>
        </div>

        <div className="px-4 py-4">
          <button 
            onClick={isRecording ? stopRecording : startRecording}
            disabled={isProcessing}
            className={cn(
              "w-full py-4 rounded-2xl font-medium flex items-center justify-center gap-3 transition-all shadow-lg",
              isRecording 
                ? "bg-red-500 text-white shadow-red-500/20 animate-pulse" 
                : "bg-indigo-600 text-white shadow-indigo-600/20 hover:bg-indigo-700",
              isProcessing && "opacity-50 cursor-not-allowed"
            )}
          >
            {isRecording ? <StopCircle className="w-5 h-5" /> : <Mic className="w-5 h-5" />}
            {isRecording ? `停止錄音 (${format(recordingTime * 1000, 'mm:ss')})` : '開始新會議'}
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-2 py-4 space-y-1">
          <div className="px-4 mb-2">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-stone-400" />
              <input 
                type="text" 
                placeholder="搜尋會議..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-10 pr-4 py-2 bg-stone-100 border-none rounded-xl text-sm focus:ring-2 focus:ring-indigo-600/20 transition-all"
              />
            </div>
          </div>

          <div className="px-4 py-2 text-xs font-bold text-stone-400 uppercase tracking-widest">
            最近的會議
          </div>

          {filteredMeetings.length === 0 ? (
            <div className="px-4 py-8 text-center text-stone-400 text-sm italic">
              尚無會議記錄
            </div>
          ) : (
            filteredMeetings.map((meeting) => (
              <button
                key={meeting.id}
                onClick={() => setSelectedMeeting(meeting)}
                className={cn(
                  "w-full text-left p-4 rounded-2xl transition-all group flex items-start gap-3",
                  selectedMeeting?.id === meeting.id 
                    ? "bg-indigo-50 text-indigo-900" 
                    : "hover:bg-stone-50"
                )}
              >
                <div className={cn(
                  "mt-1 w-2 h-2 rounded-full shrink-0",
                  meeting.status === 'completed' ? "bg-emerald-500" :
                  meeting.status === 'processing' ? "bg-amber-500 animate-pulse" :
                  meeting.status === 'error' ? "bg-red-500" : "bg-stone-300"
                )} />
                <div className="flex-1 min-w-0">
                  <div className="font-medium truncate text-sm">{meeting.title}</div>
                  <div className="text-xs text-stone-500 mt-1 flex items-center gap-2">
                    <Clock className="w-3 h-3" />
                    {format(meeting.date.toDate(), 'yyyy年MM月dd日')}
                  </div>
                </div>
                <button 
                  onClick={(e) => deleteMeeting(meeting.id, e)}
                  className="opacity-0 group-hover:opacity-100 p-1 hover:text-red-500 transition-all"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </button>
            ))
          )}
        </div>


        <div className="p-4 border-t border-stone-100">
          <div className="flex items-center gap-3 p-2 rounded-2xl bg-stone-50">
            <img 
              src={user.photoURL || `https://ui-avatars.com/api/?name=${user.displayName}`} 
              alt={user.displayName || 'User'} 
              className="w-10 h-10 rounded-xl border border-white shadow-sm"
              referrerPolicy="no-referrer"
            />
            <div className="flex-1 min-w-0">
              <div className="text-sm font-bold truncate">{user.displayName}</div>
              <div className="text-xs text-stone-500 truncate">{user.email}</div>
            </div>
            <button 
              onClick={logout}
              className="p-2 hover:bg-stone-200 rounded-xl transition-all text-stone-500 hover:text-stone-900"
            >
              <LogOut className="w-5 h-5" />
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
              <header className="p-8 bg-white border-b border-stone-100 flex items-center justify-between">
                <div>
                  <h2 className="text-3xl font-bold tracking-tight">{selectedMeeting.title}</h2>
                  <div className="flex items-center gap-4 mt-2 text-stone-500 text-sm">
                    <span className="flex items-center gap-1.5">
                      <Clock className="w-4 h-4" />
                      {format(selectedMeeting.date.toDate(), 'MMMM d, yyyy • HH:mm')}
                    </span>
                    <span className="flex items-center gap-1.5">
                      <History className="w-4 h-4" />
                      {selectedMeeting.duration ? format(selectedMeeting.duration * 1000, 'mm:ss') : '--:--'}
                    </span>
                    <span className={cn(
                      "px-2.5 py-0.5 rounded-full text-xs font-bold uppercase tracking-wider",
                      selectedMeeting.status === 'completed' ? "bg-emerald-100 text-emerald-700" :
                      selectedMeeting.status === 'processing' ? "bg-amber-100 text-amber-700" :
                      "bg-red-100 text-red-700"
                    )}>
                      {selectedMeeting.status}
                    </span>
                  </div>
                </div>
                <button 
                  onClick={() => setSelectedMeeting(null)}
                  className="p-2 hover:bg-stone-100 rounded-full transition-all"
                >
                  <X className="w-6 h-6" />
                </button>
              </header>

              {/* Content Grid */}
              <div className="flex-1 overflow-y-auto p-8">
                {selectedMeeting.status === 'processing' ? (
                  <div className="h-full flex flex-col items-center justify-center text-center">
                    <motion.div 
                      animate={{ rotate: 360 }}
                      transition={{ repeat: Infinity, duration: 2, ease: "linear" }}
                      className="w-16 h-16 border-4 border-emerald-600 border-t-transparent rounded-full mb-6"
                    />
                    <h3 className="text-xl font-bold mb-2">正在轉錄您的會議...</h3>
                    <p className="text-stone-500 max-w-sm">
                      我們的 AI 正在處理音訊，生成逐字稿、摘要和行動項目。這通常需要一分鐘左右。
                    </p>
                  </div>
                ) : selectedMeeting.status === 'error' ? (
                  <div className="h-full flex flex-col items-center justify-center text-center">
                    <AlertCircle className="w-16 h-16 text-red-500 mb-6" />
                    <h3 className="text-xl font-bold mb-2">發生錯誤</h3>
                    <p className="text-stone-500 max-w-sm">
                      處理此會議時遇到錯誤。請嘗試重新錄製。
                    </p>
                  </div>
                ) : (
                  <div className="max-w-5xl mx-auto grid grid-cols-1 lg:grid-cols-3 gap-8">
                    {/* Summary & Action Items */}
                    <div className="lg:col-span-1 space-y-8">
                      <section className="bg-white p-6 rounded-3xl shadow-sm border border-stone-100">
                        <div className="flex items-center gap-2 mb-4 text-indigo-600">
                          <FileText className="w-5 h-5" />
                          <h4 className="font-bold uppercase text-xs tracking-widest">摘要</h4>
                        </div>
                        <div className="prose prose-stone prose-sm">
                          <ReactMarkdown>{selectedMeeting.summary || '尚無摘要。'}</ReactMarkdown>
                        </div>
                      </section>

                      <section className="bg-white p-6 rounded-3xl shadow-sm border border-stone-100">
                        <div className="flex items-center gap-2 mb-4 text-indigo-600">
                          <ListChecks className="w-5 h-5" />
                          <h4 className="font-bold uppercase text-xs tracking-widest">行動項目</h4>
                        </div>
                        <ul className="space-y-3">
                          {selectedMeeting.actionItems?.map((item, i) => (
                            <li key={i} className="flex items-start gap-3 text-sm">
                              <CheckCircle2 className="w-4 h-4 text-emerald-500 shrink-0 mt-0.5" />
                              <span>{item}</span>
                            </li>
                          )) || <li className="text-stone-400 italic text-sm">未發現行動項目。</li>}
                        </ul>
                      </section>
                    </div>

                    {/* Transcript */}
                    <div className="lg:col-span-2">
                      <section className="bg-white p-8 rounded-3xl shadow-sm border border-stone-100 h-full">
                        <div className="flex items-center gap-2 mb-6 text-stone-400">
                          <Mic className="w-5 h-5" />
                          <h4 className="font-bold uppercase text-xs tracking-widest">精華會議紀錄</h4>
                        </div>
                        <div className="whitespace-pre-wrap text-stone-700 leading-relaxed font-serif text-lg">
                          {selectedMeeting.transcript || '紀錄內容為空。'}
                        </div>
                      </section>
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
              className="flex-1 flex flex-col items-center justify-center p-12 text-center"
            >
              <div className="w-32 h-32 bg-stone-200 rounded-full flex items-center justify-center mb-8">
                <Mic className="text-stone-400 w-12 h-12" />
              </div>
              <h2 className="text-4xl font-bold tracking-tight mb-4">歡迎使用 SyncLingua</h2>
              <p className="text-stone-500 max-w-lg text-lg leading-relaxed">
                從側邊欄選擇一個會議來查看其逐字稿和摘要，或者開始一個新的錄音來捕捉您的下一次對話。
              </p>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-16 w-full max-w-4xl">
                {[
                  { icon: Mic, title: '錄音', desc: '從任何設備捕捉清晰的音訊。' },
                  { icon: FileText, title: '轉錄', desc: '在幾秒鐘內獲得準確的文本逐字稿。' },
                  { icon: ListChecks, title: '分析', desc: 'AI 驅動的摘要和行動項目。' }
                ].map((feature, i) => (
                  <div key={i} className="bg-white p-8 rounded-3xl border border-stone-100 shadow-sm text-left">
                    <feature.icon className="w-8 h-8 text-indigo-600 mb-4" />
                    <h4 className="font-bold text-lg mb-2">{feature.title}</h4>
                    <p className="text-stone-500 text-sm">{feature.desc}</p>
                  </div>
                ))}
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
              className="fixed inset-0 bg-stone-900/40 backdrop-blur-sm z-50 flex items-center justify-center p-6"
            >
              <div className="bg-white p-12 rounded-3xl shadow-2xl max-w-md w-full text-center">
                <motion.div 
                  animate={{ scale: [1, 1.1, 1] }}
                  transition={{ repeat: Infinity, duration: 2 }}
                  className="w-20 h-20 bg-indigo-600 rounded-2xl flex items-center justify-center mx-auto mb-8"
                >
                  <Clock className="text-white w-10 h-10" />
                </motion.div>
                <h3 className="text-2xl font-bold mb-4">正在處理音訊</h3>
                <p className="text-stone-500 mb-0">
                  我們正在使用 AI 轉錄並分析您的會議。這只需要一點時間。
                </p>
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

