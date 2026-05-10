import { useState, useRef, useEffect, useCallback } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import {
  Send, UploadCloud, Bot, User, GraduationCap, Loader2, Trash2,
  CheckCircle, TrendingUp, BookOpen, AlertTriangle,
} from 'lucide-react';

const API_BASE = 'http://127.0.0.1:8000';

// ─── Helpers ──────────────────────────────────────────────────────────────────
function gnoColor(gno) {
  if (gno == null) return 'text-slate-500';
  if (gno >= 3.0) return 'text-emerald-400';
  if (gno >= 2.0) return 'text-amber-400';
  return 'text-red-400';
}

// ─── ECTS SVG ring ────────────────────────────────────────────────────────────
function EctsRing({ current, target }) {
  const r = 48;
  const circ = 2 * Math.PI * r;
  const pct = Math.min(current / (target || 240), 1);
  const dash = pct * circ;
  const stroke = pct >= 1 ? '#10b981' : pct >= 0.75 ? '#6366f1' : '#f59e0b';

  return (
    <div className="flex flex-col items-center gap-1.5">
      <svg viewBox="0 0 112 112" className="w-28 h-28">
        <circle cx="56" cy="56" r={r} fill="none" stroke="#1e293b" strokeWidth="9" />
        <circle
          cx="56" cy="56" r={r} fill="none"
          stroke={stroke} strokeWidth="9"
          strokeDasharray={`${dash} ${circ}`}
          strokeLinecap="round"
          transform="rotate(-90 56 56)"
          style={{ transition: 'stroke-dasharray 0.7s ease' }}
        />
        <text x="56" y="50" textAnchor="middle" fill="#e2e8f0" fontSize="15" fontWeight="700">{current}</text>
        <text x="56" y="67" textAnchor="middle" fill="#475569" fontSize="9">/ {target} AKTS</text>
      </svg>
      <p className="text-[11px] text-slate-500 font-medium tracking-wide uppercase">AKTS İlerlemesi</p>
    </div>
  );
}

// ─── Stats panel (right side) ─────────────────────────────────────────────────
function StatsPanel({ summary }) {
  const { total_ects, target_ects, missing_count, failed_count, cumulative_gpa, semester_count, semesters } = summary;

  return (
    <div className="hidden lg:flex w-72 xl:w-80 flex-shrink-0 flex-col bg-[#0c1420] border-l border-white/[0.06] overflow-y-auto">
      {/* Header */}
      <div className="px-5 pt-5 pb-4 border-b border-white/[0.06]">
        <div className="flex items-center gap-2 mb-5">
          <TrendingUp size={14} className="text-indigo-400" />
          <h2 className="text-slate-300 text-xs font-semibold tracking-wide uppercase">Transkript Özeti</h2>
        </div>

        {/* ECTS Ring */}
        <div className="flex justify-center mb-4">
          <EctsRing current={total_ects} target={target_ects} />
        </div>

        {/* GNO card */}
        <div className="bg-white/[0.04] border border-white/[0.07] rounded-xl px-4 py-3 flex items-center justify-between mb-3">
          <span className="text-slate-400 text-xs">Genel GNO</span>
          <span className={`text-xl font-bold tabular-nums ${gnoColor(cumulative_gpa)}`}>
            {cumulative_gpa != null ? cumulative_gpa.toFixed(2) : '—'}
          </span>
        </div>

        {/* Badges grid */}
        <div className="grid grid-cols-2 gap-2">
          <div className={`rounded-xl px-3 py-2.5 border ${
            missing_count > 0
              ? 'bg-amber-500/[0.07] border-amber-500/20'
              : 'bg-emerald-500/[0.07] border-emerald-500/20'
          }`}>
            <p className="text-[11px] text-slate-500 mb-0.5">Eksik Zorunlu</p>
            <p className={`text-lg font-bold tabular-nums ${missing_count > 0 ? 'text-amber-400' : 'text-emerald-400'}`}>
              {missing_count}
            </p>
          </div>
          <div className={`rounded-xl px-3 py-2.5 border ${
            failed_count > 0
              ? 'bg-red-500/[0.07] border-red-500/20'
              : 'bg-emerald-500/[0.07] border-emerald-500/20'
          }`}>
            <p className="text-[11px] text-slate-500 mb-0.5">Başarısız</p>
            <p className={`text-lg font-bold tabular-nums ${failed_count > 0 ? 'text-red-400' : 'text-emerald-400'}`}>
              {failed_count}
            </p>
          </div>
        </div>

        {failed_count > 0 && (
          <div className="mt-3 flex items-start gap-2 bg-red-500/[0.06] border border-red-500/20 rounded-xl px-3 py-2.5">
            <AlertTriangle size={13} className="text-red-400 shrink-0 mt-0.5" />
            <p className="text-[11px] text-red-300 leading-relaxed">
              {failed_count} başarısız ders var — bir sonraki dönem öncelik verilmeli.
            </p>
          </div>
        )}
      </div>

      {/* Semester list */}
      {semesters && semesters.length > 0 && (
        <div className="px-5 py-4 flex-1">
          <div className="flex items-center gap-2 mb-3">
            <BookOpen size={12} className="text-slate-500" />
            <h3 className="text-[11px] text-slate-500 font-medium tracking-wide uppercase">
              Dönemler ({semester_count})
            </h3>
          </div>
          <div className="space-y-2">
            {semesters.map((sem, i) => (
              <div
                key={i}
                className="group bg-white/[0.03] hover:bg-white/[0.06] border border-white/[0.06] rounded-xl px-3 py-2.5 transition-colors duration-150"
              >
                <div className="flex items-start justify-between gap-2 mb-1">
                  <span className="text-slate-300 text-[11.5px] font-medium leading-tight">{sem.name}</span>
                  <span className={`text-sm font-bold tabular-nums flex-shrink-0 ${gnoColor(sem.gpa)}`}>
                    {sem.gpa != null ? sem.gpa.toFixed(2) : '—'}
                  </span>
                </div>
                <div className="flex items-center gap-2 text-[11px] text-slate-600">
                  <span>{sem.ects} AKTS</span>
                  <span className="text-slate-700">·</span>
                  <span>{sem.course_count} ders</span>
                  {sem.failed_count > 0 && (
                    <>
                      <span className="text-slate-700">·</span>
                      <span className="text-red-400 font-semibold">{sem.failed_count}F</span>
                    </>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Markdown renderer (dark theme) ──────────────────────────────────────────
const mdComponents = {
  h1: ({ children }) => <h1 className="text-base font-bold mt-2 mb-1 text-slate-100">{children}</h1>,
  h2: ({ children }) => <h2 className="text-sm font-semibold mt-2 mb-1 text-slate-200">{children}</h2>,
  h3: ({ children }) => <h3 className="text-sm font-semibold mt-1 mb-0.5 text-slate-200">{children}</h3>,
  p: ({ children }) => <p className="mb-1.5 last:mb-0 text-slate-300 leading-relaxed">{children}</p>,
  ul: ({ children }) => <ul className="list-disc list-inside mb-1.5 space-y-0.5">{children}</ul>,
  ol: ({ children }) => <ol className="list-decimal list-inside mb-1.5 space-y-0.5">{children}</ol>,
  li: ({ children }) => <li className="text-[13.5px] text-slate-300">{children}</li>,
  strong: ({ children }) => <strong className="font-semibold text-slate-100">{children}</strong>,
  em: ({ children }) => <em className="italic text-slate-400">{children}</em>,
  code: ({ children, inline }) =>
    inline
      ? <code className="bg-indigo-500/20 text-indigo-300 px-1.5 py-0.5 rounded text-xs font-mono">{children}</code>
      : <pre className="bg-black/30 border border-white/10 p-3 rounded-xl text-xs font-mono overflow-x-auto my-2 text-slate-300 whitespace-pre-wrap">{children}</pre>,
  table: ({ children }) => (
    <div className="overflow-x-auto my-2 rounded-xl border border-white/10">
      <table className="text-xs border-collapse w-full">{children}</table>
    </div>
  ),
  th: ({ children }) => (
    <th className="border-b border-white/10 bg-indigo-500/10 px-3 py-2 text-left font-semibold text-slate-200">{children}</th>
  ),
  td: ({ children }) => (
    <td className="border-b border-white/[0.05] px-3 py-2 text-slate-300 last:border-b-0">{children}</td>
  ),
  blockquote: ({ children }) => (
    <blockquote className="border-l-4 border-indigo-500/40 pl-3 text-slate-400 italic my-2">{children}</blockquote>
  ),
};

// ─── Message bubble ───────────────────────────────────────────────────────────
function MessageBubble({ msg }) {
  const isUser = msg.role === 'user';
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`flex max-w-[88%] md:max-w-[76%] gap-2.5 ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>
        <div
          className={`shrink-0 h-7 w-7 rounded-full flex items-center justify-center mt-0.5
            ${isUser
              ? 'bg-indigo-600 text-white'
              : 'bg-white/[0.05] border border-white/10 text-indigo-400'}`}
        >
          {isUser ? <User size={13} /> : <Bot size={14} />}
        </div>
        <div
          className={`px-4 py-3 rounded-2xl text-[14px] leading-relaxed
            ${isUser
              ? 'bg-indigo-600 text-white rounded-tr-sm'
              : 'bg-white/[0.04] border border-white/[0.08] text-slate-200 rounded-tl-sm'}`}
        >
          {isUser
            ? <p className="whitespace-pre-wrap">{msg.text}</p>
            : <ReactMarkdown components={mdComponents}>{msg.text}</ReactMarkdown>
          }
        </div>
      </div>
    </div>
  );
}

// ─── Thinking bubble with pulsing dot ────────────────────────────────────────
function ThinkingBubble({ status }) {
  return (
    <div className="flex justify-start">
      <div className="flex gap-2.5">
        <div className="shrink-0 h-7 w-7 rounded-full bg-white/[0.05] border border-white/10 text-indigo-400 flex items-center justify-center mt-0.5">
          <Bot size={14} />
        </div>
        <div className="px-4 py-3 rounded-2xl rounded-tl-sm bg-white/[0.04] border border-white/[0.08] flex items-center gap-3 min-w-[180px]">
          <span className="relative flex h-2 w-2 flex-shrink-0">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-indigo-400 opacity-75" />
            <span className="relative inline-flex rounded-full h-2 w-2 bg-indigo-500" />
          </span>
          <span className="text-slate-400 text-[13px]">{status || 'Bağlanıyor...'}</span>
        </div>
      </div>
    </div>
  );
}

// ─── App ──────────────────────────────────────────────────────────────────────
export default function App() {
  const [messages, setMessages] = useState([
    {
      role: 'system',
      text: 'Merhaba! Ben Sanal Akademik Danışmanın. Mezuniyet durumunu, eksik kredilerini veya alman gereken dersleri analiz etmek için hazırım. Sol menüden güncel transkriptini **(TXT)** yükle.',
    },
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [loadingStatus, setLoadingStatus] = useState('');
  const [fileUploaded, setFileUploaded] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [summary, setSummary] = useState(null);
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  // ── Upload ────────────────────────────────────────────────────────────────
  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setLoading(true);
    setLoadingStatus('📄 Transkript yükleniyor ve analiz ediliyor...');
    setMessages((prev) => [...prev, { role: 'user', text: `📄 Dosya yüklendi: ${file.name}` }]);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const url = sessionId
        ? `${API_BASE}/upload?session_id=${sessionId}`
        : `${API_BASE}/upload`;
      const { data } = await axios.post(url, formData);

      if (data.session_id) setSessionId(data.session_id);
      if (data.error) throw new Error(data.error);
      if (data.summary) setSummary(data.summary);

      setFileUploaded(true);
      setMessages((prev) => [...prev, { role: 'system', text: data.reply }]);
    } catch {
      setMessages((prev) => [
        ...prev,
        { role: 'system', text: '❌ Sunucuya bağlanılamadı veya dosya işlenirken hata oluştu. Backend (main.py) çalışıyor mu?' },
      ]);
    } finally {
      setLoading(false);
      setLoadingStatus('');
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  // ── Chat (SSE) ────────────────────────────────────────────────────────────
  const sendMessage = useCallback(async () => {
    if (!input.trim() || loading) return;

    const userMessage = input.trim();
    setInput('');
    setMessages((prev) => [...prev, { role: 'user', text: userMessage }]);
    setLoading(true);
    setLoadingStatus('⚡ Akademik danışman devreye giriyor...');

    try {
      const response = await fetch(`${API_BASE}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMessage, session_id: sessionId || '' }),
      });

      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let resolved = false;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          try {
            const event = JSON.parse(line.slice(6));
            if (event.type === 'status') {
              setLoadingStatus(event.data);
            } else if (event.type === 'done') {
              setMessages((prev) => [...prev, { role: 'system', text: event.data }]);
              setLoading(false);
              setLoadingStatus('');
              resolved = true;
            } else if (event.type === 'error') {
              setMessages((prev) => [...prev, { role: 'system', text: `⚠️ ${event.data}` }]);
              setLoading(false);
              setLoadingStatus('');
              resolved = true;
            }
          } catch (_) {}
        }
      }

      if (!resolved) { setLoading(false); setLoadingStatus(''); }
    } catch {
      setMessages((prev) => [...prev, { role: 'system', text: '❌ Sunucuya ulaşılamıyor.' }]);
      setLoading(false);
      setLoadingStatus('');
    }
  }, [input, loading, sessionId]);

  // ── Reset ─────────────────────────────────────────────────────────────────
  const resetChat = async () => {
    if (sessionId) {
      try { await axios.delete(`${API_BASE}/session/${sessionId}`); } catch (_) {}
    }
    setSessionId(null);
    setFileUploaded(false);
    setLoadingStatus('');
    setSummary(null);
    setMessages([{ role: 'system', text: 'Sohbet temizlendi. Yeni bir transkript yükleyebilirsin.' }]);
  };

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div className="flex h-screen bg-[#0d1017] overflow-hidden font-sans antialiased text-slate-200">

      {/* ── SIDEBAR ─────────────────────────────────────────────────────────── */}
      <div className="hidden md:flex w-56 xl:w-64 flex-shrink-0 flex-col bg-[#0c1420] border-r border-white/[0.06]">
        {/* Logo */}
        <div className="px-5 py-5 border-b border-white/[0.06] flex items-center gap-3">
          <div className="bg-indigo-600/15 border border-indigo-500/25 p-2 rounded-xl">
            <GraduationCap size={20} className="text-indigo-400" />
          </div>
          <div>
            <h1 className="text-slate-100 font-semibold text-[13.5px] leading-tight">Sanal Danışman</h1>
            <p className="text-indigo-400/70 text-[10.5px] mt-0.5">Llama 3.1 · Ajansal AI</p>
          </div>
        </div>

        {/* Body */}
        <div className="flex-1 px-4 py-5 flex flex-col gap-3">
          <label
            className={`group flex items-center justify-center gap-2 w-full py-2.5 px-3
              rounded-xl text-sm font-medium border transition-all duration-200
              ${loading
                ? 'opacity-50 cursor-not-allowed bg-indigo-600/10 border-indigo-500/20 text-indigo-300'
                : 'cursor-pointer bg-indigo-600/10 border-indigo-500/20 text-indigo-300 hover:bg-indigo-600/20 hover:border-indigo-500/40'
              }`}
          >
            <UploadCloud size={15} className="group-hover:-translate-y-0.5 transition-transform duration-200" />
            <span>Transkript Yükle (TXT)</span>
            <input
              ref={fileInputRef}
              type="file"
              accept=".txt,text/plain"
              className="hidden"
              onChange={handleFileUpload}
              disabled={loading}
            />
          </label>

          {fileUploaded && (
            <div className="bg-emerald-500/[0.06] border border-emerald-500/20 rounded-xl px-3 py-2.5">
              <div className="flex items-center gap-1.5 mb-1">
                <CheckCircle size={12} className="text-emerald-400" />
                <span className="text-emerald-300 text-[11.5px] font-medium">Transkript yüklendi</span>
              </div>
              {summary && (
                <p className="text-slate-500 text-[11px] leading-relaxed">
                  {summary.total_ects}/{summary.target_ects} AKTS · {summary.semester_count} dönem
                </p>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-4 pb-5 border-t border-white/[0.06] pt-4">
          <button
            onClick={resetChat}
            className="flex items-center gap-1.5 text-[11.5px] text-slate-600 hover:text-slate-400 transition-colors w-full justify-center py-2"
          >
            <Trash2 size={12} />
            Sohbeti Temizle
          </button>
        </div>
      </div>

      {/* ── CHAT AREA ───────────────────────────────────────────────────────── */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-4 md:px-8 py-6 space-y-4">
          {messages.map((msg, i) => (
            <MessageBubble key={i} msg={msg} />
          ))}
          {loading && <ThinkingBubble status={loadingStatus} />}
          <div ref={messagesEndRef} />
        </div>

        {/* Input bar */}
        <div className="px-4 md:px-8 pb-5 pt-3 border-t border-white/[0.06]">
          <div className="max-w-3xl mx-auto flex items-center gap-2 bg-white/[0.04] border border-white/[0.08] rounded-2xl px-4 py-1.5 focus-within:border-indigo-500/40 transition-colors duration-200">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && sendMessage()}
              disabled={loading}
              placeholder={
                fileUploaded
                  ? 'Transkriptinle ilgili bir soru sor...'
                  : 'Önce sol menüden transkript yükle...'
              }
              className="flex-1 bg-transparent border-none py-3 text-slate-200 placeholder-slate-600 focus:outline-none text-[14px] disabled:opacity-50"
            />
            <button
              onClick={sendMessage}
              disabled={loading || !input.trim()}
              className="bg-indigo-600 hover:bg-indigo-500 active:bg-indigo-700 text-white p-2.5 rounded-xl transition-colors duration-150 disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center"
            >
              {loading
                ? <Loader2 size={15} className="animate-spin" />
                : <Send size={15} />
              }
            </button>
          </div>
          <p className="text-center text-[11px] text-slate-700 mt-2.5">
            Verileriniz yalnızca yerel Llama 3.1 modeliyle işlenir — hiçbir şey cihazınızdan çıkmaz.
          </p>
        </div>
      </div>

      {/* ── STATS PANEL ─────────────────────────────────────────────────────── */}
      {summary && <StatsPanel summary={summary} />}
    </div>
  );
}
