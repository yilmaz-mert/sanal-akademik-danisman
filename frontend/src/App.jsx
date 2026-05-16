import { useState, useRef, useEffect, useCallback } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import {
  Send, UploadCloud, Bot, User, GraduationCap, Loader2, Trash2,
  CheckCircle, TrendingUp, BookOpen, AlertTriangle, X,
  BookMarked, XCircle, HelpCircle, CalendarDays, ChevronRight,
} from 'lucide-react';

const API_BASE = 'http://127.0.0.1:8000';

// ─── Helpers ──────────────────────────────────────────────────────────────────
function gnoColor(gno) {
  if (gno == null) return 'text-slate-500';
  if (gno >= 3.0) return 'text-emerald-400';
  if (gno >= 2.0) return 'text-amber-400';
  return 'text-red-400';
}

function gradeColor(g) {
  if (!g) return 'text-slate-400';
  const u = g.toUpperCase();
  if (['F', 'FD', 'VZ', 'DZ'].includes(u)) return 'text-red-400 font-bold';
  if (['AA', 'BA'].includes(u)) return 'text-emerald-400 font-semibold';
  if (['BB', 'CB', 'CC'].includes(u)) return 'text-sky-400';
  return 'text-amber-400';
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
        <circle cx="56" cy="56" r={r} fill="none" stroke={stroke} strokeWidth="9"
          strokeDasharray={`${dash} ${circ}`} strokeLinecap="round"
          transform="rotate(-90 56 56)"
          style={{ transition: 'stroke-dasharray 0.7s ease' }} />
        <text x="56" y="50" textAnchor="middle" fill="#e2e8f0" fontSize="15" fontWeight="700">{current}</text>
        <text x="56" y="67" textAnchor="middle" fill="#475569" fontSize="9">/ {target} AKTS</text>
      </svg>
      <p className="text-[11px] text-slate-500 font-medium tracking-wide uppercase">AKTS İlerlemesi</p>
    </div>
  );
}

// ─── Modal ────────────────────────────────────────────────────────────────────
function Modal({ isOpen, onClose, title, icon: Icon, accentClass = 'text-indigo-400', children }) {
  useEffect(() => {
    if (!isOpen) return;
    const handler = (e) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [isOpen, onClose]);

  if (!isOpen) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* backdrop */}
      <div className="absolute inset-0 bg-black/70 backdrop-blur-sm" onClick={onClose} />
      {/* panel */}
      <div className="relative z-10 w-full max-w-2xl max-h-[80vh] flex flex-col bg-[#0f1829] border border-white/[0.09] rounded-2xl shadow-2xl overflow-hidden">
        {/* header */}
        <div className="flex items-center gap-3 px-5 py-4 border-b border-white/[0.07] flex-shrink-0">
          {Icon && <Icon size={16} className={accentClass} />}
          <h2 className="text-slate-200 font-semibold text-sm flex-1">{title}</h2>
          <button onClick={onClose}
            className="text-slate-500 hover:text-slate-300 transition-colors p-1 rounded-lg hover:bg-white/[0.06]">
            <X size={16} />
          </button>
        </div>
        {/* body */}
        <div className="overflow-y-auto flex-1 p-5">{children}</div>
      </div>
    </div>
  );
}

// ─── Table helper ─────────────────────────────────────────────────────────────
function DataTable({ headers, rows }) {
  if (!rows || rows.length === 0)
    return <p className="text-slate-500 text-sm text-center py-6">Veri bulunamadı.</p>;
  return (
    <div className="overflow-x-auto rounded-xl border border-white/[0.07]">
      <table className="w-full text-xs border-collapse">
        <thead>
          <tr>
            {headers.map((h) => (
              <th key={h} className="bg-white/[0.04] border-b border-white/[0.07] px-3 py-2.5 text-left text-slate-400 font-semibold tracking-wide">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} className="hover:bg-white/[0.03] transition-colors">
              {row.map((cell, j) => (
                <td key={j} className="border-b border-white/[0.04] px-3 py-2.5 text-slate-300 last-of-type:border-b-0">{cell}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ─── Stats panel (right side) ─────────────────────────────────────────────────
function StatsPanel({ summary }) {
  const {
    total_ects, target_ects, missing_count, failed_count,
    unrecognized_count = 0, unrecognized_ects = 0,
    cumulative_gpa, semester_count, semesters,
    missing_courses = [], failed_courses = [], unrecognized_electives = [],
  } = summary;

  const [modal, setModal] = useState(null); // 'missing' | 'failed' | 'unrecognized' | {sem: index}
  const close = () => setModal(null);

  const selectedSem = typeof modal === 'object' && modal !== null && modal.sem != null
    ? semesters?.[modal.sem] : null;

  return (
    <>
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

          {/* Zero-data warning */}
          {total_ects === 0 && (
            <div className="mb-3 flex items-start gap-2 bg-amber-500/[0.08] border border-amber-500/25 rounded-xl px-3 py-2.5">
              <AlertTriangle size={13} className="text-amber-400 shrink-0 mt-0.5" />
              <p className="text-[11px] text-amber-300 leading-relaxed">
                Veri ayrıştırılamadı — lütfen transkript formatını kontrol edin.
              </p>
            </div>
          )}

          {/* Badges grid — clickable */}
          <div className="grid grid-cols-2 gap-2">
            <button
              onClick={() => missing_count > 0 && setModal('missing')}
              className={`group rounded-xl px-3 py-2.5 border text-left transition-all duration-150 ${
                missing_count > 0
                  ? 'bg-amber-500/[0.07] border-amber-500/20 cursor-pointer hover:bg-amber-500/[0.13] hover:border-amber-500/35'
                  : 'bg-emerald-500/[0.07] border-emerald-500/20 cursor-default'
              }`}
            >
              <p className="text-[11px] text-slate-500 mb-0.5 flex items-center gap-1">
                Eksik Zorunlu
                {missing_count > 0 && <ChevronRight size={10} className="text-amber-500/60 group-hover:translate-x-0.5 transition-transform" />}
              </p>
              <p className={`text-lg font-bold tabular-nums ${missing_count > 0 ? 'text-amber-400' : 'text-emerald-400'}`}>
                {total_ects > 0 ? missing_count : '—'}
              </p>
            </button>

            <button
              onClick={() => failed_count > 0 && setModal('failed')}
              className={`group rounded-xl px-3 py-2.5 border text-left transition-all duration-150 ${
                failed_count > 0
                  ? 'bg-red-500/[0.07] border-red-500/20 cursor-pointer hover:bg-red-500/[0.13] hover:border-red-500/35'
                  : 'bg-emerald-500/[0.07] border-emerald-500/20 cursor-default'
              }`}
            >
              <p className="text-[11px] text-slate-500 mb-0.5 flex items-center gap-1">
                Başarısız
                {failed_count > 0 && <ChevronRight size={10} className="text-red-500/60 group-hover:translate-x-0.5 transition-transform" />}
              </p>
              <p className={`text-lg font-bold tabular-nums ${failed_count > 0 ? 'text-red-400' : 'text-emerald-400'}`}>
                {total_ects > 0 ? failed_count : '—'}
              </p>
            </button>
          </div>

          {failed_count > 0 && (
            <div className="mt-3 flex items-start gap-2 bg-red-500/[0.06] border border-red-500/20 rounded-xl px-3 py-2.5">
              <AlertTriangle size={13} className="text-red-400 shrink-0 mt-0.5" />
              <p className="text-[11px] text-red-300 leading-relaxed">
                {failed_count} başarısız ders — bir sonraki dönem öncelik verilmeli.
              </p>
            </div>
          )}

          {/* Unrecognized — clickable */}
          {unrecognized_count > 0 && (
            <button
              onClick={() => setModal('unrecognized')}
              className="group mt-3 w-full text-left bg-sky-500/[0.06] hover:bg-sky-500/[0.11] border border-sky-500/20 hover:border-sky-500/35 rounded-xl px-3 py-2.5 transition-all duration-150 cursor-pointer"
            >
              <p className="text-[11px] text-slate-500 mb-0.5 flex items-center gap-1">
                Tanınmayan Seçmeli
                <ChevronRight size={10} className="text-sky-500/60 group-hover:translate-x-0.5 transition-transform" />
              </p>
              <p className="text-sm font-bold tabular-nums text-sky-400">
                {unrecognized_count} ders · {unrecognized_ects} AKTS
              </p>
              <p className="text-[10px] text-slate-600 mt-0.5 leading-relaxed">
                Müfredatta kaydı yok — AKTS toplamına dahil edildi
              </p>
            </button>
          )}
        </div>

        {/* Semester list — each item clickable */}
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
                <button
                  key={i}
                  onClick={() => setModal({ sem: i })}
                  className="group w-full text-left bg-white/[0.03] hover:bg-white/[0.07] border border-white/[0.06] hover:border-white/[0.12] rounded-xl px-3 py-2.5 transition-all duration-150 cursor-pointer"
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
                    <ChevronRight size={10} className="ml-auto text-slate-700 group-hover:text-slate-500 group-hover:translate-x-0.5 transition-all" />
                  </div>
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* ── MODALS ── */}

      {/* Missing mandatory courses */}
      <Modal isOpen={modal === 'missing'} onClose={close}
        title={`Eksik Zorunlu Dersler (${missing_courses.length})`}
        icon={BookMarked} accentClass="text-amber-400">
        <DataTable
          headers={['Kod', 'Ders Adı', 'AKTS', 'Dönem']}
          rows={missing_courses.map(c => [
            <code className="text-amber-300 bg-amber-500/10 px-1.5 py-0.5 rounded text-[11px]">{c.kod}</code>,
            c.ad,
            <span className="text-amber-400 font-semibold">{c.akts}</span>,
            c.donem || '—',
          ])}
        />
      </Modal>

      {/* Failed courses */}
      <Modal isOpen={modal === 'failed'} onClose={close}
        title={`Başarısız Dersler (${failed_courses.length})`}
        icon={XCircle} accentClass="text-red-400">
        <DataTable
          headers={['Kod', 'Ders Adı', 'Not', 'AKTS']}
          rows={failed_courses.map(c => [
            <code className="text-red-300 bg-red-500/10 px-1.5 py-0.5 rounded text-[11px]">{c.kod}</code>,
            c.ad,
            <span className={gradeColor(c.not)}>{c.not || 'F'}</span>,
            c.akts,
          ])}
        />
      </Modal>

      {/* Unrecognized electives */}
      <Modal isOpen={modal === 'unrecognized'} onClose={close}
        title={`Tanınmayan Seçmeli Dersler (${
          Array.isArray(unrecognized_electives)
            ? unrecognized_electives.length
            : Object.keys(unrecognized_electives).length
        })`}
        icon={HelpCircle} accentClass="text-sky-400">
        {(() => {
          const list = Array.isArray(unrecognized_electives)
            ? unrecognized_electives
            : Object.values(unrecognized_electives);
          return (
            <>
              <p className="text-[11.5px] text-slate-500 mb-4 leading-relaxed">
                Bu dersler müfredat veri tabanında bulunamadı, ancak AKTS değerleri mezuniyet toplamına dahil edildi. Gözden Geçiren AI her dersi ayrıca değerlendirdi.
              </p>
              <DataTable
                headers={['Kod', 'Ders Adı', 'Not', 'AKTS', 'Gözden Geçiren Notu']}
                rows={list.map(c => [
                  <code className="text-sky-300 bg-sky-500/10 px-1.5 py-0.5 rounded text-[11px]">{c.kod}</code>,
                  c.ad,
                  <span className={gradeColor(c.not)}>{c.not || '—'}</span>,
                  <span className="text-sky-400 font-semibold">{c.akts}</span>,
                  <span className="text-slate-400 text-[11px] italic">
                    {c.reviewer_note || 'Gözden geçirildi'}
                  </span>,
                ])}
              />
            </>
          );
        })()}
      </Modal>

      {/* Semester detail */}
      <Modal
        isOpen={selectedSem != null}
        onClose={close}
        title={selectedSem ? `${selectedSem.name}  ·  ${selectedSem.gpa != null ? selectedSem.gpa.toFixed(2) + ' GNO' : ''}` : ''}
        icon={CalendarDays} accentClass="text-indigo-400"
      >
        {selectedSem && (
          <>
            <div className="flex gap-4 mb-4 text-xs text-slate-500">
              <span className="bg-white/[0.04] rounded-lg px-3 py-1.5">
                <span className="text-slate-300 font-semibold">{selectedSem.ects}</span> AKTS
              </span>
              <span className="bg-white/[0.04] rounded-lg px-3 py-1.5">
                <span className="text-slate-300 font-semibold">{selectedSem.course_count}</span> ders
              </span>
              {selectedSem.failed_count > 0 && (
                <span className="bg-red-500/10 rounded-lg px-3 py-1.5 text-red-400 font-semibold">
                  {selectedSem.failed_count} başarısız
                </span>
              )}
            </div>
            <DataTable
              headers={['Kod', 'Ders Adı', 'Not', 'AKTS', 'Durum']}
              rows={(selectedSem.courses || []).map(c => [
                <code className="text-indigo-300 bg-indigo-500/10 px-1.5 py-0.5 rounded text-[11px]">{c.kod}</code>,
                c.ad,
                <span className={gradeColor(c.not)}>{c.not || '—'}</span>,
                c.akts,
                <span className={c.durum === 'Geçti' ? 'text-emerald-400 text-[11px]' : 'text-red-400 text-[11px] font-semibold'}>
                  {c.durum || '—'}
                </span>,
              ])}
            />
          </>
        )}
      </Modal>
    </>
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
        <div className={`shrink-0 h-7 w-7 rounded-full flex items-center justify-center mt-0.5
          ${isUser ? 'bg-indigo-600 text-white' : 'bg-white/[0.05] border border-white/10 text-indigo-400'}`}>
          {isUser ? <User size={13} /> : <Bot size={14} />}
        </div>
        <div className={`px-4 py-3 rounded-2xl text-[14px] leading-relaxed
          ${isUser
            ? 'bg-indigo-600 text-white rounded-tr-sm'
            : 'bg-white/[0.04] border border-white/[0.08] text-slate-200 rounded-tl-sm'}`}>
          {isUser
            ? <p className="whitespace-pre-wrap">{msg.text}</p>
            : <ReactMarkdown components={mdComponents}>{msg.text}</ReactMarkdown>
          }
        </div>
      </div>
    </div>
  );
}

// ─── Thinking bubble ─────────────────────────────────────────────────────────
function ThinkingBubble({ status, statusLog = [] }) {
  // Show up to 2 prior steps faded above, then the active one pulsing
  const prev = statusLog.length > 1 ? statusLog.slice(-3, -1) : [];
  const current = status || (statusLog.length > 0 ? statusLog[statusLog.length - 1] : 'Bağlanıyor...');
  return (
    <div className="flex justify-start">
      <div className="flex gap-2.5">
        <div className="shrink-0 h-7 w-7 rounded-full bg-white/[0.05] border border-white/10 text-indigo-400 flex items-center justify-center mt-0.5">
          <Bot size={14} />
        </div>
        <div className="px-4 py-3 rounded-2xl rounded-tl-sm bg-white/[0.04] border border-white/[0.08] min-w-[200px] max-w-[420px]">
          {prev.length > 0 && (
            <div className="mb-2 space-y-0.5 border-b border-white/[0.05] pb-2">
              {prev.map((msg, i) => (
                <p key={i} className="text-slate-700 text-[11px] leading-tight truncate">{msg}</p>
              ))}
            </div>
          )}
          <div className="flex items-center gap-3">
            <span className="relative flex h-2 w-2 flex-shrink-0">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-indigo-400 opacity-75" />
              <span className="relative inline-flex rounded-full h-2 w-2 bg-indigo-500" />
            </span>
            <span className="text-slate-400 text-[13px]">{current}</span>
          </div>
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
  const [statusLog, setStatusLog] = useState([]);
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
    setLoadingStatus('Transkript yükleniyor...');
    setStatusLog([]);
    setMessages((prev) => [...prev, { role: 'user', text: `📄 Dosya yüklendi: ${file.name}` }]);

    const formData = new FormData();
    formData.append('file', file);
    const url = sessionId ? `${API_BASE}/upload?session_id=${sessionId}` : `${API_BASE}/upload`;

    try {
      const response = await fetch(url, { method: 'POST', body: formData });
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
            const ev = JSON.parse(line.slice(6));
            if (ev.type === 'status') {
              setLoadingStatus(ev.data);
              setStatusLog((prev) => [...prev, ev.data]);
            } else if (ev.type === 'summary') {
              const { session_id: sid, summary: s } = ev.data;
              if (sid) setSessionId(sid);
              if (s) setSummary(s);
              setFileUploaded(true);
            } else if (ev.type === 'done') {
              setMessages((prev) => [...prev, { role: 'system', text: ev.data }]);
              setLoading(false); setLoadingStatus(''); setStatusLog([]); resolved = true;
            } else if (ev.type === 'error') {
              setMessages((prev) => [...prev, { role: 'system', text: `❌ ${ev.data}` }]);
              setLoading(false); setLoadingStatus(''); setStatusLog([]); resolved = true;
            }
          } catch (_) {}
        }
      }
      if (!resolved) { setLoading(false); setLoadingStatus(''); }
    } catch {
      setMessages((prev) => [...prev, {
        role: 'system',
        text: '❌ Sunucuya bağlanılamadı veya dosya işlenirken hata oluştu. Backend (main.py) çalışıyor mu?',
      }]);
      setLoading(false); setLoadingStatus('');
    } finally {
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
    setStatusLog([]);

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
              setStatusLog((prev) => [...prev, event.data]);
            } else if (event.type === 'done') {
              setMessages((prev) => [...prev, { role: 'system', text: event.data }]);
              setLoading(false); setLoadingStatus(''); setStatusLog([]); resolved = true;
            } else if (event.type === 'error') {
              setMessages((prev) => [...prev, { role: 'system', text: `⚠️ ${event.data}` }]);
              setLoading(false); setLoadingStatus(''); setStatusLog([]); resolved = true;
            }
          } catch (_) {}
        }
      }
      if (!resolved) { setLoading(false); setLoadingStatus(''); }
    } catch {
      setMessages((prev) => [...prev, { role: 'system', text: '❌ Sunucuya ulaşılamıyor.' }]);
      setLoading(false); setLoadingStatus('');
    }
  }, [input, loading, sessionId]);

  // ── Reset ─────────────────────────────────────────────────────────────────
  const resetChat = async () => {
    if (sessionId) {
      try { await axios.delete(`${API_BASE}/session/${sessionId}`); } catch (_) {}
    }
    setSessionId(null); setFileUploaded(false); setLoadingStatus(''); setStatusLog([]); setSummary(null);
    setMessages([{ role: 'system', text: 'Sohbet temizlendi. Yeni bir transkript yükleyebilirsin.' }]);
  };

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div className="flex h-screen bg-[#0d1017] overflow-hidden font-sans antialiased text-slate-200">

      {/* ── SIDEBAR ─────────────────────────────────────────────────────────── */}
      <div className="hidden md:flex w-56 xl:w-64 flex-shrink-0 flex-col bg-[#0c1420] border-r border-white/[0.06]">
        <div className="px-5 py-5 border-b border-white/[0.06] flex items-center gap-3">
          <div className="bg-indigo-600/15 border border-indigo-500/25 p-2 rounded-xl">
            <GraduationCap size={20} className="text-indigo-400" />
          </div>
          <div>
            <h1 className="text-slate-100 font-semibold text-[13.5px] leading-tight">Sanal Danışman</h1>
            <p className="text-indigo-400/70 text-[10.5px] mt-0.5">Llama 3.1 · Ajansal AI</p>
          </div>
        </div>

        <div className="flex-1 px-4 py-5 flex flex-col gap-3">
          <label className={`group flex items-center justify-center gap-2 w-full py-2.5 px-3
            rounded-xl text-sm font-medium border transition-all duration-200
            ${loading
              ? 'opacity-50 cursor-not-allowed bg-indigo-600/10 border-indigo-500/20 text-indigo-300'
              : 'cursor-pointer bg-indigo-600/10 border-indigo-500/20 text-indigo-300 hover:bg-indigo-600/20 hover:border-indigo-500/40'
            }`}>
            <UploadCloud size={15} className="group-hover:-translate-y-0.5 transition-transform duration-200" />
            <span>Transkript Yükle (TXT)</span>
            <input ref={fileInputRef} type="file" accept=".txt,text/plain"
              className="hidden" onChange={handleFileUpload} disabled={loading} />
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

        <div className="px-4 pb-5 border-t border-white/[0.06] pt-4">
          <button onClick={resetChat}
            className="flex items-center gap-1.5 text-[11.5px] text-slate-600 hover:text-slate-400 transition-colors w-full justify-center py-2">
            <Trash2 size={12} />
            Sohbeti Temizle
          </button>
        </div>
      </div>

      {/* ── CHAT AREA ───────────────────────────────────────────────────────── */}
      <div className="flex-1 flex flex-col min-w-0">
        <div className="flex-1 overflow-y-auto px-4 md:px-8 py-6 space-y-4">
          {messages.map((msg, i) => <MessageBubble key={i} msg={msg} />)}
          {loading && <ThinkingBubble status={loadingStatus} statusLog={statusLog} />}
          <div ref={messagesEndRef} />
        </div>

        <div className="px-4 md:px-8 pb-5 pt-3 border-t border-white/[0.06]">
          <div className="max-w-3xl mx-auto flex items-center gap-2 bg-white/[0.04] border border-white/[0.08] rounded-2xl px-4 py-1.5 focus-within:border-indigo-500/40 transition-colors duration-200">
            <input type="text" value={input} onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && sendMessage()}
              disabled={loading}
              placeholder={fileUploaded ? 'Transkriptinle ilgili bir soru sor...' : 'Önce sol menüden transkript yükle...'}
              className="flex-1 bg-transparent border-none py-3 text-slate-200 placeholder-slate-600 focus:outline-none text-[14px] disabled:opacity-50" />
            <button onClick={sendMessage} disabled={loading || !input.trim()}
              className="bg-indigo-600 hover:bg-indigo-500 active:bg-indigo-700 text-white p-2.5 rounded-xl transition-colors duration-150 disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center">
              {loading ? <Loader2 size={15} className="animate-spin" /> : <Send size={15} />}
            </button>
          </div>
          <p className="text-center text-[11px] text-slate-700 mt-2.5">
            Verileriniz yalnızca yerel Llama 3.1 modeliyle işlenir — hiçbir şey cihazınızdan çıkmaz.
          </p>
        </div>
      </div>

      {/* ── STATS PANEL (+ modals rendered inside) ──────────────────────────── */}
      {summary && <StatsPanel summary={summary} />}
    </div>
  );
}
