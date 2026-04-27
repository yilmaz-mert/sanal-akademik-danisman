import { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { Send, UploadCloud, Bot, User, GraduationCap, Sparkles, Loader2, Trash2 } from 'lucide-react';

function App() {
  const [messages, setMessages] = useState([
    { 
      role: 'system', 
      text: 'Merhaba Mert! Ben senin Sanal Akademik Danışmanınım. Mezuniyet durumunu, eksik kredilerini veya alman gereken dersleri analiz etmek için hazırım. Lütfen sol menüden güncel transkriptini (PDF) yükle.' 
    }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [fileUploaded, setFileUploaded] = useState(false);
  const messagesEndRef = useRef(null);

  // Her yeni mesajda en alta kaydır
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setLoading(true);
    setFileUploaded(true);
    setMessages(prev => [...prev, { role: 'user', text: `📄 Dosya yüklendi: ${file.name}\nLütfen bu transkripti analiz et.` }]);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://127.0.0.1:8000/upload', formData);
      setMessages(prev => [...prev, { role: 'system', text: response.data.reply }]);
    } catch (error) {
      setMessages(prev => [...prev, { role: 'system', text: '❌ Sunucuya bağlanılamadı veya dosya işlenirken bir hata oluştu. Lütfen Backend (main.py) sunucusunun çalıştığından emin ol.' }]);
    }
    setLoading(false);
  };

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage = input;
    setInput('');
    setMessages(prev => [...prev, { role: 'user', text: userMessage }]);
    setLoading(true);

    try {
      const response = await axios.post('http://127.0.0.1:8000/chat', { message: userMessage });
      setMessages(prev => [...prev, { role: 'system', text: response.data.reply }]);
    } catch (error) {
      setMessages(prev => [...prev, { role: 'system', text: '❌ Sunucuya ulaşılamıyor.' }]);
    }
    setLoading(false);
  };

  const resetChat = () => {
    setMessages([{ role: 'system', text: 'Sohbet temizlendi. Yeni bir transkript yükleyebilir veya soru sorabilirsin.' }]);
    setFileUploaded(false);
  };

  return (
    <div className="flex h-screen bg-gray-50 font-sans antialiased overflow-hidden">
      
      {/* SOL MENÜ (SIDEBAR) */}
      <div className="w-80 bg-slate-900 text-slate-300 flex flex-col shadow-2xl z-10 hidden md:flex">
        <div className="p-6 border-b border-slate-700/50 flex items-center gap-3">
          <div className="bg-indigo-500 p-2 rounded-lg text-white">
            <GraduationCap size={28} />
          </div>
          <div>
            <h1 className="text-white font-bold text-lg tracking-wide">Sanal Danışman</h1>
            <p className="text-xs text-indigo-300 font-medium">Llama 3.1 Destekli</p>
          </div>
        </div>

        <div className="p-6 flex-1 flex flex-col gap-4">
          <p className="text-sm text-slate-400 mb-2">Başlamak için güncel bir transkript yükleyin.</p>
          
          <label className="cursor-pointer group relative flex items-center justify-center gap-2 w-full bg-indigo-600 hover:bg-indigo-500 text-white py-3 px-4 rounded-xl font-medium transition-all duration-200 shadow-lg shadow-indigo-900/20">
            <UploadCloud size={20} className="group-hover:-translate-y-1 transition-transform" />
            <span>Transkript Yükle (PDF)</span>
            <input type="file" accept="application/pdf" className="hidden" onChange={handleFileUpload} />
          </label>

          {fileUploaded && (
            <div className="mt-4 p-4 bg-slate-800/50 rounded-xl border border-emerald-500/30 flex items-start gap-3">
              <Sparkles className="text-emerald-400 shrink-0 mt-0.5" size={18} />
              <p className="text-sm text-slate-200">Sistem güncel transkriptini hafızaya aldı. Artık dilediğin senaryoyu sorabilirsin.</p>
            </div>
          )}
        </div>

        <div className="p-6 border-t border-slate-700/50">
          <button onClick={resetChat} className="flex items-center gap-2 text-sm text-slate-400 hover:text-white transition-colors w-full justify-center py-2">
            <Trash2 size={16} /> Sohbeti Temizle
          </button>
        </div>
      </div>

      {/* SAĞ TARAF - SOHBET ALANI */}
      <div className="flex-1 flex flex-col relative h-full">
        
        {/* Mesajlar */}
        <div className="flex-1 overflow-y-auto p-4 md:p-8 space-y-6 scroll-smooth">
          {messages.map((msg, index) => (
            <div key={index} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`flex max-w-[85%] md:max-w-[70%] gap-4 ${msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
                
                {/* Avatar */}
                <div className={`shrink-0 h-10 w-10 rounded-full flex items-center justify-center shadow-sm ${msg.role === 'user' ? 'bg-indigo-100 text-indigo-600' : 'bg-white border-2 border-indigo-100 text-indigo-600'}`}>
                  {msg.role === 'user' ? <User size={20} /> : <Bot size={22} />}
                </div>

                {/* Mesaj Balonu */}
                <div className={`p-4 md:p-5 rounded-2xl shadow-sm text-[15px] leading-relaxed ${
                  msg.role === 'user' 
                    ? 'bg-indigo-600 text-white rounded-tr-none' 
                    : 'bg-white border border-gray-100 text-gray-800 rounded-tl-none'
                }`}>
                  <p className="whitespace-pre-wrap">{msg.text}</p>
                </div>
              </div>
            </div>
          ))}
          
          {loading && (
            <div className="flex justify-start">
              <div className="flex gap-4 max-w-[70%]">
                <div className="shrink-0 h-10 w-10 rounded-full bg-white border-2 border-indigo-100 text-indigo-600 flex items-center justify-center">
                  <Bot size={22} />
                </div>
                <div className="p-5 rounded-2xl rounded-tl-none bg-white border border-gray-100 flex items-center gap-3">
                  <Loader2 className="animate-spin text-indigo-500" size={20} />
                  <span className="text-gray-500 font-medium">Ajan düşünüyor...</span>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Alt Input Alanı */}
        <div className="p-4 md:p-6 bg-transparent">
          <div className="max-w-4xl mx-auto bg-white rounded-2xl shadow-lg border border-gray-200 p-2 pl-4 flex items-center gap-2 focus-within:ring-2 focus-within:ring-indigo-500/50 transition-all">
            <input 
              type="text" 
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
              disabled={loading}
              placeholder={fileUploaded ? "Transkriptle ilgili bir soru sor..." : "Önce sol menüden transkript yüklemelisin..."}
              className="flex-1 bg-transparent border-none py-3 px-2 text-gray-700 focus:outline-none focus:ring-0 disabled:opacity-50"
            />
            <button 
              onClick={sendMessage} 
              disabled={loading || !input.trim()}
              className="bg-indigo-600 text-white p-3 md:px-6 rounded-xl hover:bg-indigo-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 font-medium"
            >
              <span className="hidden md:inline">Gönder</span>
              <Send size={18} />
            </button>
          </div>
          <p className="text-center text-xs text-gray-400 mt-3">Sanal Akademik Danışman, yerel Llama 3.1 modeli ile çalışmaktadır. Hassas verileriniz cihazınızdan dışarı çıkmaz.</p>
        </div>
      </div>
    </div>
  );
}

export default App;