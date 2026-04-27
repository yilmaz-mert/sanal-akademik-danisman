import os
import json
import re
import pdfplumber
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Model ve Hafıza Kurulumu
model = OllamaLLM(
    model="llama3.1:8b", 
    temperature=0.1,  # Yaratıcılığı azalt, mantığı artır
    num_ctx=4096      # Bağlam penceresini geniş tut
)
history = ChatMessageHistory()

# 2. Ajanın Genel Sohbet Kalıbı
template = """
SİSTEM: Sen Galatasaray Üniversitesi'nde çalışan kıdemli, samimi ve %100 dürüst bir Akademik Danışmansın.
Öğrenci senin bizzat karşında oturuyor. Lütfen ona doğrudan (Sen/Siz) hitap et.

Geçmiş Konuşmalar:
{chat_history}

Öğrenci: {question}
Danışman:"""
prompt_template = PromptTemplate(input_variables=["chat_history", "question"], template=template)

class ChatRequest(BaseModel):
    message: str

@app.post("/upload")
async def upload_transcript(file: UploadFile = File(...)):
    print("\n--- YENİ TRANSKRİPT ANALİZİ BAŞLADI ---")
    
    # 1. MÜFREDATI YÜKLE (Kök dizinden)
    mufredat_yolu = os.path.join("..", "mufredat_tam_chunked.json")
    try:
        with open(mufredat_yolu, "r", encoding="utf-8") as f:
            mufredat_data = json.load(f)
            mufredat = mufredat_data.get("dersler", [])
            print(f"✅ Müfredat yüklendi. Toplam Ders: {len(mufredat)}")
    except FileNotFoundError:
        print("❌ HATA: Müfredat dosyası bulunamadı!")
        return {"reply": "Sistem Hatası: Müfredat dosyasına ulaşılamıyor. Lütfen dosyayı kontrol edin."}

    # 2. PDF'DEN METNİ ÇIKAR
    text = ""
    with pdfplumber.open(file.file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    
    text_lower = text.lower()
    
    # 3. KODLARI BUL (Regex)
    bulunan_kodlar = set(re.findall(r'[A-Z]{3}\d{3}(?:-[A-Z])?', text))
    
    # 4. HİBRİT MANTIK MOTORU (Kod + Semantik Eşleşme)
    eksik_zorunlu_dersler = []
    eslesen_dersler_log = [] # Debug için
    toplam_akts = 0
    
    for ders in mufredat:
        ders_kodu = ders["kod"]
        ders_adi = ders["ad"]
        ders_adi_lower = ders_adi.lower()
        
        # A) Kesin Eşleşme (Ders Koduyla)
        if ders_kodu in bulunan_kodlar:
            toplam_akts += ders.get("akts", 0)
            eslesen_dersler_log.append(f"{ders_kodu} (Kod ile eşleşti)")
            
        # B) Esnek/Semantik Eşleşme (Ders İsmiyle - Eski Transkriptler İçin)
        elif ders_adi_lower in text_lower:
            toplam_akts += ders.get("akts", 0)
            eslesen_dersler_log.append(f"{ders_kodu} - {ders_adi} (İSİM ile eşleşti)")
            
        # C) Ders Yok ve Zorunluysa -> Eksik Listesine Ekle
        elif ders.get("tip") == "Zorunlu":
            eksik_zorunlu_dersler.append(f"{ders_kodu} - {ders_adi}")

    # 5. GÜVENLİK VE KONTROL (DEBUG DOSYASI OLUŞTURMA)
    debug_verisi = {
        "bulunan_ham_kodlar": list(bulunan_kodlar),
        "eslesen_dersler": eslesen_dersler_log,
        "hesaplanan_akts": toplam_akts,
        "eksik_zorunlular": eksik_zorunlu_dersler
    }
    with open("son_ayiklanan_transcript.json", "w", encoding="utf-8") as f:
        json.dump(debug_verisi, f, ensure_ascii=False, indent=4)
    print(f"✅ Analiz bitti. Hesaplanan AKTS: {toplam_akts}")
    print("📂 Detaylar 'son_ayiklanan_transcript.json' dosyasına yazıldı.")

    # 6. LLM'E KESİN EMİR (Prompt Injection)
    ilk_soru = f"""
    ADIM ADIM DÜŞÜN VE ANALİZ ET (INTERNAL MONOLOGUE):
    1. Önce transkriptteki ders kodlarını müfredat JSON'u ile matematiksel olarak eşleştir.
    2. Eğer kod tutmuyorsa, ders isminin (Örn: Matematik) müfredattaki karşılığıyla benzerliğini kontrol et.
    3. Toplam AKTS hesabını yaparken sadece "Geçti" statüsündeki dersleri say.
    4. Kendi analizini bir kez "Hata var mı?" diye kontrol et.
    5. Sonuçları öğrenciye (Mert/Ceyhun) samimi bir dille aktar.

    SİSTEMİN TESPİT ETTİĞİ GERÇEKLER:
    - Hesaplanan AKTS: {toplam_akts}
    - Eksik Zorunlu Dersler: {', '.join(eksik_zorunlu_dersler) if eksik_zorunlu_dersler else 'Yok'}

    MÜFREDAT REFERANSI:
   

    DANIŞMANIN ÖĞRENCİYE MESAJI:
    (Kişinin transkript verilerine dayanarak doğrudan hitap et.)
    """
    
    chat_history_str = "\n".join([f"{m.type}: {m.content}" for m in history.messages])
    formatted_prompt = prompt_template.format(chat_history=chat_history_str, question=ilk_soru)
    
    response = model.invoke(formatted_prompt)
    
    # Hafızaya temiz bir şekilde kaydet (Karmaşık promptu değil, sadece öğrencinin hamlesini)
    history.add_user_message("Hocam transkriptimi getirdim, mezuniyet durumumu analiz eder misiniz?")
    history.add_ai_message(response)
    
    return {"reply": response}

@app.post("/chat")
async def chat_with_agent(request: ChatRequest):
    # 1. Verileri Oku (Kopya Kağıdı)
    try:
        with open("son_ayiklanan_transcript.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            # Ajanın sadece referans alacağı JSON verisi
            context_data = {
                "gecilen_dersler": data.get("eslesen_dersler", []),
                "toplam_akts": data.get("hesaplanan_akts", 0),
                "eksikler": data.get("eksik_zorunlular", [])
            }
    except FileNotFoundError:
        context_data = "Sistem Bilgisi: Öğrenci henüz transkript yüklemedi."

    # --- GELECEK SORUNLARA KALKAN: HAFIZA YÖNETİMİ ---
    # Eğer sohbet geçmişi 8 mesajı (4 soru - 4 cevap) geçerse, en eskileri sil.
    # Bu sayede model asla şişmez ve yavaşlamaz.
    if len(history.messages) > 8:
        history.messages = history.messages[-8:]

    chat_history_str = "\n".join([f"{m.type}: {m.content}" for m in history.messages])

    # --- TEKRARLAMA SORUNUNA KALKAN: KATI SİSTEM PROMPTU ---
    sistem_talimati = f"""
    [GİZLİ SİSTEM VERİSİ - ÖĞRENCİ BİLGİLERİ]
    {json.dumps(context_data, ensure_ascii=False)}

    KATI KURALLAR:
    1. Öğrenci spesifik olarak "bütün listeyi say" veya "neleri geçtim dök" DEMEDİĞİ SÜRECE, yukarıdaki listeyi ASLA cevabına ekleme.
    2. Sadece ve sadece öğrencinin sorduğu şeye net bir cevap ver. (Örn: Sadece INF340 soruluyorsa, sadece onun AKTS'sini söyle ve sus).
    3. Konuşmayı uzatma, laf kalabalığı yapma.
    """

    # Kopya kağıdını ve kuralları, kullanıcının sorusunun arkasına gizliyoruz
    final_prompt = f"{sistem_talimati}\n\nÖğrencinin Yeni Sorusu: {request.message}"
    
    formatted_prompt = prompt_template.format(chat_history=chat_history_str, question=final_prompt)
    response = model.invoke(formatted_prompt)
    
    # Hafızaya sadece yalın soruyu kaydediyoruz (Kopya kağıdını hafızaya yazıp şişirmiyoruz)
    history.add_user_message(request.message)
    history.add_ai_message(response)
    
    return {"reply": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)