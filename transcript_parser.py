import pdfplumber
import json
from langchain_ollama import OllamaLLM

# 1. Modeli Tanımla
model = OllamaLLM(model="llama3.1:8b")

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def parse_transcript_with_ai(raw_text):
    prompt = f"""
    Aşağıdaki metin bir öğrencinin resmi üniversite transkriptidir. 
    Lütfen bu metni analiz et ve öğrencinin aldığı dersleri, durumlarını ve harf notlarını ayıkla.
    SADECE geçerli bir JSON formatında yanıt ver. 
    Ders kodlarını tam olarak (Örn: INF101) yakalamaya dikkat et.

    JSON Formatı:
    {{
        "ogrenci_adi": "Bulabiliyorsan İsim",
        "alinan_dersler": [
            {{
                "kod": "INF101",
                "ad": "Dersin Tam Adı",
                "harf_notu": "BA",
                "durum": "Geçti veya Kaldı"
            }}
        ]
    }}

    Transkript Metni:
    {raw_text}
    """
    
    response = model.invoke(prompt)
    
    # Markdown temizliği
    if "```json" in response:
        response = response.split("```json")[1].split("```")[0].strip()
    elif "```" in response:
        response = response.split("```")[1].strip()
        
    return response

# --- ANA AKIŞ ---
pdf_file = "transkript.pdf" # Buraya kendi transkript dosyanın adını yaz
print(f"📄 {pdf_file} okunuyor...")

try:
    # PDF'den ham metni çıkar
    raw_content = extract_text_from_pdf(pdf_file)
    
    print("🧠 Llama 3.1 transkripti analiz ediyor (Bu işlem 30-60 sn sürebilir)...")
    parsed_json = parse_transcript_with_ai(raw_content)
    
    # Sonucu dosyaya kaydet
    with open("transkript_gercek.json", "w", encoding="utf-8") as f:
        f.write(parsed_json)
        
    print("✅ Başarılı! 'transkript_gercek.json' oluşturuldu.")

except Exception as e:
    print(f"❌ Bir hata oluştu: {e}")