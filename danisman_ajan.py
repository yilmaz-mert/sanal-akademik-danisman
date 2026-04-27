import json
from langchain_ollama import OllamaLLM

# 1. Modeli Başlat
model = OllamaLLM(model="llama3.1:8b")

# 2. Dosyaları Oku
print("📂 Veriler yükleniyor...")
with open("mufredat_tam_chunked.json", "r", encoding="utf-8") as f:
    mufredat = json.load(f)

with open("transkript_ornek.json", "r", encoding="utf-8") as f:
    transkript = json.load(f)

# 3. Ajana Verilecek "Prompt" (Talimat)
prompt = f"""
Sen Galatasaray Üniversitesi'nde HATA KABUL ETMEYEN bir akademik danışmansın. 

MEZUNİYET İÇİN MUTLAK ŞARTLAR:
1. Öğrenci müfredattaki TÜM 'Zorunlu' dersleri 'Geçti' statüsünde tamamlamış olmalı.
2. Toplam AKTS en az 240 olmalı.
3. Hazırlık ve Staj tamamlanmış olmalı.

VERİLER:
MÜFREDAT: {json.dumps(mufredat, ensure_ascii=False)}
TRANSKRİPT: {json.dumps(transkript, ensure_ascii=False)}

GÖREVİN:
- Tek tek 'Zorunlu' dersleri kontrol et. Eğer müfredatta 'Zorunlu' olup da transkriptte 'Geçti' olarak görünmeyen TEK BİR DERS BİLE VARSA, sonuç kesinlikle "MEZUN OLAMAZ" olmalıdır.
- AKTS toplamını matematiksel olarak doğrula.

RAPOR FORMATI:
- Eksik Zorunlu Dersler: (Varsa liste)
- AKTS Durumu: (Toplam AKTS / 240)
- KARAR: (MEZUN OLABİLİR veya MEZUN OLAMAZ)
- NEDEN: (Kararının gerekçesini tek cümlede açıkla)
"""

print("🧠 Sanal Danışman (Llama 3.1) verileri analiz ediyor, lütfen bekleyin...")

# 4. Modeli Çalıştır ve Raporu Al
rapor = model.invoke(prompt)

print("\n" + "="*50)
print("🎓 SANAL AKADEMİK DANIŞMAN RAPORU")
print("="*50)
print(rapor)