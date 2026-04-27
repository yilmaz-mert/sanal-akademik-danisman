import json
from langchain_ollama import OllamaLLM

# Llama 3.1 8B Modelimizi çağırıyoruz
model = OllamaLLM(model="llama3.1:8b")

def create_json_from_text(raw_text_chunk):
    prompt = f"""
    Aşağıdaki ham metin Galatasaray Üniversitesi Bilgisayar Mühendisliği müfredatından alınmıştır. 
    Lütfen metni baştan sona dikkatlice oku ve İÇİNDEKİ TÜM DERSLERİ ayıklayarak SADECE geçerli bir JSON formatında döndür.
    Hiçbir dersi atlama. Başka hiçbir açıklama metni yazma.
    
    Format:
    {{
        "dersler": [
            {{
                "kod": "Ders Kodu (Örn: INF101)",
                "ad": "Ders Adı",
                "akts": 6,
                "tip": "Zorunlu/Seçmeli",
                "donem": 1
            }}
        ]
    }}

    Metin:
    {raw_text_chunk}
    """
    
    response = model.invoke(prompt)
    return response

# Ham metnin TAMAMINI oku
with open("gs-akts_ham_metin.txt", "r", encoding="utf-8") as f:
    full_text = f.read() # readlines yerine read() kullanıp tamamını tek metin yapıyoruz

print("⏳ Llama 3.1 tüm müfredatı (1200 satır) inceliyor... Bu işlem RTX 2080 ile 1-2 dakika sürebilir, lütfen bekle...")
json_result = create_json_from_text(full_text)

# JSON formatını temizleme (Eğer model başına/sonuna ```json gibi markdown eklerse diye)
if "```json" in json_result:
    json_result = json_result.split("```json")[1].split("```")[0].strip()
elif "```" in json_result:
    json_result = json_result.split("```")[1].strip()

# Sonucu Kaydet
with open("mufredat_tam.json", "w", encoding="utf-8") as f:
    f.write(json_result)

print("✅ Bütün dersler başarıyla çıkarıldı! 'mufredat_tam.json' dosyasını kontrol edebilirsin.")