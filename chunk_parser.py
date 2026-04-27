import json
from langchain_ollama import OllamaLLM

model = OllamaLLM(model="llama3.1:8b")

# Dosyayı satır satır oku
with open("gs-akts_ham_metin.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Metni 150 satırlık parçalara (chunk) böl
chunk_size = 150
chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]

tum_dersler = []

print(f"Toplam {len(chunks)} parça işlenecek. Bu işlem biraz sürebilir...")

for i, chunk in enumerate(chunks):
    text_chunk = "".join(chunk)
    print(f"⏳ {i+1}. Parça Llama 3.1 tarafından inceleniyor...")
    
    prompt = f"""
    Aşağıdaki metinden SADECE ders bilgilerini çıkar ve JSON formatında ver. 
    Eğer bu metin parçasında hiç ders yoksa, sadece boş bir liste döndür: {{"dersler": []}}
    Başka hiçbir açıklama yazma.
    
    Format:
    {{
        "dersler": [
            {{"kod": "INF101", "ad": "Matematik", "akts": 6, "tip": "Zorunlu", "donem": 1}}
        ]
    }}
    
    Metin:
    {text_chunk}
    """
    
    try:
        response = model.invoke(prompt)
        
        # Markdown kod bloklarını temizle
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].strip()
            
        data = json.loads(response) # String'i Python sözlüğüne çevir
        if "dersler" in data:
            tum_dersler.extend(data["dersler"]) # Bulunan dersleri ana listeye ekle
            print(f"   -> {len(data['dersler'])} ders bulundu.")
    except Exception as e:
        print(f"⚠️ {i+1}. parçada JSON anlaşılamadı veya ders yok, atlanıyor...")

# Tüm döngü bitince her şeyi tek bir dosyaya kaydet
final_json = {"universite": "Galatasaray", "bolum": "Bilgisayar Mühendisliği", "dersler": tum_dersler}

with open("mufredat_tam_chunked.json", "w", encoding="utf-8") as f:
    json.dump(final_json, f, ensure_ascii=False, indent=4)

print(f"\n✅ İşlem bitti! Toplam {len(tum_dersler)} ders çıkarıldı. 'mufredat_tam_chunked.json' dosyasını kontrol et.")