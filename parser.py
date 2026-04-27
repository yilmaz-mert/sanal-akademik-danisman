import pdfplumber
import os

def extract_text_from_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        return f"Hata: {pdf_path} dosyası bulunamadı! Lütfen PDF'i proje klasörüne attığından emin ol."
    
    with pdfplumber.open(pdf_path) as pdf:
        full_text = ""
        for page in pdf.pages:
            # Sayfadaki metni çıkar
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n--- SAYFA AYRIMI ---\n"
    return full_text

# Dosya ismi
pdf_file = "gs-akts.pdf"

print("⏳ PDF okunuyor ve metne dönüştürülüyor...")

extracted_data = extract_text_from_pdf(pdf_file)

# Sonucu bir text dosyasına kaydedelim ki gözle kontrol edebilelim
with open("gs-akts_ham_metin.txt", "w", encoding="utf-8") as f:
    f.write(extracted_data)

print("✅ İşlem tamamlandı! 'mufredat_ham_metin.txt' dosyasını kontrol edebilirsin.")