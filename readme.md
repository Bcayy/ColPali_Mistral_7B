# ColPali: Görsel & PDF Tabanlı Belge Anlama Sistemi

ColPali, PDF veya görsel belgelerden doğal dilde bilgi çekmeyi sağlayan, OCR + embedding + LLM + RAG mimarilerini birleştiren bir belge anlama sistemidir.

## 🚀 Özellikler

- PDF/görsel içerik yükleme
- OCR ile metin çıkarımı
- Sayfa içeriğini embedding'e dönüştürme (ColQwen2)
- Doğal dil sorgusu ile belge içeriği eşleştirme (cosine similarity + Plaid index)
- Mistral-7B LLM ile RAG destekli cevap üretimi
- Cevabın bulunduğu alanı görselde highlight etme
- Gradio arayüzü ile etkileşimli kullanım

## 🔧 Kurulum

```bash
git clone https://github.com/kullaniciadi/colpali-clone.git
cd colpali-clone
pip install -r requirements.txt
```

> Not: `pytesseract` için sisteminizde Tesseract OCR'ın yüklü olması gerekir. [Yükleme Talimatları](https://github.com/tesseract-ocr/tesseract)

## 🧠 Kullanılan Modeller

- **LLM**: `Mistral-7B-Instruct` (transformers üzerinden HF token ile yükleniyor)
- **Embedding**: `ColQwen2` engine
- **OCR**: `pytesseract`

## 💡 Nasıl Çalışır?

1. Kullanıcı bir PDF veya görsel yükler.
2. Sayfalar OCR ile metne çevrilir.
3. Her sayfa embedding vektörüne dönüştürülür (ve cache'e alınır).
4. Kullanıcının sorusu embed edilir.
5. Sayfalar cosine similarity + plaid index ile sıralanır.
6. En iyi 3 sayfa seçilir, LLM'e prompt olarak verilir.
7. LLM'den gelen cevaptaki ifade, OCR kutularıyla eşleştirilerek görselde highlight edilir.

## 📦 Kod Yapısı

- `Colpali_Mistral_7B_LangChain_Final`: Gradio arayüzü ve tüm iş mantığı
- `embedding_cache.pkl`: Sayfa vektörleri için cache dosyası
- `draw_highlight`: Görsel üzerine şeffaf highlight kutusu
- `CleanHFParser`: LLM çıktısından "Answer:" sonrası metni çeker

## 🧪 Çalıştırma

Google Colab üzerinden çalıştırabilirsiniz.

## ✍️ Geliştiren

**Burak Akçay**  
[LinkedIn](https://www.linkedin.com/in/burakakcay44/)  
[burakakcay44@gmail.com](mailto:burakakcay44@gmail.com)

---

Bu proje gerçek dünya problemlerini çözmek için OCR + NLP + RAG mimarilerini birleştiren güçlü bir çözümdür.
