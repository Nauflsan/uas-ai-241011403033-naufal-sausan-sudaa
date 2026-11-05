# GitBot — Asisten Pintar Git & GitHub

GitBot adalah aplikasi chatbot cerdas berbasis pengetahuan terstruktur yang dirancang untuk membantu pengguna memahami dan menyelesaikan berbagai permasalahan terkait penggunaan perintah dan alur kerja Git & GitHub. 

Aplikasi ini dikembangkan untuk tugas Ujian Akhir Semester (UAS) Kecerdasan Buatan dan dirancang agar dapat berjalan secara sepenuhnya lokal di komputer dengan CPU saja (tanpa GPU dan tanpa memanggil API AI eksternal).

---

## Fitur Utama

- **Pencocokan Intent Semantik**: Menggunakan sentence embedding `paraphrase-multilingual-MiniLM-L12-v2` dan cosine similarity untuk menangani pertanyaan berbahasa Indonesia formal maupun gaul.
- **Model Pembanding**: Dilengkapi dengan pembanding berbasis TF-IDF (baseline) dan IndoBERT mentah (`indobenchmark/indobert-base-p1`).
- **Mekanisme Fallback**: Menghindari jawaban halusinasi dengan memastikan skor kemiripan memenuhi batas ambang minimum (*threshold*).
- **Dua Pilihan Antarmuka**:
  - Antarmuka Terminal/CLI interaktif.
  - Antarmuka Web interaktif berbasis Streamlit.

---

## Struktur Proyek

```text
uas-ai-nim-nama/
├── README.md                 # Dokumentasi proyek
├── requirements.txt          # Daftar pustaka/dependensi
├── app.py                    # Antarmuka web Streamlit
├── src/
│   ├── __init__.py
│   ├── config.py             # Konfigurasi dan konstanta
│   ├── knowledge_base.py     # Pemuat dan validator knowledge base
│   ├── preprocessing.py      # Normalisasi dan prapemrosesan teks
│   ├── matcher.py            # Modul pencocok (TF-IDF, Embedding, IndoBERT)
│   ├── chatbot.py            # Logika utama chatbot GitBot
│   └── evaluate.py           # Skrip evaluasi performa metode
├── data/
│   ├── knowledge_base.json   # Basis data pengetahuan Git
│   └── test_questions.csv    # Kumpulan pertanyaan uji evaluasi
├── tests/                    # Berkas pengujian otomatis (pytest)
├── hasil/                    # Hasil keluaran dan grafik evaluasi
└── docs/screenshots/         # Tangkapan layar bukti pengujian
```

---

## Prasyarat & Instalasi

### 1. Prasyarat Sistem
- Python 3.10 atau versi yang lebih baru.
- Sistem Operasi Windows (atau Linux/macOS yang kompatibel).
- Akses internet hanya dibutuhkan sekali di awal untuk mengunduh model embedding Hugging Face.

### 2. Pemasangan Virtual Environment
Buka terminal PowerShell pada direktori proyek, lalu aktifkan virtual environment:

```powershell
# Buat virtual environment (jika belum ada)
python -m venv venv

# Aktifkan virtual environment di Windows
venv\Scripts\activate
```

### 3. Instalasi Pustaka
Pasang seluruh dependensi yang diperlukan:

```powershell
pip install -r requirements.txt
```

---

## Cara Menjalankan

Pastikan virtual environment telah aktif sebelum menjalankan perintah-perintah berikut:

### 1. Menjalankan Chatbot di Terminal
```powershell
python -m src.chatbot
```

### 2. Menjalankan Antarmuka Web (Streamlit)
```powershell
streamlit run app.py
```

### 3. Menjalankan Pengujian Otomatis (PyTest)
```powershell
python -m pytest tests -q
```

### 4. Menjalankan Skrip Evaluasi Model
```powershell
python -m src.evaluate
```

---

## Teknologi yang Digunakan

- **Python 3.10+**: Bahasa pemrograman utama.
- **Sentence-Transformers**: Ekstraksi embedding representasi kalimat (`paraphrase-multilingual-MiniLM-L12-v2`).
- **Scikit-Learn**: Representasi TF-IDF dan kalkulasi cosine similarity.
- **PySastrawi**: Normalisasi morfologi dan stemming bahasa Indonesia.
- **Streamlit**: Antarmuka visual web interaktif.
- **Pandas & Matplotlib**: Pengolahan dataset evaluasi dan visualisasi metrik performa.
- **Pytest**: Kerangka kerja pengujian unit otomatis.
