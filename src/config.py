"""Konfigurasi dan konstanta global untuk GitBot.

Modul ini menyimpan seluruh konfigurasi terpusat untuk aplikasi GitBot,
termasuk path berkas, nama model pemrosesan bahasa alami (NLP),
ambang batas (threshold) kemiripan, batas panjang masukan teks,
serta kamus normalisasi kata percakapan (slang).
"""

from pathlib import Path

# Path direktori dasar proyek (root repository)
# Menentukan lokasi root proyek secara dinamis berdasarkan lokasi berkas config.py
BASE_DIR = Path(__file__).resolve().parent.parent

# Path berkas knowledge base JSON
# Artinya: Lokasi berkas basis pengetahuan yang memuat seluruh intent, contoh pertanyaan, dan jawaban Git.
# Alasan nilai: Mengarah ke 'data/knowledge_base.json' agar data pengetahuan terpusat dan terpisah dari logika kode.
PATH_KB = BASE_DIR / "data" / "knowledge_base.json"

# Path berkas kumpulan pertanyaan pengujian (test set) CSV
# Artinya: Lokasi berkas CSV yang berisi pertanyaan uji beserta label intent target untuk evaluasi model.
# Alasan nilai: Mengarah ke 'data/test_questions.csv' untuk memisahkan data uji dari contoh di knowledge base.
PATH_TEST = BASE_DIR / "data" / "test_questions.csv"

# Model sentence embedding utama
# Artinya: Pengenal (identifier) model SentenceTransformer yang digunakan untuk representasi vektor semantik kalimat.
# Alasan nilai: 'paraphrase-multilingual-MiniLM-L12-v2' mendukung 50+ bahasa, ringan untuk CPU laptop (i3, RAM 16 GB), dan kualitas untuk bahasa Indonesia diuji di Tahap 10-11.
MODEL_EMBEDDING = "paraphrase-multilingual-MiniLM-L12-v2"

# Model IndoBERT untuk pembanding
# Artinya: Pengenal model pretrained IndoBERT mentah yang digunakan sebagai metode pembanding representasi kalimat berbasis BERT bahasa Indonesia.
# Alasan nilai: 'indobenchmark/indobert-base-p1' adalah model representasi teks bahasa Indonesia standar untuk perbandingan kontekstual tanpa fine-tuning.
MODEL_INDOBERT = "indobenchmark/indobert-base-p1"

# Ambang batas kemiripan minimum (cosine similarity threshold)
# Artinya: Batas skor cosine similarity terendah agar pertanyaan dianggap cocok dengan suatu intent; jika di bawah nilai ini, respons dialihkan ke fallback.
# Alasan nilai: 'tfidf' disetel 0.2 karena representasi kata sparse menghasilkan skor kemiripan lebih rendah, sedangkan 'embedding' dan 'indobert' disetel 0.5 sebagai batas kemiripan semantik awal sebelum dituning pada Tahap 11.
THRESHOLD = {
    "tfidf": 0.2,
    "embedding": 0.5,
    "indobert": 0.5,
}

# Batas maksimal panjang karakter input pengguna
# Artinya: Batas atas jumlah karakter dalam satu pesan pertanyaan yang diizinkan untuk diproses oleh chatbot.
# Alasan nilai: 300 karakter sudah memadai untuk pertanyaan teknis Git (1-2 kalimat) sekaligus mencegah komputasi berlebihan dan masukan anomali.
MAKS_PANJANG_INPUT = 300

# Kamus kata percakapan/slang untuk normalisasi teks
# Artinya: Pasangan pemetaan dari kata tidak baku (gaul/singkatan) ke bentuk kata baku atau istilah Git standar.
# Alasan nilai: Dict kecil ini memuat variasi kata percakapan umum sehari-hari dan variasi istilah Git lokal (seperti ngepush, ngemerge) agar pemrosesan teks tidak merusak kata kunci penting Git.
KAMUS_SLANG = {
    "gw": "saya",
    "gua": "saya",
    "gue": "saya",
    "lu": "kamu",
    "lo": "kamu",
    "elo": "kamu",
    "gimana": "bagaimana",
    "gmn": "bagaimana",
    "bikin": "buat",
    "ngapus": "hapus",
    "udah": "sudah",
    "udh": "sudah",
    "blm": "belum",
    "belom": "belum",
    "ga": "tidak",
    "gak": "tidak",
    "nggak": "tidak",
    "kalo": "kalau",
    "kl": "kalau",
    "batalin": "batalkan",
    "ngepush": "push",
    "ngemerge": "merge",
    "ngestash": "stash",
    "ngeclone": "clone",
    "pake": "pakai",
    "banget": "sangat",
    "aja": "saja",
    "tp": "tapi",
    "trs": "terus",
      "yg": "yang",
    "msh": "masih",
    "abis": "habis",
    "temen": "teman",
    "pengen": "ingin",
    "enggak": "tidak",
    "engga": "tidak",
    "ngga": "tidak",
    "klo": "kalau",
    "dgn": "dengan",
    "utk": "untuk",
    "jd": "jadi",
    "krn": "karena",
    "kpn": "kapan",
    "dah": "sudah",
    "ngepull": "pull",
    "ngecommit": "commit",
    "ngefetch": "fetch",
    "ngecek": "cek",
}
