# AGENTS.md — Aturan Agent

Proyek: **GitBot**, chatbot asisten Git & GitHub berbasis pengetahuan terstruktur (UAS Kecerdasan Buatan, individu).
Jalan lokal di laptop CPU-only (i3 lama, RAM 16 GB). Dilarang memanggil API AI eksternal.
Rencana kerja per tahap ada di `PLAN-KODE.md`.

## Stack (jangan tambah library tanpa izin)

Python 3.10+, sentence-transformers, scikit-learn, PySastrawi, streamlit, pandas, matplotlib, pytest.

## Struktur (tetap, jangan diubah)

```
uas-ai-nim-nama/
├── README.md
├── requirements.txt
├── app.py                    # UI Streamlit
├── src/
│   ├── __init__.py
│   ├── config.py             # konstanta
│   ├── knowledge_base.py     # muat & validasi KB
│   ├── preprocessing.py      # normalisasi teks
│   ├── matcher.py            # TfidfMatcher, EmbeddingMatcher, IndoBertMatcher
│   ├── chatbot.py            # GitBot: validasi input, threshold, fallback
│   └── evaluate.py           # evaluasi & perbandingan metode
├── data/
│   ├── knowledge_base.json   # ditulis user
│   └── test_questions.csv    # ditulis user
├── tests/
├── hasil/                    # keluaran evaluasi
└── docs/screenshots/
```

## Aturan kode

- Nama file, kelas, fungsi, dan field JSON **persis** seperti di `PLAN-KODE.md`.
- Docstring bahasa Indonesia di setiap fungsi dan kelas, isinya: fungsi ini untuk apa, Input, Output, Memanggil (fungsi apa saja yang dipanggil dan untuk apa), Dipanggil (dari mana). Contoh:

  ```python
  def cari_intent(pertanyaan, threshold=0.5):
      """Cari intent yang paling mirip dengan pertanyaan user.

      Input    : teks pertanyaan, threshold kemiripan minimum
      Output   : (id_intent, skor); 'fallback' kalau skor di bawah threshold
      Memanggil: normalisasi() untuk merapikan teks, matcher.cari() untuk menghitung kemiripan
      Dipanggil: GitBot.jawab()
      """
  ```

- Komentar inline bahasa Indonesia yang sederhana di setiap langkah (blok logis), bukan di setiap baris. Isinya apa yang dilakukan dan kenapa. Di bagian inti AI (`preprocessing.py`, `matcher.py`, `chatbot.py`) komentar lebih rinci, termasuk arti rumus (misalnya cosine similarity) dan alasan memakai threshold.
- Setiap konstanta di `config.py` diberi komentar: artinya dan kenapa nilainya segitu.
- Fungsi pendek (di bawah 30 baris), satu tugas.
- Penanganan error eksplisit: `try/except` spesifik, pesan error yang jelas untuk pengguna, aplikasi tidak boleh crash.
- Model dimuat sekali lalu di-cache. Jangan pakai GPU/CUDA.
- Jangan mengarang isi jawaban Git. Isi knowledge base ditulis user.

## Larangan

- Jangan menjalankan `git commit`, `git push`, mengubah remote, atau menghapus file/folder.
- Jangan menginstall paket di luar venv proyek, dan jangan menjalankan perintah di luar folder proyek.
- Jangan membaca atau mengubah isi folder `venv/`.
- Jangan mengubah struktur, nama file/fungsi, atau ID intent. Jangan menambah fitur di luar tahap yang diminta.
- Jangan mengisi `data/knowledge_base.json` (selain kerangka kosong) dan `data/test_questions.csv`.
- Jangan memanggil layanan AI/LLM eksternal. Metode AI harus berjalan lokal dan bisa dijelaskan.
- Untuk melihat struktur proyek pakai `git ls-files` atau `Get-ChildItem src,data,tests,docs,hasil      -Recurse`. Jangan jalankan `Get-ChildItem -Recurse` dari root proyek.

## Cara kerja per tahap

1. Baca `PLAN-KODE.md` dan kerjakan **hanya tahap yang diminta**.
2. Setelah selesai: jalankan tes yang relevan, ringkas file yang dibuat/diubah dan cara menjalankannya, lalu **berhenti**. Jangan lanjut ke tahap berikutnya.
3. Kalau instruksi ambigu atau butuh keputusan di luar plan, tanya dulu. Jangan menebak.
4. Usulkan pesan commit (Conventional Commits) di akhir ringkasan, jangan dijalankan.
5. Tambahkan satu entri di `CATATAN-TAHAP.md` (root proyek, tidak di-commit). Isi hanya fakta yang kamu lakukan dan ketahui:
   - Tahap: nomor dan nama
   - Dibuat/diubah: daftar file
   - Cara menjalankan dan hasil tes yang sebenarnya (tempel output; jangan menulis "berhasil" kalau tidak dijalankan)
   - Masalah yang ditemui dan cara mengatasinya
   - Keputusan yang kamu ambil di luar `PLAN-KODE.md` (kalau tidak ada, tulis "tidak ada")
   - Baris log AI: alat dan tujuan
   Jangan menulis alasan, kesimpulan, atau pendapat atas nama user.

## Perintah (Windows)

- Aktifkan venv: `venv\Scripts\activate`
- Tes: `python -m pytest tests -q`
- Chatbot di terminal: `python -m src.chatbot`
- UI: `streamlit run app.py`
- Evaluasi: `python -m src.evaluate`
