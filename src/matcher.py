"""Modul pencocokan intent untuk GitBot.

Modul ini menyediakan kelas pencocokan intent berbasis kemiripan kosine:
BaseMatcher sebagai kelas dasar abstrak, TfidfMatcher sebagai implementasi
baseline menggunakan representasi TF-IDF dan cosine similarity, serta
EmbeddingMatcher berbasis representasi sentence embedding multibahasa.
Metode pembanding IndoBertMatcher akan diimplementasikan pada Tahap 7.
"""

from typing import Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.preprocessing import normalisasi

# Cache objek model SentenceTransformer tingkat modul per nama_model
_CACHE_MODEL_EMBEDDING: dict[str, SentenceTransformer] = {}


def _validasi_elemen_contoh(contoh: list) -> None:
    """Memvalidasi struktur dan tipe data setiap elemen dalam daftar contoh kalimat.

    Input    : contoh (list) daftar pasangan intent dan kalimat contoh
    Output   : None jika valid; melempar ValueError atau TypeError jika format salah
    Memanggil: isinstance() untuk memeriksa tipe elemen tuple/list dan string
    Dipanggil: BaseMatcher.fit()
    """
    for item in contoh:
        # Memastikan setiap elemen berbentuk tuple atau list dengan panjang tepat 2
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError(
                "Setiap elemen contoh harus berupa pasangan tuple/list (intent_id, kalimat)."
            )
        # Memastikan ID intent dan kalimat bertipe string
        if not isinstance(item[0], str) or not isinstance(item[1], str):
            raise TypeError("Elemen intent_id dan kalimat pada contoh harus bertipe string.")


class BaseMatcher:
    """Kelas dasar (base class) untuk seluruh modul pencocokan intent GitBot.

    Menyediakan antarmuka standar fit() dan cari(), validasi masukan terpusat,
    serta logika agregasi skor per intent yang deterministik.

    Input    : None
    Output   : instansi BaseMatcher
    Memanggil: None
    Dipanggil: TfidfMatcher (sebagai kelas induk), EmbeddingMatcher (sebagai kelas induk)
    """

    def __init__(self) -> None:
        """Menginisialisasi atribut dasar pencocok intent.

        Input    : None
        Output   : None
        Memanggil: None
        Dipanggil: TfidfMatcher.__init__(), EmbeddingMatcher.__init__()
        """
        self._terlatih: bool = False
        self._daftar_intent_unik: list[str] = []
        self._contoh_intents: list[str] = []

    def fit(self, contoh: list[tuple[str, str]]) -> None:
        """Melatih representasi pencocok intent menggunakan daftar contoh kalimat.

        Input    : contoh (list[tuple[str, str]]) daftar pasangan (intent_id, kalimat)
        Output   : None
        Memanggil: _validasi_elemen_contoh() untuk memvalidasi format data masukan
        Dipanggil: TfidfMatcher.fit(), EmbeddingMatcher.fit()
        """
        # Validasi struktur parameter contoh
        if not isinstance(contoh, list):
            raise TypeError("Parameter 'contoh' harus berupa list pasangan (intent_id, kalimat).")
        if not contoh:
            raise ValueError("Daftar contoh untuk fit tidak boleh kosong.")
        _validasi_elemen_contoh(contoh)

        # Merekam urutan kemunculan pertama setiap intent unik secara deterministik
        self._daftar_intent_unik = []
        for intent_id, _ in contoh:
            if intent_id not in self._daftar_intent_unik:
                self._daftar_intent_unik.append(intent_id)

        # Menyimpan pemetaan ID intent untuk setiap baris contoh kalimat
        self._contoh_intents = [intent_id for intent_id, _ in contoh]
        self._terlatih = True

    def _validasi_pencarian(self, teks: str, top_k: int) -> None:
        """Memvalidasi kesiapan model dan argumen sebelum proses pencarian kemiripan.

        Input    : teks (str) teks pertanyaan, top_k (int) kuota hasil kemiripan
        Output   : None jika valid; melempar RuntimeError, TypeError, atau ValueError
        Memanggil: isinstance() untuk pengecekan tipe data
        Dipanggil: BaseMatcher.cari(), TfidfMatcher.cari(), EmbeddingMatcher.cari()
        """
        if not self._terlatih:
            raise RuntimeError(
                "Matcher belum dilatih (fit). Panggil fit() terlebih dahulu sebelum cari()."
            )
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("Parameter 'top_k' harus berupa bilangan bulat minimal 1.")
        if not isinstance(teks, str):
            raise TypeError("Parameter 'teks' harus berupa string.")

    def _agregasi_skor_intent(self, kemiripan: np.ndarray) -> list[tuple[str, float]]:
        """Mengagregasikan skor kemiripan tiap contoh kalimat menjadi skor per intent.

        Input    : kemiripan (np.ndarray) array 1D nilai kemiripan kosine untuk tiap contoh
        Output   : list[tuple[str, float]] pasangan (intent_id, skor) urut menurun
        Memanggil: float(), sorted()
        Dipanggil: TfidfMatcher.cari(), EmbeddingMatcher.cari()
        """
        # Inisialisasi skor tiap intent unik dengan nilai 0.0 sesuai urutan kemunculan pertama
        skor_per_intent = {intent_id: 0.0 for intent_id in self._daftar_intent_unik}

        # Skor sebuah intent = skor cosine tertinggi di antara contoh kalimat intent itu.
        # Arti rumus Cosine Similarity:
        #   cos(theta) = (u . v) / (||u|| * ||v||)
        # Rumus ini mengukur sudut kosinus orientasi dua vektor dalam ruang berdimensi tinggi.
        # Nilai berkisar [0, 1] untuk vektor TF-IDF non-negatif. Nilai 1 berarti arah orientasi
        # kedua vektor identik (sangat mirip leksikalnya), sedangkan 0 berarti ortogonal (tidak
        # ada kata beririsan). Kemiripan kosinus kebal terhadap perbedaan panjang kalimat.
        # Alasan memilih skor TERTINGGI (bukan rata-rata):
        #   Satu intent memiliki banyak variasi contoh kalimat (formal, santai, teknis, singkat).
        #   Pertanyaan pengguna biasanya hanya berkorespondensi dengan salah satu variasi.
        #   Jika dirata-rata, kemiripan tinggi pada satu contoh akan terdilusi (tenggelam)
        #   oleh contoh lain yang berbeda kosakata, sehingga skor intent jatuh secara tidak adil.
        for idx, intent_id in enumerate(self._contoh_intents):
            skor = float(kemiripan[idx])
            if skor > skor_per_intent[intent_id]:
                skor_per_intent[intent_id] = skor

        # Membentuk daftar pasangan (intent, skor) berdasarkan urutan kemunculan pertama
        daftar_pasangan = [(intent, skor_per_intent[intent]) for intent in self._daftar_intent_unik]

        # Timsort Python bersifat stabil: jika skor bernilai sama (misalnya semuanya 0.0),
        # urutan kemunculan awal intent tetap dipertahankan secara konsisten (deterministik).
        return sorted(daftar_pasangan, key=lambda x: x[1], reverse=True)

    def cari(self, teks: str, top_k: int = 3) -> list[tuple[str, float]]:
        """Antarmuka pencarian kemiripan intent untuk kelas turunan.

        Input    : teks (str) pertanyaan pengguna, top_k (int) jumlah intent teratas
        Output   : list[tuple[str, float]] daftar pasangan (intent_id, skor) urut menurun
        Memanggil: self._validasi_pencarian()
        Dipanggil: belum ada (direncanakan dari chatbot.py pada Tahap 8 dan evaluate.py pada Tahap 10 sesuai PLAN-KODE.md)
        """
        self._validasi_pencarian(teks, top_k)
        raise NotImplementedError("Metode cari() harus diimplementasikan oleh kelas turunan.")


class TfidfMatcher(BaseMatcher):
    """Pencocok intent berbasis TF-IDF (Term Frequency-Inverse Document Frequency).

    Menggunakan TfidfVectorizer dari scikit-learn dengan parameter bawaan sebagai baseline,
    dilengkapi normalisasi teks dan stemming aktif (stem=True).

    Input    : None
    Output   : instansi TfidfMatcher
    Memanggil: super().__init__(), TfidfVectorizer()
    Dipanggil: belum ada (direncanakan dari chatbot.py pada Tahap 8, app.py pada Tahap 9, dan evaluate.py pada Tahap 10 sesuai PLAN-KODE.md)
    """

    def __init__(self) -> None:
        """Menginisialisasi TfidfMatcher dengan TfidfVectorizer parameter bawaan.

        Input    : None
        Output   : None
        Memanggil: super().__init__(), TfidfVectorizer()
        Dipanggil: belum ada (direncanakan dari chatbot.py pada Tahap 8, app.py pada Tahap 9, dan evaluate.py pada Tahap 10 sesuai PLAN-KODE.md)
        """
        super().__init__()
        # Menggunakan parameter bawaan default scikit-learn tanpa tuning (baseline)
        # Akibat default token_pattern=r'(?u)\b\w\w+\b': token satu huruf dibuang
        self.vectorizer = TfidfVectorizer()
        self._matriks_tfidf = None

    def fit(self, contoh: list[tuple[str, str]]) -> None:
        """Melatih model TF-IDF menggunakan daftar contoh kalimat intent.

        Input    : contoh (list[tuple[str, str]]) daftar pasangan (intent_id, kalimat)
        Output   : None
        Memanggil: super().fit(), normalisasi() dengan stem=True, vectorizer.fit_transform()
        Dipanggil: belum ada (direncanakan dari chatbot.py pada Tahap 8, app.py pada Tahap 9, dan evaluate.py pada Tahap 10 sesuai PLAN-KODE.md)
        """
        super().fit(contoh)

        # Melakukan prapemrosesan dan stemming bahasa Indonesia pada seluruh contoh kalimat
        korpus = [normalisasi(kalimat, stem=True) for _, kalimat in contoh]

        # Membentuk ruang kosakata dan menghitung matriks bobot TF-IDF seluruh contoh
        try:
            self._matriks_tfidf = self.vectorizer.fit_transform(korpus)
        except ValueError as err:
            self._terlatih = False
            raise ValueError(f"Gagal melatih TF-IDF karena kosakata kosong: {err}") from err

    def cari(self, teks: str, top_k: int = 3) -> list[tuple[str, float]]:
        """Mencari intent paling mirip menggunakan kemiripan kosine vektor TF-IDF.

        Input    : teks (str) teks pertanyaan, top_k (int) jumlah kandidat teratas
        Output   : list[tuple[str, float]] pasangan (intent_id, skor) urut menurun
        Memanggil: self._validasi_pencarian(), normalisasi() dengan stem=True, vectorizer.transform(), cosine_similarity(), self._agregasi_skor_intent()
        Dipanggil: belum ada (direncanakan dari chatbot.py pada Tahap 8 dan evaluate.py pada Tahap 10 sesuai PLAN-KODE.md)
        """
        self._validasi_pencarian(teks, top_k)

        # Lakukan prapemrosesan teks masukan dengan stemming aktif
        teks_bersih = normalisasi(teks, stem=True)

        # Jika hasil normalisasi kosong (misal masukan hanya emoji/spasi),
        # kembalikan kandidat teratas dengan skor 0.0 secara deterministik tanpa crash
        if not teks_bersih.strip():
            return [(intent_id, 0.0) for intent_id in self._daftar_intent_unik[:top_k]]

        # Mengubah teks masukan menjadi representasi vektor TF-IDF pada ruang kosakata contoh
        vektor_query = self.vectorizer.transform([teks_bersih])

        # Menghitung cosine similarity antara vektor pertanyaan dengan seluruh vektor contoh
        kemiripan = cosine_similarity(vektor_query, self._matriks_tfidf).flatten()

        # Mengambil skor tertinggi per intent dan mengurutkan secara menurun
        daftar_skor = self._agregasi_skor_intent(kemiripan)
        return daftar_skor[:top_k]


class EmbeddingMatcher(BaseMatcher):
    """Pencocok intent berbasis sentence embedding dan kemiripan kosine semantik.

    Memanfaatkan model SentenceTransformer untuk merepresentasikan teks ke dalam
    vektor padat (dense embedding), berjalan pada CPU tanpa CUDA, serta
    menerapkan cache model tingkat modul agar model tidak dimuat ulang.

    Input    : nama_model (str) nama atau identifier model SentenceTransformer
    Output   : instansi EmbeddingMatcher
    Memanggil: super().__init__(), self._muat_model()
    Dipanggil: belum ada (direncanakan dari chatbot.py pada Tahap 8, app.py pada Tahap 9, dan evaluate.py pada Tahap 10 sesuai PLAN-KODE.md)
    """

    def __init__(self, nama_model: str) -> None:
        """Menginisialisasi matcher embedding dan memuat model pada perangkat CPU.

        Input    : nama_model (str) identifier model pretrained SentenceTransformer
        Output   : None
        Memanggil: super().__init__(), self._muat_model()
        Dipanggil: belum ada (direncanakan dari chatbot.py pada Tahap 8, app.py pada Tahap 9, dan evaluate.py pada Tahap 10 sesuai PLAN-KODE.md)
        """
        super().__init__()
        # Validasi parameter nama_model wajib string dan tidak kosong
        if not isinstance(nama_model, str) or not nama_model.strip():
            raise TypeError("Parameter 'nama_model' harus berupa string nama model yang valid.")

        self.nama_model: str = nama_model
        # Memuat model sekali pada perangkat CPU (tanpa GPU/CUDA) dengan cache modul
        self.model: SentenceTransformer = self._muat_model(nama_model)
        self._matriks_embedding: Optional[np.ndarray] = None
        # Dimensi vektor diambil langsung dari atribut model tanpa hardcode
        dim_method = getattr(self.model, "get_embedding_dimension", None)
        if dim_method is None:
            dim_method = getattr(self.model, "get_sentence_embedding_dimension")
        self.dimensi: int = int(dim_method())

    def _muat_model(self, nama_model: str) -> SentenceTransformer:
        """Memuat model SentenceTransformer pada perangkat CPU dengan cache tingkat modul.

        Input    : nama_model (str) identifier model pretrained
        Output   : instansi SentenceTransformer siap pakai
        Memanggil: SentenceTransformer()
        Dipanggil: __init__()
        """
        # Kembalikan model dari cache tingkat modul jika sudah pernah dimuat
        if nama_model in _CACHE_MODEL_EMBEDDING:
            return _CACHE_MODEL_EMBEDDING[nama_model]

        try:
            model = SentenceTransformer(nama_model, device="cpu")
            _CACHE_MODEL_EMBEDDING[nama_model] = model
            return model
        except OSError as err:
            # Dasar OSError (hasil probe penulis pada library terpasang):
            # (a) Nama model tidak ada, online: RepositoryNotFoundError (turunan OSError);
            # (b) Offline tanpa cache (HF_HUB_OFFLINE=1): OSError.
            # ValueError, RuntimeError, MemoryError tidak pernah terobservasi jadi jangan ditangkap.
            # Galat selain OSError dibiarkan lewat apa adanya.
            raise RuntimeError(
                f"Gagal memuat model embedding '{nama_model}': {err}. "
                "Kemungkinan penyebab: nama model salah atau tidak ditemukan di Hugging Face, "
                "model belum diunduh dan tidak ada koneksi internet, "
                "berkas cache lokal rusak, atau memori RAM tidak mencukupi. "
                "Solusi: pastikan koneksi internet aktif saat inisialisasi pertama atau bersihkan cache Hugging Face."
            ) from err

    def _encode(self, daftar_teks: list[str]) -> np.ndarray:
        """Menghitung representasi vektor embedding untuk kumpulan teks masukan.

        Input    : daftar_teks (list[str]) daftar kalimat yang akan diubah ke vektor
        Output   : np.ndarray matriks embedding 2D berdimensi (jumlah_teks, dimensi)
        Memanggil: self.model.encode()
        Dipanggil: fit(), cari()
        """
        vektor = self.model.encode(daftar_teks, convert_to_numpy=True)
        arr = np.asarray(vektor, dtype=np.float32)
        # Menjamin bentuk keluaran selalu 2D (jumlah_baris, dimensi_fitur)
        return arr.reshape(1, -1) if arr.ndim == 1 else arr

    def fit(self, contoh: list[tuple[str, str]]) -> None:
        """Melatih representasi embedding dengan menghitung vektor seluruh contoh kalimat.

        Input    : contoh (list[tuple[str, str]]) daftar pasangan (intent_id, kalimat)
        Output   : None
        Memanggil: super().fit(), normalisasi() dengan stem=False, self._encode()
        Dipanggil: belum ada (direncanakan dari chatbot.py pada Tahap 8, app.py pada Tahap 9, dan evaluate.py pada Tahap 10 sesuai PLAN-KODE.md)
        """
        super().fit(contoh)

        # Keputusan di luar plan: Gunakan stem=False karena SentenceTransformer dilatih
        # pada teks alami; pemotongan afiks Sastrawi berisiko merusak informasi semantik.
        korpus = [normalisasi(kalimat, stem=False) for _, kalimat in contoh]

        # Seluruh vektor embedding contoh dihitung SEKALI dalam satu panggilan batch
        # agar komputasi tidak diulang pada setiap pertanyaan saat proses pencarian.
        self._matriks_embedding = self._encode(korpus)
        self.dimensi = self._matriks_embedding.shape[1]

    def cari(self, teks: str, top_k: int = 3) -> list[tuple[str, float]]:
        """Mencari intent paling relevan menggunakan cosine similarity vektor embedding.

        Input    : teks (str) teks pertanyaan, top_k (int) kuota kandidat teratas
        Output   : list[tuple[str, float]] pasangan (intent_id, skor) urut menurun
        Memanggil: self._validasi_pencarian(), normalisasi() dengan stem=False, self._encode(), cosine_similarity(), self._agregasi_skor_intent()
        Dipanggil: belum ada (direncanakan dari chatbot.py pada Tahap 8 dan evaluate.py pada Tahap 10 sesuai PLAN-KODE.md)
        """
        self._validasi_pencarian(teks, top_k)

        # Lakukan prapemrosesan teks tanpa stemming (stem=False)
        teks_bersih = normalisasi(teks, stem=False)

        # Jika masukan kosong setelah dinormalisasi, kembalikan skor 0.0 secara deterministik
        if not teks_bersih.strip():
            return [(intent_id, 0.0) for intent_id in self._daftar_intent_unik[:top_k]]

        # Encode teks pertanyaan menjadi satu vektor baris (1, dimensi)
        vektor_query = self._encode([teks_bersih])

        # Hitung kemiripan kosine antara vektor pertanyaan dengan seluruh vektor contoh.
        # Arti rumus Cosine Similarity pada vektor embedding:
        #   cos(theta) = (u . v) / (||u|| * ||v||)
        # Pada vektor padat (dense), nilai berkisar [-1.0, 1.0]. Nilai 1.0 berarti arah
        # makna semantik identik, 0.0 berarti tidak berkorelasi, dan -1.0 berarti berlawanan.
        # Berbeda dari TF-IDF yang hanya mencocokkan irisan kata leksikal, embedding menangkap
        # kesamaan makna parafrase walaupun tidak ada kata yang sama persis.
        # Catatan: _agregasi_skor_intent menginisialisasi skor awal dengan 0.0, sehingga
        # skor kosine negatif (arah berlawanan) secara implisit terpotong menjadi 0.0.
        kemiripan = cosine_similarity(vektor_query, self._matriks_embedding).flatten()

        daftar_skor = self._agregasi_skor_intent(kemiripan)
        return daftar_skor[:top_k]


if __name__ == "__main__":
    import sys
    import time
    from src.config import MODEL_EMBEDDING, PATH_KB
    from src.knowledge_base import daftar_contoh, muat_kb

    print("=== Perbandingan Matcher TF-IDF vs Embedding (GitBot) ===")

    # 1. Muat Knowledge Base asli proyek
    print(f"\nMemuat data knowledge base dari: {PATH_KB}")
    kb_data = muat_kb(PATH_KB)
    contoh_kb = daftar_contoh(kb_data)
    print(f"Total intent: {len(kb_data.get('intents', []))}, Total contoh kalimat: {len(contoh_kb)}")

    # 2. Inisialisasi dan fitting TfidfMatcher
    t0_tfidf_init = time.perf_counter()
    matcher_tfidf = TfidfMatcher()
    t_tfidf_init = time.perf_counter() - t0_tfidf_init

    t0_tfidf_fit = time.perf_counter()
    matcher_tfidf.fit(contoh_kb)
    t_tfidf_fit = time.perf_counter() - t0_tfidf_fit

    print(f"\n[TF-IDF] Waktu inisialisasi: {t_tfidf_init:.4f} detik | Waktu fit: {t_tfidf_fit:.4f} detik")

    # 3. Inisialisasi dan fitting EmbeddingMatcher
    t0_emb_init = time.perf_counter()
    matcher_emb = EmbeddingMatcher(MODEL_EMBEDDING)
    t_emb_init = time.perf_counter() - t0_emb_init

    t0_emb_fit = time.perf_counter()
    matcher_emb.fit(contoh_kb)
    t_emb_fit = time.perf_counter() - t0_emb_fit

    print(f"[Embedding] Waktu muat model: {t_emb_init:.4f} detik | Waktu fit: {t_emb_fit:.4f} detik (dimensi: {matcher_emb.dimensi})")

    # 4. Tentukan daftar pertanyaan uji (dari CLI argumen atau pertanyaan bawaan netral)
    if len(sys.argv) > 1:
        pertanyaan_uji = sys.argv[1:]
        print(f"\nMenggunakan {len(pertanyaan_uji)} pertanyaan dari argumen baris perintah.")
    else:
        pertanyaan_uji = [
            "bagaimana cara memperbarui repositori lokal dari server?",
            "saya ingin membatalkan perubahan file yang belum di-stage",
            "tahap awal setting identitas programmer di git",
        ]
        print(f"\nMenggunakan 3 pertanyaan bawaan netral (bukan contoh KB).")

    # 5. Jalankan evaluasi pencarian per pertanyaan dan bandingkan hasilnya
    for no, tanya in enumerate(pertanyaan_uji, start=1):
        print(f"\n--------------------------------------------------")
        print(f"Pertanyaan {no}: '{tanya}'")

        # TF-IDF cari
        t0_tfidf_cari = time.perf_counter()
        hasil_tfidf = matcher_tfidf.cari(tanya, top_k=3)
        t_tfidf_cari = time.perf_counter() - t0_tfidf_cari

        print(f"  [TF-IDF] (waktu: {t_tfidf_cari * 1000:.2f} ms):")
        for rank, (intent, skor) in enumerate(hasil_tfidf, start=1):
            print(f"    {rank}. {intent:<25} Skor: {skor:.4f}")

        # Embedding cari
        t0_emb_cari = time.perf_counter()
        hasil_emb = matcher_emb.cari(tanya, top_k=3)
        t_emb_cari = time.perf_counter() - t0_emb_cari

        print(f"  [Embedding] (waktu: {t_emb_cari * 1000:.2f} ms):")
        for rank, (intent, skor) in enumerate(hasil_emb, start=1):
            print(f"    {rank}. {intent:<25} Skor: {skor:.4f}")
