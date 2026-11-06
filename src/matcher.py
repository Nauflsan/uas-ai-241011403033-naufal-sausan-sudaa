"""Modul pencocokan intent untuk GitBot.

Modul ini menyediakan kelas pencocokan intent berbasis kemiripan kosine:
BaseMatcher sebagai kelas dasar abstrak, dan TfidfMatcher sebagai implementasi
baseline menggunakan representasi TF-IDF dan cosine similarity.
Metode EmbeddingMatcher dan IndoBertMatcher akan diimplementasikan pada Tahap 6 dan 7.
"""

from typing import Union
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.preprocessing import normalisasi


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
    Dipanggil: TfidfMatcher (sebagai kelas induk)
    """

    def __init__(self) -> None:
        """Menginisialisasi atribut dasar pencocok intent.

        Input    : None
        Output   : None
        Memanggil: None
        Dipanggil: TfidfMatcher.__init__()
        """
        self._terlatih: bool = False
        self._daftar_intent_unik: list[str] = []
        self._contoh_intents: list[str] = []

    def fit(self, contoh: list[tuple[str, str]]) -> None:
        """Melatih representasi pencocok intent menggunakan daftar contoh kalimat.

        Input    : contoh (list[tuple[str, str]]) daftar pasangan (intent_id, kalimat)
        Output   : None
        Memanggil: _validasi_elemen_contoh() untuk memvalidasi format data masukan
        Dipanggil: TfidfMatcher.fit()
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
        Dipanggil: BaseMatcher.cari(), TfidfMatcher.cari()
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
        Dipanggil: TfidfMatcher.cari()
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


if __name__ == "__main__":
    print("=== Demonstrasi Modul Matcher TF-IDF GitBot ===")
    contoh_data = [
        ("setup_awal", "bagaimana cara setting nama dan email git"),
        ("setup_awal", "konfigurasi awal git pengguna baru"),
        ("bikin_pindah_branch", "cara membuat cabang atau branch baru"),
        ("bikin_pindah_branch", "pindah ke branch lain dengan checkout"),
        ("batalin_commit", "batalkan commit terakhir yang salah"),
        ("batalin_commit", "cara undo commit dengan git reset"),
    ]

    matcher = TfidfMatcher()
    matcher.fit(contoh_data)

    pertanyaan_uji = [
        "gimana setting config user git?",
        "buat branch baru",
        "batalin commit",
    ]

    for no, tanya in enumerate(pertanyaan_uji, start=1):
        print(f"\nPertanyaan {no}: '{tanya}'")
        hasil_cari = matcher.cari(tanya, top_k=3)
        for rank, (intent, skor) in enumerate(hasil_cari, start=1):
            print(f"  {rank}. Intent: {intent:<22} Skor: {skor:.4f}")
