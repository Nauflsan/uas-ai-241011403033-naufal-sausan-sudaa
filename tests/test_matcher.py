"""Pengujian unit untuk modul matcher intent GitBot.

Menguji BaseMatcher, TfidfMatcher, dan EmbeddingMatcher menggunakan kumpulan data kecil
di dalam tes tanpa membaca data/knowledge_base.json.
"""

import pytest
from src.config import MODEL_EMBEDDING
from src.matcher import (
    BaseMatcher,
    EmbeddingMatcher,
    TfidfMatcher,
    _CACHE_MODEL_EMBEDDING,
)


@pytest.fixture
def data_contoh_kecil():
    """Menyediakan dataset kecil berisi 3 intent untuk pengujian matcher."""
    return [
        ("setup_awal", "bagaimana cara mengatur konfigurasi nama dan email git"),
        ("setup_awal", "langkah pertama inisialisasi dan pengaturan akun git"),
        ("bikin_pindah_branch", "cara membuat cabang baru dan berpindah branch"),
        ("bikin_pindah_branch", "perintah checkout switch cabang git baru"),
        ("batalin_commit", "cara membatalkan commit terakhir dengan git reset"),
        ("batalin_commit", "undo commit yang belum dipush ke remote"),
    ]


@pytest.fixture
def matcher_terlatih(data_contoh_kecil):
    """Menyediakan instansi TfidfMatcher yang sudah dilatih dengan data kecil."""
    matcher = TfidfMatcher()
    matcher.fit(data_contoh_kecil)
    return matcher


@pytest.fixture(scope="module")
def matcher_embedding_terlatih():
    """Menyediakan instansi EmbeddingMatcher yang dilatih sekali per modul pengujian."""
    contoh_data = [
        ("setup_awal", "bagaimana cara mengatur konfigurasi nama dan email git"),
        ("setup_awal", "langkah pertama inisialisasi dan pengaturan akun git"),
        ("bikin_pindah_branch", "cara membuat cabang baru dan berpindah branch"),
        ("bikin_pindah_branch", "perintah checkout switch cabang git baru"),
        ("batalin_commit", "cara membatalkan commit terakhir dengan git reset"),
        ("batalin_commit", "undo commit yang belum dipush ke remote"),
    ]
    matcher = EmbeddingMatcher(MODEL_EMBEDDING)
    matcher.fit(contoh_data)
    return matcher


# =====================================================================
# Pengujian BaseMatcher dan TfidfMatcher (Tahap 5)
# =====================================================================


def test_kecocokan_benar_di_urutan_pertama(matcher_terlatih):
    """Memastikan pertanyaan yang relevan menghasilkan intent yang tepat di urutan pertama."""
    hasil = matcher_terlatih.cari("cara mengatur nama dan email git", top_k=3)
    assert len(hasil) > 0
    top_intent, top_score = hasil[0]
    assert top_intent == "setup_awal"
    assert top_score > 0.0


def test_satu_entri_per_intent(matcher_terlatih):
    """Memastikan hasil pencarian hanya memuat tepat satu entri per intent (tidak ada duplikat)."""
    hasil = matcher_terlatih.cari("konfigurasi git cabang baru", top_k=5)
    daftar_intent = [intent for intent, _ in hasil]
    # Jumlah entri harus sama dengan jumlah elemen uniknya
    assert len(daftar_intent) == len(set(daftar_intent))
    assert len(daftar_intent) <= 3


def test_skor_terurut_menurun(matcher_terlatih):
    """Memastikan skor kemiripan seluruh kandidat selalu terurut menurun monoton."""
    hasil = matcher_terlatih.cari("buat cabang baru di git", top_k=3)
    skor_list = [skor for _, skor in hasil]
    for i in range(len(skor_list) - 1):
        assert skor_list[i] >= skor_list[i + 1]


def test_top_k_lebih_besar_dari_jumlah_intent(matcher_terlatih):
    """Memastikan top_k yang melebihi jumlah intent unik tidak menyebabkan crash dan mengembalikan seluruh intent unik."""
    # Data kecil memiliki 3 intent unik, kita minta top_k=10
    hasil = matcher_terlatih.cari("konfigurasi", top_k=10)
    assert len(hasil) == 3
    daftar_intent = [intent for intent, _ in hasil]
    assert set(daftar_intent) == {"setup_awal", "bikin_pindah_branch", "batalin_commit"}


def test_teks_kosong_dan_hanya_spasi(matcher_terlatih):
    """Memastikan teks kosong atau hanya spasi mengembalikan top_k intent dengan skor 0.0 tanpa crash."""
    hasil_kosong = matcher_terlatih.cari("", top_k=2)
    assert len(hasil_kosong) == 2
    assert hasil_kosong == [("setup_awal", 0.0), ("bikin_pindah_branch", 0.0)]

    hasil_spasi = matcher_terlatih.cari("   \t\n  ", top_k=3)
    assert len(hasil_spasi) == 3
    assert all(skor == 0.0 for _, skor in hasil_spasi)
    assert [intent for intent, _ in hasil_spasi] == ["setup_awal", "bikin_pindah_branch", "batalin_commit"]


def test_teks_hanya_emoji(matcher_terlatih):
    """Memastikan teks yang hanya berisi emoji (hasil normalisasi kosong) mengembalikan skor 0.0 secara deterministik."""
    hasil = matcher_terlatih.cari("🚀 🎉 🔥", top_k=2)
    assert len(hasil) == 2
    assert hasil == [("setup_awal", 0.0), ("bikin_pindah_branch", 0.0)]


def test_kata_asing_di_luar_vocabulary(matcher_terlatih):
    """Memastikan kata-kata di luar kosakata menghasilkan skor 0.0 dengan urutan deterministik."""
    hasil = matcher_terlatih.cari("xyzzyqw qwertyasdf zzzzz", top_k=3)
    assert len(hasil) == 3
    assert all(skor == 0.0 for _, skor in hasil)
    # Urutan mengikuti kemunculan pertama pada daftar contoh
    assert [intent for intent, _ in hasil] == ["setup_awal", "bikin_pindah_branch", "batalin_commit"]


def test_cari_sebelum_fit_melempar_runtime_error():
    """Memastikan memanggil cari() sebelum fit() melempar RuntimeError."""
    matcher = TfidfMatcher()
    with pytest.raises(RuntimeError, match="belum dilatih"):
        matcher.cari("pertanyaan uji")

    base = BaseMatcher()
    with pytest.raises(RuntimeError, match="belum dilatih"):
        base.cari("pertanyaan uji")


def test_fit_daftar_kosong_melempar_value_error():
    """Memastikan melatih matcher dengan daftar contoh kosong melempar ValueError."""
    matcher = TfidfMatcher()
    with pytest.raises(ValueError, match="tidak boleh kosong"):
        matcher.fit([])

    base = BaseMatcher()
    with pytest.raises(ValueError, match="tidak boleh kosong"):
        base.fit([])


def test_skor_intent_tertinggi_dari_contoh_berbeda():
    """Membuktikan skor intent adalah nilai cosine tertinggi di antara contohnya, bukan rata-rata.

    Menggunakan dua contoh kalimat pada satu intent dengan derajat kecocokan yang berbeda
    secara signifikan terhadap kalimat kueri.
    """
    contoh = [
        ("setup_awal", "mengatur konfigurasi nama email git"),
        ("setup_awal", "konfigurasi editor teks default git"),
    ]
    matcher = TfidfMatcher()
    matcher.fit(contoh)

    # Kueri ini cocok kuat dengan contoh 1 (nama, email, git), dan hanya sebagian kecil dengan contoh 2
    kueri = "konfigurasi nama email git"

    # Hitung nilai kemiripan kueri terhadap matriks internal contoh
    from sklearn.metrics.pairwise import cosine_similarity
    from src.preprocessing import normalisasi

    vektor_q = matcher.vectorizer.transform([normalisasi(kueri, stem=True)])
    kemiripan = cosine_similarity(vektor_q, matcher._matriks_tfidf).flatten()

    skor_contoh_1 = float(kemiripan[0])
    skor_contoh_2 = float(kemiripan[1])

    # Buktikan bahwa kedua contoh memiliki skor berbeda
    assert skor_contoh_1 > 0.0
    assert skor_contoh_2 > 0.0
    assert skor_contoh_1 != skor_contoh_2
    assert skor_contoh_1 > skor_contoh_2

    # Hasil pencarian intent gabungan harus tepat sama dengan skor_contoh_1 (tertinggi),
    # dan secara tegas lebih besar daripada rata-rata kedua contoh
    hasil = matcher.cari(kueri, top_k=1)
    skor_gabungan = hasil[0][1]

    rata_rata = (skor_contoh_1 + skor_contoh_2) / 2.0
    assert skor_gabungan == pytest.approx(skor_contoh_1, rel=1e-5)
    assert skor_gabungan > rata_rata


def test_top_k_kurang_dari_satu(matcher_terlatih):
    """Memastikan parameter top_k kurang dari 1 melempar ValueError."""
    with pytest.raises(ValueError, match="top_k"):
        matcher_terlatih.cari("halo", top_k=0)

    with pytest.raises(ValueError, match="top_k"):
        matcher_terlatih.cari("halo", top_k=-2)


def test_validasi_tipe_parameter(matcher_terlatih):
    """Memastikan parameter masukan dengan tipe salah melempar pengecualian spesifik."""
    with pytest.raises(TypeError, match="string"):
        matcher_terlatih.cari(12345)

    with pytest.raises(TypeError, match="list"):
        matcher_terlatih.fit("bukan list")

    with pytest.raises(ValueError, match="pasangan"):
        matcher_terlatih.fit([("hanya_satu_elemen",)])

    with pytest.raises(TypeError, match="string"):
        matcher_terlatih.fit([("intent", 9999)])


def test_base_matcher_cari_not_implemented():
    """Memastikan pemanggilan cari() pada instansi langsung BaseMatcher melempar NotImplementedError."""
    base = BaseMatcher()
    base.fit([("intent_1", "contoh kalimat")])
    with pytest.raises(NotImplementedError):
        base.cari("halo")


# =====================================================================
# Pengujian EmbeddingMatcher (Tahap 6)
# =====================================================================


def test_embedding_kecocokan_semantik_parafrase(matcher_embedding_terlatih):
    """Memastikan kueri parafrase tanpa kata yang sama persis tetap mencocokkan intent yang benar di peringkat pertama."""
    # Kueri ini tidak berbagi kata eksak dengan 'langkah pertama inisialisasi dan pengaturan akun git'
    kueri_parafrase = "identitas pemilik pemula"
    hasil = matcher_embedding_terlatih.cari(kueri_parafrase, top_k=3)
    assert len(hasil) > 0
    top_intent, top_score = hasil[0]
    assert top_intent == "setup_awal"
    assert top_score > 0.0


def test_embedding_satu_entri_per_intent(matcher_embedding_terlatih):
    """Memastikan hasil pencarian embedding hanya memuat satu entri per intent tanpa duplikasi."""
    hasil = matcher_embedding_terlatih.cari("cara membuat branch baru", top_k=5)
    daftar_intent = [intent for intent, _ in hasil]
    assert len(daftar_intent) == len(set(daftar_intent))
    assert len(daftar_intent) <= 3


def test_embedding_skor_terurut_menurun(matcher_embedding_terlatih):
    """Memastikan skor seluruh kandidat pada EmbeddingMatcher selalu terurut menurun."""
    hasil = matcher_embedding_terlatih.cari("batalkan revisi commit", top_k=3)
    skor_list = [skor for _, skor in hasil]
    for i in range(len(skor_list) - 1):
        assert skor_list[i] >= skor_list[i + 1]


def test_embedding_top_k_lebih_besar_dari_jumlah_intent(matcher_embedding_terlatih):
    """Memastikan top_k yang melebihi jumlah intent unik tidak crash dan mengembalikan seluruh intent unik."""
    hasil = matcher_embedding_terlatih.cari("perintah git", top_k=10)
    assert len(hasil) == 3
    daftar_intent = [intent for intent, _ in hasil]
    assert set(daftar_intent) == {"setup_awal", "bikin_pindah_branch", "batalin_commit"}


def test_embedding_teks_kosong_dan_hanya_spasi(matcher_embedding_terlatih):
    """Memastikan teks kosong atau hanya spasi mengembalikan top_k intent dengan skor 0.0 tanpa crash."""
    hasil_kosong = matcher_embedding_terlatih.cari("", top_k=2)
    assert len(hasil_kosong) == 2
    assert hasil_kosong == [("setup_awal", 0.0), ("bikin_pindah_branch", 0.0)]

    hasil_spasi = matcher_embedding_terlatih.cari("   \t\n  ", top_k=3)
    assert len(hasil_spasi) == 3
    assert all(skor == 0.0 for _, skor in hasil_spasi)
    assert [intent for intent, _ in hasil_spasi] == ["setup_awal", "bikin_pindah_branch", "batalin_commit"]


def test_embedding_teks_hanya_emoji(matcher_embedding_terlatih):
    """Memastikan teks yang hanya berisi emoji mengembalikan kandidat dengan skor 0.0 secara deterministik."""
    hasil = matcher_embedding_terlatih.cari("🚀 🎉 🔥", top_k=2)
    assert len(hasil) == 2
    assert hasil == [("setup_awal", 0.0), ("bikin_pindah_branch", 0.0)]


def test_embedding_cari_sebelum_fit_melempar_runtime_error():
    """Memastikan pemanggilan cari() pada EmbeddingMatcher sebelum fit() melempar RuntimeError."""
    matcher = EmbeddingMatcher(MODEL_EMBEDDING)
    with pytest.raises(RuntimeError, match="belum dilatih"):
        matcher.cari("pertanyaan uji")


def test_embedding_top_k_kurang_dari_satu(matcher_embedding_terlatih):
    """Memastikan parameter top_k kurang dari 1 melempar ValueError pada EmbeddingMatcher."""
    with pytest.raises(ValueError, match="top_k"):
        matcher_embedding_terlatih.cari("halo", top_k=0)

    with pytest.raises(ValueError, match="top_k"):
        matcher_embedding_terlatih.cari("halo", top_k=-1)


def test_embedding_fit_daftar_kosong_melempar_value_error():
    """Memastikan fit() dengan daftar contoh kosong pada EmbeddingMatcher melempar ValueError."""
    matcher = EmbeddingMatcher(MODEL_EMBEDDING)
    with pytest.raises(ValueError, match="tidak boleh kosong"):
        matcher.fit([])


def test_embedding_validasi_tipe_parameter(matcher_embedding_terlatih):
    """Memastikan pengecualian bertipe salah dilempar jika parameter tidak sesuai tipe data."""
    with pytest.raises(TypeError, match="nama_model"):
        EmbeddingMatcher(12345)

    with pytest.raises(TypeError, match="nama_model"):
        EmbeddingMatcher("")

    with pytest.raises(TypeError, match="string"):
        matcher_embedding_terlatih.cari(12345)

    with pytest.raises(TypeError, match="list"):
        matcher_embedding_terlatih.fit("bukan list")


def test_embedding_contoh_dihitung_sekali(matcher_embedding_terlatih, monkeypatch):
    """Membuktikan bahwa embedding contoh dihitung sekali saat fit dan _encode tidak dipanggil ulang untuk contoh saat cari."""
    panggilan_encode = []
    encode_asli = matcher_embedding_terlatih._encode

    def mock_encode(daftar_teks):
        panggilan_encode.append(daftar_teks)
        return encode_asli(daftar_teks)

    monkeypatch.setattr(matcher_embedding_terlatih, "_encode", mock_encode)

    # Lakukan pencarian kueri
    hasil = matcher_embedding_terlatih.cari("bagaimana cara buat branch", top_k=3)
    assert len(hasil) > 0

    # _encode hanya boleh dipanggil 1 kali dan hanya untuk teks pertanyaan pengguna (panjang list 1)
    assert len(panggilan_encode) == 1
    assert len(panggilan_encode[0]) == 1
    assert panggilan_encode[0][0] == "bagaimana cara buat branch"


def test_embedding_kegagalan_memuat_model(monkeypatch):
    """Memastikan kegagalan memuat model dibungkus menjadi RuntimeError dengan pesan informatif tanpa mengunduh."""
    def mock_init_gagal(*args, **kwargs):
        raise OSError("Koneksi gagal atau model tidak ditemukan di repositori.")

    monkeypatch.setattr("src.matcher.SentenceTransformer", mock_init_gagal)

    with pytest.raises(RuntimeError) as exc_info:
        EmbeddingMatcher("model_gagal_uji")

    pesan = str(exc_info.value)
    assert "model_gagal_uji" in pesan
    assert "Kemungkinan penyebab" in pesan
    assert "model_gagal_uji" not in _CACHE_MODEL_EMBEDDING


def test_embedding_kegagalan_memuat_subclass_oserror(monkeypatch):
    """Memastikan kegagalan memuat dengan subclass OSError buatan tetap dibungkus RuntimeError."""
    class GalatUji(OSError):
        """Subclass buatan OSError untuk memverifikasi hierarki penanganan galat OSError."""
        pass

    def mock_init_gagal(*args, **kwargs):
        raise GalatUji("Koneksi ditolak oleh endpoint HF.")

    monkeypatch.setattr("src.matcher.SentenceTransformer", mock_init_gagal)

    with pytest.raises(RuntimeError) as exc_info:
        EmbeddingMatcher("model_subclass_gagal")

    pesan = str(exc_info.value)
    assert "model_subclass_gagal" in pesan
    assert "Kemungkinan penyebab" in pesan
    assert "nama model salah atau tidak ditemukan di Hugging Face" in pesan
    assert "model_subclass_gagal" not in _CACHE_MODEL_EMBEDDING


def test_embedding_cache_model_tingkat_modul():
    """Memastikan dua instansi EmbeddingMatcher dengan nama model sama berbagi objek model yang sama dari cache."""
    m1 = EmbeddingMatcher(MODEL_EMBEDDING)
    m2 = EmbeddingMatcher(MODEL_EMBEDDING)
    assert m1.model is m2.model
