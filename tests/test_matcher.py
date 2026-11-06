"""Pengujian unit untuk modul matcher intent GitBot.

Menguji BaseMatcher dan TfidfMatcher menggunakan kumpulan data kecil
di dalam tes tanpa membaca data/knowledge_base.json.
"""

import pytest
from src.matcher import BaseMatcher, TfidfMatcher


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
