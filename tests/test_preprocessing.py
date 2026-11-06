"""Modul pengujian unit untuk prapemrosesan teks GitBot.

Pengujian ini memastikan normalisasi huruf kecil, pembersihan karakter kontrol/emoji,
penggantian kamus slang, serta proses stemming selektif untuk istilah teknis Git
berjalan sesuai dengan spesifikasi proyek pada PLAN-KODE.md dan instruksi Tahap 4.
"""

import pytest
from src.preprocessing import (
    _ISTILAH_GIT_TERLINDUNGI,
    _dapatkan_stemmer,
    normalisasi,
)


def test_normalisasi_huruf_kecil_dan_spasi():
    """Uji pengubahan teks menjadi huruf kecil dan perapian spasi berlebih.

    Input    : string dengan variasi huruf kapital dan spasi berlebih
    Output   : None (assertion)
    Memanggil: normalisasi() dengan stem=False
    Dipanggil: pytest runner
    """
    masukan = "   BAGAIMANA   cara   Git  PUSH   ke   MAIN?   "
    harapan = "bagaimana cara git push ke main?"
    assert normalisasi(masukan, stem=False) == harapan


def test_hapus_emoji_dan_karakter_kontrol():
    """Uji penghapusan karakter emoji dan karakter kontrol perusak.

    Input    : string yang mengandung beragam emoji dan byte kontrol
    Output   : None (assertion)
    Memanggil: normalisasi() dengan stem=False
    Dipanggil: pytest runner
    """
    masukan = "git push origin main 🚀🔥 \x00\x07 sukses selalu 👍❤️"
    harapan = "git push origin main sukses selalu"
    assert normalisasi(masukan, stem=False) == harapan


def test_karakter_dan_simbol_git_dipertahankan():
    """Uji bahwa simbol dan tanda baca teknis Git tetap dipertahankan.

    Input    : string berisi seluruh simbol teknis Git yang diizinkan
    Output   : None (assertion)
    Memanggil: normalisasi() dengan stem=False
    Dipanggil: pytest runner
    """
    simbol_teks = ". - _ / ~ @ { } [ ] < > = ! : # ( ) \" ' , 123"
    assert normalisasi(simbol_teks, stem=False) == simbol_teks


def test_kamus_slang_pencocokan_kata_utuh():
    """Uji penggantian kata slang per kata utuh dan bukan substring.

    Input    : string berisi kata slang umum
    Output   : None (assertion)
    Memanggil: normalisasi() dengan stem=False
    Dipanggil: pytest runner
    """
    masukan = "gw mau bikin branch baru trs ngepush commit"
    harapan = "saya mau buat branch baru terus push commit"
    assert normalisasi(masukan, stem=False) == harapan


def test_kamus_slang_tidak_mengganti_substring():
    """Uji bahwa kamus slang tidak mengganti kemunculan kata sebagai substring.

    Input    : string berisi kata dengan substring mirip slang (gagal, bikinan, gagal gak)
    Output   : None (assertion)
    Memanggil: normalisasi() dengan stem=False
    Dipanggil: pytest runner
    """
    assert normalisasi("gagal", stem=False) == "gagal"
    assert normalisasi("bikinan", stem=False) == "bikinan"
    assert normalisasi("gagal gak", stem=False) == "gagal tidak"


def test_kamus_slang_kata_tambahan():
    """Uji penggantian kata slang tambahan yang terdaftar di KAMUS_SLANG.

    Input    : string berisi kata slang 'yg', 'msh', 'abis', 'temen', 'pengen'
    Output   : None (assertion)
    Memanggil: normalisasi() dengan stem=False
    Dipanggil: pytest runner
    """
    masukan = "yg msh abis temen pengen"
    harapan = "yang masih habis teman ingin"
    assert normalisasi(masukan, stem=False) == harapan


def test_kamus_slang_tanda_baca_tepi():
    """Uji bahwa tanda baca di tepi kata tidak menggagalkan penggantian slang.

    Input    : string berisi kata slang yang dikelilingi tanda baca di tepinya
    Output   : None (assertion)
    Memanggil: normalisasi() dengan stem=False
    Dipanggil: pytest runner
    """
    masukan = "gimana, bro? (udah) blm... trs 'ngepush'!"
    harapan = "bagaimana, bro? (sudah) belum... terus 'push'!"
    assert normalisasi(masukan, stem=False) == harapan


def test_stemming_default_false():
    """Uji bahwa stemming tidak dijalankan saat opsi stem=False.

    Input    : string kalimat bahasa Indonesia dengan afiks
    Output   : None (assertion)
    Memanggil: normalisasi() dengan stem=False
    Dipanggil: pytest runner
    """
    masukan = "saya ingin membatalkan perubahan konfigurasi"
    harapan = "saya ingin membatalkan perubahan konfigurasi"
    assert normalisasi(masukan, stem=False) == harapan


def test_stemming_kata_indonesia():
    """Uji bahwa kata-kata bahasa Indonesia distem saat stem=True.

    Input    : string kalimat bahasa Indonesia berimbuhan
    Output   : None (assertion)
    Memanggil: normalisasi() dengan stem=True
    Dipanggil: pytest runner
    """
    masukan = "saya membatalkan semua perubahan dan membuat pengaturan"
    harapan = "saya batal semua ubah dan buat atur"
    assert normalisasi(masukan, stem=True) == harapan


def test_stemming_istilah_git_umum():
    """Uji 18 istilah teknis Git umum tidak mengalami perubahan saat stem=True.

    Input    : 18 istilah Git umum (push, pull, fetch, clone, commit, dll.)
    Output   : None (assertion)
    Memanggil: normalisasi() untuk setiap istilah dengan stem=True
    Dipanggil: pytest runner
    """
    istilah_umum = [
        "push", "pull", "fetch", "clone", "commit", "branch", "merge",
        "rebase", "stash", "checkout", "switch", "reset", "revert",
        "remote", "origin", "main", "repository", "conflict",
    ]

    for istilah in istilah_umum:
        assert normalisasi(istilah, stem=True) == istilah
        assert istilah in _ISTILAH_GIT_TERLINDUNGI


def test_stemming_kata_mengandung_angka_dan_simbol():
    """Uji kata yang mengandung angka atau simbol tidak distem saat stem=True.

    Input    : string berisi token berangka/simbol seperti user.name dan HEAD~1
    Output   : None (assertion)
    Memanggil: normalisasi() dengan stem=True
    Dipanggil: pytest runner
    """
    masukan = "atur user.name pada HEAD~1 dan origin/main versi v1.0.0 issue#42"
    harapan = "atur user.name pada head~1 dan origin/main versi v1.0.0 issue#42"
    assert normalisasi(masukan, stem=True) == harapan


def test_stemming_dengan_tanda_baca_tepi():
    """Uji stemming kata berimbuhan dengan tanda baca di tepi kata tetap utuh.

    Input    : string kata berimbuhan yang diapit tanda baca tepi
    Output   : None (assertion)
    Memanggil: normalisasi() dengan stem=True
    Dipanggil: pytest runner
    """
    masukan = "kenapa membatalkan, (perubahan)? sangat membantu!"
    harapan = "kenapa batal, (ubah)? sangat bantu!"
    assert normalisasi(masukan, stem=True) == harapan


def test_input_kosong_dan_hanya_spasi():
    """Uji penanganan masukan berupa string kosong atau hanya berisi spasi.

    Input    : string kosong dan string spasi saja
    Output   : None (assertion)
    Memanggil: normalisasi() dengan stem=False dan stem=True
    Dipanggil: pytest runner
    """
    assert normalisasi("", stem=False) == ""
    assert normalisasi("   \t  \n  ", stem=True) == ""


def test_input_hanya_emoji():
    """Uji masukan yang hanya berisi karakter emoji menghasilkan string kosong.

    Input    : string berisi kumpulan emoji saja
    Output   : None (assertion)
    Memanggil: normalisasi()
    Dipanggil: pytest runner
    """
    assert normalisasi("🚀🔥👍", stem=False) == ""
    assert normalisasi("😀🎉💯", stem=True) == ""


def test_validasi_tipe_parameter():
    """Uji bahwa fungsi melempar TypeError jika tipe argumen tidak sesuai.

    Input    : nilai non-string untuk teks atau non-boolean untuk stem
    Output   : None (assertion pytest.raises)
    Memanggil: normalisasi() dengan argumen tidak valid
    Dipanggil: pytest runner
    """
    with pytest.raises(TypeError) as exc_teks:
        normalisasi(12345)
    assert "teks" in str(exc_teks.value)

    with pytest.raises(TypeError) as exc_none:
        normalisasi(None)
    assert "teks" in str(exc_none.value)

    with pytest.raises(TypeError) as exc_stem:
        normalisasi("git push", stem="bukan-boolean")
    assert "stem" in str(exc_stem.value)


def test_stemmer_singleton_cache():
    """Uji bahwa stemmer Sastrawi di-cache sebagai objek singleton.

    Input    : None
    Output   : None (assertion)
    Memanggil: _dapatkan_stemmer() dua kali
    Dipanggil: pytest runner
    """
    stemmer_1 = _dapatkan_stemmer()
    stemmer_2 = _dapatkan_stemmer()
    assert stemmer_1 is stemmer_2
