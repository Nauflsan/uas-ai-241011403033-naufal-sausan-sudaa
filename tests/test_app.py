"""Modul pengujian unit untuk fungsi-fungsi murni antarmuka Streamlit GitBot.

Pengujian ini menguji fungsi murni _escape_markdown dan _sapaan pada app.py
secara terisolasi tanpa memerlukan server Streamlit berjalan.
"""

import string
import pytest

from app import _escape_markdown, _sapaan


def test_escape_markdown_head_tilde():
    """Uji pelolosan tilde pada kasus perintah Git HEAD~1 agar tidak tercoret.

    Input    : None
    Output   : None (assertion)
    Memanggil: _escape_markdown()
    Dipanggil: pytest runner
    """
    teks = "git reset HEAD~1 dan HEAD~1"
    diharapkan = "git reset HEAD\\~1 dan HEAD\\~1"
    assert _escape_markdown(teks) == diharapkan


def test_escape_markdown_semua_tanda_baca():
    """Uji bahwa seluruh karakter tanda baca ASCII ber-backslash.

    Input    : None
    Output   : None (assertion)
    Memanggil: _escape_markdown()
    Dipanggil: pytest runner
    """
    semua_tanda_baca = string.punctuation
    hasil = _escape_markdown(semua_tanda_baca)
    diharapkan = "".join(f"\\{karakter}" for karakter in string.punctuation)
    assert hasil == diharapkan


def test_escape_markdown_huruf_angka_spasi_dan_baris_baru():
    """Uji bahwa huruf, angka, spasi, dan baris baru tidak diubah.

    Input    : None
    Output   : None (assertion)
    Memanggil: _escape_markdown()
    Dipanggil: pytest runner
    """
    teks = "Halo GitBot 123 baris 1\nbaris 2 tetap sama"
    assert _escape_markdown(teks) == teks


def test_escape_markdown_string_kosong():
    """Uji penanganan string kosong tetap menghasilkan string kosong.

    Input    : None
    Output   : None (assertion)
    Memanggil: _escape_markdown()
    Dipanggil: pytest runner
    """
    assert _escape_markdown("") == ""


def test_escape_markdown_backslash():
    """Uji karakter backslash di-escape dengan benar menjadi ganda.

    Input    : None
    Output   : None (assertion)
    Memanggil: _escape_markdown()
    Dipanggil: pytest runner
    """
    assert _escape_markdown("\\") == "\\\\"
    assert _escape_markdown("path\\ke\\file") == "path\\\\ke\\\\file"


@pytest.mark.parametrize(
    "jam,teks_diharapkan",
    [
        (3, "Selamat malam! Lagi lembur ngoding? Tanya aja."),
        (4, "Selamat pagi! Sambil ngopi, mau tanya apa soal Git hari ini?"),
        (10, "Selamat pagi! Sambil ngopi, mau tanya apa soal Git hari ini?"),
        (11, "Selamat siang! Udah makan? Yuk, tanya soal Git."),
        (14, "Selamat siang! Udah makan? Yuk, tanya soal Git."),
        (15, "Selamat sore! Ada error Git yang bikin pusing?"),
        (17, "Selamat sore! Ada error Git yang bikin pusing?"),
        (18, "Selamat malam! Lagi lembur ngoding? Tanya aja."),
        (0, "Selamat malam! Lagi lembur ngoding? Tanya aja."),
        (23, "Selamat malam! Lagi lembur ngoding? Tanya aja."),
    ],
)
def test_sapaan_rentang_jam(jam: int, teks_diharapkan: str):
    """Uji kesesuaian kalimat sapaan pada seluruh batas rentang jam lokal.

    Input    : jam (int) jam pengujian, teks_diharapkan (str) sapaan lengkap
    Output   : None (assertion)
    Memanggil: _sapaan()
    Dipanggil: pytest runner
    """
    assert _sapaan(jam) == teks_diharapkan
