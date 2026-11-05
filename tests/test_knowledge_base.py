"""Modul pengujian unit untuk loader dan validator knowledge base GitBot.

Pengujian ini memastikan fungsi muat_kb, validasi_kb, dan daftar_contoh
berjalan sesuai dengan spesifikasi proyek pada PLAN-KODE.md.
"""

import json
import pytest
from pathlib import Path

from src.knowledge_base import (
    DAFTAR_ID_INTENT,
    KBError,
    daftar_contoh,
    muat_kb,
    validasi_kb,
)
from src.config import PATH_KB


def test_daftar_id_intent_total():
    """Uji jumlah dan integritas konstanta DAFTAR_ID_INTENT.

    Input    : None
    Output   : None (assertion)
    Memanggil: len() untuk memeriksa total intent final
    Dipanggil: pytest runner
    """
    assert len(DAFTAR_ID_INTENT) == 18
    assert "setup_awal" in DAFTAR_ID_INTENT
    assert "secret_kepush" in DAFTAR_ID_INTENT


def test_muat_kb_sukses(tmp_path: Path):
    """Uji pemuatan berkas knowledge base JSON yang valid.

    Input    : tmp_path (fixture pytest untuk direktori sementara)
    Output   : None (assertion)
    Memanggil: muat_kb() untuk membaca berkas JSON valid
    Dipanggil: pytest runner
    """
    file_kb = tmp_path / "kb_valid.json"
    data = {
        "intents": [
            {
                "id": "setup_awal",
                "nama": "Setup Awal Git",
                "contoh": ["c1", "c2", "c3", "c4", "c5"],
                "jawaban": "Gunakan git config",
                "perintah": ["git config --global user.name 'Nama'"],
                "sumber": "Pro Git"
            }
        ],
        "fallback": {"jawaban": "Maaf belum paham."}
    }
    file_kb.write_text(json.dumps(data), encoding="utf-8")

    hasil = muat_kb(file_kb)
    assert hasil["fallback"]["jawaban"] == "Maaf belum paham."
    assert len(hasil["intents"]) == 1


def test_muat_kb_file_tidak_ditemukan():
    """Uji bahwa muat_kb melempar KBError jika berkas tidak ditemukan.

    Input    : None
    Output   : None (assertion)
    Memanggil: muat_kb() dengan path non-existent
    Dipanggil: pytest runner
    """
    with pytest.raises(KBError) as exc_info:
        muat_kb("path/yang/pasti/tidak/pernah/ada.json")
    assert "tidak ditemukan" in str(exc_info.value)


def test_muat_kb_json_rusak(tmp_path: Path):
    """Uji bahwa muat_kb melempar KBError jika isi berkas JSON rusak.

    Input    : tmp_path (fixture pytest)
    Output   : None (assertion)
    Memanggil: muat_kb() dengan berkas JSON sintaks rusak
    Dipanggil: pytest runner
    """
    file_rusak = tmp_path / "kb_rusak.json"
    file_rusak.write_text("{ intents: [ rusak ]", encoding="utf-8")

    with pytest.raises(KBError) as exc_info:
        muat_kb(file_rusak)
    assert "rusak" in str(exc_info.value)


def test_muat_kb_skema_salah_intents_bukan_list(tmp_path: Path):
    """Uji bahwa muat_kb melempar KBError jika field intents bukan list.

    Input    : tmp_path (fixture pytest)
    Output   : None (assertion)
    Memanggil: muat_kb() dengan skema data salah
    Dipanggil: pytest runner
    """
    file_salah = tmp_path / "kb_salah.json"
    data = {"intents": "bukan list", "fallback": {"jawaban": "ada"}}
    file_salah.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(KBError) as exc_info:
        muat_kb(file_salah)
    assert "kunci 'intents' harus ada dan berupa list" in str(exc_info.value)


def test_muat_kb_skema_salah_fallback_hilang(tmp_path: Path):
    """Uji bahwa muat_kb melempar KBError jika field fallback tidak ada.

    Input    : tmp_path (fixture pytest)
    Output   : None (assertion)
    Memanggil: muat_kb() tanpa field fallback
    Dipanggil: pytest runner
    """
    file_tanpa_fallback = tmp_path / "kb_no_fallback.json"
    data = {"intents": []}
    file_tanpa_fallback.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(KBError) as exc_info:
        muat_kb(file_tanpa_fallback)
    assert "kunci 'fallback' harus ada" in str(exc_info.value)


def test_muat_kb_default_proyek():
    """Uji pemuatan berkas knowledge base bawaan proyek pada PATH_KB.

    Input    : None
    Output   : None (assertion)
    Memanggil: muat_kb() dengan PATH_KB dari config
    Dipanggil: pytest runner
    """
    data = muat_kb(PATH_KB)
    assert isinstance(data, dict)
    assert "intents" in data
    assert "fallback" in data


def test_validasi_kb_sukses():
    """Uji validasi_kb mengembalikan list kosong jika data valid.

    Input    : None
    Output   : None (assertion)
    Memanggil: validasi_kb() dengan data contoh lengkap
    Dipanggil: pytest runner
    """
    kb_valid = {
        "intents": [
            {
                "id": "setup_awal",
                "nama": "Setup Awal",
                "contoh": ["bagaimana setup git", "cara konfigurasi nama", "atur email git", "inisialisasi akun", "konfigurasi awal"],
                "jawaban": "Gunakan git config --global.",
                "perintah": ["git config --global user.name"],
                "sumber": "Pro Git"
            }
        ],
        "fallback": {"jawaban": "Maaf tidak paham."}
    }
    galat = validasi_kb(kb_valid)
    assert galat == []


def test_validasi_kb_id_tidak_ada_di_daftar_final():
    """Uji validasi_kb melaporkan ID intent yang tidak terdaftar di DAFTAR_ID_INTENT.

    Input    : None
    Output   : None (assertion)
    Memanggil: validasi_kb() dengan ID tidak resmi
    Dipanggil: pytest runner
    """
    kb_salah = {
        "intents": [
            {
                "id": "intent_ngawur_palsu",
                "contoh": ["1", "2", "3", "4", "5"],
                "jawaban": "jawaban",
            }
        ],
        "fallback": {"jawaban": "Maaf."}
    }
    galat = validasi_kb(kb_salah)
    assert any("ID tidak ada di daftar final" in g for g in galat)


def test_validasi_kb_id_duplikat():
    """Uji validasi_kb melaporkan jika terdapat ID intent yang terduplikasi.

    Input    : None
    Output   : None (assertion)
    Memanggil: validasi_kb() dengan ID ganda
    Dipanggil: pytest runner
    """
    kb_duplikat = {
        "intents": [
            {
                "id": "clone_repo",
                "contoh": ["1", "2", "3", "4", "5"],
                "jawaban": "jawaban 1",
            },
            {
                "id": "clone_repo",
                "contoh": ["a", "b", "c", "d", "e"],
                "jawaban": "jawaban 2",
            }
        ],
        "fallback": {"jawaban": "Maaf."}
    }
    galat = validasi_kb(kb_duplikat)
    assert any("ID duplikat ditemukan" in g for g in galat)


def test_validasi_kb_contoh_kurang():
    """Uji validasi_kb melaporkan intent dengan jumlah contoh kurang dari 5.

    Input    : None
    Output   : None (assertion)
    Memanggil: validasi_kb() dengan contoh < 5
    Dipanggil: pytest runner
    """
    kb_kurang_contoh = {
        "intents": [
            {
                "id": "push_pertama",
                "contoh": ["cuma satu", "cuma dua"],
                "jawaban": "jawaban push",
            }
        ],
        "fallback": {"jawaban": "Maaf."}
    }
    galat = validasi_kb(kb_kurang_contoh)
    assert any("contoh kurang dari 5" in g for g in galat)


def test_validasi_kb_jawaban_kosong():
    """Uji validasi_kb melaporkan intent dengan field jawaban kosong atau hanya spasi.

    Input    : None
    Output   : None (assertion)
    Memanggil: validasi_kb() dengan string jawaban kosong
    Dipanggil: pytest runner
    """
    kb_jawaban_kosong = {
        "intents": [
            {
                "id": "stash",
                "contoh": ["1", "2", "3", "4", "5"],
                "jawaban": "   ",
            }
        ],
        "fallback": {"jawaban": "Maaf."}
    }
    galat = validasi_kb(kb_jawaban_kosong)
    assert any("jawaban kosong" in g for g in galat)


def test_daftar_contoh():
    """Uji fungsi daftar_contoh mengekstrak pasangan (intent_id, kalimat).

    Input    : None
    Output   : None (assertion)
    Memanggil: daftar_contoh() dengan data KB
    Dipanggil: pytest runner
    """
    kb = {
        "intents": [
            {
                "id": "clone_repo",
                "contoh": ["cara clone", "gimana clone repo"]
            },
            {
                "id": "stash",
                "contoh": ["simpan sementara perubahan"]
            }
        ]
    }
    pasangan = daftar_contoh(kb)
    assert len(pasangan) == 3
    assert pasangan[0] == ("clone_repo", "cara clone")
    assert pasangan[1] == ("clone_repo", "gimana clone repo")
    assert pasangan[2] == ("stash", "simpan sementara perubahan")


def test_muat_kb_unicode_decode_error(tmp_path: Path):
    """Uji bahwa muat_kb menangkap UnicodeDecodeError dan melempar KBError.

    Input    : tmp_path (fixture pytest)
    Output   : None (assertion)
    Memanggil: muat_kb() dengan berkas byte non-UTF8
    Dipanggil: pytest runner
    """
    file_invalid_utf8 = tmp_path / "invalid_utf8.json"
    file_invalid_utf8.write_bytes(b"\x80\x81\x82\xff")

    with pytest.raises(KBError) as exc_info:
        muat_kb(file_invalid_utf8)
    assert "bukan UTF-8 valid" in str(exc_info.value)


def test_validasi_kb_fallback_jawaban_bukan_string():
    """Uji bahwa validasi_kb tidak crash jika fallback.jawaban bukan string.

    Input    : None
    Output   : None (assertion)
    Memanggil: validasi_kb() dengan tipe data jawaban fallback selain str
    Dipanggil: pytest runner
    """
    kb_non_string_fb = {
        "intents": [],
        "fallback": {"jawaban": 12345}
    }
    galat = validasi_kb(kb_non_string_fb)
    assert any("Fallback memiliki jawaban kosong atau bukan string" in g for g in galat)

    kb_none_fb = {
        "intents": [],
        "fallback": {"jawaban": None}
    }
    galat_none = validasi_kb(kb_none_fb)
    assert any("Fallback memiliki jawaban kosong atau bukan string" in g for g in galat_none)


def test_daftar_contoh_lewati_contoh_bukan_list():
    """Uji bahwa daftar_contoh melewati intent dengan field contoh bukan list.

    Input    : None
    Output   : None (assertion)
    Memanggil: daftar_contoh() dengan entri contoh non-list
    Dipanggil: pytest runner
    """
    kb = {
        "intents": [
            {
                "id": "setup_awal",
                "contoh": "bukan sebuah list"
            },
            {
                "id": "login_github",
                "contoh": None
            },
            {
                "id": "clone_repo",
                "contoh": ["git clone url"]
            }
        ]
    }
    hasil = daftar_contoh(kb)
    assert len(hasil) == 1
    assert hasil[0] == ("clone_repo", "git clone url")

