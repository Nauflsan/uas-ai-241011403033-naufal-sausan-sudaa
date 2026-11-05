"""Modul pemuat dan validasi knowledge base untuk GitBot.

Modul ini bertanggung jawab untuk membaca berkas JSON basis pengetahuan,
memvalidasi integritas data dan format intent, serta mengekstrak daftar
pasangan intent dan contoh pertanyaan untuk proses pencocokan intent.
"""

import json
from pathlib import Path
from typing import Union

# Daftar 18 ID intent final yang sah sesuai spesifikasi proyek di PLAN-KODE.md
DAFTAR_ID_INTENT = (
    "setup_awal",
    "login_github",
    "clone_repo",
    "push_pertama",
    "saran_commit",
    "bikin_pindah_branch",
    "fetch_vs_pull",
    "push_ditolak",
    "divergent_branches",
    "merge_conflict",
    "merge_branch",
    "local_changes_overwritten",
    "alur_tim_aman",
    "pull_request",
    "batalin_commit",
    "stash",
    "gitignore",
    "secret_kepush",
)


class KBError(Exception):
    """Pengecualian khusus untuk kesalahan berkas atau skema knowledge base.

    Input    : pesan kesalahan (str)
    Output   : instansi KBError
    Memanggil: Exception.__init__() untuk inisialisasi pesan kesalahan
    Dipanggil: muat_kb() saat terjadi kesalahan berkas, JSON rusak, atau skema
    """
    pass


def _validasi_skema(data: dict) -> None:
    """Validasi skema dasar dictionary berkas knowledge base.

    Input    : data (dict) hasil pembacaan JSON knowledge base
    Output   : None jika skema valid; melempar KBError jika salah
    Memanggil: isinstance() untuk memeriksa tipe dictionary dan list
    Dipanggil: muat_kb()
    """
    # Memeriksa tipe data root knowledge base
    if not isinstance(data, dict):
        raise KBError("Format knowledge base tidak valid: root harus dictionary.")

    # Memeriksa keberadaan dan tipe field intents
    if "intents" not in data or not isinstance(data["intents"], list):
        raise KBError("Skema salah: kunci 'intents' harus ada dan berupa list.")

    # Memeriksa keberadaan dan tipe field fallback
    if "fallback" not in data or not isinstance(data["fallback"], dict):
        raise KBError("Skema salah: kunci 'fallback' harus ada dan berupa dictionary.")

    # Memeriksa keberadaan jawaban pada fallback
    if not data["fallback"].get("jawaban"):
        raise KBError("Skema salah: 'fallback' harus memiliki field 'jawaban'.")


def muat_kb(path: Union[str, Path]) -> dict:
    """Muat berkas JSON basis pengetahuan dan validasi struktur dasarnya.

    Input    : path (str atau Path) lokasi berkas knowledge base
    Output   : dict berisi data knowledge base yang telah dimuat
    Memanggil: open(), json.load(), _validasi_skema() untuk validasi struktur
    Dipanggil: GitBot.__init__(), modul pengujian, dan skrip evaluasi
    """
    path_obj = Path(path)

    # Membaca berkas JSON dengan penanganan error spesifik
    try:
        with open(path_obj, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as err:
        raise KBError(f"Berkas knowledge base tidak ditemukan: {path_obj}") from err
    except json.JSONDecodeError as err:
        raise KBError(f"Format JSON knowledge base rusak: {err}") from err
    except UnicodeDecodeError as err:
        raise KBError(f"Encoding berkas knowledge base bukan UTF-8 valid: {err}") from err
    except OSError as err:
        raise KBError(f"Gagal membaca berkas knowledge base: {err}") from err

    # Memvalidasi skema dasar data JSON yang terbaca
    _validasi_skema(data)
    return data


def _periksa_intent(intent: dict, idx: int, id_tercatat: set, galat: list[str]) -> None:
    """Periksa kesesuaian satu entri intent terhadap kriteria validasi.

    Input    : intent (dict), indeks (int), id_tercatat (set), galat (list[str])
    Output   : None (menambahkan pesan kesalahan ke list galat)
    Memanggil: isinstance(), str.strip() untuk memeriksa tipe dan teks
    Dipanggil: validasi_kb()
    """
    if not isinstance(intent, dict):
        galat.append(f"Intent ke-{idx + 1} bukan merupakan dictionary.")
        return

    intent_id = intent.get("id")
    # Periksa ID tidak ada di daftar final atau bernilai kosong
    if not intent_id or intent_id not in DAFTAR_ID_INTENT:
        galat.append(f"ID tidak ada di daftar final: '{intent_id}'.")
    elif intent_id in id_tercatat:
        galat.append(f"ID duplikat ditemukan: '{intent_id}'.")
    else:
        id_tercatat.add(intent_id)

    # Periksa jumlah contoh kalimat minimal 5
    contoh = intent.get("contoh")
    if not isinstance(contoh, list) or len(contoh) < 5:
        galat.append(f"Intent '{intent_id}': contoh kurang dari 5.")

    # Periksa ketersediaan jawaban tidak kosong
    jawaban = intent.get("jawaban")
    if not isinstance(jawaban, str) or not jawaban.strip():
        galat.append(f"Intent '{intent_id}': jawaban kosong.")


def validasi_kb(kb: dict) -> list[str]:
    """Validasi integritas konten knowledge base terhadap aturan proyek.

    Input    : kb (dict) data knowledge base
    Output   : list[str] berisi daftar pesan kesalahan (kosong jika valid)
    Memanggil: _periksa_intent() untuk memvalidasi tiap elemen intent
    Dipanggil: Skrip CLI knowledge base, matcher, dan modul pengujian
    """
    # Memeriksa apakah root berupa dict dan memiliki intents list
    if not isinstance(kb, dict) or not isinstance(kb.get("intents"), list):
        return ["Skema salah: root harus dictionary dengan kunci 'intents' list."]

    galat = []
    id_tercatat = set()

    # Periksa setiap entri intent dalam list
    for idx, intent in enumerate(kb["intents"]):
        _periksa_intent(intent, idx, id_tercatat, galat)

    # Periksa field fallback pada root tanpa crash bila jawaban bukan string
    fallback = kb.get("fallback")
    jawaban_fb = fallback.get("jawaban") if isinstance(fallback, dict) else None
    if not isinstance(jawaban_fb, str) or not jawaban_fb.strip():
        galat.append("Fallback memiliki jawaban kosong atau bukan string.")

    return galat


def daftar_contoh(kb: dict) -> list[tuple[str, str]]:
    """Ekstrak semua pasangan ID intent dan kalimat contoh dari knowledge base.

    Input    : kb (dict) data knowledge base
    Output   : list[tuple[str, str]] pasangan (intent_id, kalimat_contoh)
    Memanggil: isinstance() untuk validasi list, dict.get() untuk membaca data
    Dipanggil: Matcher.fit() saat membentuk representasi vektor/indeks kata
    """
    hasil = []
    intents = kb.get("intents", []) if isinstance(kb, dict) else []

    # Iterasi setiap intent dan lewati yang contohnya bukan list
    for intent in intents:
        if isinstance(intent, dict):
            contoh_list = intent.get("contoh")
            if not isinstance(contoh_list, list):
                continue
            intent_id = intent.get("id", "")
            for kalimat in contoh_list:
                if isinstance(kalimat, str) and kalimat.strip():
                    hasil.append((intent_id, kalimat.strip()))

    return hasil


if __name__ == "__main__":
    from src.config import PATH_KB

    print(f"Memuat knowledge base dari: {PATH_KB}")
    try:
        kb_data = muat_kb(PATH_KB)
        daftar_galat = validasi_kb(kb_data)
        if daftar_galat:
            print(f"Validasi gagal dengan {len(daftar_galat)} kesalahan:")
            for g in daftar_galat:
                print(f" - {g}")
        else:
            total_intent = len(kb_data.get("intents", []))
            total_contoh = len(daftar_contoh(kb_data))
            print("Knowledge base valid!")
            print(f"Total intent: {total_intent}, Total contoh kalimat: {total_contoh}")
    except KBError as err:
        print(f"Kesalahan knowledge base: {err}")
