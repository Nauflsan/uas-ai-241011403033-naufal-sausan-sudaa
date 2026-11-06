"""Modul prapemrosesan teks untuk GitBot.

Modul ini menangani pembersihan teks masukan pengguna, normalisasi huruf kecil,
penghapusan karakter kontrol dan emoji, penggantian kata tidak baku (slang),
serta proses stemming opsional bahasa Indonesia dengan menjaga keutuhan istilah
teknis dan sintaks perintah Git.
"""

import re
import string
import unicodedata
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from src.config import KAMUS_SLANG

# Regex untuk mendeteksi karakter emoji (cakupan Supplementary Multilingual Plane dan blok emoji BMP)
_POLA_EMOJI = re.compile(
    "["
    "\U00010000-\U0010FFFF"  # SMP: ragam emoji wajah, gestur, simbol modern, dan piktograf
    "\u2600-\u26FF"          # Miscellaneous Symbols: simbol cuaca, bahaya (⚠️), dll.
    "\u2700-\u27BF"          # Dingbats: centang, silang, tanda panah hiasan, dll.
    "\u2B50"                 # Simbol bintang khusus
    "\u2300-\u23FF"          # Miscellaneous Technical: jam pasir, simbol tombol
    "\u2B00-\u2BFF"          # Miscellaneous Symbols and Arrows
    "\u200d"                # Karakter zero-width joiner (penggabung urutan emoji)
    "\ufe0e-\ufe0f"          # Karakter variation selector pemilih tampilan emoji
    "]+",
    flags=re.UNICODE,
)

# Kumpulan karakter tanda baca pembatas di tepi kata
_TANDA_BACA_TEPI = frozenset(string.punctuation + "“”‘’¿¡…")

# Daftar istilah teknis Git yang dilindungi dari pemotongan afiks (stemming)
# Istilah-istilah ini adalah kata kunci domain inti yang tidak boleh dirusak bentuknya
_ISTILAH_GIT_TERLINDUNGI = frozenset({
    "push", "pull", "fetch", "clone", "commit", "branch", "merge",
    "rebase", "stash", "checkout", "switch", "reset", "revert",
    "remote", "origin", "main", "repository", "conflict",
    "git", "github", "gitlab", "master", "head", "status", "diff",
    "log", "tag", "init", "add", "divergent", "secret", "token",
    "repo", "stage", "staged", "untracked", "patch", "pop", "apply",
    "drop", "amend", "gitignore", "pr", "config", "global",
})

# Cache objek stemmer tunggal (singleton) agar tidak diinisialisasi berulang kali
_STEMMER = None


def _dapatkan_stemmer():
    """Mengambil atau menginisialisasi stemmer Sastrawi tunggal (cache).

    Input    : None
    Output   : objek Stemmer dari library PySastrawi
    Memanggil: StemmerFactory().create_stemmer() jika belum ada dalam cache
    Dipanggil: _stem_kata()
    """
    global _STEMMER
    # Inisialisasi malas (lazy): stemmer hanya dibuat sekali saat pertama kali dibutuhkan
    if _STEMMER is None:
        factory = StemmerFactory()
        _STEMMER = factory.create_stemmer()
    return _STEMMER


def _hapus_kontrol_dan_emoji(teks: str) -> str:
    """Menghapus karakter kontrol dan emoji dari teks masukan.

    Input    : teks (str) teks mentah yang akan dibersihkan
    Output   : str teks bersih tanpa karakter kontrol dan emoji
    Memanggil: _POLA_EMOJI.sub() untuk membuang emoji, unicodedata.category()
    Dipanggil: normalisasi()
    """
    # Menghapus simbol grafis emoji agar tidak menjadi derau (noise) pada pencocokan
    teks_tanpa_emoji = _POLA_EMOJI.sub("", teks)

    karakter_valid = []
    # Memfilter karakter kontrol Unicode (kategori Cc, Cf, Cs) selain whitespace standar
    for ch in teks_tanpa_emoji:
        kategori = unicodedata.category(ch)
        if kategori in ("Cc", "Cf", "Cs"):
            if ch in "\t\n\r":
                karakter_valid.append(" ")
            continue
        karakter_valid.append(ch)

    return "".join(karakter_valid)


def _pisah_tepi(token: str) -> tuple[str, str, str]:
    """Memisahkan tanda baca di tepi kiri dan kanan kata inti.

    Input    : token (str) token kata beserta kemungkinan tanda baca di tepinya
    Output   : tuple[str, str, str] berupa (tanda_depan, kata_inti, tanda_belakang)
    Memanggil: None
    Dipanggil: _ganti_slang(), _lakukan_stemming()
    """
    # Memindai tanda baca di awal token
    awal = 0
    while awal < len(token) and token[awal] in _TANDA_BACA_TEPI:
        awal += 1

    # Jika seluruh token berisi tanda baca saja (misalnya '...', '???')
    if awal == len(token):
        return token, "", ""

    # Memindai tanda baca di akhir token dari arah belakang
    akhir = len(token)
    while akhir > awal and token[akhir - 1] in _TANDA_BACA_TEPI:
        akhir -= 1

    return token[:awal], token[awal:akhir], token[akhir:]


def _mengandung_angka_atau_simbol(kata: str) -> bool:
    """Memeriksa apakah kata mengandung karakter angka atau simbol/tanda baca.

    Input    : kata (str) kata yang akan diperiksa
    Output   : bool True jika mengandung angka atau simbol, False jika huruf murni
    Memanggil: unicodedata.category() untuk mengenali tanda baca dan simbol Unicode
    Dipanggil: _stem_kata()
    """
    # Mendeteksi keberadaan digit angka, simbol ASCII, maupun simbol Unicode (P, S)
    for ch in kata:
        if ch.isdigit() or ch in string.punctuation:
            return True
        kategori = unicodedata.category(ch)
        if kategori.startswith(("P", "S")):
            return True
    return False


def _stem_kata(kata: str) -> str:
    """Melakukan proses stemming pada satu kata dengan perlindungan istilah teknis.

    Input    : kata (str) kata inti yang akan diproses
    Output   : str kata hasil stemming atau kata asli jika dilindungi
    Memanggil: _mengandung_angka_atau_simbol(), _dapatkan_stemmer(), stemmer.stem()
    Dipanggil: _lakukan_stemming()
    """
    if not kata:
        return ""

    # Kata dengan angka atau simbol (user.name, HEAD~1) dilindungi dari stemming
    if _mengandung_angka_atau_simbol(kata):
        return kata

    # Istilah teknis Git dilindungi agar semantik perintah tetap utuh
    if kata.lower() in _ISTILAH_GIT_TERLINDUNGI:
        return kata

    # Menjalankan stemmer Sastrawi; pertahankan except Exception karena Sastrawi
    # tidak menyediakan hierarki eksepsi kustom dan rantai visitor internalnya
    # dapat memicu galat tak terduga, sehingga aplikasi tetap aman dari crash
    try:
        stemmer = _dapatkan_stemmer()
        return stemmer.stem(kata)
    except Exception:
        return kata


def _ganti_slang(teks: str) -> str:
    """Mengganti kata tidak baku/slang dengan bentuk baku berbasis KAMUS_SLANG.

    Input    : teks (str) teks berhuruf kecil yang sudah dipisahkan spasi
    Output   : str teks dengan kata-kata slang digantikan bentuk baku
    Memanggil: _pisah_tepi() untuk memisahkan tanda baca di tepi kata
    Dipanggil: normalisasi()
    """
    tokens = teks.split()
    hasil_tokens = []

    # Pencocokan per kata utuh dengan menjaga tanda baca di tepi kata
    for token in tokens:
        depan, kata, belakang = _pisah_tepi(token)
        if kata in KAMUS_SLANG:
            kata = KAMUS_SLANG[kata]
        hasil_tokens.append(f"{depan}{kata}{belakang}")

    return " ".join(hasil_tokens)


def _lakukan_stemming(teks: str) -> str:
    """Melakukan proses stemming bahasa Indonesia per kata pada kalimat.

    Input    : teks (str) kalimat masukan hasil normalisasi slang
    Output   : str kalimat dengan kata dasar bahasa Indonesia yang distem
    Memanggil: _pisah_tepi(), _stem_kata()
    Dipanggil: normalisasi()
    """
    tokens = teks.split()
    hasil_tokens = []

    # Memproses stemming kata per kata dan menyusun kembali tanda baca tepi
    for token in tokens:
        depan, kata, belakang = _pisah_tepi(token)
        kata_stem = _stem_kata(kata)
        hasil_tokens.append(f"{depan}{kata_stem}{belakang}")

    return " ".join(hasil_tokens)


def normalisasi(teks: str, stem: bool = False) -> str:
    """Melakukan rangkaian prapemrosesan teks masukan untuk GitBot.

    Input    : teks (str) teks masukan pengguna, stem (bool) opsi untuk stemming
    Output   : str teks hasil prapemrosesan siap pakai
    Memanggil: _hapus_kontrol_dan_emoji(), _ganti_slang(), _lakukan_stemming()
    Dipanggil: belum ada (direncanakan dari matcher.py pada Tahap 5 dan chatbot.py pada Tahap 8 sesuai PLAN-KODE.md)
    """
    # Validasi parameter tipe data secara eksplisit
    if not isinstance(teks, str):
        raise TypeError("Parameter 'teks' harus berupa string.")
    if not isinstance(stem, bool):
        raise TypeError("Parameter 'stem' harus bertipe boolean.")

    # Langkah 1: Buang emoji dan karakter kontrol perusak
    teks_bersih = _hapus_kontrol_dan_emoji(teks)

    # Langkah 2: Ubah seluruh karakter menjadi huruf kecil (case folding)
    teks_kecil = teks_bersih.lower()

    # Langkah 3: Normalisasi kata slang per kata utuh
    teks_slang = _ganti_slang(teks_kecil)

    # Langkah 4: Lakukan stemming opsional jika diminta
    if stem:
        teks_final = _lakukan_stemming(teks_slang)
    else:
        teks_final = teks_slang

    # Langkah 5: Rapikan spasi berlebih menjadi spasi tunggal yang rapi
    return " ".join(teks_final.split())


if __name__ == "__main__":
    print("=== Demonstrasi Modul Prapemrosesan Teks GitBot ===")
    contoh_uji = [
        "Halo bro! Gimana cara git push ke origin main? 🚀",
        "Kenapa ya HEAD~1 saya bermasalah waktu ngemerge branch?",
        "Batalin commit terakhir dengan git reset --soft HEAD~1...",
        "user.name dan email konfigurasi blm diatur, trs solusinya apa?",
        "Push ditolak gara-gara divergent branches, apa yg harus dilakukan?!",
    ]

    for no, kalimat in enumerate(contoh_uji, start=1):
        print(f"\nContoh {no}:")
        print(f"Asli       : {kalimat}")
        print(f"Tanpa stem : {normalisasi(kalimat, stem=False)}")
        print(f"Dengan stem: {normalisasi(kalimat, stem=True)}")
