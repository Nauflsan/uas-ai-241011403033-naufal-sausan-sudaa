"""Modul logika utama chatbot GitBot.

Modul ini menangani validasi masukan pertanyaan pengguna, integrasi knowledge base
dengan matcher pilihan, evaluasi ambang batas kemiripan (threshold), penanganan
respons fallback, serta antarmuka interaktif pada mode terminal CLI.
"""

import argparse
import sys
from typing import Any, Optional

from src.config import MAKS_PANJANG_INPUT, MODEL_EMBEDDING, PATH_KB, THRESHOLD
from src.knowledge_base import KBError, daftar_contoh, muat_kb
from src.matcher import BaseMatcher, EmbeddingMatcher, IndoBertMatcher, TfidfMatcher


def validasi_input(pesan: Any) -> tuple[bool, str]:
    """Memvalidasi masukan teks pertanyaan dari pengguna sebelum diproses.

    Input    : pesan (Any) teks pesan pengguna yang akan divalidasi
    Output   : tuple[bool, str] status kelayakan (True jika valid) dan pesan kesalahan
    Memanggil: isinstance(), str.strip(), len(), any(), str.isalnum()
    Dipanggil: GitBot.jawab(), belum ada (direncanakan dari app.py pada Tahap 9 sesuai PLAN-KODE.md)
    """
    if not isinstance(pesan, str):
        return False, "Pesan masukan harus berupa teks string."

    teks_bersih = pesan.strip()
    if not teks_bersih:
        return False, "Pesan tidak boleh kosong atau hanya berisi spasi."

    if len(teks_bersih) > MAKS_PANJANG_INPUT:
        return (
            False,
            f"Panjang pesan melebihi batas maksimal {MAKS_PANJANG_INPUT} karakter.",
        )

    # Memastikan pesan memiliki minimal satu karakter alfanumerik (huruf atau angka)
    # Simbol saja (misal '???') atau emoji saja (misal '😀👍') otomatis ditolak
    if not any(c.isalnum() for c in teks_bersih):
        return False, "Pesan harus mengandung setidaknya satu huruf atau angka."

    return True, ""


def _validasi_kb(kb: Any) -> None:
    """Memeriksa integritas struktur dan skema dasar kamus knowledge base.

    Input    : kb (Any) data knowledge base yang akan diperiksa
    Output   : None jika valid; melempar ValueError jika struktur salah
    Memanggil: isinstance(), dict.get(), str.strip()
    Dipanggil: GitBot.__init__()
    """
    if not isinstance(kb, dict):
        raise ValueError("Parameter kb harus berupa dictionary.")
    if "intents" not in kb or not isinstance(kb["intents"], list):
        raise ValueError("Knowledge base harus memiliki field 'intents' bertipe list.")
    fallback = kb.get("fallback")
    if (
        not isinstance(fallback, dict)
        or not isinstance(fallback.get("jawaban"), str)
        or not fallback["jawaban"].strip()
    ):
        raise ValueError(
            "Knowledge base harus memiliki field 'fallback' dengan 'jawaban' string tidak kosong."
        )


def _validasi_matcher(matcher: Any) -> None:
    """Memeriksa keberadaan dan keabsahan antarmuka metode cari pada matcher.

    Input    : matcher (Any) objek pencocok intent
    Output   : None jika valid; melempar TypeError jika tidak memiliki metode cari
    Memanggil: hasattr(), getattr(), callable()
    Dipanggil: GitBot.__init__()
    """
    if not hasattr(matcher, "cari") or not callable(getattr(matcher, "cari")):
        raise TypeError("Parameter matcher harus memiliki metode callable 'cari'.")


def _validasi_threshold(threshold: Any) -> float:
    """Memvalidasi tipe data dan rentang numerik ambang batas kemiripan.

    Input    : threshold (Any) nilai ambang batas kemiripan yang akan diperiksa
    Output   : float nilai ambang batas valid dalam rentang [0.0, 1.0]
    Memanggil: isinstance(), float()
    Dipanggil: GitBot.__init__()
    """
    # bool adalah turunan int di Python sehingga harus ditolak secara eksplisit
    if isinstance(threshold, bool) or not isinstance(threshold, (float, int)):
        raise ValueError("Parameter threshold harus berupa bilangan float.")
    nilai = float(threshold)
    if not (0.0 <= nilai <= 1.0):
        raise ValueError("Nilai threshold harus berada dalam rentang 0.0 hingga 1.0.")
    return nilai


class GitBot:
    """Kelas utama asisten GitBot berbasis knowledge base dan pencocok kemiripan.

    Menangani validasi pertanyaan, integrasi pencarian kemiripan intent, evaluasi
    ambang batas (threshold), penanganan respons fallback, serta proteksi galat.

    Input    : kb (dict), matcher (objek dengan metode cari), threshold (float 0.0-1.0)
    Output   : instansi GitBot
    Memanggil: _validasi_kb(), _validasi_matcher(), _validasi_threshold()
    Dipanggil: main(), belum ada (direncanakan dari app.py pada Tahap 9 dan evaluate.py pada Tahap 10 sesuai PLAN-KODE.md)
    """

    def __init__(self, kb: dict, matcher: Any, threshold: float) -> None:
        """Menginisialisasi GitBot dan membangun pemetaan indeks intent.

        Input    : kb (dict), matcher (Any), threshold (float)
        Output   : None
        Memanggil: _validasi_kb(), _validasi_matcher(), _validasi_threshold()
        Dipanggil: main(), belum ada (direncanakan dari app.py pada Tahap 9 dan evaluate.py pada Tahap 10 sesuai PLAN-KODE.md)
        """
        _validasi_kb(kb)
        _validasi_matcher(matcher)
        self.threshold = _validasi_threshold(threshold)
        self.kb = kb
        self.matcher = matcher
        # Membangun pemetaan id intent -> entri intent satu kali saat inisialisasi
        self._peta_intent: dict[str, dict] = {
            item["id"]: item
            for item in kb["intents"]
            if isinstance(item, dict) and "id" in item
        }

    @staticmethod
    def _buat_respons(
        status: str,
        intent: Optional[str],
        skor: float,
        jawaban: str,
        perintah: list[str],
        kandidat: list[tuple[str, float]],
        pesan_error: str,
    ) -> dict:
        """Membentuk dictionary respons standar GitBot dengan tepat tujuh kunci.

        Input    : status, intent, skor, jawaban, perintah, kandidat, pesan_error
        Output   : dict berisi tepat 7 field standar respons GitBot
        Memanggil: dict literal
        Dipanggil: GitBot.jawab(), GitBot._evaluasi_kandidat()
        """
        return {
            "status": status,
            "intent": intent,
            "skor": skor,
            "jawaban": jawaban,
            "perintah": perintah,
            "kandidat": kandidat,
            "pesan_error": pesan_error,
        }

    def _evaluasi_kandidat(self, kandidat: list[tuple[str, float]]) -> dict:
        """Mengevaluasi daftar kandidat intent terhadap threshold dan knowledge base.

        Input    : kandidat (list[tuple[str, float]]) pasangan id intent dan skor
        Output   : dict respons dengan status ok, fallback, atau error
        Memanggil: self._buat_respons(), list(), dict.get()
        Dipanggil: GitBot.jawab()
        """
        if not kandidat:
            return self._buat_respons(
                "fallback", None, 0.0, self.kb["fallback"]["jawaban"], [], [], ""
            )

        id_teratas, skor_teratas = kandidat[0]
        skor_f = float(skor_teratas)
        if id_teratas not in self._peta_intent:
            return self._buat_respons(
                "error",
                None,
                skor_f,
                "",
                [],
                list(kandidat),
                f"Intent '{id_teratas}' tidak ditemukan dalam knowledge base.",
            )

        # Evaluasi threshold kemiripan kosine:
        # Skor kemiripan kosine adalah ukuran kedekatan sudut antar-vektor kalimat,
        # BUKAN probabilitas Bayesian atau probabilitas softmax (tidak harus berjumlah 1.0).
        # Pertanyaan dengan skor di bawah ambang batas (threshold) dialihkan ke fallback
        # agar chatbot tidak memberikan jawaban yang keliru (menghindari false positive).
        # Nilai threshold dari config.py adalah baseline heuristik yang dituning pada Tahap 11.
        if skor_f >= self.threshold:
            entri = self._peta_intent[id_teratas]
            cmd = entri.get("perintah", [])
            perintah = list(cmd) if isinstance(cmd, list) else []
            return self._buat_respons(
                "ok",
                id_teratas,
                skor_f,
                entri.get("jawaban", ""),
                perintah,
                list(kandidat),
                "",
            )

        return self._buat_respons(
            "fallback",
            None,
            skor_f,
            self.kb["fallback"]["jawaban"],
            [],
            list(kandidat),
            "",
        )

    def jawab(self, pesan: Any) -> dict:
        """Memproses pesan masukan pengguna dan menghasilkan respons jawaban GitBot.

        Input    : pesan (Any) teks pertanyaan pengguna yang diajukan ke chatbot
        Output   : dict berisi status, intent, skor, jawaban, perintah, kandidat, pesan_error
        Memanggil: validasi_input(), self.matcher.cari(), self._buat_respons(), self._evaluasi_kandidat()
        Dipanggil: main(), _jalankan_loop(), belum ada (direncanakan dari app.py pada Tahap 9 dan evaluate.py pada Tahap 10 sesuai PLAN-KODE.md)
        """
        valid, pesan_err = validasi_input(pesan)
        if not valid:
            return self._buat_respons("error", None, 0.0, "", [], [], pesan_err)

        try:
            kandidat = self.matcher.cari(pesan, top_k=3)
        except (RuntimeError, ValueError, TypeError) as err:
            return self._buat_respons(
                "error",
                None,
                0.0,
                "",
                [],
                [],
                f"Kesalahan saat pencocokan intent: {err}",
            )

        return self._evaluasi_kandidat(kandidat)


def _buat_matcher(metode: str) -> BaseMatcher:
    """Membuat instansi matcher berdasarkan pengenal metode yang dipilih.

    Input    : metode (str) pengenal metode pencocokan ('tfidf', 'embedding', 'indobert')
    Output   : instansi BaseMatcher siap di-fit
    Memanggil: TfidfMatcher(), EmbeddingMatcher(), IndoBertMatcher()
    Dipanggil: main(), app.py
    """
    nama = metode.lower().strip()
    if nama == "tfidf":
        return TfidfMatcher()
    if nama == "embedding":
        return EmbeddingMatcher(MODEL_EMBEDDING)
    if nama == "indobert":
        return IndoBertMatcher()
    raise ValueError(f"Metode pencocokan tidak dikenal: '{metode}'")


def _cetak_respons(hasil: dict) -> None:
    """Mencetak dictionary respons jawaban GitBot ke terminal secara terstruktur.

    Input    : hasil (dict) dictionary respons hasil GitBot.jawab()
    Output   : None
    Memanggil: print()
    Dipanggil: main(), _jalankan_loop()
    """
    print(f"Status  : {hasil['status']}")
    intent_str = hasil["intent"] if hasil["intent"] is not None else "-"
    print(f"Intent  : {intent_str} (skor: {hasil['skor']:.4f})")
    if hasil["jawaban"]:
        print(f"Jawaban : {hasil['jawaban']}")
    if hasil["perintah"]:
        print("Perintah:")
        for cmd in hasil["perintah"]:
            print(f"  $ {cmd}")
    if hasil["kandidat"]:
        print("Kandidat:")
        for cid, cskor in hasil["kandidat"]:
            print(f"  - {cid}: {cskor:.4f}")
    if hasil["pesan_error"]:
        print(f"Galat   : {hasil['pesan_error']}")


def _jalankan_loop(bot: GitBot) -> None:
    """Menjalankan loop interaktif tanya-jawab GitBot pada mode konsol terminal.

    Input    : bot (GitBot) instansi GitBot yang telah siap melayani pertanyaan
    Output   : None
    Memanggil: input(), print(), str.strip(), str.lower(), bot.jawab(), _cetak_respons()
    Dipanggil: main()
    """
    print("GitBot siap membantu Anda seputar Git & GitHub! (ketik 'keluar' untuk selesai)")
    while True:
        try:
            masukan = input("\nAnda: ")
        except (EOFError, KeyboardInterrupt):
            print("\nSampai jumpa!")
            break

        if masukan.strip().lower() in ("keluar", "exit", "quit"):
            print("Sampai jumpa!")
            break

        hasil = bot.jawab(masukan)
        _cetak_respons(hasil)


def _parse_args(argv: Optional[list[str]]) -> argparse.Namespace:
    """Mem-parsing argumen baris perintah untuk mode eksekusi CLI GitBot.

    Input    : argv (Optional[list[str]]) daftar argumen baris perintah
    Output   : argparse.Namespace berisi hasil parsing argumen CLI
    Memanggil: argparse.ArgumentParser(), add_argument(), parse_args()
    Dipanggil: main()
    """
    parser = argparse.ArgumentParser(description="GitBot - Chatbot Asisten Git & GitHub")
    parser.add_argument(
        "--metode",
        choices=["tfidf", "embedding", "indobert"],
        default="embedding",
        help="Metode pencocokan intent (default: embedding)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Ambang batas kemiripan minimum (default: THRESHOLD[metode])",
    )
    parser.add_argument(
        "--tanya",
        type=str,
        default=None,
        help="Pertanyaan tunggal untuk dijawab langsung lalu selesai",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Fungsi titik masuk (entry point) utama untuk menjalankan GitBot di terminal.

    Input    : argv (Optional[list[str]]) argumen baris perintah (None untuk sys.argv[1:])
    Output   : int kode keluar (0 jika sukses, 1 jika inisialisasi gagal)
    Memanggil: _parse_args(), muat_kb(), _buat_matcher(), matcher.fit(), daftar_contoh(), GitBot(), bot.jawab(), _cetak_respons(), _jalankan_loop()
    Dipanggil: modul chatbot (__main__)
    """
    args = _parse_args(argv)
    try:
        kb = muat_kb(PATH_KB)
        matcher = _buat_matcher(args.metode)
        matcher.fit(daftar_contoh(kb))
        ambang = (
            args.threshold if args.threshold is not None else THRESHOLD[args.metode]
        )
        bot = GitBot(kb=kb, matcher=matcher, threshold=ambang)
    except (KBError, RuntimeError, ValueError, TypeError) as err:
        print(f"Kesalahan inisialisasi GitBot: {err}", file=sys.stderr)
        return 1

    if args.tanya is not None:
        hasil = bot.jawab(args.tanya)
        _cetak_respons(hasil)
        return 0

    _jalankan_loop(bot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
