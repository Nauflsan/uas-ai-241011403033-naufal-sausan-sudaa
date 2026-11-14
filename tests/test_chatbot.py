"""Modul pengujian unit dan integrasi untuk modul chatbot GitBot.

Pengujian ini menguji fungsi validasi_input, konstruktor GitBot, metode jawab,
serta titik masuk CLI main() menggunakan matcher tiruan (dummy) dan KB kecil
tanpa memuat atau mengunduh model sentence embedding yang berat.
"""

from typing import Any
import pytest

from src.chatbot import GitBot, _buat_matcher, main, validasi_input
from src.config import MAKS_PANJANG_INPUT
from src.knowledge_base import KBError, daftar_contoh
from src.matcher import TfidfMatcher

# Basis pengetahuan dummy kecil untuk pengujian cepat dan terisolasi
KB_UJI = {
    "intents": [
        {
            "id": "setup_awal",
            "nama": "Konfigurasi Awal Git",
            "contoh": [
                "bagaimana cara mengatur nama di git",
                "cara konfigurasi user git pertama kali",
                "setup identitas git",
                "cara setting email git",
                "konfigurasi awal git",
            ],
            "jawaban": "Gunakan perintah git config --global user.name dan user.email.",
            "perintah": [
                'git config --global user.name "Nama Anda"',
                'git config --global user.email "email@anda.com"',
            ],
            "sumber": "Pro Git Book Bab 1.6",
        },
        {
            "id": "push_pertama",
            "nama": "Push Pertama Kali",
            "contoh": [
                "bagaimana push pertama kali ke remote",
                "cara upload commit pertama ke github",
                "push branch main pertama kali",
                "perintah push perdana",
                "kirim commit pertama ke origin",
            ],
            "jawaban": "Gunakan git push -u origin main untuk push pertama kali.",
            "perintah": ["git push -u origin main"],
            "sumber": "GitHub Docs",
        },
    ],
    "fallback": {
        "jawaban": "Maaf, saya belum memahami pertanyaan Anda seputar Git. Silakan gunakan kalimat lain."
    },
}

KUNCI_STANDAR_RESPONS = {
    "status",
    "intent",
    "skor",
    "jawaban",
    "perintah",
    "kandidat",
    "pesan_error",
}


class DummyMatcher:
    """Matcher tiruan untuk pengujian tanpa komputasi tensor atau model NLP berat.

    Input    : kembalian_cari (list), galat_cari (Exception)
    Output   : instansi DummyMatcher
    Memanggil: None
    Dipanggil: fungsi-fungsi pengujian di test_chatbot.py
    """

    def __init__(
        self,
        kembalian_cari: Any = None,
        galat_cari: Any = None,
    ) -> None:
        """Menginisialisasi matcher tiruan dengan respon tetap dan pelacak panggilan.

        Input    : kembalian_cari (Any), galat_cari (Any)
        Output   : None
        Memanggil: None
        Dipanggil: pytest
        """
        self.kembalian_cari = (
            kembalian_cari
            if kembalian_cari is not None
            else [("setup_awal", 0.90)]
        )
        self.galat_cari = galat_cari
        self.call_count = 0
        self.fit_call_count = 0

    def fit(self, contoh: list) -> None:
        """Mencatat pemanggilan fit tanpa komputasi vektor sebenarnya.

        Input    : contoh (list) daftar pasangan intent dan kalimat
        Output   : None
        Memanggil: None
        Dipanggil: pytest
        """
        self.fit_call_count += 1

    def cari(self, teks: str, top_k: int = 3) -> list[tuple[str, float]]:
        """Mencatat jumlah pemanggilan cari dan mengembalikan hasil atau galat terkonfigurasi.

        Input    : teks (str), top_k (int)
        Output   : list[tuple[str, float]] daftar kandidat intent teratas
        Memanggil: None
        Dipanggil: pytest, GitBot.jawab()
        """
        self.call_count += 1
        if self.galat_cari is not None:
            raise self.galat_cari
        return self.kembalian_cari[:top_k]


def test_validasi_input_teks_valid() -> None:
    """Menguji bahwa teks pertanyaan normal yang valid diterima dengan benar.

    Input    : None
    Output   : None
    Memanggil: validasi_input()
    Dipanggil: pytest
    """
    valid, pesan = validasi_input("Bagaimana cara membuat commit baru?")
    assert valid is True
    assert pesan == ""


def test_validasi_input_kosong_dan_spasi() -> None:
    """Menguji penolakan string kosong dan string yang hanya berisi spasi atau tab.

    Input    : None
    Output   : None
    Memanggil: validasi_input()
    Dipanggil: pytest
    """
    valid_kosong, pesan_k = validasi_input("")
    assert valid_kosong is False
    assert "tidak boleh kosong" in pesan_k

    valid_spasi, pesan_s = validasi_input("   \t  \n  ")
    assert valid_spasi is False
    assert "tidak boleh kosong" in pesan_s


def test_validasi_input_panjang_maksimal() -> None:
    """Menguji batas panjang karakter: tepat 300 masih valid, 301 ditolak.

    Input    : None
    Output   : None
    Memanggil: validasi_input()
    Dipanggil: pytest
    """
    teks_300 = "a" * MAKS_PANJANG_INPUT
    valid_300, pesan_300 = validasi_input(teks_300)
    assert valid_300 is True
    assert pesan_300 == ""

    teks_301 = "a" * (MAKS_PANJANG_INPUT + 1)
    valid_301, pesan_301 = validasi_input(teks_301)
    assert valid_301 is False
    assert "melebihi batas maksimal" in pesan_301


def test_validasi_input_simbol_dan_emoji_saja() -> None:
    """Menguji penolakan masukan yang hanya terdiri atas simbol atau emoji saja.

    Input    : None
    Output   : None
    Memanggil: validasi_input()
    Dipanggil: pytest
    """
    valid_simbol, pesan_simbol = validasi_input("???!!!###$$$")
    assert valid_simbol is False
    assert "setidaknya satu huruf atau angka" in pesan_simbol

    valid_emoji, pesan_emoji = validasi_input("😀🤖🎉🚀")
    assert valid_emoji is False
    assert "setidaknya satu huruf atau angka" in pesan_emoji


def test_validasi_input_angka_saja_dan_kombinasi_simbol() -> None:
    """Menguji masukan angka saja dan kombinasi huruf dengan simbol teknis Git.

    Input    : None
    Output   : None
    Memanggil: validasi_input()
    Dipanggil: pytest
    """
    valid_angka, pesan_angka = validasi_input("123456")
    assert valid_angka is True
    assert pesan_angka == ""

    valid_kombinasi, pesan_kombinasi = validasi_input("git push -u origin main?!")
    assert valid_kombinasi is True
    assert pesan_kombinasi == ""


def test_validasi_input_bukan_string() -> None:
    """Menguji penolakan masukan bertipe selain string tanpa melempar exception.

    Input    : None
    Output   : None
    Memanggil: validasi_input()
    Dipanggil: pytest
    """
    for masukan in [12345, None, True, False, ["perintah"], {"teks": "halo"}]:
        valid, pesan = validasi_input(masukan)
        assert valid is False
        assert "harus berupa teks string" in pesan


def test_kunci_hasil_jawab_tepat_tujuh_kunci() -> None:
    """Menguji bahwa respons jawab selalu berisi tepat tujuh kunci pada semua status.

    Input    : None
    Output   : None
    Memanggil: GitBot(), bot.jawab()
    Dipanggil: pytest
    """
    matcher = DummyMatcher(kembalian_cari=[("setup_awal", 0.85)])
    bot = GitBot(kb=KB_UJI, matcher=matcher, threshold=0.5)

    # 1. Status OK
    hasil_ok = bot.jawab("bagaimana setup git?")
    assert set(hasil_ok.keys()) == KUNCI_STANDAR_RESPONS
    assert hasil_ok["status"] == "ok"

    # 2. Status Fallback
    bot.threshold = 0.95
    hasil_fb = bot.jawab("bagaimana setup git?")
    assert set(hasil_fb.keys()) == KUNCI_STANDAR_RESPONS
    assert hasil_fb["status"] == "fallback"

    # 3. Status Error
    hasil_err = bot.jawab("   ")
    assert set(hasil_err.keys()) == KUNCI_STANDAR_RESPONS
    assert hasil_err["status"] == "error"


def test_ok_tepat_di_threshold() -> None:
    """Menguji bahwa skor yang sama persis dengan threshold menghasilkan status ok.

    Input    : None
    Output   : None
    Memanggil: GitBot(), bot.jawab()
    Dipanggil: pytest
    """
    matcher = DummyMatcher(kembalian_cari=[("setup_awal", 0.50)])
    bot = GitBot(kb=KB_UJI, matcher=matcher, threshold=0.50)
    hasil = bot.jawab("cara setup git")

    assert hasil["status"] == "ok"
    assert hasil["intent"] == "setup_awal"
    assert hasil["skor"] == 0.50
    assert hasil["jawaban"] == KB_UJI["intents"][0]["jawaban"]
    assert hasil["perintah"] == KB_UJI["intents"][0]["perintah"]
    assert hasil["pesan_error"] == ""


def test_fallback_di_bawah_threshold() -> None:
    """Menguji bahwa skor di bawah threshold menghasilkan status fallback.

    Input    : None
    Output   : None
    Memanggil: GitBot(), bot.jawab()
    Dipanggil: pytest
    """
    matcher = DummyMatcher(kembalian_cari=[("setup_awal", 0.4999)])
    bot = GitBot(kb=KB_UJI, matcher=matcher, threshold=0.50)
    hasil = bot.jawab("masak nasi goreng")

    assert hasil["status"] == "fallback"
    assert hasil["intent"] is None
    assert hasil["skor"] == 0.4999
    assert hasil["jawaban"] == KB_UJI["fallback"]["jawaban"]
    assert hasil["perintah"] == []
    assert hasil["kandidat"] == [("setup_awal", 0.4999)]
    assert hasil["pesan_error"] == ""


def test_validasi_gagal_tidak_memanggil_matcher() -> None:
    """Menguji bahwa kegagalan validasi masukan tidak memanggil metode matcher.cari.

    Input    : None
    Output   : None
    Memanggil: GitBot(), bot.jawab()
    Dipanggil: pytest
    """
    matcher = DummyMatcher()
    bot = GitBot(kb=KB_UJI, matcher=matcher, threshold=0.50)
    hasil = bot.jawab("   ")

    assert hasil["status"] == "error"
    assert hasil["intent"] is None
    assert hasil["skor"] == 0.0
    assert hasil["jawaban"] == ""
    assert hasil["perintah"] == []
    assert hasil["kandidat"] == []
    assert hasil["pesan_error"] != ""
    assert matcher.call_count == 0


def test_matcher_melempar_galat_menjadi_status_error() -> None:
    """Menguji penanganan galat saat matcher melempar RuntimeError, ValueError, atau TypeError.

    Input    : None
    Output   : None
    Memanggil: GitBot(), bot.jawab()
    Dipanggil: pytest
    """
    for galat in [
        RuntimeError("Koneksi tensor terputus"),
        ValueError("Dimensi vektor tidak sesuai"),
        TypeError("Tipe data masukan salah"),
    ]:
        matcher = DummyMatcher(galat_cari=galat)
        bot = GitBot(kb=KB_UJI, matcher=matcher, threshold=0.50)
        hasil = bot.jawab("cara setup git")

        assert hasil["status"] == "error"
        assert hasil["intent"] is None
        assert hasil["skor"] == 0.0
        assert hasil["jawaban"] == ""
        assert hasil["perintah"] == []
        assert hasil["kandidat"] == []
        assert str(galat) in hasil["pesan_error"]


def test_intent_tidak_ada_di_kb_menjadi_error() -> None:
    """Menguji bahwa intent hasil pencocokan yang tidak terdaftar di KB menghasilkan error.

    Input    : None
    Output   : None
    Memanggil: GitBot(), bot.jawab()
    Dipanggil: pytest
    """
    matcher = DummyMatcher(kembalian_cari=[("intent_tidak_dikenal", 0.95)])
    bot = GitBot(kb=KB_UJI, matcher=matcher, threshold=0.50)
    hasil = bot.jawab("pertanyaan acak")

    assert hasil["status"] == "error"
    assert hasil["intent"] is None
    assert hasil["skor"] == 0.95
    assert "intent_tidak_dikenal" in hasil["pesan_error"]
    assert hasil["kandidat"] == [("intent_tidak_dikenal", 0.95)]


def test_perintah_hasil_jawab_adalah_salinan() -> None:
    """Menguji bahwa modifikasi pada perintah respons tidak merusak data knowledge base.

    Input    : None
    Output   : None
    Memanggil: GitBot(), bot.jawab()
    Dipanggil: pytest
    """
    matcher = DummyMatcher(kembalian_cari=[("setup_awal", 0.90)])
    bot = GitBot(kb=KB_UJI, matcher=matcher, threshold=0.50)
    hasil = bot.jawab("setup git")

    perintah = hasil["perintah"]
    perintah.append("rm -rf /")

    hasil_kedua = bot.jawab("setup git")
    assert "rm -rf /" not in hasil_kedua["perintah"]
    assert "rm -rf /" not in KB_UJI["intents"][0]["perintah"]


def test_konstruktor_menolak_kb_tanpa_fallback() -> None:
    """Menguji bahwa konstruktor GitBot menolak struktur knowledge base tanpa fallback valid.

    Input    : None
    Output   : None
    Memanggil: GitBot()
    Dipanggil: pytest
    """
    matcher = DummyMatcher()
    with pytest.raises(ValueError, match="field 'fallback'"):
        GitBot(kb={"intents": []}, matcher=matcher, threshold=0.5)

    with pytest.raises(ValueError, match="field 'fallback'"):
        GitBot(kb={"intents": [], "fallback": {}}, matcher=matcher, threshold=0.5)

    with pytest.raises(ValueError, match="field 'fallback'"):
        GitBot(
            kb={"intents": [], "fallback": {"jawaban": "   "}},
            matcher=matcher,
            threshold=0.5,
        )


def test_konstruktor_menolak_kb_bukan_dict_atau_tanpa_intents() -> None:
    """Menguji penolakan tipe kb bukan dict atau tidak memiliki list intents.

    Input    : None
    Output   : None
    Memanggil: GitBot()
    Dipanggil: pytest
    """
    matcher = DummyMatcher()
    with pytest.raises(ValueError, match="dictionary"):
        GitBot(kb="bukan_dict", matcher=matcher, threshold=0.5)

    with pytest.raises(ValueError, match="field 'intents'"):
        GitBot(
            kb={"fallback": {"jawaban": "ada"}},
            matcher=matcher,
            threshold=0.5,
        )


def test_konstruktor_menolak_threshold_salah() -> None:
    """Menguji penolakan threshold di luar rentang [0.0, 1.0], tipe bool, dan tipe string.

    Input    : None
    Output   : None
    Memanggil: GitBot()
    Dipanggil: pytest
    """
    matcher = DummyMatcher()
    with pytest.raises(ValueError, match="rentang 0.0 hingga 1.0"):
        GitBot(kb=KB_UJI, matcher=matcher, threshold=-0.01)

    with pytest.raises(ValueError, match="rentang 0.0 hingga 1.0"):
        GitBot(kb=KB_UJI, matcher=matcher, threshold=1.01)

    with pytest.raises(ValueError, match="float"):
        GitBot(kb=KB_UJI, matcher=matcher, threshold=True)

    with pytest.raises(ValueError, match="float"):
        GitBot(kb=KB_UJI, matcher=matcher, threshold="0.5")


def test_konstruktor_menolak_matcher_tanpa_cari() -> None:
    """Menguji penolakan matcher yang tidak memiliki metode callable cari.

    Input    : None
    Output   : None
    Memanggil: GitBot()
    Dipanggil: pytest
    """
    class TanpaCari:
        pass

    class CariBukanMetode:
        cari = "bukan_callable"

    with pytest.raises(TypeError, match="metode callable 'cari'"):
        GitBot(kb=KB_UJI, matcher=TanpaCari(), threshold=0.5)

    with pytest.raises(TypeError, match="metode callable 'cari'"):
        GitBot(kb=KB_UJI, matcher=CariBukanMetode(), threshold=0.5)


def test_integrasi_ringan_tfidf_matcher_dan_data_kecil() -> None:
    """Menguji integrasi end-to-end GitBot dengan TfidfMatcher asli pada dataset kecil.

    Input    : None
    Output   : None
    Memanggil: TfidfMatcher(), matcher.fit(), GitBot(), bot.jawab()
    Dipanggil: pytest
    """
    matcher = TfidfMatcher()
    matcher.fit(daftar_contoh(KB_UJI))
    bot = GitBot(kb=KB_UJI, matcher=matcher, threshold=0.1)

    hasil = bot.jawab("bagaimana cara mengatur konfigurasi awal git?")
    assert hasil["status"] == "ok"
    assert hasil["intent"] == "setup_awal"
    assert hasil["skor"] >= 0.1
    assert len(hasil["kandidat"]) > 0
    assert len(hasil["perintah"]) == 2


def test_main_tanya_dengan_monkeypatch(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Menguji eksekusi CLI main dengan argumen --tanya dan matcher tiruan.

    Input    : monkeypatch, capsys
    Output   : None
    Memanggil: main()
    Dipanggil: pytest
    """
    dummy = DummyMatcher(kembalian_cari=[("setup_awal", 0.95)])
    monkeypatch.setattr("src.chatbot._buat_matcher", lambda m: dummy)
    monkeypatch.setattr("src.chatbot.muat_kb", lambda p: KB_UJI)

    kode = main(["--tanya", "cara setup git", "--metode", "tfidf", "--threshold", "0.5"])
    assert kode == 0
    assert dummy.call_count == 1
    assert dummy.fit_call_count == 1

    out = capsys.readouterr().out
    assert "Status  : ok" in out
    assert "Intent  : setup_awal" in out
    assert "Jawaban :" in out
    assert "Kandidat:" in out


def test_main_mengembalikan_1_bila_buat_matcher_gagal(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Menguji bahwa main mengembalikan kode 1 dan mencetak pesan jika inisialisasi matcher gagal.

    Input    : monkeypatch, capsys
    Output   : None
    Memanggil: main()
    Dipanggil: pytest
    """
    def _gagal(metode: str) -> Any:
        raise RuntimeError(f"Gagal memuat model '{metode}': koneksi terputus")

    monkeypatch.setattr("src.chatbot._buat_matcher", _gagal)
    monkeypatch.setattr("src.chatbot.muat_kb", lambda p: KB_UJI)

    kode = main(["--tanya", "halo"])
    assert kode == 1

    err = capsys.readouterr().err
    assert "Kesalahan inisialisasi GitBot" in err
    assert "koneksi terputus" in err


def test_main_mengembalikan_1_bila_muat_kb_gagal(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Menguji bahwa main mengembalikan kode 1 jika berkas knowledge base gagal dimuat.

    Input    : monkeypatch, capsys
    Output   : None
    Memanggil: main()
    Dipanggil: pytest
    """
    def _gagal_kb(p: Any) -> Any:
        raise KBError("Berkas knowledge base tidak ditemukan")

    monkeypatch.setattr("src.chatbot.muat_kb", _gagal_kb)
    kode = main(["--tanya", "halo"])
    assert kode == 1

    err = capsys.readouterr().err
    assert "Kesalahan inisialisasi GitBot" in err
    assert "tidak ditemukan" in err


def test_main_loop_keluar_interaktif(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Menguji loop interaktif main yang langsung berhenti saat pengguna mengetik 'keluar'.

    Input    : monkeypatch, capsys
    Output   : None
    Memanggil: main()
    Dipanggil: pytest
    """
    dummy = DummyMatcher()
    monkeypatch.setattr("src.chatbot._buat_matcher", lambda m: dummy)
    monkeypatch.setattr("src.chatbot.muat_kb", lambda p: KB_UJI)
    monkeypatch.setattr("builtins.input", lambda prompt="": "keluar")

    kode = main([])
    assert kode == 0
    out = capsys.readouterr().out
    assert "Sampai jumpa!" in out


def test_main_loop_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Menguji penanganan KeyboardInterrupt pada loop interaktif main agar keluar rapi.

    Input    : monkeypatch, capsys
    Output   : None
    Memanggil: main()
    Dipanggil: pytest
    """
    dummy = DummyMatcher()
    monkeypatch.setattr("src.chatbot._buat_matcher", lambda m: dummy)
    monkeypatch.setattr("src.chatbot.muat_kb", lambda p: KB_UJI)

    def _interupsi(prompt: str = "") -> str:
        raise KeyboardInterrupt()

    monkeypatch.setattr("builtins.input", _interupsi)
    kode = main([])
    assert kode == 0
    out = capsys.readouterr().out
    assert "Sampai jumpa!" in out


def test_buat_matcher_pemilihan_metode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Menguji fungsi privat _buat_matcher untuk instansiasi matcher yang sesuai.

    Input    : monkeypatch
    Output   : None
    Memanggil: _buat_matcher()
    Dipanggil: pytest
    """
    # Menguji TF-IDF tanpa monkeypatch karena tidak mengunduh model berat
    m_tfidf = _buat_matcher("tfidf")
    assert isinstance(m_tfidf, TfidfMatcher)

    # Menguji embedding dan indobert dengan monkeypatch agar tidak mengunduh model
    monkeypatch.setattr("src.chatbot.EmbeddingMatcher", lambda m: DummyMatcher())
    monkeypatch.setattr("src.chatbot.IndoBertMatcher", lambda: DummyMatcher())

    assert isinstance(_buat_matcher("embedding"), DummyMatcher)
    assert isinstance(_buat_matcher("indobert"), DummyMatcher)

    with pytest.raises(ValueError, match="tidak dikenal"):
        _buat_matcher("metode_gaib")


def test_main_threshold_tidak_valid(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Menguji bahwa main mengembalikan 1 jika threshold CLI di luar rentang valid.

    Input    : monkeypatch, capsys
    Output   : None
    Memanggil: main()
    Dipanggil: pytest
    """
    dummy = DummyMatcher()
    monkeypatch.setattr("src.chatbot._buat_matcher", lambda m: dummy)
    monkeypatch.setattr("src.chatbot.muat_kb", lambda p: KB_UJI)

    kode = main(["--tanya", "halo", "--threshold", "5"])
    assert kode == 1

    err = capsys.readouterr().err
    assert "Kesalahan inisialisasi GitBot" in err
    assert "rentang 0.0 hingga 1.0" in err


def test_main_mengembalikan_1_bila_matcher_fit_melempar_value_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Menguji bahwa main mengembalikan 1 jika matcher.fit melempar ValueError.

    Input    : monkeypatch, capsys
    Output   : None
    Memanggil: main()
    Dipanggil: pytest
    """
    class FitGalatMatcher(DummyMatcher):
        def fit(self, contoh: list) -> None:
            raise ValueError("Daftar contoh untuk fit tidak boleh kosong")

    monkeypatch.setattr("src.chatbot._buat_matcher", lambda m: FitGalatMatcher())
    monkeypatch.setattr("src.chatbot.muat_kb", lambda p: KB_UJI)

    kode = main(["--tanya", "halo"])
    assert kode == 1

    err = capsys.readouterr().err
    assert "Kesalahan inisialisasi GitBot" in err
    assert "tidak boleh kosong" in err


def test_jawab_kandidat_kosong_menghasilkan_fallback() -> None:
    """Menguji respons fallback ketika matcher.cari mengembalikan list kosong.

    Input    : None
    Output   : None
    Memanggil: GitBot(), bot.jawab()
    Dipanggil: pytest
    """
    matcher = DummyMatcher(kembalian_cari=[])
    bot = GitBot(kb=KB_UJI, matcher=matcher, threshold=0.50)
    hasil = bot.jawab("pertanyaan tanpa kandidat")

    assert hasil["status"] == "fallback"
    assert hasil["intent"] is None
    assert hasil["skor"] == 0.0
    assert hasil["kandidat"] == []
    assert hasil["jawaban"] == KB_UJI["fallback"]["jawaban"]
    assert hasil["perintah"] == []
    assert hasil["pesan_error"] == ""

