"""Antarmuka pengguna web berbasis Streamlit untuk asisten GitBot.

Modul ini menyediakan antarmuka interaktif bagi pengguna untuk berkonsultasi seputar
perintah dan alur kerja Git & GitHub menggunakan metode pencocokan kemiripan teks lokal
(TF-IDF, MiniLM sentence embedding, dan IndoBERT pembanding).
"""

from datetime import datetime
import string
from typing import Any

import streamlit as st

from src.chatbot import GitBot, _buat_matcher
from src.config import PATH_KB, THRESHOLD
from src.knowledge_base import KBError, daftar_contoh, muat_kb
from src.style import inject_css

# Pemetaan label ramah pengguna pada selectbox ke kode metode pencocok internal
PETA_METODE: dict[str, str] = {
    "TF-IDF": "tfidf",
    "MiniLM (embedding)": "embedding",
    "IndoBERT": "indobert",
}


@st.cache_resource
def _muat_sumber(metode: str) -> tuple[dict, Any]:
    """Memuat data basis pengetahuan dan melatih pencocok teks secara ter-cache.

    Input    : metode (str) kode pengenal metode ('tfidf', 'embedding', 'indobert')
    Output   : tuple[dict, Any] pasangan basis pengetahuan dan matcher terlatih
    Memanggil: muat_kb(), _buat_matcher(), daftar_contoh(), matcher.fit()
    Dipanggil: _siapkan_bot()
    """
    # Memuat berkas basis pengetahuan dari direktori data
    kb = muat_kb(PATH_KB)

    # Menginisialisasi objek pencocok sesuai algoritma yang dipilih
    matcher = _buat_matcher(metode)

    # Melatih pencocok menggunakan seluruh contoh pertanyaan pada basis pengetahuan
    # Representasi vektor kalimat atau TF-IDF dihitung sekali di sini dan di-cache
    matcher.fit(daftar_contoh(kb))

    return kb, matcher


def _siapkan_bot(metode: str, threshold: float) -> tuple[GitBot, dict[str, str]]:
    """Menyiapkan instansi GitBot baru dan memetakan nama intent dengan proteksi galat.

    Input    : metode (str) kode metode, threshold (float) ambang batas kemiripan
    Output   : tuple[GitBot, dict[str, str]] pasangan objek GitBot dan peta id->nama intent
    Memanggil: _muat_sumber(), GitBot(), st.error(), st.stop()
    Dipanggil: main()
    """
    try:
        kb, matcher = _muat_sumber(metode)
        peta_nama = {
            item["id"]: item.get("nama", item["id"])
            for item in kb.get("intents", [])
            if isinstance(item, dict) and "id" in item
        }
        # GitBot tidak di-cache karena threshold dapat diubah dinamis lewat slider
        return GitBot(kb, matcher, threshold), peta_nama
    except (KBError, RuntimeError, ValueError, TypeError) as galat:
        pesan = f"Gagal memuat basis pengetahuan atau model: {galat}"
        if metode == "indobert":
            pesan += (
                "\n\nCatatan untuk IndoBERT: Model perlu diunduh sekali dari Hugging Face "
                "dan membutuhkan koneksi internet pada pemakaian pertama."
            )
        st.error(pesan)
        st.stop()


def _sapaan(jam: int) -> str:
    """Menghasilkan kalimat sapaan santai sesuai rentang waktu lokal.

    Input    : jam (int) nilai jam lokal dalam rentang 0 sampai 23
    Output   : str kalimat sapaan ramah satu kalimat
    Memanggil: tidak memanggil fungsi eksternal (fungsi murni)
    Dipanggil: _tampilkan_beranda()
    """
    if 4 <= jam <= 10:
        return "Selamat pagi! Sambil ngopi, mau tanya apa soal Git hari ini?"
    if 11 <= jam <= 14:
        return "Selamat siang! Udah makan? Yuk, tanya soal Git."
    if 15 <= jam <= 17:
        return "Selamat sore! Ada error Git yang bikin pusing?"
    return "Selamat malam! Lagi lembur ngoding? Tanya aja."


def _escape_markdown(teks: str) -> str:
    """Menambahkan backslash pada karakter tanda baca ASCII untuk tampilan Markdown.

    Input    : teks (str) teks masukan yang akan ditampilkan pada Markdown
    Output   : str teks dengan setiap karakter tanda baca ASCII diberi awalan backslash
    Memanggil: tidak memanggil fungsi eksternal (fungsi murni)
    Dipanggil: _tampilkan_respons(), _tampilkan_riwayat(), _proses_pertanyaan()
    """
    # Memberi backslash pada tanda baca ASCII agar format Markdown tidak aktif
    return "".join(
        f"\\{karakter}" if karakter in string.punctuation else karakter
        for karakter in teks
    )


def _tangani_reset() -> None:
    """Mereset riwayat percakapan bila tombol reset pada sidebar ditekan.

    Input    : tidak ada
    Output   : None
    Memanggil: st.sidebar.button(), st.rerun()
    Dipanggil: _atur_sidebar()
    """
    if st.sidebar.button("Reset percakapan", use_container_width=True):
        st.session_state["riwayat"] = []
        st.rerun()


def _atur_sidebar() -> tuple[str, float]:
    """Mengatur konfigurasi metode pencocokan dan slider threshold pada sidebar.

    Input    : tidak ada
    Output   : tuple[str, float] pasangan kode metode terpilih dan nilai threshold
    Memanggil: st.sidebar.header(), st.sidebar.selectbox(), st.sidebar.slider(), st.sidebar.caption(), _tangani_reset()
    Dipanggil: main()
    """
    st.sidebar.header("Pengaturan GitBot")
    label_metode = st.sidebar.selectbox(
        "Pilih Metode Pencocokan", list(PETA_METODE.keys()), index=1
    )
    kode_metode = PETA_METODE[label_metode]

    # Key slider mengikuti metode agar nilai reset ke nilai awal saat metode diganti
    threshold = st.sidebar.slider(
        "Ambang Batas Kemiripan (Threshold)",
        min_value=0.00,
        max_value=1.00,
        value=float(THRESHOLD.get(kode_metode, 0.5)),
        step=0.01,
        key=f"threshold_{kode_metode}",
    )
    st.sidebar.caption(
        "Catatan: Skor adalah ukuran kemiripan kosine (kedekatan vektor), bukan probabilitas."
    )
    _tangani_reset()
    return kode_metode, threshold


def _tampilkan_beranda(riwayat_kosong: bool) -> None:
    """Menampilkan judul, deskripsi, petunjuk penggunaan, dan sapaan waktu lokal.

    Input    : riwayat_kosong (bool) penanda apakah riwayat percakapan masih kosong
    Output   : None
    Memanggil: st.title(), st.write(), st.markdown(), st.info(), datetime.now(), _sapaan()
    Dipanggil: main()
    """
    st.title("GitBot")
    st.write(
        "GitBot adalah asisten interaktif berbasis kecerdasan buatan lokal "
        "yang membantu Anda memahami perintah dan alur kerja Git & GitHub."
    )
    st.markdown(
        "- **Contoh pertanyaan**: Ajukan kendala Git seperti *'cara bikin branch baru'* atau *'batalin commit terakhir'*.\n"
        "- **Pilih metode**: Tentukan model pencocokan (TF-IDF, MiniLM, atau IndoBERT) pada bilah samping.\n"
        "- **Atur threshold**: Sesuaikan ambang batas kemiripan minimum untuk toleransi jawaban fallback."
    )

    # Sapaan ramah hanya ditampilkan sebelum chat pertama dilakukan
    if riwayat_kosong:
        st.info(_sapaan(datetime.now().hour))


def _tampilkan_kandidat(
    kandidat: list[tuple[str, float]], peta_nama: dict[str, str]
) -> None:
    """Menampilkan 3 kandidat intent teratas beserta skor kemiripannya.

    Input    : kandidat (list[tuple[str, float]]) pasangan id dan skor, peta_nama (dict)
    Output   : None
    Memanggil: st.caption()
    Dipanggil: _tampilkan_respons()
    """
    if not kandidat:
        return
    st.caption("3 Kandidat teratas:")
    for cid, cskor in kandidat[:3]:
        nama = peta_nama.get(cid, cid)
        st.caption(f"- **{nama}** (`{cid}`): {cskor:.4f}")


def _tampilkan_respons(
    hasil: dict, threshold: float, peta_nama: dict[str, str]
) -> None:
    """Menampilkan isi respons asisten GitBot sesuai status hasil evaluasi.

    Input    : hasil (dict) respons bot, threshold (float) ambang, peta_nama (dict)
    Output   : None
    Memanggil: st.markdown(), _escape_markdown(), st.code(), st.caption(), st.error(), _tampilkan_kandidat()
    Dipanggil: _tampilkan_riwayat(), _proses_pertanyaan()
    """
    status = hasil.get("status")
    if status == "ok":
        nama = peta_nama.get(hasil["intent"], str(hasil["intent"]))
        st.markdown(f"**Intent**: {nama} (skor: `{hasil['skor']:.4f}`)")
        st.markdown(_escape_markdown(hasil["jawaban"]))
        if hasil.get("perintah"):
            st.code("\n".join(hasil["perintah"]), language="bash")
        _tampilkan_kandidat(hasil.get("kandidat", []), peta_nama)
    elif status == "fallback":
        st.markdown(_escape_markdown(hasil["jawaban"]))
        st.caption(
            f"Skor tertinggi: `{hasil['skor']:.4f}` | Ambang batas: `{threshold:.2f}`"
        )
        _tampilkan_kandidat(hasil.get("kandidat", []), peta_nama)
    elif status == "error":
        st.error(hasil.get("pesan_error", "Terjadi kesalahan pada masukan."))


def _tampilkan_riwayat(peta_nama: dict[str, str]) -> None:
    """Menampilkan seluruh riwayat percakapan yang tersimpan di session state.

    Input    : peta_nama (dict[str, str]) kamus pemetaan id ke nama intent
    Output   : None
    Memanggil: st.chat_message(), st.markdown(), _escape_markdown(), _tampilkan_respons()
    Dipanggil: main()
    """
    for pesan in st.session_state["riwayat"]:
        with st.chat_message(pesan["role"]):
            if pesan["role"] == "user":
                st.markdown(_escape_markdown(pesan["konten"]))
            else:
                _tampilkan_respons(pesan["hasil"], pesan["threshold"], peta_nama)


def _proses_pertanyaan(
    prompt: str, bot: GitBot, threshold: float, peta_nama: dict[str, str]
) -> None:
    """Memproses pertanyaan pengguna, memperbarui riwayat, dan menampilkan respons.

    Input    : prompt (str) masukan pengguna, bot (GitBot), threshold (float), peta_nama (dict)
    Output   : None
    Memanggil: st.chat_message(), st.markdown(), _escape_markdown(), bot.jawab(), _tampilkan_respons(), list.append()
    Dipanggil: main()
    """
    st.session_state["riwayat"].append({"role": "user", "konten": prompt})
    with st.chat_message("user"):
        st.markdown(_escape_markdown(prompt))

    # Hasil evaluasi intent disimpan di riwayat agar tidak dihitung ulang saat rerun
    hasil = bot.jawab(prompt)
    st.session_state["riwayat"].append(
        {"role": "assistant", "hasil": hasil, "threshold": threshold}
    )
    with st.chat_message("assistant"):
        _tampilkan_respons(hasil, threshold, peta_nama)


def main() -> None:
    """Fungsi utama pengatur alur antarmuka aplikasi Streamlit GitBot.

    Input    : tidak ada
    Output   : None
    Memanggil: st.set_page_config(), _atur_sidebar(), _siapkan_bot(), _tampilkan_beranda(), _tampilkan_riwayat(), st.chat_input(), _proses_pertanyaan()
    Dipanggil: dieksekusi saat skrip app.py dijalankan oleh Streamlit
    """
    st.set_page_config(page_title="GitBot - Asisten Git & GitHub", page_icon="🤖")
    inject_css()

    if "riwayat" not in st.session_state:
        st.session_state["riwayat"] = []

    kode_metode, threshold = _atur_sidebar()
    bot, peta_nama = _siapkan_bot(kode_metode, threshold)

    riwayat_kosong = len(st.session_state["riwayat"]) == 0
    _tampilkan_beranda(riwayat_kosong)
    _tampilkan_riwayat(peta_nama)

    # st.chat_input tanpa max_chars agar validasi_input GitBot menangani batas panjang
    if prompt := st.chat_input("Tanyakan sesuatu tentang Git atau GitHub..."):
        _proses_pertanyaan(prompt, bot, threshold, peta_nama)


if __name__ == "__main__":
    main()
