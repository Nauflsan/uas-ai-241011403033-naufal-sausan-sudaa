import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, .stApp, [data-testid="stSidebar"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* bersihin chrome bawaan Streamlit */
[data-testid="stToolbarActions"],
[data-testid="stAppDeployButton"],
[data-testid="stMainMenu"],
[data-testid="stDecoration"],
#MainMenu,
footer {
    display: none !important;
}
header[data-testid="stHeader"] {
    background: transparent;
}

/* pastiin tombol buka/tutup sidebar selalu keliatan */
[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] {
    display: flex !important;
    visibility: visible !important;
}

/* area konten lebih sempit biar enak dibaca */
.block-container {
    max-width: 760px;
    padding-top: 2.5rem;
    padding-bottom: 7rem;
}

h1 {
    font-weight: 700;
    letter-spacing: -0.02em;
}

/* sidebar */
[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid #E1E5EF;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-size: 1.05rem;
    font-weight: 600;
}

/* bubble chat: bot di kiri */
[data-testid="stChatMessage"] {
    background: #FFFFFF;
    border: 1px solid #E1E5EF;
    border-radius: 4px 18px 18px 18px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.9rem;
    max-width: 88%;
    box-shadow: none;
}

/* bubble chat: user di kanan, warna gelap */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    flex-direction: row-reverse;
    margin-left: auto;
    background: #1B2133;
    border-color: #1B2133;
    border-radius: 18px 4px 18px 18px;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) * {
    color: #FFFFFF;
}

/* info intent / skor di bawah jawaban */
[data-testid="stCaptionContainer"] {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #6B738A;
}

/* code block */
pre, code {
    font-family: 'JetBrains Mono', monospace !important;
}
pre {
    border-radius: 12px !important;
}

/* input chat bentuk pil */
[data-testid="stChatInput"] > div {
    border-radius: 999px;
    background: #FFFFFF;
    border: 1px solid #D5DAE8;
    box-shadow: none;
}
[data-testid="stChatInput"] > div:focus-within {
    border-color: #F05133;
}

/* tombol */
.stButton > button {
    border-radius: 10px;
    border: 1px solid #D5DAE8;
    background: #FFFFFF;
    font-weight: 500;
}
.stButton > button:hover {
    border-color: #F05133;
    color: #F05133;
}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)
