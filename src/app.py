# Pokretanje:
# streamlit run src/app.py

from pathlib import Path

import pandas as pd
import streamlit as st

import config
import predict as pr



# Streamlit podešavanja
st.set_page_config(
    page_title="Procena momenta asinhronog motora",
    page_icon="⚙️",
    layout="wide"
)

# Stil aplikacije
st.markdown("""
<style>

    /* Glavna pozadina */
    .stApp {
        background-color: #F5EFE6;
    }

    /* Glavni sadržaj */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Naslov */
    h1 {
        color: #5A4033;
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }

    /* Podnaslovi */
    h2, h3 {
        color: #765747;
    }

    /* Tekst */
    p, label, .stMarkdown {
        color: #4E4038;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #E8D8C8;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #5A4033;
    }

    /* Dugmad */
    .stButton > button {
        background-color: #8B6F5A;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
    }

    .stButton > button:hover {
        background-color: #6F5545;
        color: white;
    }

    /* Input polja */
    .stNumberInput input {
        background-color: #FFFDF9;
        border: 1px solid #C8B29F;
        border-radius: 8px;
    }

    /* Selectbox */
    div[data-baseweb="select"] > div {
        background-color: #FFFDF9;
        border-radius: 8px;
        border: 1px solid #C8B29F;
    }

    /* Metric */
    div[data-testid="stMetric"] {
        background-color: #FFFDF9;
        padding: 1rem;
        border-radius: 14px;
        border: 1px solid #D8C5B4;
        box-shadow: 0 3px 10px rgba(90, 64, 51, 0.08);
    }

    /* Info / warning box */
    .stAlert {
        border-radius: 10px;
    }

</style>
""", unsafe_allow_html=True)



# Naslov i opis aplikacije
st.title("Procena momenta asinhronog motora")

st.write(
    "Aplikacija za procenu obrtnog momenta asinhrone mašine "
    "na osnovu ulaznih parametara."
)


# Pronalaženje dostupnih modela
available = [
    e
    for e in (
        "no_power",
        "with_power",
        "merged",
        "merged_power_over_omega"
    )
    if pr.model_path(e).exists()
]

if not available:
    st.error(
        "Nijedan model nije pronađen. "
        "Pokreni: python src/main.py"
    )
    st.stop()


# Izbor modela
default_idx = (
    available.index(config.EXPERIMENT)
    if config.EXPERIMENT in available
    else 0
)

exp = st.sidebar.selectbox(
    "Set atributa / model",
    available,
    index=default_idx
)

# Učitavanje modela
@st.cache_resource
def get_model(e):
    return pr.load_model(e), pr.load_meta(e)


model, meta = get_model(exp)

raw_cols = pr.raw_columns_for(model)


# Informacije o modelu u sidebar-u
if meta:
    st.sidebar.write(
        f"Algoritam: **{meta['model']}**"
        + (" (tjuniran)" if meta.get("tuned") else "")
    )

st.sidebar.write(
    f"Potrebne ulazne veličine: {len(raw_cols)}"
)

# Funkcija za prikaz rezultata
def show_results(df, preds):

    for warning in pr.collect_warnings(
        df,
        model,
        meta
    ):
        st.warning(warning)

    return df.assign(
        torque_pred_Nm=preds
    )

# Tabovi aplikacije
tab_manual, tab_csv = st.tabs(
    ["Ručni unos", "CSV upload"]
)

# RUČNI UNOS

with tab_manual:

    st.subheader("Ulazni parametri")

    sample = pr.default_sample().iloc[0]

    values = {}

    cols = st.columns(2)

    for i, c in enumerate(raw_cols):

        values[c] = cols[i % 2].number_input(
            c,
            value=float(sample[c]),
            format="%.4f"
        )

    if st.button("Izračunaj moment"):

        try:

            df = pd.DataFrame([values])

            preds = pr.predict_torque(
                model,
                df
            )

            show_results(
                df,
                preds
            )

            st.subheader("Rezultat")

            st.metric(
                "Procenjeni moment",
                f"{preds[0]:.3f} Nm"
            )

        except (
            pr.InputValidationError,
            ValueError
        ) as exc:

            st.error(str(exc))


# CSV UPLOAD
with tab_csv:

    st.subheader("Predikcija iz CSV fajla")

    st.caption(
        "CSV mora da sadrži kolone: "
        + ", ".join(raw_cols)
    )

    up = st.file_uploader(
        "CSV fajl",
        type="csv"
    )

    if up is not None:

        try:

            df = pd.read_csv(up)

            preds = pr.predict_torque(
                model,
                df
            )

            out = show_results(
                df,
                preds
            )

            st.dataframe(
                out,
                use_container_width=True
            )

            st.download_button(
                "Preuzmi predikcije (CSV)",
                out.to_csv(index=False).encode("utf-8"),
                "predikcije.csv",
                "text/csv"
            )

        except (
            pr.InputValidationError,
            ValueError
        ) as exc:

            st.error(str(exc))