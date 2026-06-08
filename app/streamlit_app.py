from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"

modelo = joblib.load(MODEL_DIR / "modelo_final.pkl")
label_encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")
features = joblib.load(MODEL_DIR / "colunas_modelo.pkl")


def criar_dataframe_entrada(
    latitude,
    longitude,
    bright_ti4,
    bright_ti5,
    frp,
    scan,
    track,
    confidence_num,
    daynight_num,
    data,
):
    ano = data.year
    mes = data.month
    dia = data.day
    dia_do_ano = data.timetuple().tm_yday

    risco_fogo = bright_ti4 * confidence_num
    diferenca_brilho = bright_ti4 - bright_ti5
    frp_por_brilho = frp / (bright_ti4 + 1)
    periodo_seco = 1 if mes in [7, 8, 9, 10] else 0

    entrada = pd.DataFrame([{
        "latitude": latitude,
        "longitude": longitude,
        "bright_ti4": bright_ti4,
        "bright_ti5": bright_ti5,
        "frp": frp,
        "scan": scan,
        "track": track,
        "confidence_num": confidence_num,
        "daynight_num": daynight_num,
        "ano": ano,
        "mes": mes,
        "dia": dia,
        "dia_do_ano": dia_do_ano,
        "risco_fogo": risco_fogo,
        "diferenca_brilho": diferenca_brilho,
        "frp_por_brilho": frp_por_brilho,
        "periodo_seco": periodo_seco,
    }])

    return entrada[features]


def completar_colunas_derivadas(df):
    df = df.copy()

    if "risco_fogo" not in df.columns and {"bright_ti4", "confidence_num"}.issubset(df.columns):
        df["risco_fogo"] = df["bright_ti4"] * df["confidence_num"]

    if "diferenca_brilho" not in df.columns and {"bright_ti4", "bright_ti5"}.issubset(df.columns):
        df["diferenca_brilho"] = df["bright_ti4"] - df["bright_ti5"]

    if "frp_por_brilho" not in df.columns and {"frp", "bright_ti4"}.issubset(df.columns):
        df["frp_por_brilho"] = df["frp"] / (df["bright_ti4"] + 1)

    if "periodo_seco" not in df.columns and "mes" in df.columns:
        df["periodo_seco"] = df["mes"].isin([7, 8, 9, 10]).astype(int)

    return df


st.set_page_config(
    page_title="Detector de Queimadas",
    page_icon="🔥",
    layout="wide",
)

st.title("🔥 Detector de Queimadas com IA")
st.write(
    "Aplicação de Machine Learning para classificar uma região como "
    "**normal** ou com **indício de queimada**, usando dados de satélite da NASA FIRMS."
)

st.info(
    "Modelo carregado: Random Forest. "
    "O modelo foi treinado com dados da NASA FIRMS e amostras normais geradas."
)

aba_manual, aba_csv, aba_sobre = st.tabs([
    "Previsão manual",
    "Previsão por CSV",
    "Sobre o projeto",
])

with aba_manual:
    st.subheader("Entrada manual dos dados")

    col1, col2, col3 = st.columns(3)

    with col1:
        latitude = st.number_input("Latitude", value=-22.79, format="%.6f")
        longitude = st.number_input("Longitude", value=-49.33, format="%.6f")
        data = st.date_input("Data da observação")

    with col2:
        bright_ti4 = st.number_input("Bright TI4", value=330.0)
        bright_ti5 = st.number_input("Bright TI5", value=295.0)
        frp = st.number_input("FRP", value=5.0)

    with col3:
        scan = st.number_input("Scan", value=0.5)
        track = st.number_input("Track", value=0.6)
        confidence_num = st.slider("Confiança numérica", 0, 100, 60)
        daynight_num = st.selectbox(
            "Período",
            options=[1, 0],
            format_func=lambda x: "Dia" if x == 1 else "Noite",
        )

    entrada = criar_dataframe_entrada(
        latitude=latitude,
        longitude=longitude,
        bright_ti4=bright_ti4,
        bright_ti5=bright_ti5,
        frp=frp,
        scan=scan,
        track=track,
        confidence_num=confidence_num,
        daynight_num=daynight_num,
        data=data,
    )

    st.subheader("Dados enviados ao modelo")
    st.dataframe(entrada, use_container_width=True)

    if st.button("Classificar região"):
        predicao = modelo.predict(entrada)
        probabilidades = modelo.predict_proba(entrada)

        classe_prevista = label_encoder.inverse_transform(predicao)[0]

        if classe_prevista == "queimada":
            st.error(f"Classe prevista: **{classe_prevista.upper()}**")
        else:
            st.success(f"Classe prevista: **{classe_prevista.upper()}**")

        df_prob = pd.DataFrame({
            "Classe": label_encoder.classes_,
            "Probabilidade": probabilidades[0],
        })

        df_prob["Probabilidade (%)"] = df_prob["Probabilidade"] * 100
        df_prob = df_prob[["Classe", "Probabilidade (%)"]]

        st.subheader("Probabilidades por classe")
        st.dataframe(df_prob, use_container_width=True)
        st.bar_chart(df_prob.set_index("Classe")["Probabilidade (%)"])

with aba_csv:
    st.subheader("Previsão em lote por arquivo CSV")

    st.write(
        "Envie um CSV contendo as colunas usadas pelo modelo. "
        "As colunas derivadas podem ser enviadas prontas ou serão calculadas quando possível."
    )

    st.code(", ".join(features))

    arquivo = st.file_uploader("Enviar CSV", type=["csv"])

    if arquivo is not None:
        df_upload = pd.read_csv(arquivo)
        df_upload = completar_colunas_derivadas(df_upload)

        st.write("Prévia do arquivo enviado:")
        st.dataframe(df_upload.head(), use_container_width=True)

        colunas_faltando = [col for col in features if col not in df_upload.columns]

        if colunas_faltando:
            st.error("O CSV não possui todas as colunas necessárias.")
            st.write("Colunas faltando:")
            st.write(colunas_faltando)
        else:
            X_upload = df_upload[features].copy()

            for coluna in features:
                X_upload[coluna] = pd.to_numeric(X_upload[coluna], errors="coerce")

            X_upload = X_upload.fillna(0)

            predicoes = modelo.predict(X_upload)
            probabilidades = modelo.predict_proba(X_upload)
            classes_previstas = label_encoder.inverse_transform(predicoes)

            resultado = df_upload.copy()
            resultado["classe_prevista"] = classes_previstas

            for i, classe in enumerate(label_encoder.classes_):
                resultado[f"prob_{classe}"] = probabilidades[:, i]

            st.success("Classificação concluída.")
            st.dataframe(resultado, use_container_width=True)

            csv_resultado = resultado.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Baixar resultado em CSV",
                data=csv_resultado,
                file_name="resultado_classificacao_queimadas.csv",
                mime="text/csv",
            )

with aba_sobre:
    st.subheader("Sobre o projeto")

    st.write(
        """
        Este projeto desenvolve um pipeline completo de Inteligência Artificial aplicado à Economia Espacial,
        com o objetivo de classificar amostras geográficas em duas classes:

        - Normal
        - Queimada

        A principal fonte utilizada foi a NASA FIRMS, que fornece dados de focos de calor e anomalias térmicas
        detectadas por satélites. Também foram usadas amostras normais geradas dentro de uma área geográfica definida.

        Foram treinados três modelos:

        - Logistic Regression
        - Random Forest
        - XGBoost

        O melhor modelo foi selecionado com base no F1-score da classe queimada.
        """
    )

    st.warning(
        "Limitação: o projeto possui finalidade acadêmica e demonstrativa. "
        "Os resultados não substituem sistemas oficiais de monitoramento ambiental ou defesa civil."
    )
