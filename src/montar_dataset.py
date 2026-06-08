from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def converter_confidence(valor):
    """
    Converte a coluna confidence da NASA FIRMS para valor numérico.

    Alguns sensores retornam confidence como:
    l = low
    n = nominal
    h = high

    Outros retornam valores numéricos.
    """
    if pd.isna(valor):
        return 0

    valor_str = str(valor).strip().lower()

    mapa = {
        "l": 30,
        "low": 30,
        "n": 60,
        "nominal": 60,
        "h": 90,
        "high": 90,
    }

    if valor_str in mapa:
        return mapa[valor_str]

    try:
        return float(valor)
    except ValueError:
        return 0


def preparar_queimadas(caminho):
    df = pd.read_csv(caminho)

    df = df.rename(columns={"acq_date": "data"})
    df["classe"] = "queimada"
    df["fonte"] = "NASA_FIRMS"

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    df["bright_ti4"] = pd.to_numeric(df.get("bright_ti4", 0), errors="coerce").fillna(0)
    df["bright_ti5"] = pd.to_numeric(df.get("bright_ti5", 0), errors="coerce").fillna(0)
    df["frp"] = pd.to_numeric(df.get("frp", 0), errors="coerce").fillna(0)
    df["scan"] = pd.to_numeric(df.get("scan", 0), errors="coerce").fillna(0)
    df["track"] = pd.to_numeric(df.get("track", 0), errors="coerce").fillna(0)
    df["confidence_num"] = df.get("confidence", 0).apply(converter_confidence)

    df["daynight_num"] = df.get("daynight", "D").map({"D": 1, "N": 0}).fillna(0)

    colunas = [
        "latitude",
        "longitude",
        "data",
        "bright_ti4",
        "bright_ti5",
        "frp",
        "scan",
        "track",
        "confidence_num",
        "daynight_num",
        "fonte",
        "classe",
    ]

    return df[colunas]


def preparar_normais(caminho):
    df = pd.read_csv(caminho)

    df["classe"] = "normal"
    df["fonte"] = "GERADO"

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    # Amostras normais não possuem sinais térmicos de foco de calor.
    df["bright_ti4"] = 0
    df["bright_ti5"] = 0
    df["frp"] = 0
    df["scan"] = 0
    df["track"] = 0
    df["confidence_num"] = 0
    df["daynight_num"] = 0

    colunas = [
        "latitude",
        "longitude",
        "data",
        "bright_ti4",
        "bright_ti5",
        "frp",
        "scan",
        "track",
        "confidence_num",
        "daynight_num",
        "fonte",
        "classe",
    ]

    return df[colunas]


def criar_features_temporais(df):
    df["data_original"] = df["data"]
    df["data"] = pd.to_datetime(df["data"], errors="coerce", utc=True, format="mixed")

    df["ano"] = df["data"].dt.year
    df["mes"] = df["data"].dt.month
    df["dia"] = df["data"].dt.day
    df["dia_do_ano"] = df["data"].dt.dayofyear

    df["data"] = df["data"].dt.strftime("%Y-%m-%d")

    return df


def criar_features_derivadas(df):
    df["risco_fogo"] = df["bright_ti4"] * df["confidence_num"]
    df["diferenca_brilho"] = df["bright_ti4"] - df["bright_ti5"]
    df["frp_por_brilho"] = df["frp"] / (df["bright_ti4"] + 1)
    df["periodo_seco"] = df["mes"].isin([7, 8, 9, 10]).astype(int)

    return df


def montar_dataset_a_partir_de_raw():
    caminho_queimadas = RAW_DIR / "queimadas_firms.csv"
    caminho_normais = RAW_DIR / "amostras_normais.csv"

    if not caminho_queimadas.exists() or not caminho_normais.exists():
        raise FileNotFoundError(
            "Arquivos brutos não encontrados. Execute primeiro testar_firms.py e gerar_normais.py, "
            "ou utilize o dataset já processado em data/processed/dataset_queimadas.csv."
        )

    queimadas = preparar_queimadas(caminho_queimadas)
    normais = preparar_normais(caminho_normais)

    print("Quantidade original:")
    print("Queimadas:", queimadas.shape)
    print("Normais:", normais.shape)

    n_amostras = min(len(queimadas), len(normais), 500)

    queimadas = queimadas.sample(n=n_amostras, random_state=42)
    normais = normais.sample(n=n_amostras, random_state=42)

    df = pd.concat([queimadas, normais], ignore_index=True)
    df = criar_features_temporais(df)
    df = criar_features_derivadas(df)

    df = df.dropna(subset=["latitude", "longitude", "data", "ano", "mes", "dia", "dia_do_ano"])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = montar_dataset_a_partir_de_raw()

    caminho_saida = PROCESSED_DIR / "dataset_queimadas.csv"
    df.to_csv(caminho_saida, index=False)

    print("\nDataset de queimadas criado com sucesso.")
    print("Arquivo:", caminho_saida)
    print("Dimensão final:", df.shape)
    print("Distribuição das classes:")
    print(df["classe"].value_counts())
    print("\nColunas finais:")
    print(list(df.columns))


if __name__ == "__main__":
    main()
