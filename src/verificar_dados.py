from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]


def verificar_csv(caminho, nome):
    if not caminho.exists():
        print(f"Arquivo não encontrado: {caminho}")
        return

    df = pd.read_csv(caminho)

    print("=" * 60)
    print(nome)
    print("=" * 60)
    print("Linhas e colunas:", df.shape)
    print("Colunas:")
    print(list(df.columns))

    if "classe" in df.columns:
        print("\nDistribuição das classes:")
        print(df["classe"].value_counts())

    print("\nPrimeiras linhas:")
    print(df.head())
    print()


def main():
    verificar_csv(BASE_DIR / "data" / "processed" / "dataset_queimadas.csv", "DATASET FINAL - QUEIMADAS")


if __name__ == "__main__":
    main()
