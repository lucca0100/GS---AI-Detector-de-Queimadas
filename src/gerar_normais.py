import random
from datetime import datetime, timedelta

import pandas as pd


def gerar_amostras_normais(n=800):
    """
    Gera amostras normais aleatórias dentro de uma bounding box aproximada do Brasil.

    Observação:
    Para o projeto acadêmico, essas amostras representam locais/datas sem registro direto
    nas APIs de queimadas ou enchentes utilizadas.
    """

    linhas = []

    # Bounding box aproximado do Brasil
    latitude_min = -34
    latitude_max = 6
    longitude_min = -74
    longitude_max = -34

    data_inicio = datetime(2024, 1, 1)
    data_fim = datetime(2026, 12, 31)
    intervalo_dias = (data_fim - data_inicio).days

    for _ in range(n):
        latitude = random.uniform(latitude_min, latitude_max)
        longitude = random.uniform(longitude_min, longitude_max)

        data_aleatoria = data_inicio + timedelta(
            days=random.randint(0, intervalo_dias)
        )

        linhas.append({
            "latitude": latitude,
            "longitude": longitude,
            "data": data_aleatoria.strftime("%Y-%m-%d"),
            "classe": "normal"
        })

    df = pd.DataFrame(linhas)

    caminho_saida = "data/raw/amostras_normais.csv"
    df.to_csv(caminho_saida, index=False)

    print("Amostras normais geradas com sucesso.")
    print("\nPrimeiras linhas:")
    print(df.head())

    print("\nQuantidade de linhas e colunas:")
    print(df.shape)

    print(f"\nArquivo salvo em: {caminho_saida}")


if __name__ == "__main__":
    gerar_amostras_normais(n=800)