import os
from io import StringIO

import pandas as pd
import requests
from dotenv import load_dotenv


load_dotenv()

FIRMS_MAP_KEY = os.getenv("FIRMS_MAP_KEY")


def testar_firms():
    if not FIRMS_MAP_KEY:
        raise ValueError(
            "FIRMS_MAP_KEY não encontrada. Verifique se o arquivo .env existe "
            "e se a chave foi preenchida corretamente."
        )

    # Bounding box aproximado do Brasil:
    # formato exigido pela FIRMS: west,south,east,north
    area_brasil = "-74,-34,-34,6"

    # Sensor recomendado para começar
    sensor = "VIIRS_SNPP_NRT"

    # A documentação da FIRMS indica DAY_RANGE de 1 a 5 dias.
    dias = 5

    url = (
        f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
        f"{FIRMS_MAP_KEY}/{sensor}/{area_brasil}/{dias}"
    )

    print("Consultando API NASA FIRMS...")
    print("Sensor:", sensor)
    print("Área:", area_brasil)
    print("Dias:", dias)

    response = requests.get(url, timeout=60)

    if response.status_code != 200:
        print("Erro na requisição.")
        print("Status code:", response.status_code)
        print("Resposta:")
        print(response.text[:1000])
        response.raise_for_status()

    df = pd.read_csv(StringIO(response.text))

    print("\nPrimeiras linhas:")
    print(df.head())

    print("\nQuantidade de linhas e colunas:")
    print(df.shape)

    if df.empty:
        print("\nA API respondeu, mas não retornou dados.")
        print("Isso pode acontecer dependendo do sensor, área ou período.")
        return

    df["classe"] = "queimada"

    caminho_saida = "data/raw/queimadas_firms.csv"
    df.to_csv(caminho_saida, index=False)

    print(f"\nArquivo salvo em: {caminho_saida}")
    print("\nColunas retornadas:")
    print(list(df.columns))


if __name__ == "__main__":
    testar_firms()