from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "data" / "processed" / "dataset_queimadas.csv"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"


def normalizar_shap_values(shap_values, n_features):
    """Retorna os SHAP values da classe positiva, queimada."""
    if isinstance(shap_values, list):
        # Em classificação binária com árvores, algumas versões retornam lista por classe.
        return shap_values[1] if len(shap_values) > 1 else shap_values[0]

    if isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 2:
            return shap_values
        if shap_values.ndim == 3:
            # Formato comum: linhas, features, classes.
            if shap_values.shape[1] == n_features:
                return shap_values[:, :, 1]
            # Formato alternativo: classes, linhas, features.
            if shap_values.shape[0] == 2:
                return shap_values[1, :, :]

    raise ValueError(f"Formato inesperado de shap_values: {type(shap_values)}")


def main():
    REPORTS_DIR.mkdir(exist_ok=True)

    modelo = joblib.load(MODELS_DIR / "modelo_final.pkl")
    label_encoder = joblib.load(MODELS_DIR / "label_encoder.pkl")
    features = joblib.load(MODELS_DIR / "colunas_modelo.pkl")

    df = pd.read_csv(DATASET_PATH)
    X = df[features].copy()

    for coluna in features:
        X[coluna] = pd.to_numeric(X[coluna], errors="coerce")

    X = X.fillna(0)
    X_sample = X.sample(n=min(300, len(X)), random_state=42)

    print("Modelo carregado:", type(modelo))
    print("Classes:", list(label_encoder.classes_))
    print("Amostra SHAP:", X_sample.shape)

    explainer = shap.TreeExplainer(modelo)
    shap_values_raw = explainer.shap_values(X_sample)
    shap_values = normalizar_shap_values(shap_values_raw, n_features=len(features))

    importancia = np.abs(shap_values).mean(axis=0)
    df_importancia = pd.DataFrame({
        "feature": features,
        "importancia_media_abs_shap": importancia,
    }).sort_values(by="importancia_media_abs_shap", ascending=False)

    df_importancia.to_csv(REPORTS_DIR / "shap_importancia_global.csv", index=False)

    print("\nImportância global das variáveis:")
    print(df_importancia)

    top_features = df_importancia.head(15).sort_values(by="importancia_media_abs_shap", ascending=True)

    plt.figure(figsize=(10, 7))
    plt.barh(top_features["feature"], top_features["importancia_media_abs_shap"])
    plt.xlabel("Importância média absoluta SHAP")
    plt.title("Importância Global das Variáveis - Classe Queimada")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "shap_importancia_global.png", dpi=300)
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, X_sample, feature_names=features, show=False)
    plt.title("SHAP Summary Plot - Classe Queimada")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "shap_summary_queimada.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\nArquivos SHAP gerados:")
    print(REPORTS_DIR / "shap_importancia_global.csv")
    print(REPORTS_DIR / "shap_importancia_global.png")
    print(REPORTS_DIR / "shap_summary_queimada.png")


if __name__ == "__main__":
    main()
