from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "data" / "processed" / "dataset_queimadas.csv"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"


FEATURES = [
    "latitude",
    "longitude",
    "bright_ti4",
    "bright_ti5",
    "frp",
    "scan",
    "track",
    "confidence_num",
    "daynight_num",
    "ano",
    "mes",
    "dia",
    "dia_do_ano",
    "risco_fogo",
    "diferenca_brilho",
    "frp_por_brilho",
    "periodo_seco",
]


def carregar_dataset():
    df = pd.read_csv(DATASET_PATH)

    print("Dataset carregado:", df.shape)
    print("Distribuição das classes:")
    print(df["classe"].value_counts())

    return df


def preparar_dados(df):
    X = df[FEATURES].copy()

    for coluna in FEATURES:
        X[coluna] = pd.to_numeric(X[coluna], errors="coerce")

    X = X.fillna(0)

    y = df["classe"].copy()

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print("\nMapeamento das classes:")
    for classe, codigo in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)):
        print(f"{classe} -> {codigo}")

    return X, y_encoded, label_encoder


def avaliar_modelo(nome, modelo, X_test, y_test, label_encoder):
    y_pred = modelo.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="binary", pos_label=label_encoder.transform(["queimada"])[0])
    precision = precision_score(y_test, y_pred, average="binary", pos_label=label_encoder.transform(["queimada"])[0], zero_division=0)
    recall = recall_score(y_test, y_pred, average="binary", pos_label=label_encoder.transform(["queimada"])[0], zero_division=0)

    roc_auc = None
    if hasattr(modelo, "predict_proba"):
        y_proba = modelo.predict_proba(X_test)[:, label_encoder.transform(["queimada"])[0]]
        roc_auc = roc_auc_score(y_test, y_proba)

    print("\n" + "=" * 70)
    print(f"Modelo: {nome}")
    print("=" * 70)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Queimada: {f1:.4f}")
    print(f"Precision Queimada: {precision:.4f}")
    print(f"Recall Queimada: {recall:.4f}")
    if roc_auc is not None:
        print(f"ROC AUC: {roc_auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))

    matriz = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=label_encoder.classes_)
    disp.plot(values_format="d")
    plt.title(f"Matriz de Confusão - {nome}")
    plt.xticks(rotation=45)
    plt.tight_layout()

    nome_arquivo = nome.lower().replace(" ", "_")
    caminho_figura = REPORTS_DIR / f"matriz_confusao_{nome_arquivo}.png"
    plt.savefig(caminho_figura, dpi=300)
    plt.close()

    return {
        "modelo": nome,
        "accuracy": accuracy,
        "f1_queimada": f1,
        "precision_queimada": precision,
        "recall_queimada": recall,
        "roc_auc": roc_auc,
    }


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    df = carregar_dataset()
    X, y, label_encoder = preparar_dados(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    modelos = {
        "Logistic Regression": Pipeline(steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight="balanced",
            n_jobs=1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=80,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=1,
        ),
    }

    resultados = []
    melhor_modelo = None
    melhor_nome = None
    melhor_f1 = -1

    for nome, modelo in modelos.items():
        print("\nTreinando:", nome)
        modelo.fit(X_train, y_train)

        resultado = avaliar_modelo(nome, modelo, X_test, y_test, label_encoder)
        resultados.append(resultado)

        if (resultado["f1_queimada"] > melhor_f1) or (resultado["f1_queimada"] == melhor_f1 and nome == "Random Forest"):
            melhor_f1 = resultado["f1_queimada"]
            melhor_modelo = modelo
            melhor_nome = nome

    df_resultados = pd.DataFrame(resultados).sort_values(by="f1_queimada", ascending=False)
    df_resultados.to_csv(REPORTS_DIR / "resultados_modelos.csv", index=False)

    joblib.dump(melhor_modelo, MODELS_DIR / "modelo_final.pkl")
    joblib.dump(label_encoder, MODELS_DIR / "label_encoder.pkl")
    joblib.dump(FEATURES, MODELS_DIR / "colunas_modelo.pkl")

    print("\nComparação final dos modelos:")
    print(df_resultados)
    print("\nMelhor modelo:", melhor_nome)
    print(f"Melhor F1 Queimada: {melhor_f1:.4f}")


if __name__ == "__main__":
    main()
