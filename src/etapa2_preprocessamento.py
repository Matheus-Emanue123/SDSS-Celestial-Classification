import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def tratar_missing(X: pd.DataFrame) -> pd.DataFrame:
    total = len(X)
    colunas_ruim = [c for c in X.columns if X[c].isnull().sum() / total > 0.50]
    if colunas_ruim:
        print(f"  [2.1] Removendo colunas com >50% missing: {colunas_ruim}")
        X = X.drop(columns=colunas_ruim)

    n_missing_antes = X.isnull().sum().sum()
    if n_missing_antes > 0:
        imputer = SimpleImputer(strategy="median")
        X_arr = imputer.fit_transform(X)
        X = pd.DataFrame(X_arr, columns=X.columns, index=X.index)
        print(f"  [2.1] Imputação pela mediana: {n_missing_antes} valores ausentes corrigidos.")
    else:
        print("  [2.1] Nenhum valor ausente encontrado.")
    return X

def detectar_e_remover_outliers(X: pd.DataFrame, y: pd.Series):
    _boxplot_outliers(X, prefixo="antes", titulo="Boxplots ANTES da remoção de outliers")
    mask_ok = pd.Series([True] * len(X), index=X.index)
    for col in X.columns:
        mu = X[col].mean()
        sigma = X[col].std()
        mask_ok &= (X[col] >= mu - 3 * sigma) & (X[col] <= mu + 3 * sigma)

    n_out = (~mask_ok).sum()
    print(f"  [2.2] Outliers detectados (µ±3σ): {n_out} amostras "
          f"({100*n_out/len(X):.1f}%) → removidas.")
    X_clean = X[mask_ok].reset_index(drop=True)
    y_clean = y[mask_ok].reset_index(drop=True)

    _boxplot_outliers(X_clean, prefixo="depois", titulo="Boxplots APÓS remoção de outliers")
    return X_clean, y_clean

def _boxplot_outliers(X: pd.DataFrame, prefixo: str, titulo: str):
    n = len(X.columns)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(13, rows * 3))
    axes = axes.flatten()
    for idx, col in enumerate(X.columns):
        axes[idx].boxplot(X[col].dropna(), vert=True, patch_artist=True,
                          boxprops=dict(facecolor="#4A90D9", alpha=0.6))
        axes[idx].set_title(col, fontsize=9)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle(titulo, fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/05_boxplot_{prefixo}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

def codificar_categoricos(X: pd.DataFrame) -> pd.DataFrame:
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        print(f"  [2.3] One-Hot Encoding aplicado em: {cat_cols}")
        X = pd.get_dummies(X, columns=cat_cols, drop_first=False)
    else:
        print("  [2.3] Nenhum atributo categórico encontrado — etapa não necessária.")
    return X

def padronizar_zscore(X_train: np.ndarray, X_test: np.ndarray):

    scaler = StandardScaler()
    X_train_z = scaler.fit_transform(X_train)
    X_test_z  = scaler.transform(X_test)
    print("  [2.4] Padronização Z-score aplicada (fit no treino, transform no teste).")
    return X_train_z, X_test_z, scaler

def selecionar_atributos(X: pd.DataFrame, y: pd.Series, feature_names: list,
                         k: int | str = "all"):

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    redundantes = [col for col in upper.columns if any(upper[col] > 0.95)]
    if redundantes:
        print(f"  [2.5] Removendo features redundantes (|corr|>0.95): {redundantes}")
        X = X.drop(columns=redundantes)
        feature_names = [f for f in feature_names if f not in redundantes]

    selector = SelectKBest(f_classif, k=k)
    X_sel_arr = selector.fit_transform(X.values, y_enc)
    scores = pd.Series(selector.scores_, index=feature_names).sort_values(ascending=False)

    _plot_feature_importance(scores)
    print(f"  [2.5] F-scores das features (ANOVA):")
    for nome, score in scores.items():
        print(f"         {nome:>15s}: {score:.1f}")

    feature_names_sel = list(scores.index)
    X_sel = pd.DataFrame(X_sel_arr, columns=feature_names_sel)
    return X_sel, feature_names_sel, selector


def _plot_feature_importance(scores: pd.Series):
    fig, ax = plt.subplots(figsize=(8, 4))
    scores.sort_values().plot(kind="barh", ax=ax, color="#4A90D9", edgecolor="white")
    ax.set_title("Importância das Features — ANOVA F-Score", fontsize=12, fontweight="bold")
    ax.set_xlabel("F-Score")
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/06_feature_importance.png", dpi=150)
    plt.close(fig)

def preprocessar(X: pd.DataFrame, y: pd.Series, feature_names: list):
    
    print("  ── Sub-etapa 2.1: Missing Values ──")
    X = tratar_missing(X)

    print("  ── Sub-etapa 2.2: Outliers ──")
    X, y = detectar_e_remover_outliers(X, y)

    print("  ── Sub-etapa 2.3: Variáveis Categóricas ──")
    X = codificar_categoricos(X)
    feature_names = list(X.columns)

    print("  ── Sub-etapa 2.4: Normalização ──")
    print("  (Z-score será aplicado dentro de cada fold do k-fold para evitar leakage)")

    print("  ── Sub-etapa 2.5: Seleção de Atributos ──")
    X, feature_names, selector = selecionar_atributos(X, y, feature_names)

    print(f"\n  ✔ Pré-processamento concluído: {X.shape[0]} amostras × {X.shape[1]} features")
    print(f"  Distribuição final de classes:\n{y.value_counts().to_string()}")

    return X, y, StandardScaler(), selector
