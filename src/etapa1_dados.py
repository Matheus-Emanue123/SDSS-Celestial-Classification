import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification


PALETTE = {"STAR": "#F5A623", "GALAXY": "#4A90D9", "QSO": "#7B68EE"}
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def _tentar_carregar_csv() -> pd.DataFrame | None:
    candidatos = [
        "Skyserver_SQL2_27_2018 6_51_39 PM.csv",
        "sdss.csv",
        "star_classification.csv",
        "skyserver.csv",
    ]
    caminhos = []
    for nome in candidatos:
        caminhos.append(nome)
        caminhos.append(os.path.join("base", nome))
        caminhos.append(os.path.join(BASE_DIR, nome))
        caminhos.append(os.path.join(BASE_DIR, "base", nome))

    for caminho in caminhos:
        if os.path.exists(caminho):
            print(f"  ✔ Arquivo encontrado: {caminho}")
            return pd.read_csv(caminho)
    return None


def carregar_e_explorar():
    print("  Carregando base de dados SDSS...")
    df = _tentar_carregar_csv()
    if df is None:
        raise FileNotFoundError(
            "Nenhum CSV do SDSS foi encontrado. Coloque o arquivo em 'base/' "
            "ou na raiz do projeto."
        )

    col_alvo_candidatos = ["class", "Class", "CLASS", "objtype"]
    for c in col_alvo_candidatos:
        if c in df.columns:
            df = df.rename(columns={c: "class"})
            break

    print(f"\n  Dimensões do dataset : {df.shape[0]} amostras × {df.shape[1]} atributos")
    print(f"  Distribuição de classes:\n{df['class'].value_counts().to_string()}")

    colunas_drop = ["objid", "specobjid", "run", "rerun", "camcol",
                    "field", "plate", "mjd", "fiberid"]
    colunas_drop = [c for c in colunas_drop if c in df.columns]
    df = df.drop(columns=colunas_drop)
    features = [c for c in df.columns if c != "class" and pd.api.types.is_numeric_dtype(df[c])]
    feature_names = features

    print(f"  Features selecionadas : {feature_names}")
    print(f"  Missing values por coluna:\n{df[features].isnull().sum().to_string()}")

    _plotar_exploratorio(df, features)

    X = df[features].copy()
    y = df["class"].copy()
    return X, y, feature_names


def _plotar_exploratorio(df: pd.DataFrame, features: list):
    _plot_distribuicao_classes(df)
    _plot_histogramas(df, features)
    _plot_dispersao(df, features)
    _plot_correlacao(df, features)
    print(f"  Gráficos exploratórios salvos em '{OUTPUT_DIR}/'")


def _plot_distribuicao_classes(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["class"].value_counts()
    colors = [PALETTE.get(c, "#888") for c in counts.index]
    counts.plot(kind="bar", ax=ax, color=colors, edgecolor="white", width=0.6)
    ax.set_title("Distribuição das Classes", fontsize=13, fontweight="bold")
    ax.set_xlabel("Classe"); ax.set_ylabel("Número de amostras")
    ax.set_xticklabels(counts.index, rotation=0)
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width()/2, p.get_height()),
                    ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/01_distribuicao_classes.png", dpi=150)
    plt.close(fig)


def _plot_histogramas(df, features):
    n = len(features)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 3))
    axes = axes.flatten()
    for idx, feat in enumerate(features):
        ax = axes[idx]
        for cls, grp in df.groupby("class"):
            ax.hist(grp[feat].dropna(), bins=40, alpha=0.55,
                    label=cls, color=PALETTE.get(cls, "#888"), density=True)
        ax.set_title(feat, fontsize=9)
        ax.set_ylabel("Densidade")
        ax.legend(fontsize=7)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle("Histogramas por Classe", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/02_histogramas.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_dispersao(df, features):
    if len(features) < 2:
        return
    # Scatter entre as duas features mais discriminantes (variância máxima)
    variancias = df[features].var().sort_values(ascending=False)
    f1, f2 = variancias.index[0], variancias.index[1]
    fig, ax = plt.subplots(figsize=(7, 5))
    sample = df.sample(min(2000, len(df)), random_state=42)
    for cls, grp in sample.groupby("class"):
        ax.scatter(grp[f1], grp[f2], s=8, alpha=0.5,
                   label=cls, color=PALETTE.get(cls, "#888"))
    ax.set_xlabel(f1); ax.set_ylabel(f2)
    ax.set_title(f"Dispersão: {f1} × {f2}", fontsize=12, fontweight="bold")
    ax.legend(markerscale=2)
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/03_dispersao.png", dpi=150)
    plt.close(fig)


def _plot_correlacao(df, features):
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df[features].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.5, ax=ax, annot_kws={"size": 8})
    ax.set_title("Matriz de Correlação de Pearson", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/04_correlacao.png", dpi=150)
    plt.close(fig)