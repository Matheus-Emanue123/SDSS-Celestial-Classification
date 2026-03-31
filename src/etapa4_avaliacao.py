"""
Etapa 4 — Avaliação dos Resultados
  4.1 Métricas de desempenho: Matriz de Confusão, Precisão, Recall, F1-Score
  4.2 Testes e Comparações: Boxplots de múltiplas execuções + tabela resumo
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CORES_MODELOS = {
    "KNN":              "#F5A623",
    "Random Forest":    "#4A90D9",
    "SVM (RBF, OvO)":   "#7B68EE",
}


# ─────────────────────────────────────────────────────────────────────────────
# 4.1  Matrizes de Confusão
# ─────────────────────────────────────────────────────────────────────────────
def _plot_matrizes_confusao(resultados: dict):
    nomes = list(resultados.keys())
    fig, axes = plt.subplots(1, len(nomes), figsize=(5 * len(nomes), 4))
    if len(nomes) == 1:
        axes = [axes]

    for ax, nome in zip(axes, nomes):
        cm = resultados[nome]["cm_agregada"]
        classes = resultados[nome]["classes"]
        # Normaliza por linha (recall por classe)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=classes, yticklabels=classes,
                    ax=ax, linewidths=0.5, cbar=False,
                    annot_kws={"size": 11})
        # Valores absolutos no título
        total = cm.sum()
        corretos = np.trace(cm)
        ax.set_title(f"{nome}\n(acerto: {corretos}/{total})",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Predito"); ax.set_ylabel("Real")

    fig.suptitle("Matrizes de Confusão (Agregadas — normalizadas por linha)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/07_matrizes_confusao.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✔ Matrizes de confusão salvas.")


# ─────────────────────────────────────────────────────────────────────────────
# 4.2  Boxplots de desempenho (múltiplas execuções)
# ─────────────────────────────────────────────────────────────────────────────
def _plot_boxplots_comparativo(resultados: dict):
    metricas_plot = ["acuracia", "f1_macro", "precisao", "recall"]
    titulos = ["Acurácia", "F1-Score (Macro)", "Precisão (Macro)", "Revocação (Macro)"]

    fig, axes = plt.subplots(1, len(metricas_plot), figsize=(16, 5))

    for ax, metrica, titulo in zip(axes, metricas_plot, titulos):
        dados   = [resultados[n][metrica] for n in resultados]
        labels  = list(resultados.keys())
        cores   = [CORES_MODELOS.get(n, "#888") for n in labels]

        bp = ax.boxplot(dados, patch_artist=True, notch=False,
                        medianprops=dict(color="black", linewidth=2))
        for patch, cor in zip(bp["boxes"], cores):
            patch.set_facecolor(cor)
            patch.set_alpha(0.7)

        ax.set_title(titulo, fontsize=12, fontweight="bold")
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels([n.replace(" ", "\n") for n in labels], fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.yaxis.grid(True, alpha=0.4)
        ax.set_ylabel("Score")

    fig.suptitle("Comparação de Desempenho — Boxplots (50 avaliações por modelo)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/08_boxplots_comparativo.png", dpi=150)
    plt.close(fig)
    print("  ✔ Boxplots comparativos salvos.")


# ─────────────────────────────────────────────────────────────────────────────
# F1 por classe
# ─────────────────────────────────────────────────────────────────────────────
def _plot_f1_por_classe(resultados: dict):
    nomes   = list(resultados.keys())
    classes = resultados[nomes[0]]["classes"]

    fig, axes = plt.subplots(1, len(classes),
                             figsize=(5 * len(classes), 4), sharey=True)
    if len(classes) == 1:
        axes = [axes]

    for ax, cls in zip(axes, classes):
        dados  = [resultados[n]["f1_per_class"][cls] for n in nomes]
        cores  = [CORES_MODELOS.get(n, "#888") for n in nomes]
        bp = ax.boxplot(dados, patch_artist=True,
                        medianprops=dict(color="black", linewidth=2))
        for patch, cor in zip(bp["boxes"], cores):
            patch.set_facecolor(cor)
            patch.set_alpha(0.7)
        ax.set_title(f"F1 — Classe: {cls}", fontsize=11, fontweight="bold")
        ax.set_xticks(range(1, len(nomes) + 1))
        ax.set_xticklabels([n.replace(" ", "\n") for n in nomes], fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.yaxis.grid(True, alpha=0.4)
        ax.set_ylabel("F1-Score")

    fig.suptitle("F1-Score por Classe Astronômica (Boxplots)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/09_f1_por_classe.png", dpi=150)
    plt.close(fig)
    print("  ✔ F1 por classe salvo.")


# ─────────────────────────────────────────────────────────────────────────────
# Tempo computacional
# ─────────────────────────────────────────────────────────────────────────────
def _plot_tempo_computacional(resultados: dict):
    nomes  = list(resultados.keys())
    tempos_med = [np.mean(resultados[n]["tempo_treino"]) * 1000 for n in nomes]
    tempos_std = [np.std(resultados[n]["tempo_treino"]) * 1000  for n in nomes]
    cores  = [CORES_MODELOS.get(n, "#888") for n in nomes]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(nomes, tempos_med, yerr=tempos_std, capsize=6,
                  color=cores, alpha=0.8, edgecolor="white")
    ax.set_title("Custo Computacional — Tempo médio de treino por fold",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Tempo (ms)")
    ax.set_yscale("log")
    for bar, val in zip(bars, tempos_med):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.15,
                f"{val:.1f} ms", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/10_tempo_computacional.png", dpi=150)
    plt.close(fig)
    print("  ✔ Gráfico de tempo computacional salvo.")


# ─────────────────────────────────────────────────────────────────────────────
# Tabela resumo final
# ─────────────────────────────────────────────────────────────────────────────
def _imprimir_tabela_resumo(resultados: dict):
    print("\n" + "═" * 72)
    print("  TABELA RESUMO — COMPARAÇÃO DOS ALGORITMOS")
    print("═" * 72)
    header = f"{'Modelo':<24} {'Acurácia':>10} {'F1-Macro':>10} {'Precisão':>10} {'Recall':>10} {'Tempo(ms)':>10}"
    print(header)
    print("─" * 72)
    for nome, m in resultados.items():
        acc = f"{np.mean(m['acuracia']):.4f}±{np.std(m['acuracia']):.4f}"
        f1  = f"{np.mean(m['f1_macro']):.4f}±{np.std(m['f1_macro']):.4f}"
        pre = f"{np.mean(m['precisao']):.4f}"
        rec = f"{np.mean(m['recall']):.4f}"
        t   = f"{np.mean(m['tempo_treino'])*1000:.1f}"
        print(f"{nome:<24} {acc:>10} {f1:>10} {pre:>10} {rec:>10} {t:>10}")
    print("═" * 72)

    # Relatório por classe (modelo de melhor F1)
    melhor = max(resultados, key=lambda n: np.mean(resultados[n]["f1_macro"]))
    print(f"\n  Relatório detalhado do melhor modelo ({melhor}):")
    print("─" * 72)
    report = resultados[melhor]["classification_report"]
    for cls, vals in report.items():
        if isinstance(vals, dict):
            print(f"  {cls:<20} precision={vals['precision']:.3f}  "
                  f"recall={vals['recall']:.3f}  f1={vals['f1-score']:.3f}  "
                  f"support={int(vals['support'])}")
    print("═" * 72)


def _salvar_csv_resultados(resultados: dict):
    rows = []
    for nome, m in resultados.items():
        for i, (acc, f1, pre, rec, t) in enumerate(zip(
                m["acuracia"], m["f1_macro"],
                m["precisao"], m["recall"],
                m["tempo_treino"])):
            rows.append({
                "modelo": nome, "fold": i + 1,
                "acuracia": acc, "f1_macro": f1,
                "precisao": pre, "recall": rec,
                "tempo_ms": t * 1000
            })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_DIR, "resultados_metricas.csv")
    df.to_csv(csv_path, index=False)
    print(f"  ✔ Resultados salvos em '{csv_path}'")


# ─────────────────────────────────────────────────────────────────────────────
# Função principal da etapa
# ─────────────────────────────────────────────────────────────────────────────
def gerar_relatorio_final(resultados: dict):
    _plot_matrizes_confusao(resultados)
    _plot_boxplots_comparativo(resultados)
    _plot_f1_por_classe(resultados)
    _plot_tempo_computacional(resultados)
    _imprimir_tabela_resumo(resultados)
    _salvar_csv_resultados(resultados)
    print(f"\n  Todos os gráficos foram salvos na pasta '{OUTPUT_DIR}/'")
