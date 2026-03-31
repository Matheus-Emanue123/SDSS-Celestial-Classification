import warnings
warnings.filterwarnings("ignore")

from etapa1_dados import carregar_e_explorar
from etapa2_preprocessamento import preprocessar
from etapa3_modelagem import treinar_e_avaliar
from etapa4_avaliacao import gerar_relatorio_final


def main():
    print("=" * 65)
    print("  ATIVIDADE PRÁTICA 01 — METODOLOGIA EXPERIMENTAL")
    print("  Disciplina de Inteligência Computacional")
    print("=" * 65)

    print("\n[1/4] ESCOLHA E COMPREENSÃO DA BASE DE DADOS")
    X, y, feature_names = carregar_e_explorar()

    print("\n[2/4] PRÉ-PROCESSAMENTO DOS DADOS")
    X_proc, y_proc, scaler, selector = preprocessar(X, y, feature_names)

    print("\n[3/4] METODOLOGIA EXPERIMENTAL — TREINAMENTO")
    resultados = treinar_e_avaliar(X_proc, y_proc)

    print("\n[4/4] AVALIAÇÃO DOS RESULTADOS")
    gerar_relatorio_final(resultados)

    print("\n" + "=" * 65)
    print("  Pipeline concluído! Verifique os gráficos gerados.")
    print("=" * 65)


if __name__ == "__main__":
    main()
