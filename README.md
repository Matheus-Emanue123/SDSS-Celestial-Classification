# 🌌 Pipeline de Classificação Astronômica do SDSS

> **Nota Acadêmica:** Este projeto foi desenvolvido como parte de um trabalho prático para a disciplina de **Inteligência Computacional** no **CEFET-MG**.

Este repositório contém o pipeline de Machine Learning desenvolvido para a classificação automatizada de objetos celestes do Sloan Digital Sky Survey (SDSS). O objetivo principal é distinguir com alta precisão entre **Galáxias (GALAXY)**, **Estrelas (STAR)** e **Quasares (QSO)**, viabilizando a análise de dados em larga escala na astrometria moderna.

---

## 📋 1. O Problema e a Contextualização

O SDSS gera um volume massivo de dados de observação do céu profundo. A classificação manual desses corpos celestes é impraticável, exigindo abordagens automatizadas. O desafio técnico reside em lidar com:
* **Desbalanceamento de Classes:** Galáxias dominam o dataset, superando os Quasares em uma proporção de quase 3:1.
* **Sobreposição Fotométrica ("Photometric Crowding"):** Bandas de luz frequentemente apresentam densidades sobrepostas entre diferentes classes.
* **Ruídos e Artefatos:** Erros de leitura de sensores e anomalias instrumentais que não representam fenômenos físicos.

![Distribuição das Classes](graficos/01_distribuicao_classes.png)

---

## ⚙️ 2. Pipeline de Dados e Pré-processamento

Para garantir que os modelos aprendam padrões físicos reais e não ruídos, implementamos um pipeline de pré-processamento robusto.

### Análise Exploratória (EDA) e Correlação
A análise de distribuição e dispersão revelou que características como o `redshift` possuem bimodalidade clara, sendo vitais para separar Quasares. Variáveis de indexação (`obj_ID`, `spec_obj_ID`) não possuem valor físico.

![Histogramas](graficos/02_histogramas.jpg)
![Dispersão](graficos/03_dispersao.png)

A Matriz de Correlação de Pearson evidenciou redundâncias críticas que precisavam ser tratadas para otimizar o treinamento:
* **Redundância Instrumental:** `obj_ID` e `run_ID` (correlação de 1.00).
* **Redundância Temporal:** `spec_obj_ID` e `MJD` (correlação de 0.97).
* **Multicolinearidade Fotométrica:** Bandas de luz como `u` e `g`.

![Matriz de Correlação](graficos/04_correlacao.png)

### Saneamento e Engenharia de Features
1. **Tratamento de Outliers:** Aplicamos a regra de $\mu \pm 3\sigma$ para remover artefatos instrumentais severos (valores anômalos próximos a -10.000), preservando a integridade das distribuições.
    * *Antes:* ![Boxplot Antes](graficos/05_boxplot_antes.png)
    * *Depois:* ![Boxplot Depois](graficos/05_boxplot_depois.png)
2. **Normalização (Z-score):** Fundamental para estabilizar o gradiente em redes neurais e evitar que features com grandes grandezas dominem o cálculo de distância no KNN.
3. **Seleção de Atributos:** Utilizando ANOVA F-Score, mantivemos as features de maior relevância física (como o `redshift`, com score > 100.000) e descartamos metadados irrelevantes.

![Importância das Features](graficos/06_feature_importance.png)

---

## 🧠 3. Modelagem e Algoritmos

Adotamos três paradigmas distintos de modelagem, validados através de **Stratified K-Fold Cross-Validation** (com normalização aplicada estritamente dentro de cada fold para evitar *data leakage*). Uma seed fixa (42) foi utilizada em todos os componentes estocásticos para garantir reprodutibilidade.

### K-Nearest Neighbors (KNN)
* **Configuração:** $k=7$ com pesos ponderados pela distância.
* **Por que foi escolhido:** Serve como um excelente baseline baseado em instâncias (distância espacial).
* **Limitação:** Sofre com a sobreposição de bandas fotométricas, o que afeta sua precisão ao separar estrelas de galáxias.

### Random Forest (Ensemble)
* **Configuração:** 150 estimadores, seed = 42.
* **Por que foi escolhido:** Lida nativamente com não-linearidades (como o `redshift`), é altamente resiliente a outliers remanescentes e minimiza o *overfitting* através de amostragem estocástica.

### Support Vector Machine (SVM) — Kernel RBF com One-vs-One
* **Configuração:** Kernel RBF, estratégia de decisão one-vs-one (OvO), $C=1.0$, $\gamma=\texttt{scale}$.
* **Por que foi escolhido:** Paradigma de margem máxima capaz de capturar fronteiras de decisão não-lineares entre as classes astronômicas, teoricamente eficaz para dados multi-dimensionais com sobreposição fotométrica.
* **Trade-off:** Alcança F1-Score competitivo (0.9552), porém com custo computacional significativamente mais alto (~35.7s por fold).

---

## 📊 4. Resultados e Avaliação de Desempenho

O **F1-Score (Macro)** foi eleito a métrica principal de avaliação devido ao desbalanceamento das classes, garantindo que o desempenho na classe minoritária (QSO) não fosse ofuscado. Os resultados abaixo refletem a média e desvio padrão de 10 avaliações (2 repetições × 5-fold).

| Modelo | Acurácia Média | F1-Score (Macro) | Tempo Médio/fold |
| :--- | :--- | :--- | :--- |
| **Random Forest** | **0.9773 ± 0.0008** | **0.9722 ± 0.0010** | **4953.7 ms** |
| SVM (RBF, OvO) | 0.9618 ± 0.0007 | 0.9552 ± 0.0009 | 35705.4 ms |
| KNN | 0.9174 ± 0.0018 | 0.9051 ± 0.0022 | 119.4 ms |

### Matrizes de Confusão
O KNN confundiu 15% das estrelas com galáxias devido ao "photometric crowding". Em contrapartida, Random Forest e SVM atingiram desempenho excelente na classe STAR (≥99.9% recall) e segregação superior para Quasares.

![Matrizes de Confusão](graficos/07_matrizes_confusao.png)

### Estabilidade
Em 10 avaliações distintas (2 repetições × 5-fold), o Random Forest apresentou a menor dispersão de erro e maior consistência geral, comprovando ser o modelo mais confiável. O SVM, apesar de bom desempenho, sofre com tempo de treinamento proibitivo em datasets desta escala (~96k amostras). O KNN permanece como baseline rápido mas menos preciso.

![Comparativo Boxplots](graficos/08_boxplots_comparativo.png)
![F1 por Classe](graficos/09_f1_por_classe.png)
![Tempo Computacional](graficos/10_tempo_computacional.png)

---

## 🚀 5. Conclusão e Recomendação

O modelo **Random Forest** permanece como o campeão da experimentação, equilibrando desempenho preditivo (F1-Macro: 0.9722) com custo computacional aceitável (~4.95s por fold).

O **SVM com kernel RBF e estratégia one-vs-one**, embora alcance F1-Macro respeitável (0.9552), apresenta latência de treinamento impraticável para escala de produção (~35.7s por fold). Isso ilustra um princípio fundamental: nem sempre o algoritmo teoricamente superior é a melhor escolha prática — o Random Forest oferece um balanço superior entre acurácia, estabilidade e eficiência computacional.

O pipeline demonstra que um pré-processamento rigoroso, seed fixa para reprodutibilidade e validação cruzada estratificada são fundamentos mais determinantes para o sucesso da classificação astronômica do que a mera sofisticação algorítmica.

---

## 💻 6. Como Executar Localmente

Siga os passos abaixo para reproduzir os experimentos e análises gerados neste projeto:

1. **Clone o repositório:**
   ```bash
   git clone [https://github.com/SEU_USUARIO/NOME_DO_REPOSITORIO.git](https://github.com/SEU_USUARIO/NOME_DO_REPOSITORIO.git)
   cd NOME_DO_REPOSITORIO
   ```

2. **Crie um ambiente virtual (opcional, mas recomendado):**
   ```bash
   python -m venv venv
   
   # Para ativar no Windows:
   venv\Scripts\activate
   # Para ativar no Linux/Mac:
   source venv/bin/activate
   ```

3. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Execute o pipeline:**
   ```bash
   # Substitua pelo nome do seu script principal
   python main.py 
   ```

---

## 📞 7. Contato

Sinta-se à vontade para entrar em contato caso tenha dúvidas sobre o projeto, o pipeline de dados ou os resultados obtidos!

* **Nome:** [Seu Nome Aqui]
* **LinkedIn:** [Link para o seu perfil](https://linkedin.com/in/seu-perfil)
* **GitHub:** [Link para o seu GitHub](https://github.com/SEU_USUARIO)
* **E-mail:** [seu-email@exemplo.com](mailto:seu-email@exemplo.com)