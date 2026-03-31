
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

SEED_FIXA = 42

def _criar_modelos():
    return {
        "KNN": KNeighborsClassifier(
            n_neighbors=7,
            metric="euclidean",
            weights="distance",
            n_jobs=-1
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=None,
            min_samples_split=5,
            random_state=SEED_FIXA,
            n_jobs=-1
        ),
        "SVM (RBF, OvO)": SVC(
            kernel="rbf",
            decision_function_shape="ovo",
            C=1.0,
            gamma="scale",
            cache_size=500,
            tol=1e-3
        ),
    }

def _avaliar_modelo_kfold(nome, modelo, X: np.ndarray, y: np.ndarray,
                           classes: list, n_splits: int = 10,
                           n_repeticoes: int = 5) -> dict:
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    classes_enc = le.transform(classes)

    metricas = {
        "acuracia":  [],
        "f1_macro":  [],
        "f1_per_class": {c: [] for c in classes},
        "precisao":  [],
        "recall":    [],
        "tempo_treino": [],
        "confusion_matrices": [],
    }

    total_folds = n_splits * n_repeticoes
    print(f"    → {nome}: {n_repeticoes} repetições × {n_splits}-fold "
          f"({total_folds} avaliações)...")

    for rep in range(n_repeticoes):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                              random_state=SEED_FIXA + rep)
        for fold_idx, (idx_tr, idx_te) in enumerate(skf.split(X, y_enc)):
            X_tr, X_te = X[idx_tr], X[idx_te]
            y_tr, y_te = y_enc[idx_tr], y_enc[idx_te]

            scaler = StandardScaler()
            X_tr_z = scaler.fit_transform(X_tr)
            X_te_z  = scaler.transform(X_te)
            t0 = time.perf_counter()
            modelo.fit(X_tr_z, y_tr)
            metricas["tempo_treino"].append(time.perf_counter() - t0)

            y_pred = modelo.predict(X_te_z)

            metricas["acuracia"].append(accuracy_score(y_te, y_pred))
            metricas["f1_macro"].append(f1_score(y_te, y_pred, average="macro",
                                                  zero_division=0))
            metricas["precisao"].append(precision_score(y_te, y_pred, average="macro",
                                                         zero_division=0))
            metricas["recall"].append(recall_score(y_te, y_pred, average="macro",
                                                    zero_division=0))
            metricas["confusion_matrices"].append(confusion_matrix(y_te, y_pred))

            f1_classes = f1_score(y_te, y_pred, average=None,
                                   labels=classes_enc, zero_division=0)
            for c, f1_c in zip(classes, f1_classes):
                metricas["f1_per_class"][c].append(f1_c)

    acc_med  = np.mean(metricas["acuracia"])
    acc_std  = np.std(metricas["acuracia"])
    f1_med   = np.mean(metricas["f1_macro"])
    f1_std   = np.std(metricas["f1_macro"])
    t_med    = np.mean(metricas["tempo_treino"])

    print(f"       Acurácia  : {acc_med:.4f} ± {acc_std:.4f}")
    print(f"       F1-Macro  : {f1_med:.4f} ± {f1_std:.4f}")
    print(f"       Tempo/fit : {t_med*1000:.1f} ms")

    cm_agg = np.sum(metricas["confusion_matrices"], axis=0)
    metricas["cm_agregada"] = cm_agg

    return metricas

def treinar_e_avaliar(X: pd.DataFrame, y: pd.Series):
    np.random.seed(SEED_FIXA)
    modelos = _criar_modelos()
    classes_ordenadas = sorted(y.unique().tolist())
    X_arr = X.values
    y_arr = y.values

    print(f"  Classes: {classes_ordenadas}")
    print(f"  Shape final: {X_arr.shape}\n")

    resultados = {}
    for nome, modelo in modelos.items():
        metricas = _avaliar_modelo_kfold(
            nome, modelo, X_arr, y_arr,
            classes=classes_ordenadas,
            n_splits=10,
            n_repeticoes=5
        )
        metricas["classes"] = classes_ordenadas
        resultados[nome] = metricas
        print()

    resultados = _treinar_modelo_final(resultados, modelos, X_arr, y_arr,
                                       classes_ordenadas)

    return resultados


def _treinar_modelo_final(resultados, modelos, X_arr, y_arr, classes):
    le = LabelEncoder()
    y_enc = le.fit_transform(y_arr)

    scaler = StandardScaler()
    X_z = scaler.fit_transform(X_arr)

    print("  Treinando modelos finais no dataset completo para relatório...")
    for nome, modelo in modelos.items():
        modelo.fit(X_z, y_enc)
        y_pred = modelo.predict(X_z)
        report = classification_report(y_enc, y_pred,
                                        target_names=classes,
                                        output_dict=True, zero_division=0)
        resultados[nome]["classification_report"] = report

    return resultados
