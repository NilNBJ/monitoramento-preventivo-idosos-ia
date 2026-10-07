from typing import Any

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "tempo_inatividade_min",
    "freq_transicao_comodos",
    "hora_dia_sin",
    "hora_dia_cos",
    "desvio_media_historica",
]


def pre_processar_dados(df_raw: pd.DataFrame) -> tuple[Any, StandardScaler]:
    """Ajusta o normalizador somente sobre as janelas de treinamento."""
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(df_raw[FEATURES])
    return x_scaled, scaler


def treinar_isolation_forest(x_train: Any) -> IsolationForest:
    """Treina o modelo não supervisionado com a configuração do TCC."""
    model = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train)
    return model


def avaliar_janela_comportamental(
    model: IsolationForest,
    scaler: StandardScaler,
    nova_janela_df: pd.DataFrame,
) -> list[dict[str, float | str]]:
    """Classifica janelas e converte outliers em payloads de alerta."""
    x_novos = scaler.transform(nova_janela_df[FEATURES])
    predicoes = model.predict(x_novos)
    scores = model.decision_function(x_novos)

    alertas: list[dict[str, float | str]] = []
    for predicao, score in zip(predicoes, scores, strict=True):
        score_float = float(score)
        if predicao == -1:
            alertas.append(
                {
                    "status": "ANOMALIA_DETECTADA",
                    "score": score_float,
                    "severidade": "ALTA" if score_float < -0.25 else "MEDIA",
                }
            )
        else:
            alertas.append({"status": "NORMAL", "score": score_float})
    return alertas
