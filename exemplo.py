import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from monitoramento_idosos import (  # noqa: E402
    Alerta,
    GerenciadorDados,
    JanelaComportamental,
    ModeloMetadata,
    avaliar_janela_comportamental,
    pre_processar_dados,
    treinar_isolation_forest,
)


def gerar_rotina_normal(n_amostras: int = 5_000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    horas = rng.uniform(0, 24, n_amostras)
    return pd.DataFrame(
        {
            "tempo_inatividade_min": np.clip(rng.normal(12, 5, n_amostras), 0, None),
            "freq_transicao_comodos": np.clip(rng.normal(5, 1.5, n_amostras), 0, None),
            "hora_dia_sin": np.sin(2 * np.pi * horas / 24),
            "hora_dia_cos": np.cos(2 * np.pi * horas / 24),
            "desvio_media_historica": rng.normal(0, 0.25, n_amostras),
        }
    )


def main() -> None:
    rotina = gerar_rotina_normal()
    x_train, scaler = pre_processar_dados(rotina)
    modelo = treinar_isolation_forest(x_train)

    janela = JanelaComportamental(
        dispositivo_id="casa_01_sala",
        tempo_inatividade_min=90.0,
        freq_transicao_comodos=0.0,
        hora_dia_sin=0.87,
        hora_dia_cos=-0.50,
        desvio_media_historica=3.2,
    )
    nova_janela = pd.DataFrame(
        [
            {
                "tempo_inatividade_min": janela.tempo_inatividade_min,
                "freq_transicao_comodos": janela.freq_transicao_comodos,
                "hora_dia_sin": janela.hora_dia_sin,
                "hora_dia_cos": janela.hora_dia_cos,
                "desvio_media_historica": janela.desvio_media_historica,
            }
        ]
    )
    resultado = avaliar_janela_comportamental(modelo, scaler, nova_janela)[0]
    alerta = Alerta.a_partir_da_classificacao(
        janela_id=janela.id,
        modelo_versao="isoforest_v1",
        status=str(resultado["status"]),
        score=float(resultado["score"]),
    )

    with GerenciadorDados() as db:
        db.salvar_modelo(ModeloMetadata(versao="isoforest_v1", n_amostras_treino=len(rotina)))
        db.salvar_janela(janela)
        db.salvar_alerta(alerta)

    print(resultado)


if __name__ == "__main__":
    main()
