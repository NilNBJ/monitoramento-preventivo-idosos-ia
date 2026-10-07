from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import uuid4


class StatusAlerta(str, Enum):
    NORMAL = "NORMAL"
    ANOMALIA_DETECTADA = "ANOMALIA_DETECTADA"


class Severidade(str, Enum):
    MEDIA = "MEDIA"
    ALTA = "ALTA"


@dataclass
class JanelaComportamental:
    dispositivo_id: str
    tempo_inatividade_min: float
    freq_transicao_comodos: float
    hora_dia_sin: float
    hora_dia_cos: float
    desvio_media_historica: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    id: str = field(default_factory=lambda: str(uuid4()))

    def to_feature_array(self) -> list[float]:
        return [
            self.tempo_inatividade_min,
            self.freq_transicao_comodos,
            self.hora_dia_sin,
            self.hora_dia_cos,
            self.desvio_media_historica,
        ]


@dataclass
class Alerta:
    janela_id: str
    modelo_versao: str
    status: StatusAlerta
    score: float
    severidade: Optional[Severidade] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    id: str = field(default_factory=lambda: str(uuid4()))

    @staticmethod
    def a_partir_da_classificacao(
        janela_id: str,
        modelo_versao: str,
        status: str,
        score: float,
    ) -> "Alerta":
        status_enum = StatusAlerta(status)
        severidade = None
        if status_enum == StatusAlerta.ANOMALIA_DETECTADA:
            severidade = Severidade.ALTA if score < -0.25 else Severidade.MEDIA
        return Alerta(
            janela_id=janela_id,
            modelo_versao=modelo_versao,
            status=status_enum,
            score=score,
            severidade=severidade,
        )


@dataclass
class ModeloMetadata:
    versao: str
    treinado_em: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    n_estimators: int = 100
    contamination: float = 0.05
    max_samples: str = "auto"
    n_amostras_treino: int = 0
    features_usadas: list[str] = field(
        default_factory=lambda: [
            "tempo_inatividade_min",
            "freq_transicao_comodos",
            "hora_dia_sin",
            "hora_dia_cos",
            "desvio_media_historica",
        ]
    )
