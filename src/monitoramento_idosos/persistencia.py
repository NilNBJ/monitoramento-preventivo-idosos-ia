import json
import sqlite3
from dataclasses import asdict
from pathlib import Path

from .dominio import Alerta, JanelaComportamental, ModeloMetadata


class GerenciadorDados:
    def __init__(self, caminho_db: str | Path = "pipeline_anomalias.db") -> None:
        self.conn = sqlite3.connect(caminho_db)
        self.conn.row_factory = sqlite3.Row
        self._criar_tabelas()

    def _criar_tabelas(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS janelas (
                id TEXT PRIMARY KEY,
                dispositivo_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                tempo_inatividade_min REAL NOT NULL,
                freq_transicao_comodos REAL NOT NULL,
                hora_dia_sin REAL NOT NULL,
                hora_dia_cos REAL NOT NULL,
                desvio_media_historica REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS alertas (
                id TEXT PRIMARY KEY,
                janela_id TEXT NOT NULL,
                modelo_versao TEXT NOT NULL,
                status TEXT NOT NULL,
                score REAL NOT NULL,
                severidade TEXT,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (janela_id) REFERENCES janelas (id)
            );

            CREATE TABLE IF NOT EXISTS modelos (
                versao TEXT PRIMARY KEY,
                treinado_em TEXT NOT NULL,
                n_estimators INTEGER NOT NULL,
                contamination REAL NOT NULL,
                max_samples TEXT NOT NULL,
                n_amostras_treino INTEGER NOT NULL,
                features_usadas TEXT NOT NULL
            );
            """
        )
        self.conn.commit()

    def salvar_janela(self, janela: JanelaComportamental) -> None:
        dados = asdict(janela)
        dados["timestamp"] = janela.timestamp.isoformat()
        self.conn.execute(
            """
            INSERT INTO janelas VALUES (
                :id, :dispositivo_id, :timestamp, :tempo_inatividade_min,
                :freq_transicao_comodos, :hora_dia_sin, :hora_dia_cos,
                :desvio_media_historica
            )
            """,
            dados,
        )
        self.conn.commit()

    def salvar_alerta(self, alerta: Alerta) -> None:
        dados = asdict(alerta)
        dados["timestamp"] = alerta.timestamp.isoformat()
        dados["status"] = alerta.status.value
        dados["severidade"] = alerta.severidade.value if alerta.severidade else None
        self.conn.execute(
            """
            INSERT INTO alertas VALUES (
                :id, :janela_id, :modelo_versao, :status,
                :score, :severidade, :timestamp
            )
            """,
            dados,
        )
        self.conn.commit()

    def salvar_modelo(self, modelo: ModeloMetadata) -> None:
        dados = asdict(modelo)
        dados["treinado_em"] = modelo.treinado_em.isoformat()
        dados["features_usadas"] = json.dumps(modelo.features_usadas, ensure_ascii=False)
        self.conn.execute(
            """
            INSERT OR REPLACE INTO modelos VALUES (
                :versao, :treinado_em, :n_estimators, :contamination,
                :max_samples, :n_amostras_treino, :features_usadas
            )
            """,
            dados,
        )
        self.conn.commit()

    def alertas_por_severidade(self, severidade: str) -> list[dict]:
        cursor = self.conn.execute(
            "SELECT * FROM alertas WHERE severidade = ? ORDER BY timestamp DESC",
            (severidade,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def historico_dispositivo(self, dispositivo_id: str, limite: int = 100) -> list[dict]:
        cursor = self.conn.execute(
            """
            SELECT j.*, a.status, a.score, a.severidade
            FROM janelas AS j
            LEFT JOIN alertas AS a ON a.janela_id = j.id
            WHERE j.dispositivo_id = ?
            ORDER BY j.timestamp DESC
            LIMIT ?
            """,
            (dispositivo_id, limite),
        )
        return [dict(row) for row in cursor.fetchall()]

    def fechar(self) -> None:
        self.conn.close()

    def __enter__(self) -> "GerenciadorDados":
        return self

    def __exit__(self, *_: object) -> None:
        self.fechar()
