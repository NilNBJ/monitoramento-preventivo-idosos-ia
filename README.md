# Monitoramento Preventivo de Idosos com Isolation Forest

Protótipo acadêmico desenvolvido para avaliar a viabilidade técnica de detecção de anomalias comportamentais em ambiente residencial simulado. O projeto utiliza janelas temporais sintéticas e o algoritmo `IsolationForest`.

## Escopo

- Pré-processamento de cinco atributos comportamentais.
- Treinamento não supervisionado com dados sintéticos de rotina normal.
- Classificação por `predict()` e `decision_function()` do Scikit-Learn.
- Persistência local de janelas, alertas e metadados do modelo em SQLite.
- Exemplo executável com geração de dados sintéticos.

O protótipo não foi validado com sensores físicos, pessoas idosas ou residências reais. O adaptador de notificação utilizado no experimento foi um mock local; não houve confirmação real de entrega pelo WhatsApp.

## Critério de alerta

O retorno `-1` de `predict()` identifica uma janela anômala. Na escala de `decision_function()`, valores negativos indicam outliers e valores menores representam maior anormalidade. Alertas com score inferior a `-0.25` recebem severidade alta; os demais outliers recebem severidade média.

## Requisitos

- Python 3.12
- NumPy 1.26 ou superior
- pandas 2.1 ou superior
- scikit-learn 1.4 ou superior

## Execução

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python exemplo.py
```

O exemplo cria uma base sintética de treinamento, avalia uma janela artificialmente anômala e grava o resultado no arquivo local `pipeline_anomalias.db`.

## Estrutura

```text
.
├── exemplo.py
├── requirements.txt
└── src/
    └── monitoramento_idosos/
        ├── dominio.py
        ├── modelo.py
        └── persistencia.py
```

## Uso acadêmico

Este repositório acompanha o TCC “Sistema Inteligente com Inteligência Artificial para Monitoramento Preventivo de Idosos em Ambiente Residencial”, de Nilson Barreto de Jesus, MBA em Engenharia de Software da USP/ESALQ.

## Licença

Distribuído sob a licença MIT.
