# Tech Challenge Fase 1 — Pipeline Preditivo de Churn

Pipeline de classificação para prever o risco de cancelamento (churn) de clientes,
desde a análise exploratória até uma API REST de inferência. Projeto para o
Tech Challenge Fase 1 (FIAP POSTECH — Machine Learning Engineering).

## Estrutura do projeto
- `data/` — dataset (Telco Customer Churn)
- `notebooks/` — EDA e experimentação
- `src/` — código de pré-processamento e modelo, refatorado
- `models/` — modelo treinado (`.pkl`/`.joblib`)
- `tests/` — testes automatizados (pytest)
- `docs/` — ML Canvas, Model Card

## Setup do ambiente
```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Status do projeto
- [x] Etapa 1 — Entendimento e preparação (em andamento)
- [ ] Etapa 2 — Modelagem e avaliação
- [ ] Etapa 3 — Engenharia de software e API
- [ ] Etapa 4 — Documentação e entrega final

## Como executar (será completado na Etapa 3)
_Pendente: instruções da API após implementação._
