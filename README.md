# Tech Challenge Fase 1 — Pipeline Preditivo de Churn

Pipeline de classificação para prever o risco de cancelamento (churn) de clientes de uma operadora
de telecomunicações, desde a análise exploratória até uma API REST de inferência. Projeto para o
Tech Challenge Fase 1 (FIAP POSTECH — Machine Learning Engineering).

O modelo não responde apenas "cancela / não cancela": devolve a **probabilidade de churn**, o
**segmento de risco** e a **ação de retenção sugerida** para cada cliente.

## Equipe

| Integrante | GitHub | Responsabilidades |
|---|---|---|
| Cesar Siguenza | [@siguenzacj-cloud](https://github.com/siguenzacj-cloud) | EDA, ML Canvas, modelagem, segmentação de risco, refatoração para `src/` |
| Sabrina | [@brina-eng](https://github.com/brina-eng) | API de inferência (FastAPI) |
| Juliana Tomazi | [@julianastomazi](https://github.com/julianastomazi) | Testes, reprodutibilidade, Model Card e organização final do projeto |

Guia de handoff para a equipe: [`docs/guia_equipe.md`](docs/guia_equipe.md).

## Estrutura do projeto

```
data/                  dataset Telco Customer Churn (IBM)
notebooks/             experimentação
  01_eda.ipynb         Etapa 1 — EDA e baseline de Regressão Logística
  02_modelagem.ipynb   Etapa 2 — comparação LogReg / Random Forest / MLP, calibração, segmentação, Fairlearn
src/churn/             código produtivo (pacote instalável)
  config.py            fonte única: rotas, seed, features, hiperparâmetros, segmentos de risco
  data.py              carga, limpeza e split dos dados
  features.py          pré-processamento (StandardScaler + OneHotEncoder)
  train.py             treina o modelo campeão e salva modelo + metadados
  predict.py           probabilidade → segmento → ação
models/                champion_model.joblib + champion_metadata.json
tests/                 testes automatizados (pytest)
docs/                  ML Canvas, registro de experimentos, guia da equipe
```

## Como começar

Pré-requisitos: **Python 3.12** e **Git**.

```bash
git clone https://github.com/siguenzacj-cloud/tech-challenge-fase1-churn.git
cd tech-challenge-fase1-churn

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

pip install -r requirements.txt   # versões fixadas + pacote churn em modo editável
```

Verificação (regenera o modelo a partir do CSV e mostra as métricas):

```bash
python -m churn.train
```

Resultado esperado no teste: **AUC-ROC 0,848** e **recall da segmentação 0,856**.

## Resultados (Etapa 2)

Modelo campeão: **Regressão Logística ajustada** (`C=100`, L2), escolhida frente a Random Forest e
MLP por empate técnico em AUC-ROC (~0,86) e melhor equilíbrio entre desempenho, tempo e
simplicidade. Detalhes em `notebooks/02_modelagem.ipynb` e `docs/experimentos.md`.

| Segmento | Probabilidade de churn | Ação |
|---|---|---|
| Acompanhamento | < 20% | Nenhuma ação específica |
| Retenção conservadora | 20% a < 50% | Produtos de barreira de saída (contrato, serviço, dependentes) |
| Retenção agressiva | ≥ 50% | Produtos de barreira de saída + 1 mês grátis |

## Como colaborar

- Ninguém trabalha direto na `main`: cada tarefa em uma branch (`feature/api`, `feature/testes`, ...).
- Ao terminar, abrir um **Pull Request**; outra pessoa da equipe revisa antes do merge.
- Notebooks: apenas uma pessoa edita cada notebook por vez (conflitos em `.ipynb` são difíceis de resolver).
- Mensagens de commit em português, descrevendo o que mudou.

## Status do projeto

- [x] Etapa 1 — Entendimento e preparação
- [x] Etapa 2 — Modelagem e avaliação
- [ ] Etapa 3 — Engenharia de software e API (refatoração para `src/` concluída; testes e API em andamento)
- [ ] Etapa 4 — Documentação e entrega final

## Como executar a API

_Pendente: será documentado junto com a implementação da API (Etapa 3)._
