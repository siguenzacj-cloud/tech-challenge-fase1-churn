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
- [x] Etapa 3 — Engenharia de software e API (refatoração para `src/` concluída; testes e API em andamento)
- [ ] Etapa 4 — Documentação e entrega final

## Como executar a API

A API (FastAPI) carrega o modelo campeão uma única vez na inicialização e expõe a predição de churn
por HTTP. O código fica em `src/churn/api/` (`schemas.py` e `main.py`). O schema de entrada é gerado a partir de `src/churn/config.py`, então a validação
acompanha qualquer mudança nas features.

### 1. Pré-requisitos

```bash
pip install -r requirements.txt        # já inclui fastapi, uvicorn e httpx
python -m churn.train                  # gera models/champion_model.joblib, se ainda não existir
```

### 2. Subir o servidor

A partir da raiz do repositório:

```bash
uvicorn churn.api.main:app --reload
```

- Documentação interativa (Swagger): http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

### 3. Endpoints

| Método | Rota | Descrição |
|---|---|---|
| GET | `/health` | Status da API, se o modelo está carregado e metadados do modelo |
| POST | `/predict` | Prediz um cliente: probabilidade, segmento de risco e ação sugerida |
| POST | `/predict/batch` | Prediz vários clientes (1 a 1000) em uma chamada |

### 4. Exemplo de uso

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Tenure Months": 2,
    "Monthly Charges": 70.7,
    "Total Charges": 151.65,
    "Gender": "Female",
    "Senior Citizen": "No",
    "Partner": "No",
    "Dependents": "No",
    "Phone Service": "Yes",
    "Multiple Lines": "No",
    "Internet Service": "Fiber optic",
    "Online Security": "No",
    "Online Backup": "No",
    "Device Protection": "No",
    "Tech Support": "No",
    "Streaming TV": "No",
    "Streaming Movies": "No",
    "Contract": "Month-to-month",
    "Paperless Billing": "Yes",
    "Payment Method": "Electronic check"
  }'
```

Resposta (formato):

```json
{
  "probability": 0.0,
  "segment": "Retenção agressiva",
  "action": "Produtos de barreira de saída com bonificação de 1 mês de serviço grátis."
}
```

(`probability` é um número entre 0 e 1; o valor acima é só ilustrativo.)

### 5. Validação e erros

| Código | Quando |
|---|---|
| 200 | Predição realizada |
| 422 | Campo ausente, campo desconhecido, valor numérico negativo ou categoria fora da lista aceita (ex.: `"Contract": "Weekly"`) |
| 503 | Modelo não carregado |

Se `models/champion_model.joblib` não existir, o servidor não sobe e orienta a executar
`python -m churn.train`.

### 6. Testes da API

```bash
pytest tests/test_api.py
```

Os testes são pulados automaticamente se o modelo ainda não tiver sido gerado.
