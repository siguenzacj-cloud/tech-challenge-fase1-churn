# Guia da equipe — handoff da Etapa 3

Documento para a Sabrina (API) e a Juliana (testes, reprodutibilidade, Model Card e organização
final) começarem a partir do que já está pronto. Setup do ambiente: seção "Como começar" do
`README.md`.

## 1. Onde estamos

- **Etapas 1 e 2 concluídas:** EDA, ML Canvas (`docs/ml_canvas.md`), comparação de modelos e
  escolha do campeão (`notebooks/02_modelagem.ipynb`, `docs/experimentos.md`).
- **Etapa 3 — refatoração concluída:** o código produtivo está em `src/churn/` e reproduz
  exatamente o modelo do notebook (mesmas probabilidades no teste).
- **Falta:** testes (pytest), API (FastAPI), Model Card, README final e vídeo STAR.

Entrega: **terça-feira, 3 de novembro de 2026**.

## 2. Arquitetura do código (`src/churn/`)

```
config.py   → fonte única de verdade: rotas, SEED, features e valores aceitos,
              hiperparâmetros, segmentos de risco e ações
data.py     → load_raw_data, fix_total_charges, split_features_target, split_train_test, load_dataset
features.py → build_preprocessor (StandardScaler + OneHotEncoder)
train.py    → build_model, evaluate, train  (python -m churn.train)
predict.py  → load_model, assign_segment, predict_customers, predict_customer
```

Princípios adotados (vale manter):

- **Fonte única de verdade:** cortes, segmentos e ações existem só em `config.py`. O
  `champion_metadata.json` é apenas um registro de como o modelo foi treinado; nenhum código lê os
  cortes de lá.
- **`predict.py` não conhece a API:** recebe e devolve estruturas simples (dict / DataFrame). A
  tradução entre o formato da API e esses dados acontece só dentro de `api/`.
- **Carga explícita do modelo:** `load_model()` é chamada uma vez, ao iniciar a API (não na
  importação do módulo). Assim os testes podem usar outro modelo.
- **Teste do desenho:** mudar um corte → só `config.py`; trocar de framework web → só `api/`;
  re-treinar → só rodar `python -m churn.train`.

## 3. Sabrina — API de inferência

### Como usar a lógica que já existe

```python
from churn.predict import load_model, predict_customer

model = load_model()  # carregar uma vez, no startup da API
cliente = {
    "Tenure Months": 2, "Monthly Charges": 70.7, "Total Charges": 151.65,
    "Gender": "Female", "Senior Citizen": "No", "Partner": "No", "Dependents": "No",
    "Phone Service": "Yes", "Multiple Lines": "No", "Internet Service": "Fiber optic",
    "Online Security": "No", "Online Backup": "No", "Device Protection": "No",
    "Tech Support": "No", "Streaming TV": "No", "Streaming Movies": "No",
    "Contract": "Month-to-month", "Paperless Billing": "Yes", "Payment Method": "Electronic check",
}
predict_customer(model, cliente)
# → {"probability": 0.698, "segment": "Retenção agressiva", "action": "..."}
```

As chaves do dict são os **nomes originais das colunas do dataset** (com espaços). Os valores
aceitos por variável categórica estão em `config.CATEGORICAL_VALUES`.

### O que construir

Estrutura prevista: `src/churn/api/schemas.py` (pydantic) e `src/churn/api/main.py` (FastAPI).

- [ ] `GET /health`: status da API e se o modelo está carregado
- [ ] `POST /predict`: recebe as 19 features do cliente e devolve probabilidade + segmento + ação
- [ ] Validação com pydantic: categóricas restritas aos valores de `config.CATEGORICAL_VALUES`,
      numéricas >= 0; dado inválido → erro 422 com mensagem clara
- [ ] Modelo carregado uma única vez no startup (lifespan)
- [ ] Exemplo de cliente pré-carregado na documentação automática (`/docs`)
- [ ] Seção "Como executar a API" no README (`uvicorn churn.api.main:app --reload`)

### Decisões em aberto (combinar com a equipe)

1. Nomes dos campos de entrada/saída: inglês (padrão de código) ou português (coerente com o
   restante do entregável)? Os nomes do dataset têm espaços, então é preciso mapear (`alias` no pydantic).
2. A resposta inclui o texto da ação de negócio ou só o nome do segmento?
3. Extras opcionais: endpoint em lote (`/predict/batch`, útil para a base de gestão mensal do ML
   Canvas), logging das predições, `Dockerfile`.

## 4. Juliana — testes, reprodutibilidade, Model Card e organização

### Testes (pytest) — mínimo exigido: 1 de pré-processamento + 1 de status da API

- [ ] `tests/test_data.py`: `fix_total_charges` converte brancos em 0 e o tipo fica numérico ← **exigido**
- [ ] `tests/test_predict.py`: `assign_segment` nas bordas (0,19 / 0,20 / 0,49 / 0,50) e
      probabilidade fora de [0, 1] gera erro; `predict_customer` devolve probabilidade em [0, 1]
- [ ] `tests/test_api.py` (depois da API): `/health` responde 200 ← **exigido**; `/predict` com
      cliente válido devolve os 3 campos; com dado inválido devolve 422
- [ ] `pytest` todo verde

### Reprodutibilidade

- Versões fixadas em `requirements.txt` e `pyproject.toml`; seed única em `config.SEED` (42).
- Verificação: em um clone limpo, `pip install -r requirements.txt` → `python -m churn.train` →
  AUC-ROC 0,848 no teste.

### Model Card (`docs/model_card.md`) — performance, limitações e vieses

Conteúdo já levantado nas Etapas 1 e 2:

- **Performance:** validação cruzada AUC-ROC 0,859 ± 0,013, Brier 0,130; teste AUC-ROC 0,848,
  Brier 0,136, recall da segmentação 85,6% (objetivo 70–80%). Fonte: `models/champion_metadata.json`.
- **Uso pretendido:** segmentação mensal da base em 3 faixas de risco para ações de retenção.
- **Limitações:**
  - Faixa cinzenta: o segmento "Retenção conservadora" tem ~32% de churn real, e o modelo não
    distingue com certeza quem vai sair.
  - Viés de janela de observação: clientes com pouco tempo de contrato tiveram menos tempo de
    exposição para cancelar no trimestre (ver ML Canvas, seção 7).
  - As variáveis mais importantes estão associadas ao churn, mas isso não prova que mudá-las
    retenha o cliente (recomendado testar as ações em um piloto).
  - O ganho de recall (0,57 → 0,86) vem da segmentação, não de um algoritmo mais complexo.
- **Vieses (Fairlearn, seção 7 do notebook 02):** sem diferenças por gênero; idosos recebem mais
  ações porque seu churn real é maior (41% × 24%), com probabilidades bem calibradas nos dois
  grupos. É uma decisão de caráter social, registrada para auditoria.

### Organização final

- [ ] README final (instalação, execução da API, testes, resultados)
- [ ] Revisar a estrutura de pastas e o `.gitignore`
- [ ] Conferir se os notebooks rodam do início ao fim
