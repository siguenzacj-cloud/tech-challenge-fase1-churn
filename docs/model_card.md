# Model Card — Pipeline Preditivo de Churn

## 1. Visão geral

Este projeto desenvolve um pipeline preditivo para estimar a probabilidade de churn de clientes de uma operadora de telecomunicações.

O modelo é usado como apoio à segmentação da base por nível de risco, ajudando a direcionar ações de retenção de acordo com a probabilidade estimada de cancelamento.

O pipeline foi desenvolvido com Scikit-Learn e disponibilizado para inferência por meio de uma API REST construída com FastAPI.

---

## 2. Uso pretendido

O modelo foi desenvolvido para apoiar a identificação e priorização de clientes com maior propensão ao churn.

A probabilidade estimada é usada para distribuir os clientes em três faixas de risco:

- **Acompanhamento:** probabilidade inferior a 20%;
- **Retenção conservadora:** probabilidade entre 20% e inferior a 50%;
- **Retenção agressiva:** probabilidade igual ou superior a 50%.

Cada faixa está associada a uma estratégia de negócio definida no projeto.

O resultado deve ser usado como apoio à tomada de decisão, e não como uma decisão automática isolada.

---

## 3. Modelo selecionado

O modelo campeão selecionado na etapa de modelagem foi uma **Regressão Logística ajustada**.

Durante os experimentos foram comparadas três famílias de modelos:

- Regressão Logística;
- Random Forest;
- MLPClassifier.

A comparação usou validação cruzada estratificada em cinco folds, métricas de classificação e ajuste de hiperparâmetros.

Os modelos ajustados apresentaram resultados próximos em AUC-ROC. A Regressão Logística foi escolhida pelo equilíbrio entre desempenho preditivo, simplicidade e menor tempo de processamento.

### 3.1 Configuração do modelo campeão

- `C = 100`;
- regularização L2;
- solver `liblinear`;
- `max_iter = 1000`;
- `random_state = 42`.

### 3.2 Rastreabilidade do artefato

- **Scikit-Learn:** 1.9.1;
- **Seed:** 42;
- **Data de treinamento registrada:** 2026-10-12.

O modelo final é armazenado em:

`models/champion_model.joblib`

Os metadados do treinamento são armazenados em:

`models/champion_metadata.json`

---

## 4. Dados e variáveis

O projeto usa o dataset **Telco Customer Churn**, composto por 7.043 clientes e 33 variáveis na base original.

A variável alvo usada pelo modelo é:

`Churn Value`

Para o treinamento são utilizadas 19 features, divididas em três variáveis numéricas e dezesseis categóricas.

### 4.1 Variáveis numéricas

- `Tenure Months`
- `Monthly Charges`
- `Total Charges`

### 4.2 Variáveis categóricas

- `Gender`
- `Senior Citizen`
- `Partner`
- `Dependents`
- `Phone Service`
- `Multiple Lines`
- `Internet Service`
- `Online Security`
- `Online Backup`
- `Device Protection`
- `Tech Support`
- `Streaming TV`
- `Streaming Movies`
- `Contract`
- `Paperless Billing`
- `Payment Method`

As features usadas no pipeline estão centralizadas em `src/churn/config.py`.

Variáveis como `Churn Score`, `CLTV` e `Churn Reason` foram excluídas da modelagem para evitar vazamento de informação ou uso de informações derivadas ou posteriores ao churn.

Variáveis de localização presentes na base original também não são usadas pelo modelo.

---

## 5. Pré-processamento

O pipeline realiza:

- tratamento da variável `Total Charges`, convertendo-a para formato numérico e tratando valores em branco;
- padronização das variáveis numéricas com `StandardScaler`;
- codificação das variáveis categóricas com `OneHotEncoder`.

O pré-processamento faz parte do próprio `Pipeline` do Scikit-Learn, garantindo que as mesmas transformações sejam aplicadas no treinamento e na inferência.

O `OneHotEncoder` usa `handle_unknown="ignore"`, permitindo tratar categorias não observadas durante o treinamento sem interromper a predição.

A separação entre treino e teste é estratificada e usa a seed centralizada do projeto (`SEED = 42`), contribuindo para a consistência entre execuções.

---

## 6. Performance

### 6.1 Validação cruzada

| Métrica | Resultado |
|---|---:|
| AUC-ROC | 0,859 |
| Brier Score | 0,130 |
| Recall com corte de 50% | 0,585 |
| Recall da segmentação | 0,878 |

### 6.2 Conjunto de teste

| Métrica | Resultado |
|---|---:|
| AUC-ROC | 0,848 |
| Brier Score | 0,136 |
| Recall com corte de 50% | 0,567 |
| Recall da segmentação | 0,856 |

Os resultados estão registrados em `models/champion_metadata.json`.

Neste projeto, o **recall da segmentação** representa a proporção de churners reais alcançados por alguma das faixas que recebem ação de retenção, ou seja, clientes com probabilidade estimada de churn igual ou superior a 20%.

No conjunto de teste, essa estratégia alcançou **85,6% dos churners**, enquanto o corte único de 50% alcançou **56,7%**.

Esse ganho de cobertura vem da estratégia de uso do modelo por faixas de risco, e não da adoção de um algoritmo mais complexo.

---

## 7. Interpretação das métricas

O **AUC-ROC** avalia a capacidade do modelo de ordenar clientes de acordo com seu risco de churn. Quanto maior o valor, melhor a capacidade de discriminar clientes com e sem churn.

O **Brier Score** mede o erro das probabilidades estimadas em relação aos resultados observados. Para essa métrica, valores menores indicam melhor qualidade probabilística.

O **recall** mede a proporção dos churners reais que são identificados pela regra de decisão utilizada.

Como o objetivo do projeto é apoiar ações preventivas de retenção, o recall tem relevância direta para o problema de negócio.

As métricas foram analisadas em conjunto, já que nenhuma delas, sozinha, representa toda a qualidade do modelo.

---

## 8. Segmentação de risco

Os cortes definidos pelo projeto são:

| Segmento | Probabilidade | Direcionamento |
|---|---:|---|
| Acompanhamento | < 20% | Nenhuma ação específica |
| Retenção conservadora | 20% a < 50% | Gestão comercial para contratação de produtos de barreira de saída |
| Retenção agressiva | ≥ 50% | Produtos de barreira de saída com bonificação de um mês de serviço grátis |

Na análise feita com as probabilidades out-of-fold da validação cruzada:

| Segmento | Clientes | Churn real | % dos churners |
|---|---:|---:|---:|
| Acompanhamento | 2.962 | 6,1% | 12,2% |
| Retenção conservadora | 1.357 | 32,4% | 29,4% |
| Retenção agressiva | 1.315 | 66,5% | 58,5% |

A faixa de Retenção conservadora representa uma região de maior incerteza: aproximadamente 32,4% dos clientes desse grupo apresentaram churn real.

No conjunto de teste, o comportamento permaneceu semelhante, com taxas reais de churn de aproximadamente:

- **7,3%** em Acompanhamento;
- **32,5%** em Retenção conservadora;
- **62,4%** em Retenção agressiva.

Os resultados mostram uma progressão consistente do risco observado entre as três faixas definidas no projeto.

---

## 9. Limitações

### 9.1 Incerteza na faixa intermediária

O segmento de Retenção conservadora concentra clientes cujo risco não permite uma separação tão clara entre churn e permanência.

As ações para esse grupo devem considerar o custo da intervenção e a possibilidade de abordar clientes que permaneceriam mesmo sem uma ação de retenção.

### 9.2 Janela de observação

Clientes com menor tempo de relacionamento tiveram menos tempo de exposição para apresentar churn.

Esse ponto deve ser considerado na interpretação das probabilidades, principalmente para clientes mais recentes.

### 9.3 Associação não implica causalidade

O modelo identifica padrões associados ao churn, mas não demonstra relações causais.

Uma variável associada ao cancelamento não significa que sua alteração necessariamente reduzirá o churn.

A efetividade das ações de retenção propostas deve ser avaliada por meio de experimentos ou pilotos controlados antes de atribuir relação causal entre a intervenção e a redução do churn.

### 9.4 Dependência dos dados históricos

O modelo foi desenvolvido a partir dos padrões presentes no dataset utilizado.

Mudanças no comportamento dos clientes, produtos, preços, políticas comerciais ou composição da população podem alterar sua performance.

Em uma eventual utilização contínua, recomenda-se acompanhar o desempenho e avaliar a necessidade de recalibração ou retreinamento.

### 9.5 Generalização

As métricas apresentadas refletem o comportamento do modelo no conjunto de dados usado no projeto.

O desempenho observado não garante resultados equivalentes em outras populações, períodos ou contextos de negócio sem uma nova validação.

---

## 10. Auditoria de vieses

A etapa de modelagem incluiu uma auditoria com Fairlearn para observar possíveis diferenças na distribuição das ações de retenção entre grupos sensíveis.

Foram avaliadas as variáveis:

- `Gender`;
- `Senior Citizen`.

A análise usou as probabilidades out-of-fold obtidas na validação cruzada do conjunto de treinamento.

Foram observados:

- taxa real de churn;
- probabilidade média prevista;
- percentual de clientes direcionados para alguma ação;
- recall entre churners que receberiam alguma ação;
- percentual direcionado à Retenção agressiva;
- diferença de equalized odds.

### 10.1 Gênero

Os resultados entre os grupos de gênero ficaram próximos:

| Métrica | Female | Male |
|---|---:|---:|
| Churn real | 26,6% | 26,4% |
| Probabilidade média prevista | 26,7% | 26,4% |
| Clientes com alguma ação | 47,3% | 47,5% |
| Recall com ação | 88,7% | 86,9% |
| Retenção agressiva | 24,0% | 22,7% |

A diferença de equalized odds observada foi de **0,018**.

### 10.2 Senior Citizen

Foram observadas diferenças maiores entre clientes classificados ou não como idosos:

| Métrica | Não idosos | Idosos |
|---|---:|---:|
| Churn real | 23,7% | 41,1% |
| Probabilidade média prevista | 23,7% | 41,2% |
| Clientes com alguma ação | 42,8% | 71,2% |
| Recall com ação | 85,7% | 94,2% |
| Retenção agressiva | 19,5% | 43,3% |

A diferença de equalized odds observada foi de **0,257**.

As probabilidades médias previstas ficaram próximas das taxas reais de churn observadas nos dois grupos.

Ao mesmo tempo, a proporção de clientes idosos direcionados às ações de retenção foi maior. Essa diferença deve continuar documentada e ser monitorada em análises futuras, principalmente em uma eventual aplicação do modelo fora do contexto acadêmico.

A auditoria registra diferenças observadas entre os grupos, mas, sozinha, não determina se essas diferenças representam tratamento inadequado ou discriminatório.

Os resultados também precisam ser interpretados considerando as diferenças de prevalência de churn existentes entre os grupos.

---

## 11. API de inferência

O modelo é disponibilizado para inferência por meio de uma API REST desenvolvida com FastAPI.

A implementação está localizada em:

`src/churn/api/`

Os principais componentes são:

- `src/churn/api/main.py`;
- `src/churn/api/schemas.py`.

A API disponibiliza endpoints para:

- verificação de saúde da aplicação;
- predição individual;
- predição em lote.

O endpoint `/health` permite verificar se a aplicação está ativa e se o modelo foi carregado corretamente.

O endpoint `/predict` recebe os dados de um cliente e retorna:

- probabilidade estimada de churn;
- segmento de risco;
- ação recomendada de acordo com a segmentação definida no projeto.

O endpoint `/predict/batch` permite realizar a predição de vários clientes em uma única requisição.

A validação dos dados de entrada rejeita situações como:

- categorias não aceitas;
- campos obrigatórios ausentes;
- valores numéricos incompatíveis;
- campos desconhecidos;
- lotes vazios.

---

## 12. Testes e asseguramento de qualidade

O projeto possui testes automatizados organizados em:

- `tests/test_api.py`;
- `tests/test_data.py`;
- `tests/test_predict.py`.

### 12.1 Preparação dos dados

Os testes verificam:

- conversão de `Total Charges` para formato numérico;
- tratamento de valores em branco;
- preservação do DataFrame original;
- seleção apenas das features configuradas;
- reprodutibilidade e estratificação da divisão entre treino e teste.

### 12.2 Predição e segmentação

Os testes verificam:

- limites das faixas de segmentação;
- comportamento nos pontos de corte de 20% e 50%;
- rejeição de probabilidades fora do intervalo entre 0 e 1;
- ausência de features obrigatórias;
- estrutura da saída das predições;
- consistência entre probabilidade, segmento e ação.

### 12.3 API

Os testes verificam:

- validade dos valores categóricos usados no exemplo;
- endpoint `/health`;
- endpoint `/predict`;
- consistência entre probabilidade, segmento e ação;
- rejeição de categoria inválida;
- rejeição de campo obrigatório ausente;
- rejeição de valor negativo incompatível;
- rejeição de campo desconhecido;
- predição em lote;
- rejeição de lote vazio.

Depois da integração das contribuições do grupo na branch `main`, a suíte completa foi executada com:

**25 testes aprovados e nenhuma falha.**

Em seguida, o pipeline de treinamento foi executado novamente.

Após o treinamento, a suíte completa foi rodada mais uma vez e permaneceu com:

**25 testes aprovados e nenhuma falha.**

Durante a execução apareceu um aviso de depreciação relacionado ao `TestClient` usado pelo FastAPI/Starlette e à biblioteca `httpx`. O aviso não gerou falhas nos testes e não impediu a validação funcional do projeto.

---

## 13. Reprodutibilidade

O projeto usa versões fixadas das principais dependências em `requirements.txt` e `pyproject.toml`.

Os parâmetros de treinamento estão centralizados em `src/churn/config.py`, incluindo:

`SEED = 42`

A reprodutibilidade foi validada em um novo clone do repositório, com criação de ambiente virtual independente e reinstalação das dependências.

Após um novo treinamento, foram reproduzidos os resultados registrados para o modelo campeão:

- AUC-ROC de validação cruzada: **0,859**;
- AUC-ROC de teste: **0,848**;
- Brier Score de teste: **0,136**;
- Recall da segmentação no teste: **0,856**.

Essa validação mostrou que, no ambiente testado, as dependências, configurações e seed do projeto permitem reproduzir os resultados registrados.

Depois da integração final das contribuições do grupo na branch `main`, o pipeline de treinamento foi executado novamente e reproduziu as mesmas métricas:

- AUC-ROC de validação cruzada: **0,859**;
- AUC-ROC de teste: **0,848**;
- Brier Score de teste: **0,136**;
- Recall da segmentação no teste: **0,856**.

A suíte automatizada também foi executada antes e depois desse treinamento, com **25 testes aprovados nas duas execuções**.

Com essas execuções, validamos a reprodutibilidade em ambiente limpo e o funcionamento da versão integrada do projeto.

---

## 14. Considerações de uso

As probabilidades fornecidas pelo modelo representam estimativas de risco e não previsões determinísticas.

Para uma eventual utilização prática, recomenda-se:

- acompanhar métricas técnicas e de negócio;
- monitorar mudanças na distribuição dos dados;
- revisar periodicamente diferenças de desempenho e direcionamento de ações entre grupos;
- validar as ações de retenção por meio de testes controlados;
- acompanhar a qualidade das probabilidades estimadas;
- reavaliar o modelo caso o contexto de negócio ou a população atendida se altere;
- realizar uma nova validação antes de usar o modelo em uma população diferente daquela representada no dataset do projeto.

O modelo e a segmentação proposta devem ser usados como ferramentas de apoio, mantendo supervisão humana nas decisões de negócio.

---

## 15. Referências internas

### Análise e modelagem

- `notebooks/01_eda.ipynb`
- `notebooks/02_modelagem.ipynb`

### Modelo e metadados

- `models/champion_model.joblib`
- `models/champion_metadata.json`

### Pipeline

- `src/churn/config.py`
- `src/churn/data.py`
- `src/churn/features.py`
- `src/churn/predict.py`
- `src/churn/train.py`

### API

- `src/churn/api/__init__.py`
- `src/churn/api/main.py`
- `src/churn/api/schemas.py`

### Testes

- `tests/test_api.py`
- `tests/test_data.py`
- `tests/test_predict.py`

### Configuração e dependências

- `pyproject.toml`
- `requirements.txt`