# Model Card — Pipeline Preditivo de Churn

## 1. Visão geral

Este projeto desenvolve um pipeline preditivo para estimar a probabilidade de churn de clientes de uma operadora de telecomunicações.

O modelo é utilizado como apoio à segmentação da base por nível de risco, permitindo direcionar ações de retenção de acordo com a probabilidade estimada de cancelamento.

O pipeline foi desenvolvido com Scikit-Learn e disponibilizado para inferência por meio de uma API REST construída com FastAPI.

## 2. Uso pretendido

O modelo foi desenvolvido para apoiar a identificação e priorização de clientes com maior propensão ao churn.

A probabilidade estimada é utilizada para distribuir os clientes em três faixas de risco:

- **Acompanhamento:** probabilidade inferior a 20%;
- **Retenção conservadora:** probabilidade entre 20% e 49%;
- **Retenção agressiva:** probabilidade igual ou superior a 50%.

Cada faixa está associada a uma estratégia de negócio definida pelo projeto.

O resultado deve ser utilizado como ferramenta de apoio à tomada de decisão, e não como decisão automática isolada.

## 3. Modelo selecionado

O modelo campeão selecionado na etapa de modelagem foi uma **Regressão Logística ajustada**.

Durante a experimentação foram comparadas três famílias de modelos:

- Regressão Logística;
- Random Forest;
- MLPClassifier.

A comparação utilizou validação cruzada estratificada em cinco folds, métricas de classificação e ajuste de hiperparâmetros.

Os modelos ajustados apresentaram resultados próximos em AUC-ROC. A Regressão Logística foi selecionada pelo equilíbrio entre desempenho preditivo, simplicidade e menor tempo de processamento.

### Configuração do modelo campeão

- `C = 100`
- regularização L2;
- solver `liblinear`;
- `max_iter = 1000`;
- `random_state = 42`.

Informações de rastreabilidade do artefato atual:

- **Scikit-Learn:** 1.9.1;
- **Seed:** 42;
- **Data de treinamento registrada:** 2026-10-12.

O modelo final é armazenado em:

`models/champion_model.joblib`

Os metadados do treinamento são armazenados em:

`models/champion_metadata.json`

## 4. Dados e variáveis

O projeto utiliza o dataset **Telco Customer Churn**, composto por 7.043 clientes e 33 variáveis na base original.

A variável alvo utilizada pelo modelo é:

`Churn Value`

Para o treinamento são utilizadas 19 features, divididas em três variáveis numéricas e dezesseis categóricas.

### Variáveis numéricas

- `Tenure Months`
- `Monthly Charges`
- `Total Charges`

### Variáveis categóricas

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

As features utilizadas no pipeline estão centralizadas em `src/churn/config.py`.

Variáveis excluídas da modelagem para evitar vazamento de informação ou utilização de informações derivadas ou posteriores ao churn incluem `Churn Score`, `CLTV` e `Churn Reason`.

## 5. Pré-processamento

O pipeline realiza:

- tratamento da variável `Total Charges`, convertendo-a para formato numérico e tratando valores em branco;
- padronização das variáveis numéricas com `StandardScaler`;
- codificação das variáveis categóricas com `OneHotEncoder`.

O pré-processamento faz parte do próprio `Pipeline` do Scikit-Learn, mantendo as mesmas transformações entre treinamento e inferência.

O `OneHotEncoder` utiliza `handle_unknown="ignore"`, permitindo tratar categorias não observadas durante o treinamento sem interromper a predição.

## 6. Performance

### Validação cruzada

| Métrica | Resultado |
|---|---:|
| AUC-ROC | 0,859 |
| Brier Score | 0,130 |
| Recall com corte de 50% | 0,585 |
| Recall da segmentação | 0,878 |

### Conjunto de teste

| Métrica | Resultado |
|---|---:|
| AUC-ROC | 0,848 |
| Brier Score | 0,136 |
| Recall com corte de 50% | 0,567 |
| Recall da segmentação | 0,856 |

Os resultados estão registrados em `models/champion_metadata.json`.

Neste projeto, **recall da segmentação** representa a proporção de churners reais alcançados por alguma das faixas que recebem ação de retenção, isto é, clientes com probabilidade estimada de churn igual ou superior a 20%.

No conjunto de teste, essa estratégia alcançou **85,6% dos churners**, enquanto o corte único de 50% alcançou **56,7%**.

O ganho de cobertura decorre da estratégia de utilização do modelo por faixas de risco, e não da adoção de um algoritmo mais complexo.

## 7. Interpretação das métricas

O **AUC-ROC** avalia a capacidade do modelo de ordenar clientes de acordo com seu risco de churn. Valores maiores indicam melhor capacidade de discriminação entre clientes com e sem churn.

O **Brier Score** mede o erro das probabilidades estimadas em relação aos resultados observados. Para essa métrica, valores menores representam melhor qualidade probabilística.

O **recall** mede a proporção dos churners reais que são identificados pela regra de decisão utilizada.

Como o objetivo do projeto é apoiar ações preventivas de retenção, o recall possui relevância direta para o problema de negócio.

## 8. Segmentação de risco

Os cortes definidos pelo projeto são:

| Segmento | Probabilidade | Direcionamento |
|---|---:|---|
| Acompanhamento | < 20% | Nenhuma ação específica |
| Retenção conservadora | 20% a < 50% | Gestão comercial para contratação de produtos de barreira de saída |
| Retenção agressiva | ≥ 50% | Produtos de barreira de saída com bonificação de um mês de serviço grátis |

Na análise realizada com as probabilidades out-of-fold da validação cruzada:

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

## 9. Limitações

### Incerteza na faixa intermediária

O segmento de Retenção conservadora concentra clientes cujo risco não permite uma separação tão clara entre churn e permanência.

As ações destinadas a esse grupo devem considerar o custo da intervenção e a possibilidade de abordar clientes que permaneceriam mesmo sem uma ação de retenção.

### Janela de observação

Clientes com menor tempo de relacionamento tiveram menor período de exposição para apresentar churn.

Esse aspecto deve ser considerado na interpretação das probabilidades, especialmente para clientes recentes.

### Associação não implica causalidade

O modelo identifica padrões associados ao churn, mas não demonstra relações causais.

Uma variável associada ao cancelamento não significa que sua alteração necessariamente reduzirá o churn.

A efetividade das ações de retenção propostas deve ser avaliada por meio de experimentos ou pilotos controlados antes que seja atribuída relação causal entre a intervenção e a redução do churn.

### Dependência dos dados históricos

O modelo foi desenvolvido a partir dos padrões presentes no dataset utilizado.

Mudanças no comportamento dos clientes, produtos, preços, políticas comerciais ou composição da população podem alterar sua performance.

Em eventual utilização contínua, recomenda-se monitorar o desempenho e avaliar a necessidade de recalibração ou retreinamento.

## 10. Auditoria de vieses

A etapa de modelagem incluiu uma auditoria com Fairlearn para avaliar possíveis diferenças na distribuição das ações de retenção entre grupos sensíveis.

Foram avaliadas as variáveis:

- `Gender`;
- `Senior Citizen`.

A análise utilizou as probabilidades out-of-fold obtidas na validação cruzada do conjunto de treinamento.

Foram observados:

- taxa real de churn;
- probabilidade média prevista;
- percentual de clientes direcionados para alguma ação;
- recall entre churners que receberiam alguma ação;
- percentual direcionado à Retenção agressiva;
- diferença de equalized odds.

### Gênero

Os resultados entre os grupos de gênero ficaram próximos:

| Métrica | Female | Male |
|---|---:|---:|
| Churn real | 26,6% | 26,4% |
| Probabilidade média prevista | 26,7% | 26,4% |
| Clientes com alguma ação | 47,3% | 47,5% |
| Recall com ação | 88,7% | 86,9% |
| Retenção agressiva | 24,0% | 22,7% |

A diferença de equalized odds observada foi de **0,018**.

### Senior Citizen

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

Ao mesmo tempo, a proporção de clientes idosos direcionados às ações de retenção foi maior. Essa diferença deve permanecer documentada e ser monitorada em auditorias futuras, principalmente em eventual aplicação do modelo fora do contexto acadêmico.

A auditoria registra diferenças observadas entre os grupos, mas, isoladamente, não determina se essas diferenças representam tratamento inadequado ou discriminatório.

## 11. Reprodutibilidade

O projeto utiliza versões fixadas das principais dependências em `requirements.txt` e `pyproject.toml`.

Os parâmetros de treinamento são centralizados em `src/churn/config.py`, incluindo:

`SEED = 42`

A reprodutibilidade foi validada em um novo clone do repositório, com criação de ambiente virtual independente e reinstalação das dependências.

Após um novo treinamento, foram reproduzidos os resultados registrados para o modelo campeão:

- AUC-ROC de validação cruzada: **0,859**;
- AUC-ROC de teste: **0,848**;
- Brier Score de teste: **0,136**;
- Recall da segmentação no teste: **0,856**.

A validação confirma que, no ambiente testado, as dependências, configurações e seed definidas no projeto permitem reproduzir os resultados registrados.

Após a integração final das contribuições do grupo na branch `main`, a suíte completa de testes deverá ser executada novamente para validar o funcionamento conjunto dos componentes.

## 12. Considerações de uso

As probabilidades fornecidas pelo modelo representam estimativas de risco e não previsões determinísticas.

Para eventual utilização prática, recomenda-se:

- acompanhar métricas técnicas e de negócio;
- monitorar alterações na distribuição dos dados;
- revisar periodicamente diferenças de desempenho e direcionamento de ações entre grupos;
- validar as ações de retenção por meio de testes controlados;
- reavaliar o modelo caso o contexto de negócio ou a população atendida se altere.

## 13. Referências internas

- `notebooks/01_eda.ipynb`
- `notebooks/02_modelagem.ipynb`
- `models/champion_model.joblib`
- `models/champion_metadata.json`
- `src/churn/config.py`
- `src/churn/data.py`
- `src/churn/features.py`
- `src/churn/predict.py`
- `src/churn/train.py`