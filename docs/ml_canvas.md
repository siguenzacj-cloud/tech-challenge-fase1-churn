# ML Canvas — Pipeline Preditivo de Churn

> Framing de negócio do problema, antes de tocar código de modelagem.
> Tech Challenge Fase 1 — FIAP POSTECH.

## 1. Objetivo de negócio
*Pergunta-guia: que decisão de negócio muda se o modelo funcionar?*

Identificar de forma preventiva os clientes com maior risco real de cancelar o serviço, por meio
de um modelo com um recall objetivo de referência entre 0,70 e 0,80 (benchmark de indústria para
problemas de churn, a ser validado com os dados próprios após o treinamento do modelo baseline),
para que a equipe de clientes possa desenhar campanhas de retenção focadas e medir seu impacto na
redução da taxa de cancelamento atual (~26,5%).

## 2. Predição central
*Pergunta-guia: o que o modelo prediz, para quem, com que frequência?*

O modelo classifica mensalmente cada cliente ativo de acordo com sua probabilidade de
cancelamento em três níveis de risco — alto, médio e baixo — e gera uma base de gestão que é
entregue à gerência de clientes.

> **Nota técnica (a revisar na Etapa 2/3):** para gerar 3 níveis de risco em vez de uma simples
> classificação binária, o modelo precisa retornar uma **probabilidade** por cliente
> (`predict_proba`), não apenas sim/não. Os cortes exatos entre alto/médio/baixo risco ainda
> não estão definidos — ficam para quando o modelo estiver treinado.

## 3. Dados de entrada disponíveis
*Pergunta-guia: quais colunas estão disponíveis no momento real de prever — não depois?*

Ficam excluídas por desenho: `CustomerID` e `Count` (identificadores/contadores sem valor
preditivo); e `Churn Score`, `CLTV` e `Churn Reason` — as duas primeiras porque são saída de
**outro modelo preditivo** (IBM SPSS Modeler / fórmulas corporativas), não dados brutos do
cliente; a terceira porque só existe para quem já cancelou — um exemplo direto de vazamento de
dados (*data leakage*). `Churn Label` e `Churn Value` são o target, não um input.

Todas as demais colunas (geografia, dados demográficos, antiguidade/contrato, serviços
contratados, faturamento) ficam como **candidatas disponíveis** no momento da predição — a
decisão sobre quais realmente usar no modelo depende da EDA (Passo 4-5), não desta seção.

## 4. Ações de negócio
*Pergunta-guia: se o modelo diz "alto risco de churn", o que a empresa faz com essa informação?*

A equipe de clientes segmenta os benefícios de retenção de acordo com o nível de risco: quanto
maior o risco de cancelamento, maior o benefício oferecido. O contato é realizado via SMS,
e-mail ou WhatsApp.

## 5. Custo dos erros
*Pergunta-guia: o que é pior — contatar alguém que não ia cancelar (falso positivo) ou não detectar quem realmente ia cancelar (falso negativo)?*

Um falso negativo (não detectar um cliente que realmente ia cancelar) é considerado mais custoso
que um falso positivo (oferecer um benefício a um cliente que não ia sair), pois perder um
cliente implica o custo de atrair um novo — geralmente mais alto que o custo de retenção
(princípio amplamente aceito em gestão de clientes, embora o valor exato do custo de aquisição
não esteja disponível neste dataset e fique fora do escopo deste projeto). O custo dos falsos
positivos é mitigado pela estratificação por nível de risco: cada faixa (alto/médio/baixo) pode
ter um orçamento de retenção próprio, controlando o gasto total mesmo quando o modelo erra para
o lado do falso positivo. Essa priorização de recall sobre precisão será formalizada na escolha
da métrica técnica (Passo 6).

## 6. Métricas de sucesso
*Pergunta-guia: métrica offline (F1/recall/AUC — será definida no Passo 6) e métrica de negócio (ex.: % de clientes retidos, economia estimada).*

**Métrica técnica (offline):** recall entre 0,70 e 0,80, como referência de benchmark de
indústria — a ser validado com os dados próprios após o treinamento do modelo baseline (ver
seção 1).

**Métrica de negócio:** redução da taxa de cancelamento a partir do patamar atual de ~26,5%
(1.869 de 7.043 clientes) — apresentada como hipótese de negócio a validar, já que a conversão
real depende da efetividade da campanha de retenção (ver seções 1 e 5).

## 7. Riscos e vieses
*Pergunta-guia: alguma variável pode discriminar injustamente (ex.: gênero, tipo de contrato ligado a nível socioeconômico)?*

As variáveis `Gender`, `Senior Citizen` e as de localização (`City`, `State`, `Zip Code`) podem
funcionar como proxies de características sensíveis (gênero, idade, nível socioeconômico). O
risco principal não é um tratamento punitivo — o objetivo do modelo é gerar um benefício de
retenção, não negar algo que o cliente já tem —, mas sim uma **distribuição desigual desse
benefício**: se o modelo subestimar sistematicamente o risco real de um grupo, esse grupo
receberia sistematicamente menos benefícios de retenção que outro grupo com o mesmo risco real.
**Fairlearn** será aplicado na Etapa 2/3, uma vez que exista um modelo treinado, para auditar
taxas de seleção e erro entre grupos.
