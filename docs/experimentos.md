# Registro de Experimentos — Etapa 2

Validação cruzada estratificada (5 folds) no conjunto de treino. Valores: média ± desvio-padrão. Métrica de seleção: AUC-ROC.

| Modelo | AUC-ROC | Recall | Precision | F1 | Tempo treino (s) | Hiperparâmetros |
|---|---|---|---|---|---|---|
| Regressão Logística (baseline) | 0.859 ± 0.013 | 0.582 ± 0.042 | 0.667 ± 0.024 | 0.621 ± 0.029 | 0.03 | padrão (C=1, L2), sem class_weight |
| Regressão Logística (ajustada) | 0.859 ± 0.013 | 0.585 ± 0.043 | 0.665 ± 0.025 | 0.621 ± 0.031 | 0.03 | GridSearchCV: C=100, l1_ratio=0 |
| Random Forest (padrão) | 0.837 ± 0.013 | 0.513 ± 0.029 | 0.652 ± 0.027 | 0.574 ± 0.028 | 0.14 | padrão do sklearn (100 árvores, sem limite de profundidade) |
| Random Forest (ajustado) | 0.864 ± 0.010 | 0.551 ± 0.022 | 0.673 ± 0.031 | 0.605 ± 0.020 | 0.50 | RandomizedSearchCV (30 combinações): n_estimators=300, min_samples_leaf=10, max_features=0.5, max_depth=6 |
| MLP (padrão) | 0.850 ± 0.011 | 0.497 ± 0.063 | 0.680 ± 0.034 | 0.571 ± 0.045 | 0.15 | hidden_layer_sizes=(32,), early_stopping |
| MLP (ajustada) | 0.859 ± 0.013 | 0.580 ± 0.036 | 0.669 ± 0.032 | 0.620 ± 0.025 | 0.13 | RandomizedSearchCV (20 combinações): learning_rate_init=0.01, hidden_layer_sizes=(32,), alpha=0.001 |
