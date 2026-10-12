# Registro de Experimentos — Etapa 2

Validação cruzada estratificada (5 folds) no conjunto de treino. Valores: média ± desvio-padrão. Métrica de seleção: AUC-ROC.

| modelo | roc_auc | recall | precision | f1 | tempo_treino_s | observacao |
|---|---|---|---|---|---|---|
| Regressão Logística (baseline) | 0.859 ± 0.013 | 0.582 ± 0.042 | 0.667 ± 0.024 | 0.621 ± 0.029 | 0.04 | Mesma configuração da Etapa 1, sem class_weight |
| Random Forest (padrão) | 0.837 ± 0.013 | 0.513 ± 0.029 | 0.652 ± 0.027 | 0.574 ± 0.028 | 0.15 | Hiperparâmetros padrão do sklearn |
| Random Forest (ajustado) | 0.864 ± 0.010 | 0.551 ± 0.022 | 0.673 ± 0.031 | 0.605 ± 0.020 | 0.47 | RandomizedSearchCV, 30 combinações, seleção por AUC: {'classifier__n_estimators': 300, 'classifier__min_samples_leaf': 10, 'classifier__max_features': 0.5, 'classifier__max_depth': 6} |
| Regressão Logística (ajustada) | 0.859 ± 0.013 | 0.585 ± 0.043 | 0.665 ± 0.025 | 0.621 ± 0.031 | 0.04 | GridSearchCV, seleção por AUC: {'classifier__C': 100, 'classifier__penalty': 'l2'} |
| Random Forest (ajustado) | 0.864 ± 0.010 | 0.551 ± 0.022 | 0.673 ± 0.031 | 0.605 ± 0.020 | 0.46 | RandomizedSearchCV, 30 combinações, seleção por AUC: {'classifier__n_estimators': 300, 'classifier__min_samples_leaf': 10, 'classifier__max_features': 0.5, 'classifier__max_depth': 6} |
| Random Forest (padrão) | 0.837 ± 0.013 | 0.513 ± 0.029 | 0.652 ± 0.027 | 0.574 ± 0.028 | 0.14 | Hiperparâmetros padrão do sklearn |
| Regressão Logística (ajustada) | 0.859 ± 0.013 | 0.585 ± 0.043 | 0.665 ± 0.025 | 0.621 ± 0.031 | 0.03 | GridSearchCV, seleção por AUC: {'classifier__C': 100, 'classifier__penalty': 'l2'} |
| Random Forest (padrão) | 0.837 ± 0.013 | 0.513 ± 0.029 | 0.652 ± 0.027 | 0.574 ± 0.028 | 0.14 | Hiperparâmetros padrão do sklearn |
| Random Forest (ajustado) | 0.864 ± 0.010 | 0.551 ± 0.022 | 0.673 ± 0.031 | 0.605 ± 0.020 | 0.48 | RandomizedSearchCV, 30 combinações, seleção por AUC: {'classifier__n_estimators': 300, 'classifier__min_samples_leaf': 10, 'classifier__max_features': 0.5, 'classifier__max_depth': 6} |
