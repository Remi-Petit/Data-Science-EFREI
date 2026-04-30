# Sujet 2 – Rétention Client & Évaluation du Risque de Revenus

Système intelligent de prédiction du churn client basé sur le dataset `customer_churn.csv` (10 000 clients).

## Tâche prédictive

**Classification binaire** : prédire si un client va résilier son abonnement (`churn = 1`).

> Déséquilibre de classes : ~10 % de churn → métriques prioritaires : **Recall, F1-score, PR-AUC**.

## Structure du projet

```
Sujet_2/
├── data/
│   └── customer_churn.csv
├── models/                         # modèles entraînés (.joblib)
├── notebook/
│   └── churn_ml.ipynb              # EDA + pipeline interactif
├── results/                        # figures PNG + model_comparison.csv
├── src/
│   ├── __init__.py
│   ├── preprocessing.py            # chargement, feature engineering, split
│   ├── train.py                    # pipelines ML, cross-validation, sauvegarde
│   └── evaluate.py                 # métriques, visualisations, analyse seuil
└── train_pipeline.py               # script CLI
```

## Modèles entraînés

| Modèle | Stratégie déséquilibre |
|--------|------------------------|
| Logistic Regression | `class_weight='balanced'` |
| Random Forest | `class_weight='balanced'` |
| XGBoost | `scale_pos_weight=8` |
| MLP (Deep Learning) | `early_stopping=True` |

## Lancer le pipeline

```bash
cd IA/Sujet_2
python train_pipeline.py
```

Les modèles sont sauvegardés dans `models/` et les figures dans `results/`.

## Features

**Numériques** : age, tenure_months, monthly_logins, weekly_active_days, avg_session_time, features_used, usage_growth_rate, last_login_days_ago, monthly_fee, total_revenue, payment_failures, avg_resolution_time, csat_score, escalations, email_open_rate, marketing_click_rate, nps_score, referral_count

**Catégorielles encodées** : gender, customer_segment, signup_channel, contract_type, payment_method, discount_applied, price_increase_last_3m, survey_response, complaint_type

**Engineered** :
- `revenue_per_month` = total_revenue / (tenure_months + 1)
- `engagement_score` = weekly_active_days × avg_session_time
- `ticket_burden` = support_tickets × (avg_resolution_time + 1)

## Variables cibles alternatives (bonus)

- `total_revenue` → régression CLV (valeur vie client)
- `revenue_at_risk` = total_revenue × proba_churn → estimation du revenu à risque
