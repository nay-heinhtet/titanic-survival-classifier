# Titanic Survival Classifier

A binary classification project that predicts passenger survival on the Titanic using machine learning. Built as a foundational ML project covering the complete workflow from data exploration to model evaluation.

## What It Does

Takes passenger data (age, gender, ticket class, fare, etc.) and predicts whether they survived the Titanic disaster. Compares multiple models and analyzes where predictions go wrong.

## Workflow

1. **Exploratory Data Analysis** — Visualized survival rates by gender, passenger class, and age to identify predictive features
2. **Data Cleaning** — Handled missing values (median imputation for Age, mode for Embarked), dropped sparse columns (Cabin), encoded categorical variables
3. **Feature Engineering** — Converted text categories to numerical representations using label mapping and one-hot encoding
4. **Model Training** — Trained Logistic Regression and Random Forest classifiers on an 80/20 train-validation split
5. **Evaluation** — Compared model accuracy against the 62% baseline and analyzed prediction errors using confusion matrices

## Results

| Model | Accuracy |
|---|---|
| Baseline (always predict "died") | 62% |
| Random Forest | ~80% |
| Logistic Regression | ~81% |

Error analysis revealed both models struggled with the same 19 edge-case survivors, suggesting limitations in the available features rather than the model choice.

## Key Findings

- Gender was the strongest predictor of survival (women survived at much higher rates)
- Passenger class strongly correlated with survival (first class > second > third)
- Young children had higher survival rates across all classes
- The ~1% accuracy difference between models was not statistically significant at this sample size

## Tech Stack

| Tool | Purpose |
|---|---|
| pandas | Data loading, cleaning, and manipulation |
| matplotlib & seaborn | Exploratory data analysis and visualization |
| scikit-learn | Model training, evaluation, and train-test splitting |

## Dataset

The Titanic dataset is from [Kaggle's Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) competition.

## How to Run

```bash
pip install pandas matplotlib seaborn scikit-learn
jupyter notebook
```

Open the notebook and run all cells (Kernel → Restart & Run All).
