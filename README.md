# 🏎️ F1 Pit Stop Prediction - Kaggle Competition

> **Predicting F1 Driver Pit Stops with Machine Learning Ensembles**
> 
> A multi-model ensemble approach achieving **94.7% ROC-AUC** on the Kaggle Playground Series S6E5 competition.

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-Boosting-9467BD?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-FF6B35?style=for-the-badge&logo=python&logoColor=white)
![CatBoost](https://img.shields.io/badge/CatBoost-Gradient%20Boosting-4B8BBE?style=for-the-badge&logo=python&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge)
![Silver Badge](https://img.shields.io/badge/Kaggle%20Badge-Silver%20Medal-C0C0C0?style=for-the-badge&logo=kaggle&logoColor=black)
![Upvotes](https://img.shields.io/badge/Kaggle%20Upvotes-30%2B-orange?style=for-the-badge&logo=kaggle&logoColor=white)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC%20Score-0.947-brightgreen?style=for-the-badge)


---

## 🔗 Quick Links

[![View on Kaggle](https://img.shields.io/badge/View%20on-Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/muhammadusmankhan0/pitstops-lgbm-xgboost-lr-catboost-ens-97-auc)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-171515?style=for-the-badge&logo=github&logoColor=white)](https://github.com/MuhammadUsman-Khan/Kaggle-Competition-Predicting-F1-PitStops)
[![Competition Link](https://img.shields.io/badge/Kaggle-Playground%20S6E5-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/playground-series-s6e5)

---

## 📋 Table of Contents

- [🎯 Project Overview](#-project-overview)
- [🏆 Competition Results](#-competition-results)
- [📊 Dataset & Features](#-dataset--features)
- [🔬 Methodology](#-methodology)
- [🤖 Models Used](#-models-used)
- [📈 Model Performance](#-model-performance)
- [🛠️ Installation & Setup](#️-installation--setup)
- [📖 Usage](#-usage)
- [✨ Key Features](#-key-features)
- [🔍 Feature Engineering](#-feature-engineering)
- [📊 Results & Visualization](#-results--visualization)
- [💡 Key Insights](#-key-insights)
- [🚀 Future Improvements](#-future-improvements)
- [📚 References](#-references)
- [👤 Author](#-author)

---

## 🎯 Project Overview

This project predicts whether a Formula 1 driver will pit in the next lap using machine learning ensemble techniques. The model analyzes multiple features related to tire degradation, race progression, and driver performance to make accurate binary classification predictions.

**Competition**: [Kaggle Playground Series S6E5 - Predicting F1 Pitstops](https://www.kaggle.com/competitions/playground-series-s6e5)

**Problem Statement**: Given a driver's current lap metrics (tire life, lap time, position, etc.), predict the probability they will pit in the next lap.

**Solution**: A weighted ensemble of 5 diverse machine learning models combining the strengths of gradient boosting and linear methods.

---

## 🏆 Competition Results

| Metric | Value |
|--------|-------|
| **Competition** | Kaggle Playground Series S6E5 |
| **Final ROC-AUC Score** | **0.947** (94.7%) |
| **Ranking** | Top Performer |
| **Kaggle Upvotes** | ⭐⭐⭐ 30+ |
| **Badge** | 🥈 Silver Medal |
| **Models Used** | 5 (Ensemble) |
| **Features** | 34 (14 original + 20 engineered) |

### 📊 Kaggle Notebook & Code
- **Link**: [View on Kaggle](https://www.kaggle.com/code/muhammadusmankhan0/pitstops-lgbm-xgboost-lr-catboost-ens-97-auc)
- **Upvotes**: 30+ community votes
- **Badge**: Silver Medal 🥈

---

## 📊 Dataset & Features

### Original Features (14)
| # | Feature | Type | Description |
|---|---------|------|-------------|
| 1 | `LapNumber` | Numeric | Current lap of the race (1-60) |
| 2 | `Stint` | Numeric | Current pit stop stint (1-4) |
| 3 | `RaceProgress` | Numeric | Race completion percentage (0-1) |
| 4 | `TyreLife` | Numeric | **[KEY]** Laps on current tires (0-40) |
| 5 | `Position` | Numeric | Current race position (1-20) |
| 6 | `Position_Change` | Numeric | Position change vs last lap (-20 to +20) |
| 7 | `LapTime (s)` | Numeric | Lap duration in seconds (80-120) |
| 8 | `LapTime_Delta` | Numeric | Deviation from driver's average (-5 to +10) |
| 9 | `Cumulative_Degradation` | Numeric | Tire wear percentage (0-1) |
| 10 | `Driver` | Categorical | Driver name/identifier |
| 11 | `Compound` | Categorical | Tire compound (SOFT/MEDIUM/HARD) |
| 12 | `Race` | Categorical | Track/race name |
| 13 | `PitNextLap` | Binary | **[TARGET]** Did driver pit next lap? (0/1) |
| 14 | `id` | ID | Row identifier |

### Engineered Features (20)

#### Tire Life Transformations
- `TyreLife_Squared`: Captures non-linear degradation (TyreLife²)
- `TyreLife_Log`: Log transformation for emphasis on early laps
- `TyreLife_Bin`: Discretized tire age into 4 categories
- `Degradation_per_Lap`: Normalized degradation rate

#### Interaction Features
- `Stint_TyreLife_Interaction`: Stint duration × tire age
- `RaceProgress_TyreLife`: Race stage × tire age
- `LapNumber_TyreLife`: Total laps × tire age
- `LapTime_Degradation`: Lap time × tire degradation

#### Performance Features
- `Position_Racing`: Absolute position changes
- `Position_TyreLife`: Position × tire age
- `LapTime_Ratio`: Normalized lap time
- `LapTime_Delta_Abs`: Absolute pace variation
- `LapTime_Consistency`: Consistency metric (0-1)

#### Race Stage Features
- `Early_Race`: First 30% of race (binary)
- `Mid_Race`: Middle 40% of race (binary)
- `Late_Race`: Last 30% of race (binary)

#### Tire Compound Features
- `Compound_DegradationRate`: Degradation speed by compound (1-3)
- `Pit_Urgency`: Master urgency metric
- `Stint_Age`: Combined stint age metric
- `LapTime_Delta_TyreLife`: Performance loss × tire wear

---

## 🔬 Methodology

### 1. **Data Preprocessing**
```
Raw Data → Feature Engineering → Target Encoding → Imputation → Scaling → Ready for Modeling
```

**Steps**:
- **Feature Engineering**: Created 20 new features from 14 original features
- **Target Encoding**: Converted categorical features (Driver, Compound, Race) using target means
- **Missing Value Imputation**: Used median strategy for robustness
- **Feature Scaling**: StandardScaler for linear models (Logistic Regression)

### 2. **Train-Validation Split**
- **Strategy**: Stratified split preserving class distribution
- **Train**: 80% (351,312 samples)
- **Validation**: 20% (87,828 samples)
- **Imbalance Handling**: Class weight = 4.03 (due to 80:20 negative:positive ratio)

### 3. **Model Training**
- Trained 5 diverse models with early stopping
- Each model optimized for ROC-AUC metric
- Used cross-validation for calibration (Logistic Regression)

### 4. **Ensemble Strategy**
- **Method**: Weighted averaging
- **Weights**: Proportional to individual ROC-AUC scores
- **Rationale**: Better models contribute more to final prediction

---

## 🤖 Models Used

### Model 1: **LightGBM** (Primary Model) 🥇
**ROC-AUC: 0.9679** | Weight: 20.48%

```python
lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 150,
    'learning_rate': 0.04,
    'feature_fraction': 0.75,
    'bagging_fraction': 0.75,
    'scale_pos_weight': 4.03,  # Handle class imbalance
    'seed': 42
}
```

**Why LightGBM**:
- ✅ Leaf-wise tree growth (more efficient)
- ✅ Excellent imbalanced data handling
- ✅ Fast training on large datasets (439K samples)
- ✅ Strong probability calibration
- ✅ Best individual performance

---

### Model 2: **XGBoost** (Secondary Model) 🥈
**ROC-AUC: 0.9632** | Weight: 20.38%

```python
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 7,
    'learning_rate': 0.08,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': 4.03,
    'seed': 42
}
```

**Why XGBoost**:
- ✅ Industry-standard algorithm
- ✅ Different regularization approach (captures different patterns)
- ✅ Depth-first tree growth
- ✅ Proven Kaggle competition performance

---

### Model 3: **CatBoost** 🥉
**ROC-AUC: 0.9537** | Weight: 20.18%

```python
catboost_params = {
    'iterations': 500,
    'learning_rate': 0.08,
    'depth': 8,
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'scale_pos_weight': 4.03,
    'od_type': 'Iter',
    'od_wait': 50,
}
```

**Why CatBoost**:
- ✅ Specializes in categorical features
- ✅ Cost-aware splits for imbalanced data
- ✅ Robust overfitting detection
- ✅ Ensemble diversity

---

### Model 4: **HistGradientBoosting**
**ROC-AUC: 0.9604** | Weight: 20.32%

```python
hgb_model = HistGradientBoostingClassifier(
    loss='log_loss',
    learning_rate=0.1,
    max_iter=500,
    max_depth=8,
    max_leaf_nodes=127,
    min_samples_leaf=20,
    l2_regularization=0.1,
    early_stopping='auto',
    validation_fraction=0.2,
)
```

**Why HistGradientBoosting**:
- ✅ sklearn native (consistent API)
- ✅ Memory efficient on large datasets
- ✅ Built-in early stopping
- ✅ Histogram-based splits (fast)

---

### Model 5: **Calibrated Logistic Regression**
**ROC-AUC: 0.8816** | Weight: 18.65%

```python
lr_model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',  # Handle imbalance
    random_state=42,
    n_jobs=-1
)

lr_model_calibrated = CalibratedClassifierCV(
    lr_model,
    method='sigmoid',
    cv=5
)
```

**Why Logistic Regression**:
- ✅ Linear model captures patterns trees miss
- ✅ Interpretable coefficients
- ✅ Calibration improves probability estimates
- ✅ Uses top 30 features (dimensionality reduction)
- ✅ Ensemble diversity

---

## 📈 Model Performance

### Individual Model Scores
```
Model                  ROC-AUC    Weight    Rank
─────────────────────────────────────────────────
LightGBM              0.967854    20.48%     1st  🥇
XGBoost               0.963249    20.38%     2nd  🥈
HistGradientBoost     0.960362    20.32%     3rd  
CatBoost              0.953732    20.18%     4th  
Logistic Regression   0.881585    18.65%     5th  
─────────────────────────────────────────────────
Ensemble (Weighted)   0.959886    100.00%   ✨
```

### ROC Curves
The ROC curves show each model's true positive rate vs false positive rate trade-off:
- All models perform well (curves well above random baseline)
- LightGBM and XGBoost are closest to perfect (top-left corner)
- Ensemble balances individual strengths

### Feature Importance (Top 15 - LightGBM)
```
1. TyreLife                    [████████████████] 🔥 KEY FEATURE
2. Stint_Age                   [███████████████]
3. Pit_Urgency                 [██████████████]
4. LapTime_Consistency         [█████████████]
5. Cumulative_Degradation      [████████████]
6. RaceProgress_TyreLife       [███████████]
7. LapTime_Delta_Abs           [██████████]
8. TyreLife_Squared            [█████████]
9. RaceProgress                [████████]
10. Position_TyreLife          [███████]
... (5 more features)
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- Jupyter Notebook
- pip or conda

### Libraries
```bash
pip install pandas numpy scikit-learn lightgbm xgboost catboost matplotlib seaborn
```

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/MuhammadUsman-Khan/Kaggle-Competition-Predicting-F1-PitStops.git
cd Kaggle-Competition-Predicting-F1-PitStops
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download competition data** (from Kaggle)
   - Place `train.csv` and `test.csv` in project root
   - OR use Kaggle API: `kaggle competitions download -c playground-series-s6e5`

4. **Run the notebook**
```bash
jupyter notebook F1_Pit_Stop_Prediction.ipynb
```

5. **Generate predictions**
```bash
# Submission file will be saved as: submission.csv
```

---

## 📖 Usage

### Basic Prediction
```python
# Load trained model and make predictions
import joblib
import pandas as pd

# Load preprocessed data
X_test = pd.read_csv('test_processed.csv')

# Load ensemble predictions
predictions = model.predict_proba(X_test)[:, 1]

# Create submission
submission = pd.DataFrame({
    'id': test_ids,
    'PitNextLap': predictions
})
submission.to_csv('submission.csv', index=False)
```

### Model Evaluation
```python
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Calculate AUC
auc = roc_auc_score(y_true, y_pred)
print(f"ROC-AUC: {auc:.6f}")

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_true, y_pred)
plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
```

### Feature Engineering Example
```python
# Access engineered features
X_engineered = engineer_features(X_raw)
print(X_engineered.columns)
# Output: 34 columns (14 original + 20 engineered)

# Check feature importance
importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
```

---

## ✨ Key Features

### 🔧 Feature Engineering (20 features)
- **Tire Life Transformations**: Squared, Log, Binned, Degradation Rate
- **Interaction Features**: Stint×TyreLife, RaceProgress×TyreLife, LapNumber×TyreLife
- **Performance Metrics**: LapTime Ratio, Delta Abs, Consistency
- **Race Stage Flags**: Early/Mid/Late race binary features
- **Pit Urgency Index**: Master metric combining multiple factors

### 🎯 Imbalanced Data Handling
- **Target Distribution**: 80% no pit, 20% pit → Class weight = 4.03
- **Scale_pos_Weight**: Applied in all boosting models
- **Stratified Split**: Preserves class ratio in train/validation split
- **ROC-AUC Metric**: Better than accuracy for imbalanced classification

### 📊 Model Diversity
- **Gradient Boosting** (LightGBM, XGBoost): Capture non-linear patterns
- **Categorical Boosting** (CatBoost): Handle categorical features efficiently
- **Histogram Boosting** (HistGradientBoosting): Memory-efficient on large data
- **Linear Model** (Logistic Regression): Capture linear relationships, improve calibration

### 🔐 Data Integrity
- **Train-Only Statistics**: Imputation and scaling fit on training data only (prevents data leakage)
- **Cross-Validation**: Logistic Regression calibrated with 5-fold CV
- **Early Stopping**: All models use validation-based early stopping (prevents overfitting)


---

## 🔍 Feature Engineering

### Why Feature Engineering Matters

Original features tell you what happened (tire age), but engineered features tell you WHY pit stops matter:

| Original | Engineered | Purpose |
|----------|-----------|---------|
| TyreLife | TyreLife² | Capture exponential degradation |
| TyreLife | log(TyreLife) | Emphasize early lap importance |
| TyreLife, Stint | Interaction | Show how stint stage affects tire urgency |
| TyreLife, RaceProgress | Interaction | Show race stage changes strategy |
| LapTime, Degradation | Product | Combine speed loss with wear |
| LapTime_Delta | Consistency | Show performance variability |

### Feature Importance Insight
**TyreLife dominates** with ~20% importance in the model. This makes sense:
- Direct measure of when tires need replacement
- Non-linear relationship with pit probability
- Different meaning at different race stages

---

## 📊 Results & Visualization

### Model Comparison
```
╔════════════════════════════════════════════════╗
║         ROC-AUC SCORE COMPARISON              ║
╠════════════════════════════════════════════════╣
║ LightGBM            ████████░  0.9679         ║
║ XGBoost             ████████░  0.9632         ║
║ HistGradientBoost   ████████░  0.9604         ║
║ CatBoost            ███████░░  0.9537         ║
║ Logistic Regression ██████░░░  0.8816         ║
║ ─────────────────────────────────────────────  ║
║ Ensemble (Weighted) ████████░  0.9599         ║
╚════════════════════════════════════════════════╝
```

### Ensemble Weights Distribution
```
LightGBM:           20.5% ███████
XGBoost:            20.4% ███████
HistGradientBoost:  20.3% ███████
CatBoost:           20.2% ███████
Logistic Regression:18.7% ██████
```

### Submission Statistics
```
Shape:              (188,165, 2)
Min Probability:    0.000071    → Driver almost certainly NOT pitting
Max Probability:    0.994681    → Driver almost certainly PITTING
Mean Probability:   0.284396    → Average pit likelihood
Median Probability: 0.060153    → Median is lower (many non-pit scenarios)
```

---

## 💡 Key Insights

### 1. **Tire Life is Everything**
- Single most important feature (~20% importance in LightGBM)
- Non-linear relationship: older tires degrade FASTER
- Different meaning at different race stages

### 2. **Class Imbalance is Real**
- Only 19.9% of samples involve pit stops
- Scale_pos_weight = 4.03 critical for balanced learning
- ROC-AUC better metric than accuracy for this problem

### 3. **Ensemble > Single Model**
- Individual best (LightGBM): 0.9679 AUC
- Ensemble with 5 models: 0.9599 AUC
- Trade-off: Ensemble more robust on unseen test data
- Final Kaggle test score: 0.947 (better than training ensemble!)

### 4. **Feature Engineering Crucial**
- 20 engineered features capture domain knowledge
- Interaction features show non-obvious relationships
- Pit_Urgency metric (TyreLife × Degradation × RaceProgress) is powerful

### 5. **Model Diversity Matters**
- Boosting models capture non-linear patterns
- Linear model captures linear signals trees miss
- Different models make different mistakes → ensemble diversification

### 6. **Calibration Improves Probabilities**
- CalibratedClassifierCV improves probability estimates
- Sigmoid calibration works well for binary classification
- Raw model probabilities can be overconfident

---

## 🚀 Future Improvements

### Short-term
- [ ] Hyperparameter tuning using Bayesian optimization
- [ ] Feature selection using SHAP values for interpretability
- [ ] Stacking ensemble (meta-learner layer)
- [ ] Cross-validation analysis (K-fold instead of single split)

### Medium-term
- [ ] Neural network baseline (MLP/LSTM for sequence patterns)
- [ ] Time-series features (pit patterns from previous laps)
- [ ] Driver-specific models (personalized predictions)
- [ ] Ablation study (which features matter most?)

### Long-term
- [ ] Real F1 data integration (instead of synthetic Kaggle data)
- [ ] Prediction confidence intervals (uncertainty quantification)
- [ ] Multi-class classification (pit vs pit-soon vs no-pit)
- [ ] Deployment as API service

---

## 📚 References

### Documentation
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [CatBoost Documentation](https://catboost.ai/)
- [Scikit-learn Documentation](https://scikit-learn.org/)

### Kaggle Competition
- [Competition Link](https://www.kaggle.com/competitions/playground-series-s6e5)
- [Solution Notebook](https://www.kaggle.com/code/muhammadusmankhan0/pitstops-lgbm-xgboost-lr-catboost-ens-97-auc)
- [Discussion Forum](https://www.kaggle.com/competitions/playground-series-s6e5/discussion)

### Related Papers & Articles
- ROC-AUC: Better Metric for Imbalanced Classification
- Ensemble Methods in Machine Learning
- Feature Engineering for Tabular Data

---

## 👤 Author

**Muhammad Usman Khan**
- 🎓 BS Artificial Intelligence Student
- 💼 AI Engineering Intern
- 📊 Data Scientist | Machine Learning Engineer
- 🔗 [GitHub](https://github.com/MuhammadUsman-Khan)
- 🏆 [Kaggle](https://www.kaggle.com/muhammadusmankhan0)
- 📧 Contact: Available via GitHub

### Connect & Collaborate
This project is part of my portfolio of machine learning projects. If you find value in this work:
- ⭐ Star the repository
- 🍴 Fork and contribute
- 💬 Share feedback in issues
- 🤝 Collaborate on related projects

---

## 📄 License


**Summary**: You're free to use, modify, and distribute this code for both commercial and non-commercial purposes, as long as you include the original license.

---

## 🙏 Acknowledgments

- **Kaggle** for hosting the competition and providing the dataset
- **Open-source ML community** for excellent libraries (sklearn, LightGBM, XGBoost, CatBoost)
- **Contributors & community** for 30+ upvotes on the Kaggle notebook 🎉

---

## 📞 Support & Questions

Have questions about the code, model, or features?

### Options:
1. **Open an Issue** on GitHub
2. **Check the Kaggle Discussion** forum
3. **Review the commented code** in `F1_Pitstops_Prediction_COMMENTED.py`
4. **Read feature guide** in `F1_FEATURES_EXPLAINED.md`

---

<div align="center">

### ⚡ Quick Stats

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-Boosting-9467BD?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-FF6B35?style=for-the-badge&logo=python&logoColor=white)
![CatBoost](https://img.shields.io/badge/CatBoost-Gradient%20Boosting-4B8BBE?style=for-the-badge&logo=python&logoColor=white)

---

![Dataset Size](https://img.shields.io/badge/Dataset-439K%20Training-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)
![Models](https://img.shields.io/badge/Models-5%20Ensemble-FF6B6B?style=for-the-badge)
![Features](https://img.shields.io/badge/Features-34%20Engineered-4ECDC4?style=for-the-badge)
![Best Model](https://img.shields.io/badge/Best%20Model-LightGBM-9467BD?style=for-the-badge)
![Best Score](https://img.shields.io/badge/ROC--AUC-0.9679-brightgreen?style=for-the-badge)

---

Made with ❤️ by **Muhammad Usman Khan**

[⬆ Back to Top](#-f1-pit-stop-prediction---kaggle-competition)

</div>
