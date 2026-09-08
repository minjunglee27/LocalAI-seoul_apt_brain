import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score
import catboost as cb
import warnings
warnings.filterwarnings('ignore')

# 1. Load Data
train = pd.read_csv('seoul_apt_train.csv')
test = pd.read_csv('seoul_apt_test.csv')

print(f"Train Shape: {train.shape}, Test Shape: {test.shape}")
print("Target Mean:", train['올랐나'].mean())

def get_features(df_train, df_test):
    df_all = pd.concat([df_train.assign(is_train=1), df_test.assign(is_train=0, 올랐나=np.nan)], ignore_index=True)
    
    # Basic location grade
    loc_map = {'강남구': 3, '서초구': 3, '송파구': 3, '용산구': 2, '성동구': 2, '마포구': 2, '광진구': 2, '동작구': 1, '강동구': 1, '양천구': 1, '영등포구': 1}
    df_all['입지_등급'] = df_all['구'].map(loc_map).fillna(0)
    
    # Ratios and Differences
    df_all['평당가_동평당가_비율'] = df_all['평당가'] / (df_all['동평당가'] + 1e-5)
    df_all['평당가_구평당가_비율'] = df_all['평당가'] / (df_all['구평당가'] + 1e-5)
    df_all['동평당가_구평당가_비율'] = df_all['동평당가'] / (df_all['구평당가'] + 1e-5)
    df_all['평당가_동평당가_차이'] = df_all['평당가'] - df_all['동평당가']
    df_all['평당가_구평당가_차이'] = df_all['평당가'] - df_all['구평당가']
    
    # 직전변동
    df_all['직전변동_결측'] = df_all['직전변동'].isna().astype(int)
    df_all['직전변동_0'] = df_all['직전변동'].fillna(0)
    
    # 건물 특성
    df_all['is_신축'] = (df_all['연식'] <= 5).astype(int)
    df_all['is_구축'] = (df_all['연식'] >= 25).astype(int)
    df_all['평수'] = df_all['면적'] / 3.30578
    df_all['고층여부'] = (df_all['층'] >= 10).astype(int)
    
    # 상호작용
    df_all['평당가_x_직전변동'] = df_all['평당가'] * df_all['직전변동_0'] / 100.0
    
    # One-hot
    gu_dummies = pd.get_dummies(df_all['구'], prefix='구', drop_first=False)
    df_all = pd.concat([df_all, gu_dummies], axis=1)
    
    train_out = df_all[df_all['is_train'] == 1].drop(columns=['is_train'])
    test_out = df_all[df_all['is_train'] == 0].drop(columns=['is_train', '올랐나'])
    
    return train_out, test_out

train_f, test_f = get_features(train, test)
feature_cols = [c for c in train_f.columns if c not in ['번호', '이름', '구', '동', '올랐나']]

X = train_f[feature_cols].fillna(0)
y = train['올랐나'].values
X_test = test_f[feature_cols].fillna(0)

print(f"Number of Features: {len(feature_cols)}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'Logistic Regression (All Features)': LogisticRegression(C=0.1, max_iter=1000),
    'Logistic Regression (L1)': LogisticRegression(penalty='l1', solver='liblinear', C=0.1),
    'MLP Classifier (1 Layer 16)': MLPClassifier(hidden_layer_sizes=(16,), max_iter=300, random_state=42),
    'HistGradientBoosting': HistGradientBoostingClassifier(max_iter=100, max_depth=3, learning_rate=0.03, random_state=42),
    'RandomForest (depth=5)': RandomForestClassifier(n_estimators=300, max_depth=5, random_state=42),
    'ExtraTrees (depth=5)': ExtraTreesClassifier(n_estimators=300, max_depth=5, random_state=42),
    'CatBoost (depth=3)': cb.CatBoostClassifier(iterations=300, depth=3, learning_rate=0.03, verbose=0, random_seed=42),
    'CatBoost (depth=4)': cb.CatBoostClassifier(iterations=300, depth=4, learning_rate=0.03, verbose=0, random_seed=42)
}

print("\n--- 5-Fold Cross Validation Results ---")
for name, model in models.items():
    if 'Logistic' in name or 'MLP' in name:
        scores = cross_val_score(model, X_scaled, y, cv=skf, scoring='accuracy')
    else:
        scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    print(f"{name:40s}: CV = {scores.mean()*100:.2f}% (std: {scores.std()*100:.2f}%)")
