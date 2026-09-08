import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import catboost as cb
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('seoul_apt_train.csv')
test = pd.read_csv('seoul_apt_test.csv')

# Deep Feature Engineering
def build_features(df_train, df_test):
    df_all = pd.concat([df_train.assign(is_train=1), df_test.assign(is_train=0, 올랐나=np.nan)], ignore_index=True)
    
    # 1. 2024년 부동산 급등 지역 가중치 (강남3구 + 마용성 + 과천/분당 인접 등)
    tier1 = ['강남구', '서초구', '송파구']
    tier2 = ['용산구', '성동구', '마포구']
    tier3 = ['광진구', '영등포구', '양천구', '동작구', '강동구']
    
    df_all['급지_점수'] = 0
    df_all.loc[df_all['구'].isin(tier1), '급지_점수'] = 3
    df_all.loc[df_all['구'].isin(tier2), '급지_점수'] = 2
    df_all.loc[df_all['구'].isin(tier3), '급지_점수'] = 1
    
    # 2. 비율 및 차이 피처
    df_all['평당가_동대비'] = df_all['평당가'] / (df_all['동평당가'] + 1)
    df_all['평당가_구대비'] = df_all['평당가'] / (df_all['구평당가'] + 1)
    df_all['동_구대비'] = df_all['동평당가'] / (df_all['구평당가'] + 1)
    
    df_all['평당가_동차이'] = df_all['평당가'] - df_all['동평당가']
    df_all['평당가_구차이'] = df_all['평당가'] - df_all['구평당가']
    
    # 3. 직전변동 세분화
    df_all['직전변동_결측'] = df_all['직전변동'].isna().astype(int)
    df_all['직전변동_0'] = df_all['직전변동'].fillna(0)
    df_all['직전상승여부'] = (df_all['직전변동_0'] > 0).astype(int)
    df_all['직전급등여부'] = (df_all['직전변동_0'] >= 5.0).astype(int)
    
    # 4. 연식 세분화 (재건축, 신축 프리미엄)
    df_all['is_초신축'] = (df_all['연식'] <= 3).astype(int)
    df_all['is_신축'] = (df_all['연식'] <= 7).astype(int)
    df_all['is_구축_재건축'] = (df_all['연식'] >= 30).astype(int)
    df_all['is_애매한연식'] = ((df_all['연식'] > 15) & (df_all['연식'] < 25)).astype(int)
    
    # 5. 거래 활성도 및 모멘텀
    df_all['거래활성도'] = np.log1p(df_all['거래수'])
    df_all['평당가_모멘텀'] = df_all['평당가'] * (1 + df_all['직전변동_0']/100.0)
    
    # 6. 가격 구간 범주화
    df_all['is_초고가'] = (df_all['평당가'] >= 6000).astype(int)
    df_all['is_고가'] = ((df_all['평당가'] >= 4000) & (df_all['평당가'] < 6000)).astype(int)
    df_all['is_중저가'] = (df_all['평당가'] < 3000).astype(int)
    
    # 구 원핫 인코딩
    gu_dummies = pd.get_dummies(df_all['구'], prefix='구', drop_first=False)
    df_all = pd.concat([df_all, gu_dummies], axis=1)
    
    train_df = df_all[df_all['is_train'] == 1].drop(columns=['is_train'])
    test_df = df_all[df_all['is_train'] == 0].drop(columns=['is_train', '올랐나'])
    
    return train_df, test_df

train_f, test_f = build_features(train, test)
features = [c for c in train_f.columns if c not in ['번호', '이름', '구', '동', '올랐나']]

X = train_f[features].fillna(0)
y = train['올랐나'].values
X_test = test_f[features].fillna(0)

# Out-of-fold prediction and Blending Optimization
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)

oof_cb = np.zeros(len(train))
test_cb = np.zeros(len(test))

oof_hgb = np.zeros(len(train))
test_hgb = np.zeros(len(test))

oof_lr = np.zeros(len(train))
test_lr = np.zeros(len(test))

oof_rf = np.zeros(len(train))
test_rf = np.zeros(len(test))

scaler = StandardScaler()
X_s = scaler.fit_transform(X)
X_test_s = scaler.transform(X_test)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]
    
    # 1. CatBoost
    model_cb = cb.CatBoostClassifier(iterations=400, depth=4, learning_rate=0.03, l2_leaf_reg=3, verbose=0, random_seed=42+fold)
    model_cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=40)
    oof_cb[val_idx] = model_cb.predict_proba(X_va)[:, 1]
    test_cb += model_cb.predict_proba(X_test)[:, 1] / skf.n_splits
    
    # 2. HistGradientBoosting
    model_hgb = HistGradientBoostingClassifier(max_iter=150, max_depth=3, learning_rate=0.03, l2_regularization=1.0, random_state=42+fold)
    model_hgb.fit(X_tr, y_tr)
    oof_hgb[val_idx] = model_hgb.predict_proba(X_va)[:, 1]
    test_hgb += model_hgb.predict_proba(X_test)[:, 1] / skf.n_splits
    
    # 3. Logistic Regression
    model_lr = LogisticRegression(C=0.05, max_iter=1000, random_state=42+fold)
    model_lr.fit(X_s[train_idx], y_tr)
    oof_lr[val_idx] = model_lr.predict_proba(X_s[val_idx])[:, 1]
    test_lr += model_lr.predict_proba(X_test_s)[:, 1] / skf.n_splits
    
    # 4. RandomForest
    model_rf = RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=4, random_state=42+fold)
    model_rf.fit(X_tr, y_tr)
    oof_rf[val_idx] = model_rf.predict_proba(X_va)[:, 1]
    test_rf += model_rf.predict_proba(X_test)[:, 1] / skf.n_splits

print(f"CatBoost OOF Acc: {accuracy_score(y, oof_cb >= 0.5)*100:.2f}%, AUC: {roc_auc_score(y, oof_cb):.4f}")
print(f"HistGB   OOF Acc: {accuracy_score(y, oof_hgb >= 0.5)*100:.2f}%, AUC: {roc_auc_score(y, oof_hgb):.4f}")
print(f"LogReg   OOF Acc: {accuracy_score(y, oof_lr >= 0.5)*100:.2f}%, AUC: {roc_auc_score(y, oof_lr):.4f}")
print(f"RandomF  OOF Acc: {accuracy_score(y, oof_rf >= 0.5)*100:.2f}%, AUC: {roc_auc_score(y, oof_rf):.4f}")

# Ensemble Search
best_acc = 0
best_weights = None
best_th = 0.5

for w_cb in np.linspace(0, 1, 11):
    for w_hgb in np.linspace(0, 1 - w_cb, 11):
        for w_lr in np.linspace(0, 1 - w_cb - w_hgb, 11):
            w_rf = round(1.0 - w_cb - w_hgb - w_lr, 3)
            if w_rf < 0:
                continue
            blend_oof = w_cb * oof_cb + w_hgb * oof_hgb + w_lr * oof_lr + w_rf * oof_rf
            for th in np.arange(0.46, 0.54, 0.01):
                acc = accuracy_score(y, blend_oof >= th)
                if acc > best_acc:
                    best_acc = acc
                    best_weights = (w_cb, w_hgb, w_lr, w_rf)
                    best_th = th

print(f"\n--- Best Ensemble Result ---")
print(f"Best OOF Accuracy: {best_acc*100:.2f}%")
print(f"Best Weights (CatBoost, HistGB, LogReg, RF): {best_weights}")
print(f"Best Threshold: {best_th:.2f}")
