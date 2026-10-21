import os, sys
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, f1_score,
    confusion_matrix, roc_curve, auc, average_precision_score,
    precision_score, recall_score
)
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import warnings
from pandas.errors import PerformanceWarning
from itertools import combinations
import seaborn as sns
from tqdm import tqdm

# ==========================================
# 設定區
# ==========================================
# 實驗標籤，用於區分輸出檔案
EXP_TAG = "no_duration"  
# 這是為了進行真實場景模擬 (Realistic Scenario)，移除事後變數

# 環境設定
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

warnings.simplefilter(action="ignore", category=PerformanceWarning)

# ==========================================
# 資料載入與特徵工程 (保持與原版一致以確保公平比較)
# ==========================================
TARGET = 'y'
NUMS = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
CATS = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

print("Loading data...")
train = pd.read_csv('../data/train.csv', index_col='id')
test = pd.read_csv('../data/test.csv', index_col='id')
orig = pd.read_csv('../data/bank-full.csv', delimiter=';')

orig[TARGET] = orig[TARGET].replace({'no': 0, 'yes': 1}).infer_objects(copy=False)

train[CATS] = train[CATS].astype('category')
test[CATS] = test[CATS].astype('category')
orig[CATS] = orig[CATS].astype('category')

# 數值變換
for df in [train, test, orig]:
    df['_duration_sin'] = np.sin(2*np.pi * df['duration'] / 540).astype('float32')
    df['_duration_cos'] = np.cos(2*np.pi * df['duration'] / 540).astype('float32')
    df['_balance_log'] = (np.sign(df['balance']) * np.log1p(np.abs(df['balance']))).astype('float32')
    df['_balance_sin'] = np.sin(2*np.pi * df['balance'] / 1000).astype('float32')
    df['_balance_cos'] = np.cos(2*np.pi * df['balance'] / 1000).astype('float32')
    df['_age_sin'] = np.sin(2*np.pi * df['age'] / 10).astype('float32')
    df['_pdays_sin'] = np.sin(2*np.pi * df['pdays'] / 7).astype('float32')

# 數值交互作用 (Interaction)
columns = NUMS
for r in [2]:
    for cols in list(combinations(columns, r)):
        name = '_x_'.join(cols)
        for df in [train, test, orig]:
            df[name] = df[cols[0]]
            for col in cols[1:]:
                df[name] = df[name] * df[col]

# 類別交互作用 (Categorical Combination)
TE_columns = []
columns = NUMS + CATS
print("Generating Categorical Interactions...")
for r in [2]:
    for cols in tqdm(list(combinations(columns, r))):
        name = '-'.join(cols)
        
        # 為了效率，先處理 orig 以外的部分，這裡簡化邏輯保持您原code結構
        train[name] = train[cols[0]].astype(str)
        test[name] = test[cols[0]].astype(str)
        orig[name] = orig[cols[0]].astype(str)
        
        for col in cols[1:]:
            train[name] = train[name] + '_' + train[col].astype(str)
            test[name] = test[name] + '_' + test[col].astype(str)
            orig[name] = orig[name] + '_' + orig[col].astype(str)

        combined = pd.concat([train[name], test[name], orig[name]], ignore_index=True)
        combined, _ = combined.factorize()
        train[name] = combined[:len(train)]
        test[name] = combined[len(train):len(train) + len(test)]
        orig[name] = combined[len(train) + len(test):]

        TE_columns.append(name)

# Target Encoding & Count Encoding
CC = TE_columns + CATS + NUMS
print(f"Applying Target/Count Encoding to {len(CC)} columns...")

for i, c in enumerate(CC):
    # Target Encoding
    te_values = orig.groupby(c)['y'].mean().astype('float32')
    te_values.name = f"TE_ORIG_{c}"
    
    train = train.merge(te_values, on=c, how='left')
    train[te_values.name] = train[te_values.name].fillna(train[te_values.name].mean())
    
    test = test.merge(te_values, on=c, how='left')
    test[te_values.name] = test[te_values.name].fillna(train[te_values.name].mean())
    
    # Count Encoding
    ce_values = orig[c].value_counts()
    
    train_ce = train[c].map(ce_values)
    test_ce = test[c].map(ce_values)
    
    if hasattr(train_ce, 'cat'): train_ce = train_ce.astype('float32')
    if hasattr(test_ce, 'cat'): test_ce = test_ce.astype('float32')
    
    train[f'CE_ORIG_{c}'] = train_ce.fillna(0)
    test[f'CE_ORIG_{c}'] = test_ce.fillna(0)

# =======================
# 1. 特徵篩選 (CRITICAL STEP)
# =======================

# 取得所有數值型欄位
all_feats = train.select_dtypes(include=["number"]).columns.tolist()
if TARGET in all_feats:
    all_feats.remove(TARGET)

# [修改點] 移除所有與 duration 相關的欄位
FEATS = [f for f in all_feats if 'duration' not in f]
removed_feats = [f for f in all_feats if 'duration' in f]

print("\n" + "="*40)
print(f" Ablation Study: Removing 'duration'")
print("="*40)
print(f"Total numeric features available: {len(all_feats)}")
print(f"Features removed (duration-related): {len(removed_feats)}")
print(f"Features remaining for training: {len(FEATS)}")
print(f"Example removed: {removed_feats[:5]}")
print("="*40 + "\n")

X = train[FEATS].values.astype("float32")
y = train[TARGET].values.astype("int8")
X_test = test[FEATS].values.astype("float32")

# =======================
# 2. 5-fold OOF 訓練
# =======================

model_dir = f"lgbm_cv_models_{EXP_TAG}"
os.makedirs(model_dir, exist_ok=True)

def fold_model_path(fold):
    return os.path.join(model_dir, f"lgbm_{EXP_TAG}_fold{fold}.txt")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(train), dtype=np.float32)
pred = np.zeros(len(test), dtype=np.float32)
models = []
fold_scores = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dvalid = lgb.Dataset(X_va, label=y_va)

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.02,
        "num_leaves": 64,
        "max_depth": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": 50,
        "lambda_l1": 0.0,
        "lambda_l2": 0.0,
        "verbose": -1,
        "seed": 42 + fold,
        "n_jobs": -1,
    }

    model_path = fold_model_path(fold)

    if os.path.exists(model_path):
        print(f"[LGBM-{EXP_TAG}] Fold {fold}: load from {model_path}")
        model = lgb.Booster(model_file=model_path)
    else:
        print(f"[LGBM-{EXP_TAG}] Fold {fold}: train new model")
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=5000,
            valid_sets=[dtrain, dvalid],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100),
                lgb.log_evaluation(period=200),
            ]
        )
        model.save_model(model_path)

    oof[va_idx] = model.predict(X_va, num_iteration=model.best_iteration)
    pred += model.predict(X_test, num_iteration=model.best_iteration) / skf.n_splits

    fold_auc = roc_auc_score(y_va, oof[va_idx])
    fold_scores.append(fold_auc)
    models.append(model)
    print(f"[LGBM-{EXP_TAG}] Fold {fold} AUC: {fold_auc:.6f}")

cv_auc = roc_auc_score(y, oof)
print(f"\n[LGBM-{EXP_TAG}] CV AUC: {cv_auc:.6f}")
print("Per-fold AUC:", [f"{s:.5f}" for s in fold_scores])

# =======================
# 3. F1 / Precision-Recall 分析
# =======================

y_true = y.astype(np.float32)
y_prob = oof.astype(np.float32)

print(f"[METRIC] OOF AUROC: {roc_auc_score(y_true, y_prob):.6f}")

prec, rec, thr = precision_recall_curve(y_true, y_prob)
thr_for_f1 = np.r_[thr, 1.0]
f1 = 2 * prec * rec / (prec + rec + 1e-12)
best_idx = np.nanargmax(f1)
best_thr = float(thr_for_f1[best_idx])

print(f"[METRIC] Best F1: {f1[best_idx]:.6f} @ threshold={best_thr:.6f}")

# =======================
# 4. SHAP (TreeExplainer)
# =======================
# 用 AUC 最好的那一折模型
best_fold = int(np.argmax(fold_scores))
best_model = models[best_fold]

print(f"[SHAP] Generating SHAP for {EXP_TAG}...")

# 抽樣
N_EXP = 3000
idx_all = np.arange(len(train))
np.random.seed(123)
exp_idx = np.random.choice(idx_all, size=min(N_EXP, len(idx_all)), replace=False)
X_exp = X[exp_idx]

explainer = shap.TreeExplainer(best_model)
shap_vals = explainer.shap_values(X_exp)

if isinstance(shap_vals, list):
    shap_vals = shap_vals[1]

mean_abs_shap = np.abs(shap_vals).mean(axis=0)

shap_importance = pd.DataFrame({
    "feature": FEATS,
    "mean_|SHAP|": mean_abs_shap
}).sort_values("mean_|SHAP|", ascending=False)

# 儲存的重要性檔案也要改名
imp_csv_name = f"lgbm_{EXP_TAG}_shap_importance.csv"
print(f"[SHAP] Top-20 features (No Duration):")
print(shap_importance.head(20).to_string(index=False))
shap_importance.to_csv(imp_csv_name, index=False)

# SHAP Plots (改名)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_vals, X_exp, feature_names=FEATS, show=False, max_display=30)
plt.tight_layout()
plt.savefig(f"lgbm_{EXP_TAG}_shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_vals, X_exp, feature_names=FEATS, plot_type="bar", show=False, max_display=20)
plt.tight_layout()
plt.savefig(f"lgbm_{EXP_TAG}_shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"Saved SHAP plots with tag: {EXP_TAG}")

# =======================
# 5. ROC / PR Plots (改名)
# =======================

fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(5,4))
plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
plt.plot([0,1],[0,1],'--',lw=1)
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve ({EXP_TAG})")
plt.legend(); plt.tight_layout()
plt.savefig(f"lgbm_{EXP_TAG}_roc_curve.png", dpi=150); plt.close()

prec, rec, thr = precision_recall_curve(y_true, y_prob)
ap = average_precision_score(y_true, y_prob)
plt.figure(figsize=(5,4))
plt.plot(rec, prec, label=f"AP = {ap:.4f}")
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title(f"PR Curve ({EXP_TAG})")
plt.legend(); plt.tight_layout()
plt.savefig(f"lgbm_{EXP_TAG}_pr_curve.png", dpi=150); plt.close()

# Operating points
thr_for_f1 = np.r_[thr, 1.0]
f1 = 2*prec*rec/(prec+rec+1e-12)
best_idx = int(np.nanargmax(f1))
best_thr = float(thr_for_f1[best_idx])

rows = []
for t in sorted(set([best_thr, 0.5, np.quantile(y_prob, 0.9), np.quantile(y_prob, 0.95)])):
    y_pred = (y_prob >= t).astype(int)
    rows.append({
        "threshold": t,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
    })

pd.DataFrame(rows).to_csv(f"lgbm_{EXP_TAG}_pr_operating_points.csv", index=False)
print(f"Saved: lgbm_{EXP_TAG}_pr_operating_points.csv")

# =======================
# 6. Submission (改名)
# =======================
offset = 750000
pd.DataFrame({
    "id": test.index + offset,
    "y": pred
}).to_csv(f"lgbm_{EXP_TAG}_submit_f1thr.csv", index=False, float_format="%.12f")

print(f"Saved: lgbm_{EXP_TAG}_submit_f1thr.csv")


# ==========================================
# 7. [新增] 月份偏好度分析 (Month Preference Analysis)
# ==========================================

print("[ANALYSIS] Analyzing Month Preference...")

# 1. 取得 SHAP 樣本對應的「原始月份」
# exp_idx 是您在第 4 步抽樣的 index，我們用它回溯 train 資料
# 確保 month 欄位是原始的 category 格式
month_data = train.iloc[exp_idx]['month'].astype(str).values

# 定義月份順序 (Bank Marketing 資料集的標準月份)
month_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
               'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

# 2. 找出所有與 "month" 相關的特徵
# 邏輯：只要特徵名稱裡面有 "month" (例如 TE_ORIG_day-month, TE_ORIG_month)，就把它算進來
month_feat_indices = [i for i, f in enumerate(FEATS) if 'month' in f]
month_feat_names = [FEATS[i] for i in month_feat_indices]

print(f"Found {len(month_feat_names)} features related to 'month'.")
# print(month_feat_names) # 如果想看具體是哪些特徵，可以 uncomment

# 3. 計算這些特徵的 SHAP 總和 (Net Impact of Month)
# axis=1 代表把每個樣本中，所有 month 相關特徵的得分加起來
month_shap_sum = shap_vals[:, month_feat_indices].sum(axis=1)

# 4. 準備繪圖數據
plot_df = pd.DataFrame({
    'Month': month_data,
    'Month_Impact (SHAP Sum)': month_shap_sum
})

# 設定月份排序 (轉成 Categorical)
plot_df['Month'] = pd.Categorical(plot_df['Month'], categories=month_order, ordered=True)

# 5. 繪製箱型圖 (Boxplot)
plt.figure(figsize=(12, 6))
sns.boxplot(data=plot_df, x='Month', y='Month_Impact (SHAP Sum)', palette="RdBu_r")
plt.axhline(0, color='gray', linestyle='--', linewidth=1) # 0 分界線

plt.title(f"Model Preference by Month ({EXP_TAG})\n(Sum of SHAP values for all month-related features)")
plt.ylabel("Net Impact on Prediction (SHAP Sum)")
plt.xlabel("Month")

# 儲存圖表
plt.tight_layout()
plt.savefig(f"lgbm_{EXP_TAG}_month_preference.png", dpi=150)
plt.show()

print(f"Saved: lgbm_{EXP_TAG}_month_preference.png")

# 6. [選用] 計算每個月的平均影響力並印出
mean_impact = plot_df.groupby('Month', observed=False)['Month_Impact (SHAP Sum)'].mean()
print("\nAverage SHAP Impact by Month (Higher is better):")
print(mean_impact.sort_values(ascending=False))