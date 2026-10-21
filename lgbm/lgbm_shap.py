import os, tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, f1_score,
    confusion_matrix, roc_curve, auc, average_precision_score,
    precision_score, recall_score
)
# import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import warnings
from pandas.errors import PerformanceWarning
# from cuml.preprocessing import TargetEncoder
# from pytabkit import TabM_D_Classifier
from itertools import combinations
from tqdm import tqdm
import sys

# ====== [新增] 解決 LightGBM OSEnvironmentError ======
# 將 Conda 環境的 lib 目錄添加到動態連結函式庫搜索路徑中
# 這會確保載入 LightGBM 時，優先找到 Conda 環境中的 libstdc++.so.6
# CONDA_ENV_LIB_PATH = "/home/bai/miniconda3/envs/rnn-tabm/lib"

# if CONDA_ENV_LIB_PATH not in os.environ.get('LD_LIBRARY_PATH', ''):
#     # 將新路徑加到現有路徑的最前面 (優先)
#     os.environ['LD_LIBRARY_PATH'] = CONDA_ENV_LIB_PATH + ":" + os.environ.get('LD_LIBRARY_PATH', '')
# # ======================================================

import lightgbm as lgb

# 先求穩定：關 XLA / oneDNN；綁單卡 + 記憶體增長
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[0], 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

warnings.simplefilter(action="ignore", category=PerformanceWarning)
TARGET = 'y'
NUMS = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
CATS = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

train = pd.read_csv('../data/train.csv', index_col='id')
test = pd.read_csv('../data/test.csv', index_col='id')
orig = pd.read_csv('../data/bank-full.csv', delimiter=';')
# 修改 lgbm_shap.py 第 49 行：
orig[TARGET] = orig[TARGET].replace({'no': 0, 'yes': 1}).infer_objects(copy=False)

train[CATS] = train[CATS].astype('category')
test[CATS] = test[CATS].astype('category')
orig[CATS] = orig[CATS].astype('category')

train['_duration_sin'] = np.sin(2*np.pi * train['duration'] / 540).astype('float32')
train['_duration_cos'] = np.cos(2*np.pi * train['duration'] / 540).astype('float32')
train['_balance_log'] = (np.sign(train['balance']) * np.log1p(np.abs(train['balance']))).astype('float32')
train['_balance_sin'] = np.sin(2*np.pi * train['balance'] / 1000).astype('float32')
train['_balance_cos'] = np.cos(2*np.pi * train['balance'] / 1000).astype('float32')
train['_age_sin'] = np.sin(2*np.pi * train['age'] / 10).astype('float32')
train['_pdays_sin'] = np.sin(2*np.pi * train['pdays'] / 7).astype('float32')

test['_duration_sin'] = np.sin(2*np.pi * test['duration'] / 540).astype('float32')
test['_duration_cos'] = np.cos(2*np.pi * test['duration'] / 540).astype('float32')
test['_balance_log'] = (np.sign(test['balance']) * np.log1p(np.abs(test['balance']))).astype('float32')
test['_balance_sin'] = np.sin(2*np.pi * test['balance'] / 1000).astype('float32')
test['_balance_cos'] = np.cos(2*np.pi * test['balance'] / 1000).astype('float32')
test['_age_sin'] = np.sin(2*np.pi * test['age'] / 10).astype('float32')
test['_pdays_sin'] = np.sin(2*np.pi * test['pdays'] / 7).astype('float32')

orig['_duration_sin'] = np.sin(2*np.pi * orig['duration'] / 540).astype('float32')
orig['_duration_cos'] = np.cos(2*np.pi * orig['duration'] / 540).astype('float32')
orig['_balance_log'] = (np.sign(orig['balance']) * np.log1p(np.abs(orig['balance']))).astype('float32')
orig['_balance_sin'] = np.sin(2*np.pi * orig['balance'] / 1000).astype('float32')
orig['_balance_cos'] = np.cos(2*np.pi * orig['balance'] / 1000).astype('float32')
orig['_age_sin'] = np.sin(2*np.pi * orig['age'] / 10).astype('float32')
orig['_pdays_sin'] = np.sin(2*np.pi * orig['pdays'] / 7).astype('float32')

columns = NUMS

for r in [2]:
    for cols in tqdm(list(combinations(columns, r))):
        name = '_x_'.join(cols)

        train[name] = train[cols[0]]
        for col in cols[1:]:
            train[name] = train[name] * train[col]

        test[name] = test[cols[0]]
        for col in cols[1:]:
            test[name] = test[name] * test[col]

        orig[name] = orig[cols[0]]
        for col in cols[1:]:
            orig[name] = orig[name] * orig[col]

TE_columns = []

columns = NUMS + CATS

for r in [2]:
    for cols in tqdm(list(combinations(columns, r))):
        name = '-'.join(cols)

        train[name] = train[cols[0]].astype(str)
        for col in cols[1:]:
            train[name] = train[name] + '_' + train[col].astype(str)

        test[name] = test[cols[0]].astype(str)
        for col in cols[1:]:
            test[name] = test[name] + '_' + test[col].astype(str)

        orig[name] = orig[cols[0]].astype(str)
        for col in cols[1:]:
            orig[name] = orig[name] + '_' + orig[col].astype(str)

        combined = pd.concat([train[name], test[name], orig[name]], ignore_index=True)
        combined, _ = combined.factorize()
        train[name] = combined[:len(train)]
        test[name] = combined[len(train):len(train) + len(test)]
        orig[name] = combined[len(train) + len(test):]

        TE_columns.append(name)

TE_ORIG = []
CC = TE_columns + CATS + NUMS

# print("test")
# for i,c in enumerate(CC):
#     print(i,c)

# sys.exit()


print(f"Processing {len(CC)} columns... ", end="")
for i, c in enumerate(CC):
    if i % 10 == 0: 
        print(f"{i}, ", end="")
    
    # Target Encoding (TE) - 目標平均值
    te_values = orig.groupby(c)['y'].mean().astype('float32')
    te_values.name = f"TE_ORIG_{c}"
    TE_ORIG.append(f"TE_ORIG_{c}")
    
    # 合併 TE 到 train 和 test
    train = train.merge(te_values, on=c, how='left')
    train[te_values.name] = train[te_values.name].fillna(train[te_values.name].mean())
    
    test = test.merge(te_values, on=c, how='left')
    test[te_values.name] = test[te_values.name].fillna(train[te_values.name].mean())
    
    # Count Encoding (CE) - 類別計數
    ce_values = orig[c].value_counts()
    TE_ORIG.append(f"CE_ORIG_{c}")
    
    # 安全的 map 操作
    train_ce = train[c].map(ce_values)
    test_ce = test[c].map(ce_values)
    
    # 處理可能的 categorical 類型
    if hasattr(train_ce, 'cat'):
        train_ce = train_ce.astype('float32')
    if hasattr(test_ce, 'cat'):
        test_ce = test_ce.astype('float32')
    
    train[f'CE_ORIG_{c}'] = train_ce.fillna(0)
    test[f'CE_ORIG_{c}'] = test_ce.fillna(0)

print("\nEncoding complete!")


# output after preprocessing dataset
train.to_csv("lgbm_train_preprocessed.csv", index=True)
test.to_csv("lgbm_test_preprocessed.csv", index=True)
print("Saved: lgbm_train_preprocessed.csv, lgbm_test_preprocessed.csv")
sys.exit()

FEATURES = train.columns.tolist()
FEATURES.remove(TARGET)


# =======================
# 1. 決定這個模型要用哪些 features
# =======================

# 先簡單一版：用所有 numeric 欄位（不含 y）
num_cols = train.select_dtypes(include=["number"]).columns.tolist()
if TARGET in num_cols:
    num_cols.remove(TARGET)

FEATS = num_cols
print("LGBM uses", len(FEATS), "numeric features.")

X = train[FEATS].values.astype("float32")
y = train[TARGET].values.astype("int8")
X_test = test[FEATS].values.astype("float32")



# output after preprocessing dataset



# =======================
# 2. 5-fold OOF 訓練 + 預測
# =======================



model_dir = "lgbm_cv_models"
os.makedirs(model_dir, exist_ok=True)

def fold_model_path(fold):
    return os.path.join(model_dir, f"lgbm_fold{fold}.txt")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(train), dtype=np.float32)
pred = np.zeros(len(test), dtype=np.float32)
models = []
fold_scores = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n[Fold {fold}]")
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

    # 如果已經有訓練好的模型，直接讀取
    if os.path.exists(model_path):
        print(f"[LGBM] Fold {fold}: load from {model_path}")
        model = lgb.Booster(model_file=model_path)
    else:
        print(f"[LGBM] Fold {fold}: train new model")
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
        print(f"[LGBM] Fold {fold}: saved to {model_path}")

    # OOF / test 預測
    oof[va_idx] = model.predict(X_va, num_iteration=model.best_iteration)
    pred += model.predict(X_test, num_iteration=model.best_iteration) / skf.n_splits

    fold_auc = roc_auc_score(y_va, oof[va_idx])
    fold_scores.append(fold_auc)
    models.append(model)

    print(f"[LGBM] Fold {fold} AUC: {fold_auc:.6f}")

cv_auc = roc_auc_score(y, oof)
print(f"[LGBM] CV AUC: {cv_auc:.6f}")
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

print(f"[METRIC] Best F1: {f1[best_idx]:.6f} @ threshold={best_thr:.6f}, "
      f"precision={prec[best_idx]:.6f}, recall={rec[best_idx]:.6f}")

def summarize_at(t):
    y_pred = (y_prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    prec_t = tp / max(tp+fp, 1)
    rec_t  = tp / max(tp+fn, 1)
    f1_t   = 2*prec_t*rec_t / max(prec_t+rec_t, 1e-12)
    return dict(
        threshold=float(t),
        precision=float(prec_t),
        recall=float(rec_t),
        f1=float(f1_t),
        tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn)
    )

candidates = [best_thr, 0.5, np.quantile(y_prob, 0.9), np.quantile(y_prob, 0.95)]
candidates = sorted(set([float(x) for x in candidates]))
summary = [summarize_at(t) for t in candidates]
print("[METRIC] Operating points:", summary)

# =======================
# 4. SHAP (TreeExplainer)
# =======================

# 用 AUC 最好的那一折模型
best_fold = int(np.argmax(fold_scores))
best_model = models[best_fold]
print(f"[SHAP] Use fold {best_fold+1} model for explanation.")

# 抽一部分樣本來做 SHAP（避免太慢）
N_BG = 1000
N_EXP = 3000
idx_all = np.arange(len(train))
np.random.seed(123)
bg_idx = np.random.choice(idx_all, size=min(N_BG, len(idx_all)), replace=False)
exp_idx = np.random.choice(idx_all, size=min(N_EXP, len(idx_all)), replace=False)

X_bg = X[bg_idx]
X_exp = X[exp_idx]

explainer = shap.TreeExplainer(best_model)
shap_vals = explainer.shap_values(X_exp)

# 對二元分類，shap_values 通常是 list[2]，第二個是 positive class
if isinstance(shap_vals, list):
    # 取對「正類」的 SHAP
    shap_vals = shap_vals[1]

# 計算 global importance
mean_abs_shap = np.abs(shap_vals).mean(axis=0)  # shape = (num_features,)

shap_importance = pd.DataFrame({
    "feature": FEATS,
    "mean_|SHAP|": mean_abs_shap
}).sort_values("mean_|SHAP|", ascending=False)

print("[SHAP] Top-20 features:")
print(shap_importance.head(20).to_string(index=False))
shap_importance.to_csv("lgbm_shap_importance.csv", index=False)

# ==============================
# 4.1 SHAP Summary & Bar Plots
# ==============================

print("[SHAP] Generating summary plot & bar plot...")

# SHAP for LGBM: shap_vals 是 (N, F)，可以直接畫
plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_vals, 
    X_exp,
    feature_names=FEATS,
    show=False,
    max_display=30
)
plt.tight_layout()
plt.savefig("lgbm_shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()

# Bar plot
plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_vals,
    X_exp,
    feature_names=FEATS,
    plot_type="bar",
    show=False,
    max_display=20
)
plt.tight_layout()
plt.savefig("lgbm_shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()

print("Saved: lgbm_shap_summary.png, lgbm_shap_bar.png")



# =======================
# 5. ROC / PR 圖 + PR operating points
# =======================

fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(5,4))
plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
plt.plot([0,1],[0,1],'--',lw=1)
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curve (OOF, LGBM)")
plt.legend(); plt.tight_layout()
plt.savefig("lgbm_roc_curve.png", dpi=150); plt.close()
print("Saved: lgbm_roc_curve.png")

prec, rec, thr = precision_recall_curve(y_true, y_prob)
ap = average_precision_score(y_true, y_prob)
plt.figure(figsize=(5,4))
plt.plot(rec, prec, label=f"AP = {ap:.4f}")
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("Precision-Recall Curve (OOF, LGBM)")
plt.legend(); plt.tight_layout()
plt.savefig("lgbm_pr_curve.png", dpi=150); plt.close()
print("Saved: lgbm_pr_curve.png")

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

pd.DataFrame(rows).to_csv("lgbm_pr_operating_points.csv", index=False)
print("Saved: lgbm_pr_operating_points.csv")

# =======================
# 6. 產 Kaggle 提交檔
# =======================

offset = 750000

pd.DataFrame({
    "id": test.index + offset,
    "y": pred
}).to_csv(
    "lgbm_submit_f1thr.csv",
    index=False,
    float_format="%.12f"
)


print("Saved: lgbm_submit_f1thr.csv (no scientific notation)")

