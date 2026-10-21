import os
import numpy as np
import joblib
import pandas as pd
import warnings

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve,
    confusion_matrix, precision_score, recall_score, f1_score
)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from pandas.errors import PerformanceWarning
from itertools import combinations
from tqdm import tqdm

warnings.simplefilter(action="ignore", category=PerformanceWarning)

TARGET = "y"
NUMS = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
CATS = ['job', 'marital', 'education', 'default', 'housing', 'loan',
        'contact', 'month', 'poutcome']

# ========= 讀資料 =========
train = pd.read_csv('../data/train.csv', index_col='id')
test  = pd.read_csv('../data/test.csv',  index_col='id')
orig  = pd.read_csv('../data/bank-full.csv', delimiter=';')
orig[TARGET] = orig[TARGET].replace({'no': 0, 'yes': 1})

train[CATS] = train[CATS].astype('category')
test[CATS]  = test[CATS].astype('category')
orig[CATS]  = orig[CATS].astype('category')

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


FEATURES = train.columns.tolist()
FEATURES.remove(TARGET)

from tqdm import tqdm
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

class AdaBoostWithProgress(AdaBoostClassifier):
    def fit(self, X, y):
        self.progress = tqdm(total=self.n_estimators, desc="AdaBoost training")
        return super().fit(X, y)
    
    def _boost(self, iboost, X, y, sample_weight, random_state):
        result = super()._boost(iboost, X, y, sample_weight, random_state)
        self.progress.update(1)
        return result


# 只保留 numeric feature，避免 category / object
num_features = train[FEATURES].select_dtypes(include=["number"]).columns.tolist()
print("Num features for AdaBoost:", len(num_features))

# 使用驗證集版本
from sklearn.model_selection import train_test_split

X      = train[num_features].values.astype("float32")
X_test = test[num_features].values.astype("float32")
y      = train[TARGET].values.astype("int8")

# 分割訓練集和驗證集（80/20）
X_tr, X_va, y_tr, y_va = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

print(f"Training samples: {len(X_tr)}, Validation samples: {len(X_va)}")

# 建立或載入模型
model_path = "models/adaboost_full.pkl"
if os.path.isfile(model_path):
    print(f"[INFO] Loading existing model from {model_path}")
    clf = joblib.load(model_path)
else:
    print("Training AdaBoost...")
    base = DecisionTreeClassifier(max_depth=1, random_state=42)
    clf = AdaBoostWithProgress(
        estimator=base,
        n_estimators=100,
        learning_rate=0.05,
        algorithm="SAMME",
        random_state=42
    )
    clf.fit(X_tr, y_tr)
    if hasattr(clf, "progress"):
        clf.progress.close()
        del clf.progress
    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, model_path)
    print(f"[INFO] Saved model to {model_path}")

# 驗證集評估
pred_va = clf.predict_proba(X_va)[:, 1]
va_auc = roc_auc_score(y_va, pred_va)
print(f"[AdaBoost] Validation AUC: {va_auc:.6f}")

# 測試集預測
pred_test = clf.predict_proba(X_test)[:, 1]

# 使用驗證集找最佳閾值
y_true = y_va.astype(np.float32)
y_prob = pred_va.astype(np.float32)
prec, rec, thr = precision_recall_curve(y_true, y_prob)
thr_for_f1 = np.r_[thr, 1.0]
f1 = 2 * prec * rec / (prec + rec + 1e-12)
best_idx = np.nanargmax(f1)
best_thr = float(thr_for_f1[best_idx])
print(f"[METRIC] Best F1: {f1[best_idx]:.6f} @ threshold={best_thr:.6f}")

# 生成提交檔案
offset = 750000
y_test = pred_test

sub = pd.DataFrame({
    "id": np.arange(len(pred_test)) + offset,
    "y": y_test
})
sub.to_csv("ada_submit_f1thr.csv", index=False)
print("Saved: ada_submit_f1thr.csv (threshold = %.6f)" % best_thr)

# ==========================
# 7. 圖片與評估資料：ROC / PR
# ==========================
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

y_true = y_va.astype(np.float32)
y_prob = pred_va.astype(np.float32)

# -------- ROC 曲線 --------
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(5,4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0,1],[0,1],'--',lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (AdaBoost)")
plt.legend()
plt.tight_layout()
plt.savefig("ada_roc_curve.png", dpi=150)
plt.close()
print("Saved: ada_roc_curve.png")

# -------- PR 曲線 --------
prec, rec, thr = precision_recall_curve(y_true, y_prob)
ap = average_precision_score(y_true, y_prob)

plt.figure(figsize=(5,4))
plt.plot(rec, prec, label=f"AP = {ap:.4f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (AdaBoost)")
plt.legend()
plt.tight_layout()
plt.savefig("ada_pr_curve.png", dpi=150)
plt.close()
print("Saved: ada_pr_curve.png")

# -------- PR Operating Points --------
thr_for_f1 = np.r_[thr, 1.0]
f1 = 2 * prec * rec / (prec + rec + 1e-12)
best_idx = int(np.nanargmax(f1))
best_thr = float(thr_for_f1[best_idx])

rows = []
for t in sorted(set([best_thr, 0.5, np.quantile(y_prob, 0.9), np.quantile(y_prob, 0.95)])):
    y_pred = (y_prob >= t).astype(int)
    rows.append({
        "threshold": t,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0)
    })

pd.DataFrame(rows).to_csv("ada_pr_operating_points.csv", index=False)
print("Saved: ada_pr_operating_points.csv")

# ==========================
# 8. 特徵重要度分析
# ==========================
print("\n[Feature Analysis] Starting...")

# ========== 方法1：使用內建的 feature_importances_ ==========
print("[1/2] Calculating feature importances...")
feature_importance = clf.feature_importances_

importance_df = pd.DataFrame({
    "feature": num_features,
    "importance": feature_importance
}).sort_values("importance", ascending=False)

print("[Feature Importance] Top-20 features:")
print(importance_df.head(20).to_string(index=False))

importance_df.to_csv("ada_feature_importance.csv", index=False)
print("Saved: ada_feature_importance.csv")

# 繪製特徵重要度
plt.figure(figsize=(10, 8))
top_n = 30
top_features = importance_df.head(top_n)
plt.barh(range(top_n), top_features['importance'].values)
plt.yticks(range(top_n), top_features['feature'].values, fontsize=8)
plt.xlabel('Importance')
plt.title(f'Top {top_n} Feature Importances (AdaBoost)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("ada_feature_importance.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: ada_feature_importance.png")

# ========== 方法2：SHAP 分析（改進版） ==========
try:
    import shap
    from sklearn.preprocessing import StandardScaler
    
    print("\n[2/2] Generating SHAP values (this may take a while)...")
    
    # 標準化特徵以避免數值不穩定
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_va_scaled = scaler.transform(X_va)
    
    # 創建使用標準化特徵的包裝函數
    def predict_scaled(X_scaled):
        # 反標準化後預測
        X_original = scaler.inverse_transform(X_scaled)
        return clf.predict_proba(X_original)[:, 1]
    
    # 取樣以加快速度
    sample_size = min(300, len(X_va_scaled))  # 減少樣本數
    background_size = min(30, len(X_tr_scaled))  # 減少背景樣本數
    
    sample_indices = np.random.choice(len(X_va_scaled), sample_size, replace=False)
    background_indices = np.random.choice(len(X_tr_scaled), background_size, replace=False)
    
    X_sample = X_va_scaled[sample_indices]
    X_background = X_tr_scaled[background_indices]
    
    print(f"[SHAP] Using {len(X_sample)} samples and {len(X_background)} background samples...")
    
    # 使用 KernelExplainer with L1 regularization
    explainer = shap.KernelExplainer(
        predict_scaled,
        X_background,
        link="identity"  # 使用 identity link
    )
    
    # 計算 SHAP 值（增加 nsamples 和 l1_reg）
    shap_vals = explainer.shap_values(
        X_sample, 
        nsamples=200,  # 增加採樣數以提高穩定性
        l1_reg="num_features(10)"  # 添加 L1 正則化
    )
    shap_vals = np.array(shap_vals)
    
    # 檢查 SHAP 值的合理性
    print(f"[SHAP] SHAP values range: {shap_vals.min():.4f} to {shap_vals.max():.4f}")
    print(f"[SHAP] SHAP values mean abs: {np.mean(np.abs(shap_vals)):.4f}")
    
    # 平均絕對 SHAP 值
    mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
    
    shap_importance = pd.DataFrame({
        "feature": num_features,
        "mean_|SHAP|": mean_abs_shap
    }).sort_values("mean_|SHAP|", ascending=False)
    
    print("[SHAP] Top-20 features:")
    print(shap_importance.head(20).to_string(index=False))
    
    shap_importance.to_csv("ada_shap_importance.csv", index=False)
    print("Saved: ada_shap_importance.csv")
    
    # SHAP summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_vals, X_sample, feature_names=num_features, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig("ada_shap_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: ada_shap_summary.png")
    
    # SHAP bar plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_vals, X_sample, feature_names=num_features, 
                      plot_type="bar", show=False, max_display=20)
    plt.tight_layout()
    plt.savefig("ada_shap_bar.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: ada_shap_bar.png")
    
except Exception as e:
    print(f"\n[SHAP] Warning: SHAP analysis failed with error: {e}")
    print("[SHAP] Skipping SHAP analysis. Using feature_importances_ instead.")

print("\n[Feature Analysis] Complete!")