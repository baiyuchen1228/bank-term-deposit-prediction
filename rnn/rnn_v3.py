import os, tensorflow as tf
import numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# 先求穩定：關 XLA / oneDNN；綁單卡 + 記憶體增長
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[0], 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


import pandas as pd
import numpy as np
import warnings
import os
from sklearn.model_selection import StratifiedKFold
from pandas.errors import PerformanceWarning
# from cuml.preprocessing import TargetEncoder
from sklearn.metrics import roc_auc_score
# from pytabkit import TabM_D_Classifier
from itertools import combinations
from tqdm import tqdm

warnings.simplefilter(action="ignore", category=PerformanceWarning)
TARGET = 'y'
NUMS = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
CATS = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

train = pd.read_csv('../data/train.csv', index_col='id')
test = pd.read_csv('../data/test.csv', index_col='id')
orig = pd.read_csv('../data/bank-full.csv', delimiter=';')
orig[TARGET] = orig[TARGET].replace({'no': 0, 'yes': 1})

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


FEATURES = train.columns.tolist()
FEATURES.remove(TARGET)


import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from tensorflow import keras
from tensorflow.keras import layers

# 只選純數值欄位，不包含 category / object
num_cols = train.select_dtypes(include=["number"]).columns.tolist()
if TARGET in num_cols:
    num_cols.remove(TARGET)
rnn_features = num_cols

# imp = pd.read_csv("shap_importance.csv")
# topK = 100
# keep_feats = imp.head(topK)["feature"].tolist()

# # 重新定義 rnn_features
# rnn_features = keep_feats


print("RNN uses", len(rnn_features), "numeric features.")

# def prepare_numeric(df, cols):
#     X = df[cols].copy()
#     X = X.replace([np.inf, -np.inf], np.nan)
#     return X

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision

# 建議開 AMP，RTX 4060 會更快
mixed_precision.set_global_policy("mixed_float16")  # 新增

def build_rnn(input_steps, use_bn=True):
    inp = keras.Input(shape=(input_steps, 1))
    x = layers.SpatialDropout1D(0.1)(inp)            # 用 SpatialDropout1D，不會破壞 cuDNN 路徑
    x = layers.GRU(128, dropout=0.0, recurrent_dropout=0.0)(x)  # 改 GRU，避免 recurrent_dropout
    if use_bn:
        x = layers.BatchNormalization()(x)            # BN 放在 RNN 後面
    x = layers.Dense(128)(x); x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid", dtype="float32")(x)  # 輸出回到 float32
    model = keras.Model(inp, out)
    opt = keras.optimizers.Adam(1e-3)  # AMP 下自動 loss scaling
    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
        jit_compile=False
    )
    return model

import os, hashlib
from tensorflow import keras

models, scalers, fold_scores = [], [], []


# === 建議：用「特徵指紋」避免權重汙染 ===
feat_fingerprint = hashlib.md5(",".join(sorted(rnn_features)).encode()).hexdigest()[:8]
ckpt_dir = "checkpoints_rnn_v1"
ckpt_path = os.path.join(ckpt_dir, f"rnn_full_{feat_fingerprint}.weights.h5")
os.makedirs(ckpt_dir, exist_ok=True)

def fold_ckpt_path(fold):
    return os.path.join(ckpt_dir, f"rnn_fold{fold}_{feat_fingerprint}.weights.h5")

def train_or_load(model, ds_tr, ds_va, ckpt_path):
    """
    若 ckpt 存在：load 並跳過訓練。
    否則：訓練 + ModelCheckpoint(以 val_auc 儲存最佳)。
    """
    if os.path.exists(ckpt_path):
        try:
            model.load_weights(ckpt_path)
            print(f"[RNN] Loaded weights: {ckpt_path} → skip training")
            return model, True
        except Exception as e:
            print(f"[RNN] Load failed ({e}); will retrain…")

    # 訓練流程（只在沒權重或載入失敗時執行）
    cbs = [
        keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_auc", mode="max",
            save_weights_only=True, save_best_only=True, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=3, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc", mode="max", factor=0.5, patience=2, verbose=0
        ),
    ]
    model.fit(ds_tr, validation_data=ds_va, epochs=50, callbacks=cbs, verbose=1)
    # 確保最後載入最佳權重（即使 restore_best_weights=True，仍保險再讀一次）
    model.load_weights(ckpt_path)
    print(f"[RNN] Trained & saved: {ckpt_path}")
    return model, False



skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(train), dtype=np.float32)
pred = np.zeros(len(test), dtype=np.float32)

for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(train)), train[TARGET]), 1):
    X_tr = train.loc[tr_idx, rnn_features].astype("float32").copy()
    X_va = train.loc[va_idx, rnn_features].astype("float32").copy()
    y_tr = train.loc[tr_idx, TARGET].astype("float32").values
    y_va = train.loc[va_idx, TARGET].astype("float32").values

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr);  X_va = scaler.transform(X_va)
    X_te = scaler.transform(test[rnn_features].astype("float32").copy())

    X_tr = X_tr[..., None]; X_va = X_va[..., None]; X_te = X_te[..., None]

    model = build_rnn(X_tr.shape[1], use_bn=True)
    cbs = [
        keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=3, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.5, patience=3, verbose=0),
    ]

    AUTO = tf.data.AUTOTUNE

    def make_ds(X, y=None, batch=1024, shuffle=False):
        if y is None:
            ds = tf.data.Dataset.from_tensor_slices((X,))
            ds = ds.batch(batch).prefetch(AUTO)
        else:
            ds = tf.data.Dataset.from_tensor_slices((X, y))
            if shuffle:
                ds = ds.shuffle(65536, reshuffle_each_iteration=True)
            ds = ds.batch(batch).prefetch(AUTO)
        return ds

    ds_tr = make_ds(X_tr, y_tr, batch=1024, shuffle=True)
    ds_va = make_ds(X_va, y_va, batch=1024)
    ds_te = make_ds(X_te, None, batch=1024)

    ckpt_path = fold_ckpt_path(fold)          # ★ 每折自己的 checkpoint
    model, loaded = train_or_load(model, ds_tr, ds_va, ckpt_path)

    # 無論是否訓練，都要做預測以產生 oof / test
    oof[va_idx] = model.predict(ds_va, verbose=0).ravel()
    pred += model.predict(ds_te, verbose=0).ravel()

    fold_auc = roc_auc_score(y_va, oof[va_idx])
    fold_scores.append(fold_auc)
    models.append(model)                      # ★ 保存本折模型
    scalers.append(scaler)                    # ★ 保存本折 scaler

    print(f"[RNN] Fold {fold} AUC: {roc_auc_score(y_va, oof[va_idx]):.6f}")

pred /= skf.get_n_splits()
print(f"[RNN] CV AUC: {roc_auc_score(train[TARGET], oof):.6f}")
print("Per-fold AUC:", [f"{s:.5f}" for s in fold_scores])

from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, confusion_matrix, roc_curve, auc

y_true = train[TARGET].values.astype(np.float32)
y_prob = oof.astype(np.float32)

# 1) AUROC
cv_auc = roc_auc_score(y_true, y_prob)
print(f"[METRIC] OOF AUROC: {cv_auc:.6f}")

# 2) PR 曲線 + F1 最佳閾值
prec, rec, thr = precision_recall_curve(y_true, y_prob)
# 對齊閾值長度（prec/rec 比 thr 多一個點）
thr_for_f1 = np.r_[thr, 1.0]
f1 = 2 * prec * rec / (prec + rec + 1e-12)
best_idx = np.nanargmax(f1)
best_thr = thr_for_f1[best_idx]
print(f"[METRIC] Best F1: {f1[best_idx]:.6f} @ threshold={best_thr:.6f}, precision={prec[best_idx]:.6f}, recall={rec[best_idx]:.6f}")

# 也可給幾個常用 operating points（你可選一個用於最終提交）
def summarize_at(t):
    y_pred = (y_prob >= t).astype(int)
    P = (y_pred == 1).sum()
    R = (y_true == 1).sum()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    prec_t = tp / max(tp+fp, 1)
    rec_t  = tp / max(tp+fn, 1)
    f1_t   = 2*prec_t*rec_t / max(prec_t+rec_t, 1e-12)
    return dict(threshold=float(t), precision=float(prec_t), recall=float(rec_t), f1=float(f1_t), tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))

candidates = [best_thr, 0.5, np.quantile(y_prob, 0.9), np.quantile(y_prob, 0.95)]
candidates = sorted(set([float(x) for x in candidates]))
summary = [summarize_at(t) for t in candidates]
print("[METRIC] Operating points:", summary)


# ===== SHAP：穩健版 =====
import shap, tensorflow as tf
import pandas as pd
import numpy as np

# 1) 用 AUC 最佳的那折做解釋
best_fold = int(np.argmax(fold_scores))  # 0-based
best_model = models[best_fold]
best_scaler = scalers[best_fold]
print(f"[SHAP] Use fold {best_fold+1} model for explanation.")

# 2) 取那折的驗證索引，準備 background / explain 子集
#    （若你有保存每折的 val index，可以直接復用；這裡重新切一遍取相同 fold）
skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
va_idx_list = [va for _, va in skf2.split(np.zeros(len(train)), train[TARGET])]
va_idx_best = va_idx_list[best_fold]

X_va_raw = train.loc[va_idx_best, rnn_features].astype("float32").copy()

n_bg, n_exp = 256, 512
bg_idx = np.random.choice(len(X_va_raw), size=min(n_bg, len(X_va_raw)), replace=False)
ex_idx = np.random.choice(len(X_va_raw), size=min(n_exp, len(X_va_raw)), replace=False)

# === 修正版：不要重建/包一層 Keras Model; 直接用 best_model ===
# (1) 先確保用 DataFrame 維持欄名，避免 scaler 警告
X_bg_df  = pd.DataFrame(X_va_raw.iloc[bg_idx].values, columns=rnn_features)
X_exp_df = pd.DataFrame(X_va_raw.iloc[ex_idx].values, columns=rnn_features)

# 保持為 NumPy arrays（float32, shape=(N, F, 1)）
X_bg  = best_scaler.transform(X_bg_df).astype("float32")[..., None]
X_exp = best_scaler.transform(X_exp_df).astype("float32")[..., None]

explainer = shap.GradientExplainer(best_model, X_bg)   # ← 傳 numpy
shap_vals = explainer.shap_values(X_exp)               # ← 傳 numpy

if isinstance(shap_vals, list):
    shap_vals = shap_vals[0]
shap_vals = np.asarray(shap_vals)
axes_to_reduce = tuple(i for i in range(shap_vals.ndim) if i != 1)
sv = np.mean(np.abs(shap_vals), axis=axes_to_reduce)   # -> (F,)


shap_importance = pd.DataFrame({
    "feature": rnn_features,
    "mean_|SHAP|": sv.astype(float)
}).sort_values("mean_|SHAP|", ascending=False)
print("[SHAP] Top-20 features:")
print(shap_importance.head(20).to_string(index=False))
shap_importance.to_csv("shap_importance.csv", index=False)

# === RNN SHAP 圖（summary + bar）===
# SHAP 對 RNN 輸出 shape 通常是 (N, F, 1)，summary_plot 需要 (N, F)
# 所以把最後一維壓掉
shap_vals_2d = shap_vals.reshape(shap_vals.shape[0], shap_vals.shape[1])
X_exp_2d = X_exp.reshape(X_exp.shape[0], X_exp.shape[1])

# 1) summary plot（蜂群圖）
plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_vals_2d,
    X_exp_2d,
    feature_names=rnn_features,
    show=False,
    max_display=20
)
plt.tight_layout()
plt.savefig("rnn_shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()

# 2) bar plot（平均 |SHAP| 排名）
plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_vals_2d,
    X_exp_2d,
    feature_names=rnn_features,
    plot_type="bar",
    show=False,
    max_display=20
)
plt.tight_layout()
plt.savefig("rnn_shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()

print("Saved: rnn_shap_summary.png, rnn_shap_bar.png")





y_true = train[TARGET].values.astype(np.float32)
y_prob = oof.astype(np.float32)

# ROC
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(5,4))
plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
plt.plot([0,1],[0,1],'--',lw=1)
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curve (OOF)"); plt.legend(); plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150); plt.close()
print("Saved: roc_curve.png")

# PR
prec, rec, thr = precision_recall_curve(y_true, y_prob)
ap = average_precision_score(y_true, y_prob)
plt.figure(figsize=(5,4))
plt.plot(rec, prec, label=f"AP = {ap:.4f}")
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("Precision-Recall Curve (OOF)"); plt.legend(); plt.tight_layout()
plt.savefig("pr_curve.png", dpi=150); plt.close()
print("Saved: pr_curve.png")

import pandas as pd

# 這裡沿用你前面算好的 prec/rec/thr 和 best_thr, f1
thr_for_f1 = np.r_[thr, 1.0]
f1 = 2*prec*rec/(prec+rec+1e-12)
best_idx = int(np.nanargmax(f1))
best_thr = float(thr_for_f1[best_idx])

rows = []
for t in sorted(set([best_thr, 0.5, np.quantile(y_prob, 0.9), np.quantile(y_prob, 0.95)])):
    y_pred = (y_prob >= t).astype(int)
    from sklearn.metrics import precision_score, recall_score, f1_score
    rows.append({
        "threshold": t,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0)
    })

pd.DataFrame(rows).to_csv("pr_operating_points.csv", index=False)
print("Saved: pr_operating_points.csv")

offset = 750000

# pred 是你5折平均的 test 機率
y_test = pred
pd.DataFrame({"id": test.index + offset, "y": y_test}).to_csv("submit_f1thr.csv", index=False)
print("Saved: submit_f1thr.csv (threshold = %.6f)" % best_thr)


