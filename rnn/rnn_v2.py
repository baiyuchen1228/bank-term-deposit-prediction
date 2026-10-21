import os
import tensorflow as tf
import pandas as pd
import numpy as np
import warnings
import hashlib
from sklearn.model_selection import StratifiedKFold, KFold
from pandas.errors import PerformanceWarning
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, confusion_matrix, roc_curve, auc, average_precision_score
from sklearn.metrics import precision_score, recall_score
from itertools import combinations
from tqdm import tqdm
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, Model
import tensorflow.keras.backend as K
from sklearn.preprocessing import StandardScaler

# --- 環境設定 ---
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("GPU setup complete.")
    except RuntimeError as e:
        print(e)

warnings.simplefilter(action="ignore", category=PerformanceWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)

# --- 全局變數 ---
EPOCHS = 10
BATCH_SIZE = 1024
TARGET = 'y'
NUMS = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
CATS = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

# --- 1. 資料讀取與前處理 ---
def load_and_preprocess_data():
    print("Loading data...")
    train = pd.read_csv('data/train.csv', index_col='id')
    test = pd.read_csv('data/test.csv', index_col='id')
    orig = pd.read_csv('data/bank-full.csv', delimiter=';')
    orig[TARGET] = orig[TARGET].replace({'no': 0, 'yes': 1})

    for df in [train, test, orig]:
        for col in CATS:
            df[col] = df[col].astype('category')
    
    print("Data loaded and basic types converted.")
    return train, test, orig

# --- 2. 特徵工程 ---
def kfold_target_encode(train_df, test_df, col, target, n_splits=5):
    te_col_series = pd.Series(np.nan, index=train_df.index, dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for _, (train_idx, val_idx) in enumerate(kf.split(train_df)):
        tr_fold, va_fold = train_df.iloc[train_idx], train_df.iloc[val_idx]
        target_mean = tr_fold.groupby(tr_fold[col].astype('object'))[target].mean()
        te_col_series.iloc[val_idx] = va_fold[col].astype('object').map(target_mean)
    global_mean = train_df[target].mean()
    te_col_series.fillna(global_mean, inplace=True)
    train_df[f'TE_{col}'] = te_col_series
    full_train_target_mean = train_df.groupby(train_df[col].astype('object'))[target].mean()
    test_df[f'TE_{col}'] = test_df[col].astype('object').map(full_train_target_mean)
    test_df[f'TE_{col}'].fillna(global_mean, inplace=True)
    return train_df, test_df

def create_adv_features(df):
    df['balance_per_age'] = df['balance'] / (df['age'] + 1e-6)
    df['pdays_contact_ratio'] = df['pdays'] / (df['contact'].astype('category').cat.codes + 1)
    df['duration_per_campaign'] = df['duration'] / (df['campaign'] + 1e-6)
    month_map = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6, 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
    df['month_num'] = df['month'].map(month_map).astype(int)
    df['day_month_sin'] = np.sin(2 * np.pi * (df['day'] + df['month_num']*30) / 365)
    return df

def engineer_features(train, test, orig):
    print("Engineering features...")
    
    for df in [train, test, orig]:
        df['_duration_sin'] = np.sin(2*np.pi * df['duration'] / 540).astype('float32')
        df['_balance_log'] = (np.sign(df['balance']) * np.log1p(np.abs(df['balance']))).astype('float32')

    TE_columns = []
    curated_interactions = [
        ('job', 'education'), ('job', 'marital'), 
        ('education', 'housing'), ('marital', 'housing'),
        ('age', 'balance')
    ]
    for cols in tqdm(curated_interactions, desc="Generating curated interaction features"):
        name = '-'.join(str(c) for c in cols)
        for df in [train, test, orig]:
            df[name] = df[cols[0]].astype(str) + '_' + df[cols[1]].astype(str)
        
        combined = pd.concat([train[name], test[name], orig[name]], ignore_index=True)
        combined, _ = combined.factorize()
        train[name] = combined[:len(train)]
        test[name] = combined[len(train):len(train) + len(test)]
        orig[name] = combined[len(train) + len(test):]
        TE_columns.append(name)

    print("Applying KFold Target Encoding and Count Encoding...")
    encode_cols = CATS + TE_columns
    for col in tqdm(encode_cols, desc="Encoding Features"):
        train, test = kfold_target_encode(train, test, col, TARGET)
        ce_map = pd.concat([train[col], test[col], orig[col]]).value_counts().to_dict()
        train[f'CE_{col}'] = train[col].map(ce_map).astype(float).fillna(0)
        test[f'CE_{col}'] = test[col].map(ce_map).astype(float).fillna(0)

    print("Creating advanced features...")
    train = create_adv_features(train)
    test = create_adv_features(test)
    
    print("Feature engineering complete.")
    return train, test

# --- 3. 模型架構 (中等複雜度版) ---
def build_medium_rnn(input_steps):
    inp = layers.Input(shape=(input_steps, 1))
    x = layers.SpatialDropout1D(0.2)(inp)
    
    # 使用單層雙向 GRU，速度比兩層快
    x = layers.Bidirectional(layers.GRU(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(x)
    
    attention = layers.Attention()([x, x])
    avg_pool = layers.GlobalAveragePooling1D()(attention)
    max_pool = layers.GlobalMaxPooling1D()(attention)
    x = layers.concatenate([avg_pool, max_pool])
    x = layers.LayerNormalization()(x)
    
    dense_out = layers.Dense(128, activation='leaky_relu')(x)
    dense_out = layers.Dropout(0.4)(dense_out)
    
    out = layers.Dense(1, activation="sigmoid")(dense_out)
    model = Model(inp, out)
    
    opt = tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-5)
    model.compile(
        optimizer=opt,
        loss='binary_crossentropy',
        metrics=[keras.metrics.AUC(name="auc")],
        jit_compile=False
    )
    return model

# --- 4. 訓練與評估 ---
def train_and_evaluate(train, test):
    print("Starting training and evaluation...")
    num_cols = [c for c in train.columns if train[c].dtype in ['int64', 'float32', 'int32', 'float64'] and c != TARGET]
    rnn_features = num_cols
    print(f"RNN uses {len(rnn_features)} numeric features.")

    feat_fingerprint = hashlib.md5(",".join(sorted(rnn_features)).encode()).hexdigest()[:8]
    ckpt_dir = "checkpoints_rnn_medium" # 使用新的目錄
    os.makedirs(ckpt_dir, exist_ok=True)

    def fold_ckpt_path(fold):
        return os.path.join(ckpt_dir, f"rnn_fold{fold}_{feat_fingerprint}.weights.h5")

    oof = np.zeros(len(train), dtype=np.float32)
    pred = np.zeros(len(test), dtype=np.float32)
    fold_thresholds = []
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, train[TARGET]), 1):
        print(f"\n--- Fold {fold} ---")
        X_tr_df, X_va_df = train.iloc[tr_idx], train.iloc[va_idx]
        y_tr, y_va = X_tr_df[TARGET].values, X_va_df[TARGET].values

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_df[rnn_features].fillna(0))
        X_va = scaler.transform(X_va_df[rnn_features].fillna(0))
        X_te = scaler.transform(test[rnn_features].fillna(0))

        X_tr = X_tr[..., None]; X_va = X_va[..., None]; X_te = X_te[..., None]

        model = build_medium_rnn(X_tr.shape[1])
        
        ds_tr = tf.data.Dataset.from_tensor_slices((X_tr, y_tr)).shuffle(1024*8).batch(1024).prefetch(tf.data.AUTOTUNE)
        ds_va = tf.data.Dataset.from_tensor_slices((X_va, y_va)).batch(1024).prefetch(tf.data.AUTOTUNE)
        ds_te = tf.data.Dataset.from_tensor_slices(X_te).batch(1024).prefetch(tf.data.AUTOTUNE)

        ckpt_path_fold = fold_ckpt_path(fold)
        cbs = [
            keras.callbacks.ModelCheckpoint(ckpt_path_fold, monitor="val_auc", mode="max", save_weights_only=True, save_best_only=True, verbose=1),
            keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.5, patience=2, verbose=1)
        ]
        model.fit(ds_tr, validation_data=ds_va, epochs=EPOCHS, callbacks=cbs, verbose=2)

        model.load_weights(ckpt_path_fold)
        
        oof[va_idx] = model.predict(ds_va, verbose=0).ravel()
        pred += model.predict(ds_te, verbose=0).ravel() / skf.get_n_splits()

        prec, rec, thr = precision_recall_curve(y_va, oof[va_idx])
        f1s = 2 * prec * rec / (prec + rec + 1e-12)
        best_thr_fold = thr[np.nanargmax(f1s[:-1])]
        fold_thresholds.append(best_thr_fold)
        print(f"Fold {fold} AUC: {roc_auc_score(y_va, oof[va_idx]):.6f}, Best F1: {np.nanmax(f1s):.6f} @ Thr: {best_thr_fold:.6f}")

    avg_best_thr = np.mean(fold_thresholds)
    print(f"\nCV AUC: {roc_auc_score(train[TARGET], oof):.6f}")
    print(f"Average Best Threshold: {avg_best_thr:.6f}")
    
    return oof, pred, avg_best_thr

# --- 5. 報告與提交 ---
def generate_report_and_submission(y_true, oof_probs, test_preds, test_ids, best_threshold):
    print("\n--- Generating Final Report and Submission ---")
    y_pred_best = (oof_probs >= best_threshold).astype(int)
    f1_best = f1_score(y_true, y_pred_best)
    auroc = roc_auc_score(y_true, oof_probs)
    ap = average_precision_score(y_true, oof_probs)
    
    metrics_df = pd.DataFrame({
        "Metric": ["OOF AUROC", "OOF Avg Precision", "OOF Best F1", "Selected Threshold"],
        "Score": [auroc, ap, f1_best, best_threshold]
    })
    print("\n--- OOF Performance Summary ---")
    print(metrics_df.to_string(index=False))
    
    fpr, tpr, _ = roc_curve(y_true, oof_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auroc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--'); plt.title('ROC Curve (OOF)'); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.legend(); plt.tight_layout()
    plt.savefig("final_roc_curve.png", dpi=150)
    plt.close()
    print("Saved: final_roc_curve.png")

    prec, rec, _ = precision_recall_curve(y_true, oof_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, label=f'PR Curve (AP = {ap:.4f})')
    plt.title('Precision-Recall Curve (OOF)'); plt.xlabel('Recall'); plt.ylabel('Precision'); plt.legend(); plt.tight_layout()
    plt.savefig("final_pr_curve.png", dpi=150)
    plt.close()
    print("Saved: final_pr_curve.png")

    y_test_final = (test_preds >= best_threshold).astype(int)
    submission_df = pd.DataFrame({'id': test_ids, 'y': y_test_final})
    submission_df.to_csv('submission_final.csv', index=False)
    print(f"Saved: submission_final.csv (using threshold {best_threshold:.6f})")

# --- 主流程 ---
def main():
    train, test, orig = load_and_preprocess_data()
    train, test = engineer_features(train, test, orig)
    oof, pred, best_thr = train_and_evaluate(train, test)
    generate_report_and_submission(train[TARGET].values, oof, pred, test.index, best_thr)

if __name__ == "__main__":
    main()
