# Bank Marketing Term Deposit Prediction

Data Science Project 期末專題（Team 12）。

預測銀行電話行銷的客戶是否會訂閱定期存款（二元分類），資料來自
[Kaggle Playground Series S5E8](https://www.kaggle.com/competitions/playground-series-s5e8)，
並額外合併 UCI Bank Marketing 原始資料集。我們比較了 LightGBM、AdaBoost 與 RNN，
並用 SHAP 解釋模型、以 F1 選擇最佳分類閾值。

## 專案結構

```
.
├── data/              # 原始資料（需自行下載，見 data/README.md）
├── lgbm/              # LightGBM（target encoding + 特徵工程）
├── lgbm_no_encode/    # LightGBM（不做 encoding 的版本）
├── lgbm2/             # LightGBM（移除 duration，模擬真實情境）
├── AdaBoost/          # AdaBoost
├── rnn/               # RNN（v1 / v2 / v3）
├── result/            # 各模型最終結果彙整（ROC、PR、SHAP 圖與 CSV）
├── reports/           # 期中、期末報告 PDF
└── environment.yml    # Conda 環境
```

每個模型資料夾內的輸出包含：

- `*_roc_curve.png`, `*_pr_curve.png`：ROC / Precision-Recall 曲線
- `*_pr_operating_points.csv`：不同閾值下的 precision / recall / F1
- `*_shap_summary.png`, `*_shap_bar.png`, `*_shap_importance.csv`：SHAP 特徵重要性

## 環境安裝

```bash
conda env create -f environment.yml
conda activate rnn-tabm
```

## 執行方式

先依照 [`data/README.md`](data/README.md) 下載資料，接著進入各模型資料夾執行：

```bash
cd lgbm && python lgbm_shap.py
cd lgbm2 && python lgbm_shap_without_duration.py
cd AdaBoost && python AdaBoost.py
cd rnn && python rnn_v3.py
```

訓練好的模型、checkpoint 與 submission 檔案體積較大，未放入 repo，執行腳本即可重新產生。

## 報告

- 期末報告：[`reports/`](reports/)
