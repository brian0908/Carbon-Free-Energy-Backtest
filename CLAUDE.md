# CLAUDE.md — 台灣電網 24/7 CFE Score 歷史回溯分析

## Project Overview

本專案為台大「環境與能源的資料科學」課程期末個人報告。目標是回溯計算 2020–2025 年台灣電網的 hourly Carbon-Free Electricity (CFE) Score，模擬 RE100 企業在不同採購策略下的碳排表現，並以 ML 模型分析天氣對電網「綠度」的驅動因子。

本研究作為 TransitionZero (2025) 前瞻模型的歷史對照基準線。

## Tech Stack

- Python 3.10+
- pandas, numpy (資料處理)
- matplotlib, seaborn, plotly (視覺化)
- scikit-learn, xgboost (ML 建模)
- Jupyter Notebook (最終產出格式)

## 專案結構

```
project/
├── CLAUDE.md
├── data/
│   ├── raw/                    # 原始資料 (台電 CSV, 天氣 CSV)
│   ├── processed/              # 清理後的 hourly 資料
│   └── emission_factors.csv    # 燃料別排放係數查找表
├── notebooks/
│   └── Personal Report-B11208018 李適軒.ipynb      # 主 notebook (最終繳交物)
└── outputs/
    └── figures/                # 匯出的圖表
```

## 資料說明

### 1. 台電 10 分鐘發電資料

- 來源：台電官網 / 台灣電力資訊公開平台
- 範圍：2020–2025 (能拿到多少算多少)
- 格式：Parquet, 10 分鐘間隔
- 預處理：resample 成 hourly (取平均或加總，視單位而定)

### 2. 氣象資料

- 來源：中央氣象署 (CWB) / 學生已有資料
- 預處理：對齊到 hourly 時間軸，與台電資料 merge

### 3. Emission Factor 查找表 (手動建立)

```python
EMISSION_FACTORS = {
    "coal": 0.95,      # tCO2/MWh, IPCC 2006
    "gas_cc": 0.37,    # tCO2/MWh, IPCC 2006
    "oil": 0.65,       # tCO2/MWh, IPCC 2006
    "nuclear": 0.0,
    "solar": 0.0,
    "wind": 0.0,
    "hydro": 0.0,
    "geothermal": 0.0,
}
```

## 核心公式

### Grid CFE %

每小時電網中來自零碳電源的比例：

```python
grid_cfe_pct[h] = cfe_generation[h] / total_generation[h]
```

其中 CFE 電源 = 太陽能 + 風力 + 水力 + 核能 + 地熱
排除：抽蓄水力 (充放電來源混合)、汽電共生 (燃料別不明)

### Hourly Emission Intensity

```python
emission_intensity[h] = sum(generation_by_fuel[h] * ef[fuel]) / total_generation[h]
# 單位: gCO2/kWh (注意 tCO2/MWh → gCO2/kWh 的換算是 ×1000)
```

### RE100 企業 CFE Score (Google 24/7 CFE 方法論)

```python
contracted_cfe[h] = min(company_load[h], ppa_generation[h])
grid_import[h] = company_load[h] - contracted_cfe[h]
consumed_grid_cfe[h] = grid_import[h] * grid_cfe_pct[h]
cfe_score[h] = (contracted_cfe[h] + consumed_grid_cfe[h]) / company_load[h]

# 年度 CFE Score = mean(cfe_score) across all hours
```

### Annual vs Hourly Gap

```python
annual_cfe_claim = total_ppa_generation / total_company_load  # 通常設為 1.0 (100%)
hourly_cfe_average = mean(cfe_score)  # 每小時的平均
cfe_gap = annual_cfe_claim - hourly_cfe_average
```

## RE100 模擬情境 (4 種)

### Demand Profiles (2 種)

| ID | 名稱 | 建構方式 |
|---|---|---|
| D1 | 24/7 工廠 | 每小時固定 load = annual_total / 8760 |
| D2 | 9-5 辦公 | 工作日 8-18 時 load 為均值的 1.5 倍，其餘 0.5 倍；假日全天 0.3 倍 |

### Procurement Strategies (2 種)

| ID | 名稱 | PPA 發電曲線建構 |
|---|---|---|
| S1 | 純太陽能 PPA | ppa_gen[h] = actual_solar_profile[h] × scale_factor，scale 至年度 = annual_load |
| S2 | Solar 70% + Wind 30% | ppa_gen[h] = 0.7 × solar_profile[h] + 0.3 × wind_profile[h]，scale 至年度 = annual_load |

solar_profile 和 wind_profile 直接用台電資料中的太陽能/風力實際出力，正規化後 scale。

## ML 建模

### 目標

預測下一小時的 Grid CFE % (regression) 或是否高於中位數 (classification)。

### Features

```python
features = [
    # 時間特徵
    "hour_of_day",          # 0-23
    "month",                # 1-12
    "day_of_week",          # 0-6
    "is_weekend",           # 0/1
    
    # 天氣特徵
    "temperature",          # °C
    "solar_irradiance",     # W/m² 或 MJ/m²
    "wind_speed",           # m/s
    "rainfall",             # mm
    
    # Lag 特徵
    "cfe_lag_1h",           # 前 1 小時 CFE %
    "cfe_lag_24h",          # 前 24 小時 CFE %
    "cfe_rolling_7d_mean",  # 前 7 天滾動平均
]
```

### Models

1. Linear Regression (baseline)
2. Random Forest
3. XGBoost

### 評估

- Train/test split: 最後一年做 test set，其餘做 train
- Metrics: RMSE, MAE, R²
- Feature importance: 用 XGBoost 或 Random Forest 的 feature_importances_
- 視覺化: actual vs predicted scatter plot, feature importance bar chart

## Notebook 結構 (最終繳交)

Notebook 內需包含完整程式碼、註解、視覺化與文字敘述。

```
## 1. Introduction
- 研究問題與動機
- 與 TransitionZero (2025) 的關係
- CFE 定義與方法論說明

## 2. Data Loading & Preprocessing
- 載入台電 10-min 資料
- 載入天氣資料
- 10-min → hourly resampling
- 資料品質檢查 (缺值、異常值)
- 燃料分類與 CFE/non-CFE 標記
- 合併電力與天氣 DataFrame

## 3. Grid CFE % 歷史分析
- 計算 hourly Grid CFE %
- [圖] Heatmap: 24h × 12month 的 Grid CFE % (核心封面圖)
- 年度趨勢分析
- 月度 / 時段模式
- (若有 2025 資料) 核電除役前後比較

## 4. Hourly Emission Intensity
- 計算 hourly gCO2/kWh
- [圖] 時序趨勢線
- 與環境部年度公告值比較 (一段文字即可)

## 5. RE100 企業模擬
- 建構 2 種 demand profile
- 建構 2 種 PPA 策略
- 計算 4 種情境的 hourly CFE Score
- [圖] Annual claim vs Hourly reality gap (grouped bar chart)
- [圖] 某典型週的 hourly CFE 對照 (overlaid line chart)
- 發現與討論

## 6. ML: 天氣驅動因子分析
- Feature engineering
- Train/test split
- 訓練 3 個模型
- [圖] Actual vs Predicted
- [圖] Feature Importance
- 關鍵發現：哪些天氣因子最影響 Grid CFE %

## 7. Discussion & Limitations
- 排放係數精度限制 (燃料別 vs 機組級)
- 汽電共生與抽蓄排除的影響
- 單一氣象站代表性
- PPA 模擬假設的簡化
- 與 TransitionZero 2030 預測的對照

## 8. AI 聲明
- 原創性聲明
- AI 工具使用 (Claude, Claude Code)
- AI 功能：程式碼生成、資料分析建議、報告結構設計
- 代表性 prompt 範例
- 前後比對：AI 輔助前的初始構想 vs 最終產出的差異
```

## 視覺化規格

### 共通設定

```python
import matplotlib.pyplot as plt
plt.rcParams.update({
    "figure.figsize": (12, 6),
    "font.size": 12,
    "font.family": "sans-serif",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

dpi=300


### 必出圖表 (3 張核心)

1. **Grid CFE % Heatmap** — x: month, y: hour_of_day, color: mean CFE %
   - 用 seaborn heatmap, cmap="RdYlGn"
   - 一年一張，5 年平均一張

2. **Annual vs Hourly Gap** — grouped bar chart
   - x 軸: 4 種情境 (D1S1, D1S2, D2S1, D2S2)
   - 兩根 bar: Annual claim (都是 100%) vs Hourly average CFE score
   - Gap 用文字標註在圖上

3. **Feature Importance** — horizontal bar chart
   - 從 XGBoost/RF 取 feature_importances_
   - 排序後畫 barh

4. 典型一週 hourly CFE 時序 (line chart, 疊 PPA generation + grid import)
5. Emission intensity 年度趨勢 (line chart)
6. Actual vs Predicted scatter (ML 評估)



## 注意事項

- 所有程式碼加中文註解，簡要即可不需要很多註解
- Markdown cell 用中文撰寫分析敘事
- 圖表標題/軸標籤用英文 (學術慣例)，圖說用中文
- Notebook 要能從頭到尾 Run All 不報錯
- 最終產出是單一 .ipynb 檔案
