# %% [markdown]
# # 스마트워치 심박수 로그 (wHr) 심층 EDA
#
# 이 노트북은 스마트워치의 PPG 센서 데이터 `ch2025_wHr.parquet`을 심층 분석합니다.
# 심박수는 사용자의 자율신경계 반응(스트레스, 흥분) 및 신체 활성화, 수면 단계 등을 예측하는 핵심적인 신체 정보입니다.

# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Seaborn 테마 설정 시 폰트 정보를 함께 주입하여 덮어쓰기 방지 (Windows/Mac/Linux 대응)
sns.set_theme(
    style="whitegrid",
    rc={
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Malgun Gothic",
            "AppleGothic",
            "NanumGothic",
            "DejaVu Sans",
            "Arial",
            "Helvetica",
            "sans-serif",
        ],
        "axes.unicode_minus": False,
    },
)
plt.rcParams["figure.figsize"] = (12, 6)

# 현재 스크립트/노트북 위치 기준의 절대 경로 기준점 정의 (어디서 실행해도 경로 에러 방지)
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()

DATA_PATH = os.path.join(BASE_DIR, "../../data/ch2025_data_items/ch2025_wHr.parquet")
df = pd.read_parquet(DATA_PATH)

# %% [markdown]
# ## 1. 데이터 확인

# %%
print(f"데이터 크기 (Shape): {df.shape}")
df.head()

# %% [markdown]
# ## 2. 수집 주기당 측정된 심박 배열의 크기 분포

# %%
df["hr_count"] = df["heart_rate"].apply(lambda x: len(x) if x is not None else 0)

plt.figure(figsize=(10, 5))
sns.histplot(df["hr_count"], bins=20, color="crimson", kde=True)
plt.title("수집 주기당 기록된 심박 원시 데이터 개수")
plt.xlabel("심박수 데이터 수")
plt.ylabel("빈도")
plt.show()

# %% [markdown]
# ## 3. 전체 원시 심박수(BPM) 값 분포 분석
# 원시 배열에서 값들을 추출해 심박수의 정규 범위를 알아봅니다.

# %%
hr_values = []
for hr_arr in df["heart_rate"].sample(5000, random_state=42):
    hr_values.extend(hr_arr)

hr_values = np.array(hr_values)

plt.figure(figsize=(10, 5))
sns.histplot(hr_values, bins=50, kde=True, color="firebrick")
plt.title("전체 스마트워치 심박수(BPM) 분포")
plt.xlabel("Heart Rate (BPM)")
plt.ylabel("밀도")
plt.show()

# %%
print("심박수 요약 통계량:")
pd.Series(hr_values).describe()

# %% [markdown]
# ## 4. 참여자별 심박수 분포 비교

# %%
# 샘플링하여 피벗 통계 계산
sample_df = df.sample(15000, random_state=42).copy()
sample_df["mean_hr"] = sample_df["heart_rate"].apply(lambda x: np.mean(x) if (x is not None and len(x)>0) else np.nan)

plt.figure(figsize=(12, 6))
sns.boxplot(data=sample_df.dropna(), x="subject_id", y="mean_hr", hue="subject_id", palette="Reds", legend=False)
plt.title("참여자별 평균 심박수(BPM) 분포")
plt.xlabel("참여자 ID")
plt.ylabel("평균 심박수 (BPM)")
plt.show()

# %% [markdown]
# ## 5. 피처 엔지니어링 제안
# * **`hr_mean`**: 하루 전체 평균 심박수.
# * **`hr_std`**: 심박수 표준편차 (간접적인 심박 변이도 HRV 유도 피처).
# * **`hr_rest`**: 일일 최소 심박수 분위수(예: 5% 분위수)를 이용한 안정 시 심박수 추정.
