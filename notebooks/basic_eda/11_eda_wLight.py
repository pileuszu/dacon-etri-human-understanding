# %% [markdown]
# # 스마트워치 조도 센서 (wLight) 심층 EDA
#
# 이 노트북은 `ch2025_wLight.parquet` 파일을 상세히 시각화하고 특징을 파악합니다.
# 스마트워치는 손목에 착용하기 때문에, 옷소매에 가려진 어두운 환경인지 실외에 직접 노출되었는지를 판단하는 데 활용됩니다.

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

DATA_PATH = os.path.join(BASE_DIR, "../../data/ch2025_data_items/ch2025_wLight.parquet")
df = pd.read_parquet(DATA_PATH)

# %% [markdown]
# ## 1. 데이터 확인

# %%
print(f"데이터 크기 (Shape): {df.shape}")
print(f"데이터 타입 (Data Types):\n{df.dtypes}")
df.head()

# %% [markdown]
# ## 2. 스마트워치 조도 값의 로그 변환 및 분포 분석

# %%
df["log_wlight"] = np.log1p(df["w_light"])

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.histplot(df["w_light"], bins=50, ax=axes[0], color="skyblue", kde=False)
axes[0].set_title("원시 워치 조도(Lux) 분포")
axes[0].set_xlabel("Watch Illuminance (lux)")

sns.histplot(df["log_wlight"], bins=50, ax=axes[1], color="deepskyblue", kde=True)
axes[1].set_title("로그 변환 워치 조도(log1p) 분포")
axes[1].set_xlabel("Log-scaled Illuminance")
plt.show()

# %% [markdown]
# ## 3. 참여자별 워치 조도값 비교

# %%
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x="subject_id", y="log_wlight", hue="subject_id", palette="Blues", legend=False)
plt.title("참여자별 스마트워치 조도 범위 비교 (로그 스케일)")
plt.xlabel("참여자 ID")
plt.ylabel("로그 조도값")
plt.show()

# %% [markdown]
# ## 4. 시간대별 워치 조도 변화 흐름

# %%
df["hour"] = df["timestamp"].dt.hour
hourly_wlight = df.groupby(["subject_id", "hour"])["w_light"].mean().reset_index()

plt.figure(figsize=(12, 6))
sns.lineplot(data=hourly_wlight, x="hour", y="w_light", hue="subject_id", palette="tab10", marker="o")
plt.title("참여자별 시간대에 따른 평균 워치 조도값 변화")
plt.xlabel("시간대 (Hour)")
plt.ylabel("평균 조도 (lux)")
plt.xticks(range(0, 24))
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# %% [markdown]
# ## 5. 피처 엔지니어링 제안
# * **`w_light_mean`**: 스마트워치 감지 하루 평균 밝기.
# * **`w_light_zero_ratio`**: 손목이 소매 등으로 완전히 가려졌거나 불이 꺼진 상태의 점유율.
# * **`w_light_variance`**: 하루 조도 변화의 표준편차.
