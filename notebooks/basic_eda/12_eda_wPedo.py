# %% [markdown]
# # 스마트워치 만보기 로그 (wPedo) 심층 EDA
#
# 이 노트북은 스마트워치의 운동 정보인 `ch2025_wPedo.parquet`을 깊이 분석합니다.
# 걸음 수와 칼로리 소모량은 사용자의 일일 물리적 에너지 소비량을 파악하는 주된 척도입니다.

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

DATA_PATH = os.path.join(BASE_DIR, "../../data/ch2025_data_items/ch2025_wPedo.parquet")
df = pd.read_parquet(DATA_PATH)

# %% [markdown]
# ## 1. 데이터 확인

# %%
print(f"데이터 크기 (Shape): {df.shape}")
print(f"데이터 타입 (Data Types):\n{df.dtypes}")
df.head()

# %% [markdown]
# ## 2. 수집된 걸음 수(Step) 분포 탐색
# 0 걸음이 대다수를 차지하므로, 0을 제외한 유효 움직임(걸음 수 > 0)의 통계치를 분석합니다.

# %%
df_active = df[df["step"] > 0]
print(f"유효 움직임 기록 개수: {df_active.shape[0]} / {df.shape[0]} ({df_active.shape[0]/df.shape[0]*100:.2f}%)")

plt.figure(figsize=(10, 5))
sns.histplot(df_active["step"], bins=50, kde=True, color="coral")
plt.title("0 걸음을 제외한 구간별 유효 걸음 수 분포")
plt.xlabel("걸음 수 (Step)")
plt.ylabel("빈도")
plt.show()

# %% [markdown]
# ## 3. 칼로리, 거리 및 속도 간의 피어슨 상관관계

# %%
plt.figure(figsize=(8, 6))
pedo_corr = df[["step", "distance", "speed", "burned_calories"]].corr()
sns.heatmap(pedo_corr, annot=True, cmap="Oranges", fmt=".3f")
plt.title("Pedometer 지표 간 상관계수 행렬")
plt.show()

# %% [markdown]
# ## 4. 시간대별 누적 걸음 수 패턴 분석
# 하루 중 신체 활동량이 가장 높은 Golden Hour를 찾습니다.

# %%
df["hour"] = df["timestamp"].dt.hour
hourly_steps = df.groupby(["subject_id", "hour"])["step"].mean().reset_index()

plt.figure(figsize=(12, 6))
sns.lineplot(data=hourly_steps, x="hour", y="step", hue="subject_id", palette="tab10", marker="o")
plt.title("참여자별 시간대당 평균 걸음 수")
plt.xlabel("시간대 (Hour)")
plt.ylabel("평균 걸음 수")
plt.xticks(range(0, 24))
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# %% [markdown]
# ## 5. 피처 엔지니어링 제안
# * **`pedo_steps_sum`**: 하루 총 누적 걸음 수.
# * **`pedo_calories_sum`**: 하루 총 칼로리 소비량.
# * **`pedo_active_hours`**: 하루 동안 10걸음 이상 걸은 활성 시간(Hours)의 총합.
