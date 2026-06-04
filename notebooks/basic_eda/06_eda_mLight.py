# %% [markdown]
# # 모바일 조도 센서 (mLight) 심층 EDA
#
# 이 노트북은 `ch2025_mLight.parquet` 파일을 상세 분석합니다.
# 스마트폰의 전면 조도 센서는 실내외 구분 및 스마트폰이 주머니에 있는지, 혹은 어두운 방에 있는지(수면 유추) 판단하는 결정적인 근거가 됩니다.

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

DATA_PATH = os.path.join(BASE_DIR, "../../data/ch2025_data_items/ch2025_mLight.parquet")
df = pd.read_parquet(DATA_PATH)

# %% [markdown]
# ## 1. 데이터 확인

# %%
print(f"데이터 크기 (Shape): {df.shape}")
print(f"데이터 타입 (Data Types):\n{df.dtypes}")
df.head()

# %% [markdown]
# ## 2. 조도(Lux) 값 분포
# 조도 값은 일반적으로 0 주변에 매우 강하게 수렴하고 매우 큰 값을 가진 꼬리 분포를 띱니다. 로그 스케일을 적용하여 봅니다.

# %%
df["log_light"] = np.log1p(df["m_light"])

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.histplot(df["m_light"], bins=50, ax=axes[0], color="orange", kde=False)
axes[0].set_title("원시 조도(Lux) 분포")
axes[0].set_xlabel("Illuminance (lux)")

sns.histplot(df["log_light"], bins=50, ax=axes[1], color="gold", kde=True)
axes[1].set_title("로그 변환 조도(log1p) 분포")
axes[1].set_xlabel("Log-scaled Illuminance")
plt.show()

# %% [markdown]
# ## 3. 참여자별 조도값 비교

# %%
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x="subject_id", y="log_light", hue="subject_id", palette="YlOrBr", legend=False)
plt.title("참여자별 스마트폰 조도 감지 범위 (로그 스케일)")
plt.xlabel("참여자 ID")
plt.ylabel("로그 조도값")
plt.show()

# %% [markdown]
# ## 4. 시간대별 조도 변화 흐름
# 낮과 밤에 스마트폰이 접하는 빛의 양 변화 패턴을 시각화합니다.

# %%
df["hour"] = df["timestamp"].dt.hour
hourly_light = df.groupby(["subject_id", "hour"])["m_light"].mean().reset_index()

plt.figure(figsize=(12, 6))
sns.lineplot(data=hourly_light, x="hour", y="m_light", hue="subject_id", palette="tab10", marker="o")
plt.title("참여자별 시간대에 따른 평균 조도값 변화")
plt.xlabel("시간대 (Hour)")
plt.ylabel("평균 조도 (lux)")
plt.xticks(range(0, 24))
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# %% [markdown]
# ## 5. 피처 엔지니어링 제안
# * **`light_mean`**: 하루 전체 평균 조도값.
# * **`light_zero_ratio`**: 하루 동안 조도가 0 lux인 상태의 비율 (스마트폰이 주머니에 있거나 밤에 불을 끄고 자는 시간).
# * **`light_high_lux_ratio`**: 실외 직사광선 수준(예: > 5000 lux)에 노출된 비율 (실외 활동 피처).
