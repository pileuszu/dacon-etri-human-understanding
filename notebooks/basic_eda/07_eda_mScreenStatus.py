# %% [markdown]
# # 모바일 화면 사용 여부 (mScreenStatus) 심층 EDA
#
# 이 노트북은 `ch2025_mScreenStatus.parquet` 파일을 탐색합니다.
# 화면 켜짐 상태는 사용자가 깨어있고 스마트폰을 실시간으로 조작하고 있음을 직접적으로 증명하는 데이터입니다.

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

DATA_PATH = os.path.join(BASE_DIR, "../../data/ch2025_data_items/ch2025_mScreenStatus.parquet")
df = pd.read_parquet(DATA_PATH)

# %% [markdown]
# ## 1. 데이터 확인

# %%
print(f"데이터 크기 (Shape): {df.shape}")
print(f"데이터 타입 (Data Types):\n{df.dtypes}")
df.head()

# %% [markdown]
# ## 2. 참여자별 화면 사용 빈도 분석
# 스마트폰 화면이 활성화(1) 되어 있는 총 점유 비율을 비교합니다.

# %%
screen_ratio = df.groupby("subject_id")["m_screen_use"].mean().reset_index()
screen_ratio = screen_ratio.sort_values(by="m_screen_use", ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(data=screen_ratio, x="subject_id", y="m_screen_use", hue="subject_id", palette="magma", legend=False)
plt.title("참여자별 화면 활성화(1) 상태 점유 비율")
plt.ylabel("화면 On 비율")
plt.xlabel("참여자 ID")
plt.ylim(0, 1)
plt.show()

# %% [markdown]
# ## 3. 시간대별 화면 켬 비율 변화 (24시간 라이프사이클)
# 수면 주기와 스마트폰 과의존 시간대 등을 식별합니다.

# %%
df["hour"] = df["timestamp"].dt.hour
hourly_screen = df.groupby(["subject_id", "hour"])["m_screen_use"].mean().reset_index()

# %% [markdown]
# ### 3.1 참여자 전체의 시간대별 평균 스마트폰 화면 켬 흐름

# %%
plt.figure(figsize=(12, 6))
sns.lineplot(data=hourly_screen, x="hour", y="m_screen_use", errorbar=None, color="crimson", marker="o", linewidth=2.5)
plt.title("시간대별 평균 화면 활성화 비율", fontsize=14)
plt.xlabel("시간대 (Hour)")
plt.ylabel("화면 켬 비율")
plt.xticks(range(0, 24))
plt.ylim(0, 1)
plt.show()

# %% [markdown]
# ### 3.2 각 참여자별 개별 수면/스마트폰 사용 시간대 흐름

# %%
subjects = sorted(df["subject_id"].unique())
fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharex=True, sharey=True)
axes = axes.flatten()

for idx, sub in enumerate(subjects):
    sub_data = hourly_screen[hourly_screen["subject_id"] == sub]
    axes[idx].plot(sub_data["hour"], sub_data["m_screen_use"], color="purple", linewidth=2)
    axes[idx].fill_between(sub_data["hour"], 0, sub_data["m_screen_use"], color="orchid", alpha=0.4, label="On")
    axes[idx].fill_between(sub_data["hour"], sub_data["m_screen_use"], 1, color="lightgray", alpha=0.3, label="Off")
    axes[idx].set_title(f"참여자: {sub}", fontsize=11, fontweight="bold")
    axes[idx].set_xticks(range(0, 24, 4))
    axes[idx].set_ylim(0, 1)

plt.suptitle("참여자별 24시간 화면 사용/미사용(On/Off) 타임라인", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. 피처 엔지니어링 제안
# * **`screen_on_ratio`**: 하루 중 화면이 켜져 있는 시간의 비율.
# * **`screen_turns_per_hour`**: 시간당 화면을 켠 횟수 (스마트폰 체크 빈도).
# * **`screen_longest_off_duration`**: 하루 중 화면이 연속해서 가장 오랫동안 꺼져 있던 시간 (예상 연속 수면 시간 피처).
