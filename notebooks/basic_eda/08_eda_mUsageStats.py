# %% [markdown]
# # 모바일 앱 사용 통계 (mUsageStats) 심층 EDA
#
# 이 노트북은 스마트폰 내 앱 실행 시간 기록인 `ch2025_mUsageStats.parquet` 파일을 심층적으로 탐색합니다.
# 어떤 앱을 주로 사용하는지는 개인의 사회적 성향, 여가 유형 등을 직접적으로 매핑합니다.

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

DATA_PATH = os.path.join(BASE_DIR, "../../data/ch2025_data_items/ch2025_mUsageStats.parquet")
df = pd.read_parquet(DATA_PATH)

# %% [markdown]
# ## 1. 데이터 확인

# %%
print(f"데이터 크기 (Shape): {df.shape}")
df.head()

# %% [markdown]
# ## 2. 데이터 평탄화 (Flatten) 및 앱 사용 분포 분석
# 각 행의 `m_usage_stats` 어레이 내에 위치한 앱 명칭(`app_name`)과 실행 시간(`total_time`) 정보를 추출합니다.

# %%
# 샘플링하여 파싱 수행
sample_df = df.sample(min(len(df), 15000), random_state=42)
records = []
for idx, row in sample_df.iterrows():
    subj = row["subject_id"]
    ts = row["timestamp"]
    for app in row["m_usage_stats"]:
        if "app_name" in app and "total_time" in app:
            records.append({
                "subject_id": subj,
                "timestamp": ts,
                "app_name": app["app_name"],
                "total_time": float(app["total_time"])
            })

flat_df = pd.DataFrame(records)
print(f"평탄화 완료된 앱 사용 수: {flat_df.shape[0]}")
flat_df.head()

# %% [markdown]
# ### 2.1 총 사용 시간이 가장 긴 앱 TOP 15

# %%
# 인코딩 문제 등으로 깨져 보일 수 있는 한글 앱 이름을 감안하여 순위를 봅니다.
top_apps = flat_df.groupby("app_name")["total_time"].sum().reset_index()
top_apps = top_apps.sort_values(by="total_time", ascending=False).head(15)

plt.figure(figsize=(12, 6))
sns.barplot(data=top_apps, x="total_time", y="app_name", hue="app_name", palette="viridis", legend=False)
plt.title("실행 시간(Total Time) 기준 상위 15개 앱")
plt.xlabel("총 사용 시간 (초)")
plt.ylabel("앱 이름")
plt.show()

# %% [markdown]
# ## 3. 참여자별 하루 평균 스마트폰 앱 이용 시간 비교

# %%
# 참여자별 일일 총 사용 시간 분포 계산
flat_df["date"] = flat_df["timestamp"].dt.date
daily_usage = flat_df.groupby(["subject_id", "date"])["total_time"].sum().reset_index()

plt.figure(figsize=(12, 6))
sns.boxplot(data=daily_usage, x="subject_id", y="total_time", hue="subject_id", palette="Set3", legend=False)
plt.title("참여자별 일일 앱 사용 시간(초) 분포")
plt.xlabel("참여자 ID")
plt.ylabel("일일 총 사용 시간 (초)")
plt.show()

# %% [markdown]
# ## 4. 피처 엔지니어링 제안
# * **`app_total_time`**: 하루 동안 실행된 모든 앱의 누적 시간.
# * **`app_count`**: 하루 동안 한 번 이상 실행된 서로 다른 앱의 개수.
# * **`app_sns_time_ratio`**: 카카오톡, 인스타그램 등 SNS/메신저 앱의 점유율 (사회성 피처).
