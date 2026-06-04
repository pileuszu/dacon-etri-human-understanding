# %% [markdown]
# # 모바일 GPS 로그 (mGps) 심층 EDA
#
# 이 노트북은 모바일 GPS 데이터 파일인 `ch2025_mGps.parquet`을 깊이 있게 탐색합니다.
# GPS 데이터를 통해 이동성(Mobility)과 외출 패턴, 생활 반경 등을 계산할 수 있습니다.

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

DATA_PATH = os.path.join(BASE_DIR, "../../data/ch2025_data_items/ch2025_mGps.parquet")
df = pd.read_parquet(DATA_PATH)

# %% [markdown]
# ## 1. 데이터 확인

# %%
print(f"데이터 크기 (Shape): {df.shape}")
df.head()

# %% [markdown]
# ## 2. 수집된 GPS 데이터 포인트 개수 분석
# 각 스캔 위치당 기록된 위치 좌표의 개수 분포를 파악합니다.

# %%
df["gps_log_count"] = df["m_gps"].apply(lambda x: len(x) if x is not None else 0)

plt.figure(figsize=(10, 5))
sns.histplot(df["gps_log_count"], bins=20, color="teal", kde=True)
plt.title("스캔당 기록된 GPS 포인트 개수 분포")
plt.xlabel("GPS 포인트 수")
plt.ylabel("빈도 수")
plt.show()

# %% [markdown]
# ## 3. 속도(Speed) 분포 탐색
# GPS 데이터에서 측정된 이동 속도 정보를 확인하여 정지 상태와 이동 상태를 구분해 봅니다.

# %%
# 샘플로 몇 개의 행에서 속도 정보를 파싱하여 분포를 봅니다.
speeds = []
for path in df["m_gps"].sample(3000, random_state=42):
    for point in path:
        if "speed" in point:
            speeds.append(point["speed"])

speeds = np.array(speeds)
print(f"추출된 속도 데이터 수: {len(speeds)}")

plt.figure(figsize=(10, 5))
sns.histplot(speeds[speeds < 20], bins=50, kde=True, color="darkcyan")  # 이상치 제외 속도만 시각화
plt.title("GPS 감지 속도(Speed) 분포 (20 m/s 이하)")
plt.xlabel("속도 (m/s)")
plt.ylabel("밀도")
plt.show()

# %% [markdown]
# ## 4. 시간대별 이동 패턴 분석 (이동 속도가 0.5 m/s 이상인 빈도)

# %%
df["hour"] = df["timestamp"].dt.hour

# 각 로그별 최대 속도 추출
def get_max_speed(gps_array):
    if gps_array is None or len(gps_array) == 0:
        return 0.0
    return max([p.get("speed", 0.0) for p in gps_array])

# 빠른 처리를 위해 샘플 사용
sample_df = df.sample(10000, random_state=42).copy()
sample_df["max_speed"] = sample_df["m_gps"].apply(get_max_speed)
sample_df["is_moving"] = (sample_df["max_speed"] > 0.5).astype(int)

moving_pattern = sample_df.groupby("hour")["is_moving"].mean().reset_index()

plt.figure(figsize=(12, 6))
sns.barplot(data=moving_pattern, x="hour", y="is_moving", hue="hour", color="teal", legend=False)
plt.title("시간대별 이동 중인 상태(속도 > 0.5m/s) 비율")
plt.xlabel("시간대 (Hour)")
plt.ylabel("이동 비율")
plt.show()

# %% [markdown]
# ## 5. 피처 엔지니어링 제안
# * **`gps_total_records`**: 하루 동안 수집된 총 GPS 로그 개수 (기기 활성화 정도).
# * **`gps_max_speed`**: 하루 중 감지된 최대 속도 (차량 탑승 여부 판단 피처).
# * **`gps_location_entropy`**: 위경도 그리드화를 통한 위치 엔드로피 (주 생활 반경의 다양성).
