# %% [markdown]
# # 모바일 와이파이 스캔 (mWifi) 심층 EDA
#
# 이 노트북은 주변 Wi-Fi AP 스캔 정보인 `ch2025_mWifi.parquet`을 분석합니다.
# 감지되는 고유한 BSSID의 패턴은 장소의 고유성이나 이동 유무를 파악하는 실마리가 됩니다.

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

DATA_PATH = os.path.join(BASE_DIR, "../../data/ch2025_data_items/ch2025_mWifi.parquet")
df = pd.read_parquet(DATA_PATH)

# %% [markdown]
# ## 1. 데이터 확인

# %%
print(f"데이터 크기 (Shape): {df.shape}")
df.head()

# %% [markdown]
# ## 2. 스캔별 감지된 Wi-Fi AP 개수 분석
# 한 번의 스캔 주기 동안 몇 개의 Wi-Fi AP 신호가 포착되는지 파악합니다.

# %%
df["wifi_count"] = df["m_wifi"].apply(lambda x: len(x) if x is not None else 0)

plt.figure(figsize=(10, 5))
sns.histplot(df["wifi_count"], bins=30, kde=True, color="olive")
plt.title("스캔당 포착된 Wi-Fi AP 개수 분포")
plt.xlabel("Wi-Fi 기기 수")
plt.ylabel("빈도 수")
plt.show()

# %% [markdown]
# ## 3. 참여자별 감지 Wi-Fi AP 개수 비교

# %%
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x="subject_id", y="wifi_count", hue="subject_id", palette="summer", legend=False)
plt.title("참여자별 감지되는 평균 Wi-Fi 개수 분포 비교")
plt.xlabel("참여자 ID")
plt.ylabel("Wi-Fi 개수")
plt.show()

# %% [markdown]
# ## 4. Wi-Fi 신호 강도(RSSI) 분포
# 스캔된 Wi-Fi 기기 중 일부 RSSI 값을 추출하여 거리를 짐작합니다.

# %%
rssis = []
for wifi_list in df["m_wifi"].sample(5000, random_state=42):
    for ap in wifi_list:
        if "rssi" in ap:
            rssis.append(ap["rssi"])

rssis = np.array(rssis)
plt.figure(figsize=(10, 5))
sns.histplot(rssis, bins=40, kde=True, color="darkkhaki")
plt.title("Wi-Fi 신호 강도(RSSI) 분포")
plt.xlabel("RSSI (dBm)")
plt.ylabel("밀도")
plt.show()

# %% [markdown]
# ## 5. 피처 엔지니어링 제안
# * **`wifi_unique_bssids`**: 하루 동안 스캔된 전체 고유 BSSID 개수 (공간적 복잡도 피처).
# * **`wifi_max_rssi`**: 하루 동안 기록된 최대 RSSI 값 (Wi-Fi 연결 공유기와의 최단 접근 거리).
# * **`wifi_scans`**: 하루 동안 수집된 총 Wi-Fi 스캔 횟수.
