# %% [markdown]
# # 제출 양식 샘플 (ch2026_submission_sample) 분석
#
# 이 노트북은 제출 양식인 `ch2026_submission_sample.csv` 파일의 형식을 확인하고 테스트 세트의 피험자 및 날짜 구성을 파악합니다.

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

DATA_PATH = os.path.join(BASE_DIR, "../../data/ch2026_submission_sample.csv")
df = pd.read_csv(DATA_PATH)

# %% [markdown]
# ## 1. 데이터 및 형태 확인

# %%
print(f"제출 데이터 크기 (Shape): {df.shape}")
df.head()

# %% [markdown]
# ## 2. 테스트 데이터 고유 참여자 및 일수 분석
# 예측해야 하는 대상 피험자군과 타임라인 일수를 계산합니다.

# %%
print(f"테스트 데이터 고유 참여자 수: {df['subject_id'].nunique()}")
print(f"테스트 데이터 참여자 목록: {df['subject_id'].unique().tolist()}")
print(f"테스트 데이터 전체 일수: {df['lifelog_date'].nunique()}일")

# %% [markdown]
# ## 3. 학습 세트와 테스트 세트의 피험자 교집합 확인
# 학습 세트의 피험자 목록과 제출 양식의 피험자 목록이 같은지 체크합니다.

# %%
train_df = pd.read_csv(os.path.join(BASE_DIR, "../../data/ch2026_metrics_train.csv"))
train_subs = set(train_df["subject_id"].unique())
test_subs = set(df["subject_id"].unique())

print(f"학습 피험자 수: {len(train_subs)}")
print(f"테스트 피험자 수: {len(test_subs)}")
print(f"동일한 피험자인가?: {train_subs == test_subs}")
print(f"공통 피험자 수: {len(train_subs.intersection(test_subs))}")

# %% [markdown]
# ## 4. 제출 시 확인 사항 정리
# * **인덱스 및 행 수 정밀 일치**: 제출 시 행 수는 정확히 250행(헤더 제외)이어야 하며 순서가 일치해야 합니다.
# * **결측치 및 무한대 제거**: 모델의 예측값에 NaN 또는 Inf가 존재하지 않아야 제출이 성공합니다.
# * **이진 예측 범위**: 모든 Q1-Q3, S1-S4 값은 이진(0 또는 1) 범주 형태로 예측되어야 합니다.
