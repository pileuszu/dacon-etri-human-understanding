# %% [markdown]
# # 학습용 타겟 데이터 (ch2026_metrics_train) 심층 EDA
#
# 이 노트북은 예측 목표가 되는 타겟 라벨 데이터셋 `ch2026_metrics_train.csv`을 분석합니다.
# 다중 타겟 변수들의 분포 균형과 라벨 간의 연관관계를 이해합니다.

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

DATA_PATH = os.path.join(BASE_DIR, "../../data/ch2026_metrics_train.csv")
df = pd.read_csv(DATA_PATH)

# %% [markdown]
# ## 1. 데이터 확인

# %%
print(f"데이터 크기 (Shape): {df.shape}")
print(f"데이터 타입 (Data Types):\n{df.dtypes}")
df.head()

# %% [markdown]
# ## 2. 타겟 변수(Q1-Q3, S1-S4) 값 분포 분석
# 타겟 변수들의 이진 클래스 불균형 정도를 시각화합니다.

# %%
target_cols = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]

fig, axes = plt.subplots(2, 4, figsize=(18, 10))
axes = axes.flatten()

for idx, col in enumerate(target_cols):
    sns.countplot(data=df, x=col, ax=axes[idx], hue=col, palette="coolwarm", legend=False)
    axes[idx].set_title(f"{col} 분포")
    axes[idx].set_xlabel("값")
    axes[idx].set_ylabel("빈도 수")
    
# 남은 서브플롯 삭제
for j in range(len(target_cols), len(axes)):
    fig.delaxes(axes[j])

plt.suptitle("타겟 라벨 이진 분류 분포 (0 vs 1)", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. 타겟 변수 간의 연관성 분석

# %%
plt.figure(figsize=(8, 6))
label_corr = df[target_cols].corr()
sns.heatmap(label_corr, annot=True, cmap="bwr", fmt=".3f", vmin=-1, vmax=1)
plt.title("타겟 라벨 간 피어슨 상관관계")
plt.show()

# %% [markdown]
# ## 4. 참여자별 타겟 분포 편차 분석
# 특정 사용자에게 특정 응답 라벨(0 또는 1)이 편향되어 분포하는지 검토합니다. (피험자 개인 편차 유무 파악)

# %%
df_subject_target = df.groupby("subject_id")[target_cols].mean()

plt.figure(figsize=(14, 8))
sns.heatmap(df_subject_target, annot=True, cmap="YlOrRd", fmt=".2f")
plt.title("참여자별 타겟 변수의 '1' 응답 비율 분포")
plt.xlabel("타겟 변수")
plt.ylabel("참여자 ID")
plt.show()

# %% [markdown]
# ## 5. 검증 전략 제안
# * **사용자 분할 교차 검증 (GroupKFold)**: `subject_id`를 그룹으로 삼아 교차 검증을 수행하여, 학습에 보지 못한 피험자가 평가 세트에 들어올 때 모델의 일반화 성능을 보호해야 합니다.
