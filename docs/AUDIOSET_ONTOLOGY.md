# 🔊 Google AudioSet Ontology 상세 분석 및 활용 가이드

본 문서는 `mAmbience` 데이터셋에서 핵심 정보를 구성하고 있는 **Google AudioSet Ontology(구글 오디오셋 온톨로지)**의 기술적 구조와 분류 계층을 설명하고, 이를 대회 타겟 변수 예측에 활용하기 위한 피처 엔지니어링 전략을 제시합니다.

---

## 1. Google AudioSet 개요

**Google AudioSet**은 구글 연구팀이 머신러닝 기반 오디오 이벤트 분류 연구를 위해 구축한 세계 최대 규모의 정형 오디오 데이터셋이자 분류 체계(Ontology)입니다.

* **구조적 특징**: 단순 일차원 분류가 아닌, 소리 간의 부모-자식 관계를 정의하는 **계층형 방향성 그래프(Hierarchical Directed Graph)** 구조입니다.
* **스케일**: 총 632개의 세부 사운드 클래스로 구성되며, 최대 6단계의 깊이를 지닙니다.
* **작동 원리**: 모바일 기기에 내장된 오디오 감지 모델이 마이크 센서를 통해 유입되는 짧은 오디오 세그먼트를 분류하고, 각 소리별 감지 확률(Probability/Confidence)을 실시간으로 도출합니다.

---

## 2. 데이터 인프라 내의 구조 (`mAmbience`)

대회 데이터셋 중 `ch2025_mAmbience.parquet` 내의 `m_ambience` 컬럼은 다음과 같이 리스트 안에 클래스명과 감지 점수가 페어로 묶여서 기록되어 있습니다:

```python
# m_ambience 컬럼 예시 데이터
[
    ["Silence", "0.85"],
    ["Speech", "0.12"],
    ["Music", "0.03"]
]
```

* **Label (예: "Silence")**: AudioSet Ontology 상에 존재하는 표준 음향 클래스명.
* **Prob (예: "0.85")**: 오디오 모델이 해당 시점에 그 소리가 발생했을 것이라고 예측한 확률값(Confidence Score, 0.0 ~ 1.0).

---

## 3. 오디오 온톨로지 주요 계층 및 탐지 항목

AudioSet의 632개 클래스는 크게 **7가지 최상위 범주(Top-Level Categories)**로 그룹화됩니다. 각 범주와 수집 데이터에서 포착할 수 있는 구체적인 음향은 다음과 같습니다.

### 👤 3.1 사람의 소리 (Human Sounds)
사용자의 신체 활동 상태, 발성, 생리 현상을 유추하는 데 있어 가장 중요한 지표입니다.
* **말소리 (Speech)**: 일상 대화, 통화 등 타인과의 대면 소통 수준 유추.
* **웃음 (Laughter)**: 긍정적 정서 상태 및 스트레스 완화 지표.
* **울음 (Crying/Sobbing)**: 부정적 정서 상태 및 심리적 불안 지표.
* **기침/재채기 (Coughing/Sneeze)**: 호흡기 건강 상태 및 질병 징후.
* **코골이 (Snoring)**: **수면 장애 및 수면 품질(Q1) 예측의 직격 피처**.

### 🎵 3.2 음악 (Music)
사용자의 여가 활동 및 실내 체류 패턴을 반영합니다.
* **음악 (Music)**: 이어폰/스피커를 통한 음악 감상, 라디오 등 정적 휴식 및 집중도.
* **악기 연주 (Musical Instrument)**: 악기를 다루는 활동성 취미 유추.

### 🚗 3.3 물건/기계의 소리 (Sounds of Things)
인공물, 기계 장치 등에 의해 발생하는 소리로 이동 수단 탑승 여부나 기기 활용도를 나타냅니다.
* **차량/엔진음 (Vehicle / Motor Vehicle)**: 자동차, 버스, 오토바이 탑승 및 출퇴근 이동 분석.
* **사이렌/경적 (Siren / Horn)**: 도로 및 실외 복잡도, 도시 소음 노출도.
* **전화 벨소리/알림음 (Ringtone / Alarm)**: 스마트 기기 사용 활성도 및 수면 중 방해 요인.
* **문 여닫는 소리 (Door)**: 공간 이동(출입) 여부 파악.

### 🐱 3.4 동물의 소리 (Animal Sounds)
반려동물 양육 여부 및 주변 환경의 특성을 반영합니다.
* **개 짖는 소리 (Dog / Barking)**, **고양이 울음소리 (Cat / Purring / Meow)**.

### 🌧️ 3.5 자연의 소리 (Natural Sounds)
사용자가 노출된 기상 환경 및 야외 활동 여부를 보조적으로 파악합니다.
* **비/바람/천둥 (Rain / Wind / Thunder)**: 악천후 하에서의 외부 활동 제약 예측.
* **물 흐르는 소리 (Water)**: 샤워, 설거지 등의 위생 활동 유추.

### 🔇 3.6 환경 및 배경음 (Channel, environment and background)
주변 환경의 전반적인 정숙성을 파악하는 지표입니다.
* **침묵 (Silence)**: 사용자가 조용한 환경(독서실, 개인실, 야간 수면 등)에 머물고 있음을 방증.
* **소음/험음 (Noise / Environmental noise)**: 카페, 공공장소 등 대중 시설 체류 상태 유추.

---

## 4. 대회 타겟 지표 예측을 위한 피처 엔지니어링 전략

수면 품질(Q1), 피로도(Q2), 스트레스(Q3) 등을 예측하는 기계학습 모델의 강건성을 높이기 위해 다음과 같은 오디오 기반 피처 파이프라인 설계를 제안합니다.

### 💤 수면 지표 (Q1, Q4, Q5, Q7) 예측을 위한 오디오 피처
* **`sleep_snoring_intensity_night`**: 수면 시간대(23:00~07:00) 동안 감지된 **`Snoring(코골이)`**의 평균 신뢰도.
* **`sleep_silence_ratio_night`**: 수면 시간대 동안 감지된 **`Silence(침묵)`**의 비율. 이 비율이 높을수록 방해받지 않는 이상적인 수면 환경임을 의미.
* **`sleep_interrupt_events`**: 야간 시간대 중 갑작스러운 **`Siren(사이렌)`**, **`Barking(개 짖는 소리)`**, **`Alarm(알람)`** 등의 돌발 소음 노출 횟수.

### 🤯 스트레스 (Q3) 및 피로도 (Q2) 예측을 위한 오디오 피처
* **`social_interaction_ratio`**: 일간 활동 시간대(09:00~21:00) 동안 감지된 **`Speech(말소리)`**의 총 누적 시간 비율. 대화 및 사회 활동량이 낮을수록 스트레스와 고립감이 높게 예측될 가능성이 큼.
* **`positive_emotion_index`**: 하루 중 감지된 **`Laughter(웃음)`**의 빈도수. 스트레스와 강한 부적 상관관계를 가짐.
* **`noise_exposure_hours`**: 주간 동안 **`Environmental noise(소음)`**이나 **`Motor vehicle(엔진음)`** 등 피로를 유발하는 환경에 노출된 누적 시간.

---

## 5. 데이터 처리 및 피처화 예시 (Python Code)

```python
import pandas as pd
import numpy as np

def extract_audio_features(df_ambience):
    """
    df_ambience: parquet를 로드한 raw DataFrame
    """
    # 1. 중첩 리스트 평탄화
    records = []
    for _, row in df_ambience.iterrows():
        subj = row['subject_id']
        ts = row['timestamp']
        for sound in row['m_ambience']:
            if len(sound) == 2:
                records.append({
                    'subject_id': subj,
                    'timestamp': ts,
                    'sound_class': sound[0],
                    'confidence': float(sound[1])
                })
    
    flat_df = pd.DataFrame(records)
    flat_df['hour'] = flat_df['timestamp'].dt.hour
    flat_df['date'] = flat_df['timestamp'].dt.date
    
    # 2. 피처 집계 (예: 일별/참여자별 야간 침묵 비율, 주간 대화 비율)
    features = []
    grouped = flat_df.groupby(['subject_id', 'date'])
    for (subj, date), group in grouped:
        # 야간 데이터 필터 (23시 ~ 07시)
        night_group = group[group['hour'].isin([23, 0, 1, 2, 3, 4, 5, 6])]
        # 주간 데이터 필터 (09시 ~ 18시)
        day_group = group[group['hour'].isin([9, 10, 11, 12, 13, 14, 15, 16, 17])]
        
        # 피처 계산
        night_silence = night_group[night_group['sound_class'] == 'Silence']['confidence'].mean() if not night_group.empty else 0
        night_snoring = night_group[night_group['sound_class'] == 'Snoring']['confidence'].mean() if not night_group.empty else 0
        day_speech = day_group[day_group['sound_class'] == 'Speech']['confidence'].mean() if not day_group.empty else 0
        
        features.append({
            'subject_id': subj,
            'date': date,
            'audio_night_silence_avg': night_silence,
            'audio_night_snoring_avg': night_snoring,
            'audio_day_speech_avg': day_speech
        })
        
    return pd.DataFrame(features)
```
