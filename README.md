# Titanic Feature Engineering Pipeline

저장소: https://github.com/jiun3374/titanic-feature-engineering

머신러닝 특성 공학 파이프라인 — Titanic 생존 예측을 통한 전처리·변수선택 전략 비교 분석

## 개요
- **문제**: 이진 분류 (생존 여부 예측)
- **데이터**: Kaggle Titanic (891 samples × 12 columns)
- **모델**: Logistic Regression, RandomForest, XGBoost, LightGBM

## 파이프라인 단계
1. **STEP01 데이터 준비** — 로드, 구조 확인, 타겟 정의
2. **STEP02 EDA** — 결측치/이상치/분포/상관/타겟 분석 (Histogram, Boxplot, Heatmap, Countplot)
3. **STEP03 특성공학** — 결측치 처리·인코딩·스케일링 비교, 파생변수 4종(FamilySize, IsAlone, Title, FarePerPerson)
4. **STEP04 변수선택** — RF Feature Importance, SelectKBest, 제거 전/후 비교
5. **STEP05 모델학습/실험비교** — 4가지 전처리 조합(Base/Exp-1/Exp-2/Exp-3) × 4개 모델

## 가산점 항목
- Pipeline 형태의 전처리 함수(`build_dataset`)
- GridSearchCV 하이퍼파라미터 튜닝
- SHAP 기반 설명가능성 분석
- Feature Importance 시각화

## 주요 결과
| 실험 | 전처리 | Best Model | F1 |
|------|--------|-----------|-----|
| Base | Label, no scale | RandomForest | 0.7536 |
| **Exp-1** | **Mean+OneHot+Standard** | **LogReg** | **0.7939** |
| Exp-2 | Median+Label+MinMax+FSel | RandomForest | 0.7536 |
| Exp-3 | MostFreq+OneHot+Robust+FSel | LightGBM | 0.7338 |

## 실행
```bash
pip install -r requirements.txt
jupyter notebook Titanic_FeatureEngineering.ipynb
```

## 파일
- `Titanic_FeatureEngineering.ipynb` — 전체 파이프라인 노트북 (출력 포함)
- `Titanic_FeatureEngineering_Report.pdf` — 분석 리포트
