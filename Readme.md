## 2018년도 NHIS 데이터를 활용하여 당뇨 단계 예측

#### 1. 주요 기능과 목적
- 주요 기능
> 각 데이터에 사용된 변수의 히스토그램을 확인할 수 있습니다.
> 두 연속형 변수에 대해 산점도를 확인할 수 있습니다.
> 당뇨의 단계 예측에 대해 XGBoost, LightGBM 두 모델로 비교해볼 수 있습니다.
> SHAP을 활용하여 각 변수의 크기와 방향으로 반응변수인 당뇨 단계에 어떠한 영향을 미치는가를 확인할 수 있습니다.


#### 2. 작성 환경 패키지와 버전
python = "^3.9"
numpy = "1.24.4"
plotly = "^5.18.0"
dash = "^2.14.1"
scikit-learn = "^1.3.2"
xgboost = "^2.0.2"
lightgbm = "^4.1.0"
matplotlib = "^3.8.1"
pandas = "2.1.0"
dash-bootstrap-components = "^1.5.0"
shap = "^0.43.0"


#### 3. 폴더 구조
![image](https://github.com/Daw-ny/CodingTest_Study_3rd/assets/76687996/842b74b3-6368-437a-98da-9f1811b0f3bc)


#### 4. 파일 구조와 설명
> Code
>> 01_data_check_and_make_new_data.ipynb
>> - 데이터 확인 및 결측치 또는 이상치 대치 또는 제거와 파생변수 생성
>> 
>> 02_data_train_with_boostmodel.ipynb
>> - 데이터 학습 및 결과 출력
>>
>> 03_Dashboard.py
>> - 대시보드 생성 및 각 변수별 분포확인, 분석 그래프 출력
>>
> Data
>> adult_use_data.csv
>> - 학습을 위해 생성한 csv파일
>> Data_Codebook.xlsx
>> - 변수 정의를 위해 생성한 codebook

