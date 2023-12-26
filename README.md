# MFR (Music Feature Recommendation)
건국대학교 스마트ICT융합공학과 종합설계2 프로젝트

##### 음악 자체에서 특성을 추출해 음원 추천에 이용하자는 생각에서 시작되었습니다.
* 데이터셋은 kaggle의 gtzan dataset을 사용합니다.(10개의 장르, 총 1000개의 음악 샘플, 장르별 100곡)
* 음원의 특성으로 MFCCs, Chroma_STFT, Tempogram을 추출하였습니다.
* 음악의 지역적 특성과 시계열적 특성을 모두 활용하기 위해 CRNN 기반의 인공신경망을 설계하고 훈련시켰습니다.
