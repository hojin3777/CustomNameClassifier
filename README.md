# CustomNameClassifier
Classify store names with deeplearning


### 0725
- 데이터셋 50000개로 LayoutLM 재학습. 기존 1000개의 결과물이 더 좋음
- yolov8m 모델 학습 완료, 준수한 성능을 보이나 좀 더 다듬을 필요가 있음
- YOLO 모델을 발전시키는 쪽으로 개발

### 0726_0120
- 커스텀 템플릿 3가지 추가, 폰트 7종 추가 및 이미지 내용 좀 더 실제 이미지와 유사하게 개선(generate.py 변경내역 참조)
- 데이터셋 7만개로 증가
- 커밋 후 yolov8L 모델로 학습 예정

### 0727
- yolov8L모델 학습 완료, 일부 DATE 클래스를 놓치는 경향
- loss weight 를 조정하여 추가 학습 진행중
- donut 모듈 pull 및 가상환경 분리, gitignore 추가
- *.pt 파일 업로드 제외(용량제한)

### 0728
- yolov8L 추가 학습 완료, 성능 변동 없음
- donut 모듈 학습 트러블슈팅 진행 중
- torch 모듈 내 distributetd_c10d.py에서 line 1671에 분산 처리 모듈 gloo로 고정

### 0828
- yolov11l, yolov8x 모델 테스트, 성능 큰 향상 없음.
- 레이아웃 4종 추가
- 후보정 로직 생성
- 레이아웃 3종 추가된 모델 학습 전 커밋