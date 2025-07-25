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