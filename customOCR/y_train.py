import torch
from ultralytics import YOLO
import sys
import os

# --- 1. 설정 ---
# 데이터셋 설정 파일 경로
DATASET_YAML_PATH = './customOCR/yolo_dataset/dataset.yaml'
CONTINUE_TRAIN_MODEL_PATH = 'C:/code/customOCR/bank_statement_detector/yolov8l_e50_bs8_0726/weights/best.pt'
# 사용할 YOLO 모델 (n: nano, s: small, m: medium, l: large, x: extra-large)
# 작은 모델로 시작하여 빠르게 성능을 확인하고, 필요 시 더 큰 모델을 사용합니다.
MODEL = 'yolov8l'
# MODEL_NAME = 'yolov8n.pt' 
MODEL_NAME = MODEL + '.pt'
DATE = '0823'

# 학습 하이퍼파라미터
EPOCHS = 50
IMAGE_SIZE = 640
BATCH_SIZE = 8 # GPU 메모리에 따라 조절 (예: 8, 16, 32)
LOSS_WEIGHT = 1.0 # 클래스 분류 손실 가중치
PROJECT_NAME = './customOCR/bank_statement_detector/'
BASE_RUN_NAME = f'{MODEL}_e{EPOCHS}_bs{BATCH_SIZE}_{DATE}'

def train_yolo_model():
    """YOLOv8 모델을 학습시킵니다."""
    
    # --- 2. GPU 사용 가능 여부 확인 ---
    if torch.cuda.is_available():
        device = '0' # 첫 번째 GPU 사용
        print(f"GPU를 사용합니다: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("GPU를 사용할 수 없습니다. CPU로 학습을 진행합니다.")

    # ★★★ 3. 학습 모드 결정 및 모델 로드 ★★★
    # 터미널 명령어에 '--continue'가 있는지 확인
    continue_training = '--continue' in sys.argv
    
    if continue_training:
        # 이어서 학습 모드
        initial_model_path = CONTINUE_TRAIN_MODEL_PATH
        run_name = f'{BASE_RUN_NAME}_continued' # 실행 이름 변경
        print(f"\n[모드] 이전 학습에 이어서 훈련을 시작합니다.")
        print(f"  - 로드할 모델: {os.path.abspath(initial_model_path)}")
    else:
        # 신규 학습 모드
        initial_model_path = MODEL_NAME
        run_name = BASE_RUN_NAME
        print(f"\n[모드] 새로운 훈련을 시작합니다.")
        print(f"  - 로드할 모델: {initial_model_path}")

    try:
        model = YOLO(initial_model_path)
    except Exception as e:
        print(f"오류: YOLO 모델('{initial_model_path}') 로딩에 실패했습니다. 경로를 확인하세요.")
        print(e)
        return

    # --- 4. 모델 학습 ---
    print("\n모델 학습을 시작합니다...")
    print(f"  - 데이터셋: {os.path.abspath(DATASET_YAML_PATH)}")
    print(f"  - 에포크: {EPOCHS}")
    print(f"  - 이미지 크기: {IMAGE_SIZE}")
    print(f"  - 배치 사이즈: {BATCH_SIZE}")
    print(f"  - 실행 이름: {run_name}")

    try:
        results = model.train(
            data=DATASET_YAML_PATH,
            epochs=EPOCHS,
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            project=PROJECT_NAME,
            name=run_name, # 동적으로 결정된 실행 이름 사용
            device=device,
            patience=10, # 10 에포크 동안 성능 향상이 없으면 조기 종료
            exist_ok=True, # 동일한 이름의 실행이 있어도 덮어쓰기
            cls=LOSS_WEIGHT, # 클래스 분류 손실 가중치
        )
        print("\n모델 학습이 성공적으로 완료되었습니다.")
        
        # 학습된 모델의 가중치 파일 경로 출력
        # 가장 마지막에 저장된 best 모델 가중치를 찾습니다.
        best_model_path = os.path.join(PROJECT_NAME, run_name, 'weights', 'best.pt')
        if os.path.exists(best_model_path):
            print(f"\n🎉 최적의 모델 가중치가 다음 경로에 저장되었습니다:")
            print(f"   {os.path.abspath(best_model_path)}")
        else:
            print("\n학습은 완료되었으나, 최적 가중치 파일을 찾을 수 없습니다.")

    except Exception as e:
        print(f"\n모델 학습 중 오류가 발생했습니다: {e}")
        print("데이터셋 경로, 파일 권한, 또는 ultralytics 설치 상태를 확인해주세요.")

if __name__ == '__main__':
    train_yolo_model()