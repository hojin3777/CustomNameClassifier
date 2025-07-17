import torch

# --- 프로젝트 경로 설정 ---
# 이 파일(config.py)이 있는 src 폴더의 부모 디렉토리를 기준으로 경로를 설정합니다.
BASE_DIR = "c:/code/customOCR" # 사용자의 프로젝트 경로에 맞게 수정하세요.
DATA_DIR = f"{BASE_DIR}/generated_dataset"
OUTPUT_DIR = f"{BASE_DIR}/output"

# --- 데이터셋 관련 설정 ---
# 데이터 생성 스크립트(generate.py)에서 생성된 이미지와 라벨 파일 경로
IMAGE_DIR = f"{DATA_DIR}" 
LABEL_FILE = f"{DATA_DIR}/_labels.csv"

# --- 모델 설정 ---
# Hugging Face에 사전 학습된 모델 이름
# 한국어 이해 능력이 좋은 모델을 선택하는 것이 중요합니다.
# 'klue/roberta-large'는 텍스트 이해에 강점이 있고,
# 'microsoft/layoutlmv3-base'는 레이아웃과 텍스트를 함께 이해하는 데 특화되어 있습니다.
# 여기서는 LayoutLM 계열을 사용하겠습니다.
PRETRAINED_MODEL_NAME = "microsoft/layoutlmv3-base"

# --- 학습 관련 설정 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
LEARNING_RATE = 5e-5
NUM_EPOCHS = 10
TRAIN_RATIO = 0.8 # 전체 데이터 중 학습에 사용할 비율 (나머지는 검증용)

# --- 라벨(클래스) 설정 ---
# 데이터 생성 시 사용한 라벨들에 'O' (Other, 기타)를 추가합니다.
# 모델은 각 텍스트 조각을 아래 클래스 중 하나로 분류하게 됩니다.
LABELS = [
    'O', # Other (아무것도 아닌 텍스트)
    'DATE',
    'DATE_HEADER',
    'MERCHANT',
    'MEMO',
    'AMOUNT_IN',
    'AMOUNT_OUT',
    'BALANCE',
    'TIME'
]

# 라벨 이름을 정수 인덱스로, 인덱스를 라벨 이름으로 변환하기 위한 딕셔너리
label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for i, label in enumerate(LABELS)}
NUM_LABELS = len(LABELS)