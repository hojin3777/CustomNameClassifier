import pandas as pd
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import shutil
from PIL import Image

# --- 1. 설정 (사용자 환경에 맞게 경로 수정) ---
# 원본 데이터 경로
# 사용자가 지정한 경로로 수정합니다.
IMAGE_SOURCE_DIR = './customOCR/generated_dataset/'
LABELS_CSV_PATH = os.path.join(IMAGE_SOURCE_DIR, '_labels.csv') 

# 생성될 YOLO 데이터셋 경로
YOLO_DATASET_DIR = './customOCR/yolo_dataset/'

# 클래스 정의 (LayoutLM 학습 시 사용한 config의 id2label과 동일해야 함)
CLASSES = [
    'DATE_HEADER', 'DATE', 'TIME', 'MERCHANT', 
    'MEMO', 'AMOUNT_IN', 'AMOUNT_OUT', 'BALANCE'
]
CLASS_MAP = {name: i for i, name in enumerate(CLASSES)}

# --- 2. 폴더 구조 생성 ---
def create_yolo_directories():
    """YOLO 학습에 필요한 폴더 구조를 생성합니다."""
    os.makedirs(os.path.join(YOLO_DATASET_DIR, 'images/train'), exist_ok=True)
    os.makedirs(os.path.join(YOLO_DATASET_DIR, 'images/val'), exist_ok=True)
    os.makedirs(os.path.join(YOLO_DATASET_DIR, 'labels/train'), exist_ok=True)
    os.makedirs(os.path.join(YOLO_DATASET_DIR, 'labels/val'), exist_ok=True)
    print("YOLO 데이터셋 폴더 구조 생성 완료.")

# --- 3. 데이터 변환 및 저장 ---
def convert_to_yolo_format():
    """CSV 데이터를 읽어 YOLO 포맷으로 변환하고 파일로 저장합니다."""
    try:
        df = pd.read_csv(LABELS_CSV_PATH)
    except FileNotFoundError:
        print(f"오류: '{LABELS_CSV_PATH}' 파일을 찾을 수 없습니다.")
        print("이전 단계에서 생성한 라벨 CSV 파일의 경로를 확인해주세요.")
        return

    # 이미지 ID를 기준으로 데이터를 그룹화
    image_groups = df.groupby('image_id')
    
    # 전체 이미지 파일 목록
    all_image_ids = list(image_groups.groups.keys())
    
    # Train / Validation 분할
    train_ids, val_ids = train_test_split(all_image_ids, test_size=0.2, random_state=42)
    print(f"데이터 분할 완료: Train {len(train_ids)}개, Validation {len(val_ids)}개")

    datasets = {'train': train_ids, 'val': val_ids}

    for split, ids in datasets.items():
        print(f"\n'{split}' 데이터셋 변환 시작...")
        
        image_target_dir = os.path.join(YOLO_DATASET_DIR, f'images/{split}')
        label_target_dir = os.path.join(YOLO_DATASET_DIR, f'labels/{split}')

        for image_id in tqdm(ids, desc=f"Processing {split} set"):
            # 1. 이미지 복사
            src_image_path = os.path.join(IMAGE_SOURCE_DIR, image_id)
            dst_image_path = os.path.join(image_target_dir, image_id)
            if os.path.exists(src_image_path):
                shutil.copy(src_image_path, dst_image_path)
            else:
                print(f"경고: 원본 이미지 '{src_image_path}'를 찾을 수 없어 건너뜁니다.")
                continue

            # 2. 이미지 크기 정보 가져오기
            with Image.open(dst_image_path) as img:
                img_width, img_height = img.size

            # 3. 라벨 파일 생성
            group = image_groups.get_group(image_id)
            label_path = os.path.join(label_target_dir, os.path.splitext(image_id)[0] + '.txt')
            
            with open(label_path, 'w') as f:
                for _, row in group.iterrows():
                    label = row['label']
                    if label not in CLASS_MAP:
                        continue
                        
                    class_id = CLASS_MAP[label]
                    
                    x_min, y_min = row['x_min'], row['y_min']
                    x_max, y_max = row['x_max'], row['y_max']
                    
                    box_width = x_max - x_min
                    box_height = y_max - y_min
                    x_center = x_min + box_width / 2
                    y_center = y_min + box_height / 2
                    
                    norm_x_center = x_center / img_width
                    norm_y_center = y_center / img_height
                    norm_width = box_width / img_width
                    norm_height = box_height / img_height
                    
                    f.write(f"{class_id} {norm_x_center} {norm_y_center} {norm_width} {norm_height}\n")

    print("\n모든 데이터 변환 완료.")

# --- 4. dataset.yaml 파일 생성 ---
def create_dataset_yaml():
    """dataset.yaml 파일을 생성합니다."""
    # YAML 파일 내용은 절대 경로를 사용해야 YOLO 라이브러리가 안정적으로 인식합니다.
    content = f"""
path: {os.path.abspath(YOLO_DATASET_DIR)}
train: images/train
val: images/val

names:
"""
    for i, name in enumerate(CLASSES):
        content += f"  {i}: {name}\n"

    with open(os.path.join(YOLO_DATASET_DIR, 'dataset.yaml'), 'w', encoding='utf-8') as f:
        f.write(content)
    print("'dataset.yaml' 파일 생성 완료.")


if __name__ == '__main__':
    # 스크립트 실행 시 항상 폴더를 새로 만들도록 하여 이전 실행 결과를 초기화합니다.
    if os.path.exists(YOLO_DATASET_DIR):
        shutil.rmtree(YOLO_DATASET_DIR)
        print(f"기존 '{YOLO_DATASET_DIR}' 폴더를 삭제했습니다.")

    create_yolo_directories()
    convert_to_yolo_format()
    create_dataset_yaml()
    print(f"\n🎉 YOLO 데이터셋 준비가 완료되었습니다. '{YOLO_DATASET_DIR}' 폴더를 확인하세요.")