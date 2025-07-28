import pandas as pd
import os
import json
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from collections import defaultdict

# --- 1. 설정 ---
# 원본 데이터 경로 (y_convert_labels.py와 동일)
IMAGE_SOURCE_DIR = './customOCR/generated_dataset/'
LABELS_CSV_PATH = os.path.join(IMAGE_SOURCE_DIR, '_labels.csv') 

# 생성될 Donut 데이터셋 경로
DONUT_DATASET_DIR = './customOCR/donut_dataset/'

# --- 2. 데이터 변환 및 저장 ---
def convert_to_donut_format():
    """CSV 데이터를 읽어 Donut 학습용 JSONL 포맷으로 변환하고 파일로 저장합니다."""
    print("Donut 데이터셋 변환을 시작합니다.")
    
    # --- CSV 파일 로드 ---
    try:
        df = pd.read_csv(LABELS_CSV_PATH)
        # Donut은 텍스트 정보가 필수적이므로 'word' 컬럼이 없는 경우를 대비
        if 'text' not in df.columns:
            print("오류: '_labels.csv' 파일에 'word' 컬럼이 없습니다. Donut 변환을 위해서는 텍스트 정보가 필요합니다.")
            return
    except FileNotFoundError:
        print(f"오류: '{LABELS_CSV_PATH}' 파일을 찾을 수 없습니다.")
        return

    # --- 데이터 분할 (YOLO 변환 스크립트와 동일한 로직) ---
    image_groups = df.groupby('image_id') # 'filename' 또는 'image_id' 컬럼명 확인
    all_image_files = list(image_groups.groups.keys())
    
    train_files, val_files = train_test_split(all_image_files, test_size=0.2, random_state=42)
    datasets = {'train': train_files, 'validation': val_files}
    print(f"데이터 분할 완료: Train {len(train_files)}개, Validation {len(val_files)}개")

    # --- 데이터셋 생성 ---
    for split, files in datasets.items():
        print(f"\n'{split}' 데이터셋 생성 중...")
        target_dir = os.path.join(DONUT_DATASET_DIR, split)
        metadata_path = os.path.join(target_dir, 'metadata.jsonl')

        with open(metadata_path, 'w', encoding='utf-8') as f:
            for filename in tqdm(files, desc=f"Processing {split} set"):
                # 1. 이미지 파일을 해당 폴더로 복사
                src_image_path = os.path.join(IMAGE_SOURCE_DIR, filename)
                dst_image_path = os.path.join(target_dir, filename)
                if os.path.exists(src_image_path):
                    shutil.copy(src_image_path, dst_image_path)
                else:
                    print(f"경고: 원본 이미지 '{src_image_path}'를 찾을 수 없어 건너뜁니다.")
                    continue

                # 2. 해당 이미지의 ground_truth JSON 생성
                image_data = image_groups.get_group(filename)
                gt_parse = defaultdict(list)
                
                # 같은 라벨을 가진 텍스트들을 리스트로 묶음
                for _, row in image_data.iterrows():
                    label = row['label']
                    text = str(row['text']) # NaN 방지를 위해 문자열로 변환
                    gt_parse[label].append(text)
                
                # 리스트에 항목이 하나뿐인 경우, 리스트 대신 단일 값으로 저장
                final_gt_parse = {k: (v[0] if len(v) == 1 else v) for k, v in gt_parse.items()}

                # 3. 최종 JSONL 라인 생성
                ground_truth = {
                    "gt_parse": final_gt_parse
                }
                
                # Donut 형식에 맞게 ground_truth를 다시 문자열로 변환
                json_line = {
                    "file_name": filename,
                    "ground_truth": json.dumps(ground_truth, ensure_ascii=False)
                }
                
                f.write(json.dumps(json_line, ensure_ascii=False) + '\n')

    print("\n모든 데이터 변환 완료.")

if __name__ == '__main__':
    # 스크립트 실행 시 기존 metadata.jsonl 파일이 있다면 덮어쓰게 됩니다.
    # 폴더는 미리 생성했다고 가정합니다.
    convert_to_donut_format()
    print(f"\n🎉 Donut 데이터셋 준비가 완료되었습니다. '{DONUT_DATASET_DIR}' 폴더를 확인하세요.")