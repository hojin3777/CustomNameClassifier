import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
from transformers import LayoutLMv3Processor

# config.py 파일에서 설정값들을 가져옵니다.
from config import PRETRAINED_MODEL_NAME, LABEL_FILE, IMAGE_DIR, label2id

class BankStatementDataset(Dataset):
    """
    은행 거래내역서 이미지와 라벨을 LayoutLMv3 모델 입력 형식으로 변환하는 클래스.
    """
    def __init__(self, image_dir, annotations_file, processor, max_length=512):
        """
        Args:
            image_dir (string): 이미지가 저장된 디렉토리 경로.
            annotations_file (string): 라벨 정보가 담긴 CSV 파일 경로.
            processor: HuggingFace의 LayoutLMv3Processor 객체.
            max_length (int): 모델에 입력될 토큰의 최대 길이.
        """
        self.image_dir = image_dir
        self.processor = processor
        self.max_length = max_length

        # 라벨 파일을 읽고 이미지 ID별로 그룹화합니다.
        self.df = pd.read_csv(annotations_file)
        self.image_ids = self.df['image_id'].unique()

    def __len__(self):
        """데이터셋의 총 샘플(이미지) 수를 반환합니다."""
        return len(self.image_ids)

    def __getitem__(self, idx):
        """
        하나의 샘플(이미지)에 대한 데이터를 가져와 모델 입력 형식으로 변환합니다.
        """
        image_id = self.image_ids[idx]
        image_annotations = self.df[self.df['image_id'] == image_id].copy()
        
        image_path = os.path.join(self.image_dir, image_id)
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        
        words = image_annotations["text"].tolist()
        
        # --- ★★★ 수정된 핵심 로직 ★★★ ---
        # Bounding box 좌표를 [0, 1000] 범위로 정규화하고 유효성을 보장합니다.
        boxes = []
        for _, row in image_annotations.iterrows():
            # 좌표를 정수형으로 변환
            x_min, y_min = int(row['x_min']), int(row['y_min'])
            x_max, y_max = int(row['x_max']), int(row['y_max'])

            # 정규화
            norm_x_min = int(1000 * x_min / width)
            norm_y_min = int(1000 * y_min / height)
            norm_x_max = int(1000 * x_max / width)
            norm_y_max = int(1000 * y_max / height)
            
            # 유효성 검사: x_max가 x_min보다, y_max가 y_min보다 크도록 보장
            if norm_x_min >= norm_x_max:
                norm_x_max = norm_x_min + 1
            if norm_y_min >= norm_y_max:
                norm_y_max = norm_y_min + 1
            
            # 최종 좌표값이 [0, 1000] 범위를 벗어나지 않도록 clip
            final_box = [
                max(0, norm_x_min),
                max(0, norm_y_min),
                min(1000, norm_x_max),
                min(1000, norm_y_max)
            ]
            boxes.append(final_box)
        # --- ★★★ 여기까지 수정 ★★★ ---

        word_labels = [label2id[label] for label in image_annotations["label"].tolist()]

        # 이미지, 텍스트, 박스를 한 번에 processor에 전달합니다.
        encoding = self.processor(
            image,
            words,
            boxes=boxes,
            word_labels=word_labels,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        encoding = {key: val.squeeze() for key, val in encoding.items()}
        
        # bbox 텐서의 타입을 long으로 변경 (이전 수정사항 유지)
        if 'bbox' in encoding:
            encoding['bbox'] = encoding['bbox'].long()

        return encoding

# --- 이 파일이 직접 실행될 때 테스트용으로 동작하는 코드 ---
if __name__ == '__main__':
    # 프로세서 초기화
    processor = LayoutLMv3Processor.from_pretrained(PRETRAINED_MODEL_NAME, apply_ocr=False)
    
    # 데이터셋 객체 생성
    # 실제 이미지와 라벨 파일이 config.py에 지정된 경로에 있어야 합니다.
    try:
        dataset = BankStatementDataset(
            image_dir=IMAGE_DIR,
            annotations_file=LABEL_FILE,
            processor=processor
        )
        print(f"데이터셋 로딩 성공! 총 {len(dataset)}개의 이미지를 찾았습니다.")
        
        # 첫 번째 데이터를 가져와서 내용 확인
        first_item = dataset[0]
        print("\n--- 첫 번째 데이터 샘플 ---")
        for key, value in first_item.items():
            print(f"{key}: {value.shape}")
        
        print("\n'labels' 텐서의 일부:")
        print(first_item['labels'][:10])
        
        print("\n'input_ids'에 해당하는 토큰 (디코딩):")
        decoded_tokens = processor.tokenizer.decode(first_item['input_ids'])
        print(decoded_tokens)

    except FileNotFoundError as e:
        print(f"오류: 데이터 파일을 찾을 수 없습니다. {e}")
        print("데이터 생성 스크립트를 먼저 실행했는지, config.py의 경로가 올바른지 확인해주세요.")
    except Exception as e:
        print(f"데이터셋 로딩 중 예기치 않은 오류 발생: {e}")