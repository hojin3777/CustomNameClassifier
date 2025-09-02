import torch
from ultralytics import YOLO
from PIL import Image, ImageFont
import os
import sys
import cv2
import numpy as np
import re
from collections import defaultdict
import pandas as pd
import io
import classification_service

# --- 로컬 Pororo 모듈 경로 설정 ---
# 이 파일의 위치를 기준으로 경로를 다시 계산해야 합니다.
# customMydataService/backend/ocr_service.py 이므로, 두 단계 위로 올라가야 합니다.
PORORO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'pororo_easyocr_main'))
if PORORO_PATH not in sys.path:
    sys.path.append(PORORO_PATH)

try:
    from main import EasyPororoOcr
except ImportError:
    print(f"오류: '{PORORO_PATH}' 경로에서 Pororo 모듈을 찾을 수 없습니다.")
    raise

# --- 전역 변수로 예측기(predictor)를 관리 (싱글톤 패턴) ---
# 서버가 켜질 때 한 번만 로드하여 재사용합니다.
predictor = None

# --- Jupyter Notebook의 클래스와 함수들을 여기에 붙여넣고 수정 ---

class YOLOv8_OCR_Predictor:
    # ... y_predict.ipynb의 YOLOv8_OCR_Predictor 클래스 내용 전체를 여기에 복사 ...
    # 단, detect_only, run_ocr_on_detections 메서드의 입력값을 image_path 대신 image (PIL Image 객체)로 변경해야 합니다.
    # 예시: def detect_only(self, image, conf_threshold=0.5):
    #           original_image = image.convert("RGB") # Image.open() 대신 이 코드를 사용
    # ... (아래 전체 코드 제공)
    pass # 우선 빈 클래스로 두고 아래 전체 코드로 대체

# ... y_predict.ipynb의 다른 함수들도 여기에 복사 ...
# calculate_iou, is_valid_ocr_text, run_hybrid_prediction, structure_transactions_sequentially 등
# 마찬가지로 plt.show(), display() 같은 출력 코드는 모두 제거합니다.


# --- 아래는 위 설명을 바탕으로 수정한 전체 코드입니다. 이 코드를 사용하세요. ---

class YOLOv8_OCR_Predictor:
    """YOLOv8로 객체를 탐지하고, 탐지된 영역에서 OCR을 수행하는 예측기 (API용)"""
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO 모델 가중치 파일을 찾을 수 없습니다: {model_path}")
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.ocr_reader = EasyPororoOcr(gpu=(self.device == 'cuda'))

    def detect_only(self, image, conf_threshold=0.5):
        results = self.model.predict(image, conf=conf_threshold, verbose=False)
        detected_objects = []
        for res in results:
            boxes = res.boxes.cpu().numpy()
            for box in boxes:
                class_id = int(box.cls[0])
                label = self.model.names[class_id]
                confidence = float(box.conf[0])
                coords = [int(c) for c in box.xyxy[0]]
                detected_objects.append({'label': label, 'box': coords, 'confidence': confidence})
        detected_objects.sort(key=lambda obj: (obj['box'][1], obj['box'][0]))
        return detected_objects

def run_hybrid_prediction(image, predictor_instance, conf_threshold=0.4):
    yolo_dets = predictor_instance.detect_only(image, conf_threshold=conf_threshold)
    if not yolo_dets:
        return []

    # ... (Jupyter Notebook의 run_hybrid_prediction 함수 로직) ...
    # 단순화를 위해 여기서는 핵심 로직만 남기고, 복잡한 보완 로직은 생략하거나 나중에 추가합니다.
    # 지금은 YOLO 탐지 -> OCR 까지만 구현하여 빠르게 테스트하는 것이 중요합니다.
    
    # --- 2. 탐지 영역 일괄 OCR ---
    yolo_predictions = []
    if yolo_dets:
        cropped_images_with_info = []
        padding = 10
        for det in yolo_dets:
            box = det['box']
            crop_box = (
                max(0, box[0] - padding), max(0, box[1] - padding),
                min(image.width, box[2] + padding), min(image.height, box[3] + padding)
            )
            cropped_img = image.crop(crop_box)
            cropped_images_with_info.append({'image': cropped_img, 'original_det': det})

        max_width = max(img.width for img in [item['image'] for item in cropped_images_with_info]) if cropped_images_with_info else 0
        total_height = sum(img.height for img in [item['image'] for item in cropped_images_with_info])
        
        if max_width == 0 or total_height == 0:
            return []

        composite_image = Image.new('RGB', (max_width, total_height), (0, 0, 0))
        y_offset = 0
        crop_y_boundaries = []
        for item in cropped_images_with_info:
            img = item['image']
            composite_image.paste(img, (0, y_offset))
            crop_y_boundaries.append((y_offset, y_offset + img.height))
            y_offset += img.height

        composite_cv_image = cv2.cvtColor(np.array(composite_image), cv2.COLOR_RGB2BGR)
        predictor_instance.ocr_reader.run_ocr(composite_cv_image, debug=False)
        composite_ocr_results = predictor_instance.ocr_reader.get_ocr_result()

        for i, (y_start, y_end) in enumerate(crop_y_boundaries):
            texts_for_this_crop = []
            if composite_ocr_results:
                for res in composite_ocr_results:
                    ocr_box_y_center = (res[0][0][1] + res[0][2][1]) / 2
                    if y_start <= ocr_box_y_center < y_end:
                        texts_for_this_crop.append(res[1])
            
            text = ' '.join(texts_for_this_crop)
            original_det = cropped_images_with_info[i]['original_det']
            
            yolo_predictions.append({
                'label': original_det['label'], 'text': text.strip(), 'box': original_det['box'],
                'confidence': f"{original_det['confidence']:.2f}", 'source': 'YOLO-Primary'
            })
            
    final_predictions = sorted(yolo_predictions, key=lambda p: (p['box'][1], p['box'][0]))
    return final_predictions

def structure_transactions_sequentially(predictions):
    # ... y_predict.ipynb의 structure_transactions_sequentially 함수 내용 전체를 여기에 복사 ...
    # 단, 입력값 predictions는 image_id가 없으므로 관련 로직을 단순화합니다.
    def parse_amount(text, label):
        if not text: return None
        try:
            cleaned_text = re.sub(r'[^\d-]', '', str(text))
            if not cleaned_text: return None
            amount = float(cleaned_text)
            if label == 'AMOUNT_OUT' and amount > 0: return -amount
            return amount
        except (ValueError, TypeError): return None

    all_transactions = []
    sorted_predictions = sorted(predictions, key=lambda p: (p['box'][1], p['box'][0]))
    current_transaction = {}
    last_known_date = None

    for item in sorted_predictions:
        label, text = item['label'], item['text']
        if label == 'DATE':
            last_known_date = text
            if current_transaction and not current_transaction.get('date'):
                current_transaction['date'] = last_known_date
        elif label == 'MERCHANT':
            if 'merchant' in current_transaction: current_transaction = {}
            current_transaction['merchant'] = text
        elif label in ['AMOUNT_IN', 'AMOUNT_OUT']:
            if 'amount' in current_transaction: current_transaction = {}
            current_transaction['amount'] = parse_amount(text, label)
        elif label == 'MEMO':
            current_transaction['memo'] = current_transaction.get('memo', '') + ' ' + text.strip()
        elif label == 'BALANCE':
            current_transaction['balance'] = text.strip()

        if 'merchant' in current_transaction and 'amount' in current_transaction:
            if not current_transaction.get('date'):
                current_transaction['date'] = last_known_date
            all_transactions.append(current_transaction)
            current_transaction = {}

    if not all_transactions: return pd.DataFrame()
    final_df = pd.DataFrame(all_transactions)
    final_columns = ['date', 'merchant', 'amount', 'balance', 'memo']
    existing_columns = [col for col in final_columns if col in final_df.columns]
    return final_df[existing_columns]


# --- API를 위한 메인 함수들 ---

def initialize_predictor(model_path):
    """서버 시작 시 예측기 인스턴스를 초기화합니다."""
    global predictor
    if predictor is None:
        predictor = YOLOv8_OCR_Predictor(model_path=model_path)

def process_image_to_transactions(image_bytes):
    """이미지 바이트를 입력받아 최종 거래 내역(JSON)을 반환합니다."""
    if predictor is None:
        raise Exception("OCR 예측기가 초기화되지 않았습니다.")

    # Bytes -> PIL Image
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # 1. 하이브리드 예측 실행
    # Jupyter Notebook의 복잡한 로직 대신 단순화된 버전을 사용합니다.
    predictions = run_hybrid_prediction(image, predictor)
    
    # 2. 결과 구조화
    df = structure_transactions_sequentially(predictions)

    # --- ✨새로운 단계: 업종 분류 추가 ---
    if not df.empty and 'merchant' in df.columns:
        # 'merchant' 열의 각 값에 대해 분류 함수를 적용하여 'category' 열을 생성합니다.
        df['category'] = df['merchant'].apply(classification_service.classify_merchant_category)
    else:
        df['category'] = "분류 안됨"

    # 3. DataFrame -> JSON 변환
    # orient='records'는 [{column: value}, ...] 형태의 리스트로 만들어줍니다.
    return df.to_dict(orient='records')
