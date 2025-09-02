from flask import Flask, jsonify, request
from flask_cors import CORS
import ocr_service
import classification_service
import database  # database.py 임포트
import os

# Flask 앱 초기화
app = Flask(__name__)
CORS(app)  # 모든 도메인에서의 요청 허용 (개발 단계에서만 사용 권장)

# --- 서버 시작 시 한 번만 모델 및 DB 로드 ---
print("Starting server...")

# 1. DB 초기화
try:
    database.init_db()
except Exception as e:
    print(f"ERROR: Database initialization failed - {e}")

# 1. OCR 서비스 초기화
print("Initializing OCR service...")
# ★★★ 모델 경로를 실제 best.pt 파일 위치로 수정해야 합니다. ★★★
OCR_MODEL_PATH = 'C:/code/customOCR/bank_statement_detector/yolov8l_e50_bs8_0828/weights/best.pt'
try:
    ocr_service.initialize_predictor(model_path=OCR_MODEL_PATH)
    print("OCR service initialized successfully.")
except Exception as e:
    print(f"ERROR: OCR service initialization failed - {e}")

# 2. 업종 분류 서비스 초기화
print("Initializing Classification service...")
try:
    classification_service.initialize_classifier()
    print("Classification service initialized successfully.")
except Exception as e:
    print(f"ERROR: Classification service initialization failed - {e}")
# -----------------------------------------

# 기본 API 엔드포인트 정의
@app.route('/')
def home():
    return jsonify({"message": "Backend server is running successfully!"})

# 이미지 처리를 위한 API 엔드포인트
@app.route('/api/ocr', methods=['POST'])
def process_image_ocr():
    """
    업로드된 이미지를 받아 OCR을 수행하고 구조화된 데이터를 반환합니다.
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image file found in the request."}), 400

    file = request.files['image']
    
    if file.filename == '':
        return jsonify({"error": "No file selected."}), 400

    try:
        image_bytes = file.read()
        transactions = ocr_service.process_image_to_transactions(image_bytes)
        
        # --- ✨새로운 단계: 결과를 DB에 저장 ---
        for trans in transactions:
            database.add_transaction(trans)
        # ------------------------------------
        all_transactions = database.get_all_transactions()
        return jsonify(all_transactions)

    except Exception as e:
        # 실제 운영 환경에서는 더 상세한 로깅이 필요
        print(f"Error during OCR processing: {e}")
        return jsonify({"error": f"An error occurred during image processing: {e}"}), 500

# ✨새로운 API: 모든 거래 내역 조회
@app.route('/api/transactions', methods=['GET'])
def get_transactions():
    """데이터베이스에 저장된 모든 거래 내역을 조회합니다."""
    try:
        all_trans = database.get_all_transactions()
        return jsonify(all_trans)
    except Exception as e:
        print(f"Error fetching transactions: {e}")
        return jsonify({"error": "Failed to fetch transactions"}), 500

# 이 파일이 직접 실행될 때만 서버를 실행
if __name__ == '__main__':
    app.run(debug=False, port=5000) # 실제 서비스 시 debug=False 권장