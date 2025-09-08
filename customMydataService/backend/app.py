from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import ocr_service
import classification_service
import database
import category_utils
import account_utils
import os
from datetime import date

# Flask 앱 초기화
app = Flask(__name__, static_folder='../frontend/dist', static_url_path='/')
CORS(app)  # 모든 도메인에서의 요청 허용 (개발 단계에서만 사용 권장)

# --- 서버 시작 시 한 번만 모델 및 DB 로드 ---
print("Starting server...")

# 1. DB 초기화
try:
    database.init_db()
    category_utils.initialize_default_categories() # 카테고리 기본값 채우기
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


# ------------------- 거래내역 API -------------------
@app.route('/api/transactions', methods=['GET'])
def get_transactions():
    """데이터베이스에 저장된 모든 거래 내역을 JOIN하여 조회합니다. (필요시 기본값 생성)"""
    try:
        all_trans = database.get_all_transactions_joined()

        # ✨ 거래내역이 비어있을 때 기본값 생성 로직
        if not all_trans:
            print("No transactions found. Creating default entry...")
            conn = database.get_db_connection()
            # 계좌 DB의 첫 번째 값을 가져옵니다.
            first_account_row = conn.execute('SELECT name FROM accounts ORDER BY id LIMIT 1').fetchone()
            conn.close()

            if first_account_row:
                default_transaction = {
                    "id": 1, # 첫 번째 거래내역이므로 ID를 1로 지정
                    "date": date.today().strftime('%Y-%m-%d'),
                    "account": first_account_row['name'],
                    "type": '이체',
                    "majorCategory": '이체분류',
                    "minorCategory": '내계좌이체',
                    "amount": 100000,
                    "payee": '계좌등록',
                    "memo": '첫 계좌의 초기 잔액을 입력하세요.'
                }
                # DB에 기본값 추가
                database.add_single_transaction(default_transaction)
                # 추가 후 데이터를 다시 불러옵니다.
                all_trans = database.get_all_transactions_joined()
            else:
                print("No accounts found. Cannot create default transaction.")


        # 프론트엔드 형식에 맞게 키 이름 변경
        renamed_trans = [
            {
                "id": t["id"],
                "checked": False,
                "date": t["trans_date"],
                "account": t["account"],
                "type": t["type"],
                "majorCategory": t["major_category"],
                "minorCategory": t.get("minor_category", ""),
                "amount": t["amount"],
                "payee": t["merchant"],
                "memo": t["memo"]
            } for t in all_trans
        ]
        return jsonify(renamed_trans)
    except Exception as e:
        print(f"Error fetching transactions: {e}")
        return jsonify({"error": "Failed to fetch transactions"}), 500

# ✨ 거래내역 저장 API 엔드포인트 추가
@app.route('/api/transactions', methods=['POST'])
def update_transactions():
    """프론트엔드에서 받은 거래내역 데이터를 DB에 저장하고 결과를 반환합니다."""
    transactions_data = request.get_json()
    try:
        database.save_all_transactions(transactions_data)
        # 저장 후, 최신 데이터를 다시 불러와서 반환 (get_transactions 로직 재사용)
        return get_transactions()
    except Exception as e:
        print(f"Error saving transactions: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/transactions/reset', methods=['POST'])
def reset_transactions():
    """거래내역 데이터를 모두 삭제하고 성공 메시지를 반환합니다."""
    try:
        database.reset_transactions_table()
        return jsonify({"message": "Transactions reset successfully"})
    except Exception as e:
        print(f"Error resetting transactions: {e}")
        return jsonify({"error": str(e)}), 500

# ------------------- 카테고리 API -------------------
@app.route('/api/categories', methods=['GET', 'POST'])
def manage_categories():
    if request.method == 'GET':
        categories = category_utils.load_categories_from_db()
        return jsonify(categories)
    if request.method == 'POST':
        data = request.get_json()
        category_utils.save_categories_to_db(data)
        return jsonify({"status": "success", "message": "Categories saved successfully"})


@app.route('/api/categories/usage', methods=['GET'])
def get_category_usage():
    uuid = request.args.get('uuid')
    if not uuid:
        return jsonify({"error": "UUID parameter is required."}), 400
    try:
        in_use = database.is_category_in_use(uuid)
        return jsonify({"in_use": in_use})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/categories/minor', methods=['PUT'])
def update_minor_category():
    """단일 소분류 카테고리의 이름을 변경합니다."""
    try:
        data = request.json
        old_major = data.get('oldMajor')
        old_minor = data.get('oldMinor')
        new_minor = data.get('newMinor')

        if not all([old_major, old_minor, new_minor]):
            return jsonify({"error": "필수 정보가 누락되었습니다."}), 400

        database.update_minor_category_name(old_major, old_minor, new_minor)
        return jsonify({"message": "카테고리 이름이 성공적으로 변경되었습니다."})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ✨ 2. 카테고리 삭제(DELETE) API 추가
@app.route('/api/categories/minor', methods=['DELETE'])
def delete_minor_category():
    """단일 소분류 카테고리를 삭제합니다. 사용 중인 경우 오류를 반환합니다."""
    try:
        data = request.json
        major = data.get('major')
        minor = data.get('minor')

        if not all([major, minor]):
            return jsonify({"error": "필수 정보가 누락되었습니다."}), 400

        database.delete_minor_category_if_unused(major, minor)
        return jsonify({"message": "카테고리가 성공적으로 삭제되었습니다."})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ------------------- 계좌 API -------------------
@app.route('/api/accounts', methods=['GET', 'POST'])
def manage_accounts():
    if request.method == 'GET':
        accounts = account_utils.load_accounts()
        return jsonify(accounts)
    if request.method == 'POST':
        data = request.get_json()
        account_utils.save_accounts(data)
        return jsonify({"status": "success"})


# 이 파일이 직접 실행될 때만 서버를 실행
if __name__ == '__main__':
    app.run(debug=False, port=5000) # 개발 시 debug=True
    # app.run(debug=False, port=5000, host='0.0.0.0') # 실제 서비스 시 debug=False 권장