import os
from PIL import Image, ImageDraw, ImageFont
import random

# --- 설정 ---
OUTPUT_DIR = "bank_templates_sketches"
IMAGE_WIDTH = 1080
IMAGE_HEIGHT = 2340
BG_COLOR = "#FFFFFF"  # 흰색 배경

# 색상 팔레트
HEADER_COLOR = "#F0F0F0"
TEXT_COLOR = "#333333"
SUBTEXT_COLOR = "#888888"
AMOUNT_IN_COLOR = "#2E86C1"
AMOUNT_OUT_COLOR = "#E74C3C"
SEPARATOR_COLOR = "#EAEAEA"

# 폰트 설정 (없을 경우 기본 폰트 사용)
try:
    font_bold = ImageFont.truetype("malgunbd.ttf", size=45)
    font_regular = ImageFont.truetype("malgun.ttf", size=40)
    font_small = ImageFont.truetype("malgun.ttf", size=35)
except IOError:
    print("경고: '맑은 고딕' 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")
    font_bold = ImageFont.load_default()
    font_regular = ImageFont.load_default()
    font_small = ImageFont.load_default()

# --- 헬퍼 함수 ---
def draw_transaction(draw, y_pos, is_deposit=True):
    """거래 내역 한 줄을 그리는 함수"""
    draw.text((80, y_pos), "거래처명", fill=TEXT_COLOR, font=font_regular)
    draw.text((80, y_pos + 55), "메모 내용", fill=SUBTEXT_COLOR, font=font_small)
    
    if is_deposit:
        amount_text = "+ 12,345원"
        color = AMOUNT_IN_COLOR
    else:
        amount_text = "- 54,321원"
        color = AMOUNT_OUT_COLOR
        
    text_width = draw.textlength(amount_text, font=font_bold)
    draw.text((IMAGE_WIDTH - 80 - text_width, y_pos), amount_text, fill=color, font=font_bold)
    draw.text((IMAGE_WIDTH - 80 - draw.textlength("잔액 1,234,567원", font_small), y_pos + 65), "잔액 1,234,567원", fill=SUBTEXT_COLOR, font=font_small)
    
    draw.line([(80, y_pos + 140), (IMAGE_WIDTH - 80, y_pos + 140)], fill=SEPARATOR_COLOR, width=2)
    return y_pos + 160

# --- 템플릿 생성 함수 ---

def create_template_date_as_separator():
    """
    [스케치 1] 날짜가 거래 내역 중간에 '구분선'처럼 들어가는 UI
    """
    # ★★★ 수정: 함수 시작 시 이미지와 그리기 객체를 새로 생성 ★★★
    img = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)
    
    # 헤더
    draw.rectangle([0, 0, IMAGE_WIDTH, 150], fill=HEADER_COLOR)
    draw.text((80, 60), "거래내역조회", fill=TEXT_COLOR, font=font_bold)
    
    y = 250
    # 첫 번째 날짜 그룹
    draw.rectangle([80, y - 10, 400, y + 50], fill="#EAECEE")
    draw.text((100, y), "2025.07.31", fill=TEXT_COLOR, font=font_regular)
    y += 80
    
    y = draw_transaction(draw, y, is_deposit=True)
    y = draw_transaction(draw, y, is_deposit=False)
    
    # 두 번째 날짜 그룹 (구분선 역할)
    y += 40
    draw.rectangle([80, y - 10, 400, y + 50], fill="#EAECEE")
    draw.text((100, y), "2025.07.30", fill=TEXT_COLOR, font=font_regular)
    y += 80
    
    y = draw_transaction(draw, y, is_deposit=False)
    y = draw_transaction(draw, y, is_deposit=True)
    y = draw_transaction(draw, y, is_deposit=True)

    img.save(os.path.join(OUTPUT_DIR, "sketch_template_separator_date.png"))
    print("스케치 1 (날짜 구분선 타입) 생성 완료.")

def create_template_date_top_right():
    """
    [스케치 2] 날짜가 헤더의 '오른쪽'에 위치하는 UI
    """
    # ★★★ 수정: 함수 시작 시 이미지와 그리기 객체를 새로 생성 ★★★
    img = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)
    
    # 헤더
    draw.rectangle([0, 0, IMAGE_WIDTH, 150], fill=HEADER_COLOR)
    draw.text((80, 60), "거래내역조회", fill=TEXT_COLOR, font=font_bold)
    
    # 헤더 오른쪽 날짜
    date_text = "조회기간: 2025.07.01 ~ 07.31"
    text_width = draw.textlength(date_text, font=font_regular)
    draw.text((IMAGE_WIDTH - 80 - text_width, 65), date_text, fill=SUBTEXT_COLOR, font=font_regular)
    
    y = 250
    y = draw_transaction(draw, y, is_deposit=True)
    y = draw_transaction(draw, y, is_deposit=False)
    y = draw_transaction(draw, y, is_deposit=False)
    y = draw_transaction(draw, y, is_deposit=True)

    img.save(os.path.join(OUTPUT_DIR, "sketch_template_top_right_date.png"))
    print("스케치 2 (헤더 오른쪽 날짜 타입) 생성 완료.")

def create_template_header_and_dates():
    """
    [스케치 3] 헤더에 '조회기간'이 있고, 거래내역에도 '세부 날짜'가 있는 UI
    - 가장 현실적인 형태의 새로운 레이아웃 학습 유도
    """
    img = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)
    
    # 헤더
    draw.rectangle([0, 0, IMAGE_WIDTH, 150], fill=HEADER_COLOR)
    draw.text((80, 60), "거래내역조회", fill=TEXT_COLOR, font=font_bold)
    
    # 헤더 오른쪽 날짜 (DATE_HEADER)
    date_header_text = "조회기간: 2025.07.30 ~ 07.31"
    text_width = draw.textlength(date_header_text, font=font_regular)
    draw.text((IMAGE_WIDTH - 80 - text_width, 65), date_header_text, fill=SUBTEXT_COLOR, font=font_regular)
    
    y = 250
    # 첫 번째 날짜 그룹 (DATE)
    draw.text((80, y), "2025.07.31", fill=TEXT_COLOR, font=font_regular)
    y += 60
    
    y = draw_transaction(draw, y, is_deposit=True)
    
    # 두 번째 날짜 그룹 (DATE)
    y += 40
    draw.text((80, y), "2025.07.30", fill=TEXT_COLOR, font=font_regular)
    y += 60
    
    y = draw_transaction(draw, y, is_deposit=False)
    y = draw_transaction(draw, y, is_deposit=False)

    img.save(os.path.join(OUTPUT_DIR, "sketch_template_header_with_dates.png"))
    print("스케치 3 (헤더+세부날짜 타입) 생성 완료.")

def draw_transaction_reversed(draw, y_pos, is_deposit=True):
    """(스케치 4용) 좌우 반전된 거래 내역 한 줄을 그리는 함수"""
    # 금액/잔액을 왼쪽에 표시
    if is_deposit:
        amount_text = "+ 12,345원"
        color = AMOUNT_IN_COLOR
    else:
        amount_text = "- 54,321원"
        color = AMOUNT_OUT_COLOR
    
    draw.text((80, y_pos), amount_text, fill=color, font=font_bold)
    draw.text((80, y_pos + 65), "잔액 1,234,567원", fill=SUBTEXT_COLOR, font=font_small)

    # 거래처명/메모를 오른쪽에 표시 (우측 정렬)
    merchant_text = "거래처명"
    memo_text = "메모 내용"
    merchant_width = draw.textlength(merchant_text, font=font_regular)
    memo_width = draw.textlength(memo_text, font=font_small)

    draw.text((IMAGE_WIDTH - 80 - merchant_width, y_pos), merchant_text, fill=TEXT_COLOR, font=font_regular)
    draw.text((IMAGE_WIDTH - 80 - memo_width, y_pos + 55), memo_text, fill=SUBTEXT_COLOR, font=font_small)
    
    draw.line([(80, y_pos + 140), (IMAGE_WIDTH - 80, y_pos + 140)], fill=SEPARATOR_COLOR, width=2)
    return y_pos + 160

def create_template_reversed_layout():
    """
    [스케치 4] 좌우가 반전된 레이아웃
    - 모델이 위치에만 의존하는 것을 방지
    """
    img = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)
    
    draw.rectangle([0, 0, IMAGE_WIDTH, 150], fill=HEADER_COLOR)
    draw.text((80, 60), "입출금내역", fill=TEXT_COLOR, font=font_bold)
    
    y = 250
    y = draw_transaction_reversed(draw, y, is_deposit=False)
    y = draw_transaction_reversed(draw, y, is_deposit=True)
    y = draw_transaction_reversed(draw, y, is_deposit=False)

    img.save(os.path.join(OUTPUT_DIR, "sketch_template_reversed_layout.png"))
    print("스케치 4 (좌우 반전 타입) 생성 완료.")

def create_template_datetime_combined():
    """
    [스케치 5] 날짜와 시간이 결합된 레이아웃
    - DATE와 TIME 클래스를 더 정교하게 분리하도록 학습
    """
    img = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)

    draw.rectangle([0, 0, IMAGE_WIDTH, 150], fill=HEADER_COLOR)
    draw.text((80, 60), "상세조회", fill=TEXT_COLOR, font=font_bold)

    y = 250
    # 날짜와 시간을 한 줄에 표시
    draw.text((80, y), "2025.07.31 15:30:22", fill=TEXT_COLOR, font=font_regular)
    y += 60
    y = draw_transaction(draw, y, is_deposit=True)
    
    y += 40
    draw.text((80, y), "2025.07.31 11:15:05", fill=TEXT_COLOR, font=font_regular)
    y += 60
    y = draw_transaction(draw, y, is_deposit=False)

    img.save(os.path.join(OUTPUT_DIR, "sketch_template_datetime_combined.png"))
    print("스케치 5 (날짜+시간 결합 타입) 생성 완료.")

def create_template_day_of_week_date():
    """
    [스케치 6] 다른 날짜 형식 (요일 포함)
    - 다양한 날짜 포맷에 대한 대응 능력 강화
    """
    img = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)

    draw.rectangle([0, 0, IMAGE_WIDTH, 150], fill=HEADER_COLOR)
    draw.text((80, 60), "입출금 알림", fill=TEXT_COLOR, font=font_bold)

    y = 250
    # '월.일 (요일)' 형식의 날짜
    draw.text((80, y), "07.31 (목)", fill=TEXT_COLOR, font=font_regular)
    y += 60
    y = draw_transaction(draw, y, is_deposit=False)
    
    y += 40
    draw.text((80, y), "07.30 (수)", fill=TEXT_COLOR, font=font_regular)
    y += 60
    y = draw_transaction(draw, y, is_deposit=True)
    y = draw_transaction(draw, y, is_deposit=False)

    img.save(os.path.join(OUTPUT_DIR, "sketch_template_day_of_week_date.png"))
    print("스케치 6 (요일 포함 날짜 타입) 생성 완료.")

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    create_template_date_as_separator()
    # create_template_date_top_right()
    create_template_header_and_dates()
    create_template_reversed_layout()
    create_template_datetime_combined()
    create_template_day_of_week_date()
    create_template_reversed_layout()
    print(f"\n모든 스케치 생성이 완료되었습니다. '{OUTPUT_DIR}' 폴더를 확인하세요.")