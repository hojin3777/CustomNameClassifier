import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import glob
import pandas as pd
import random
from datetime import datetime, timedelta
from tqdm import tqdm
import argparse

# --- 1. 전역 설정 ---
COLOR_TO_LABEL_MAP = {
    (255, 0, 255): 'DATE',
    (0, 255, 255): 'DATE_HEADER',
    (0, 255, 0):   'MERCHANT',
    (0, 0, 255):   'MEMO',
    (255, 0, 0):   'AMOUNT',
    (255, 255, 0): 'BALANCE',
    (255, 127, 255): 'TIME',
}
MULTILINE_HEIGHT_THRESHOLD = 55
DATE_WIDTH_THRESHOLD = 130

# --- 2. 유틸리티 함수 ---

def load_fonts(font_dir, font_sizes=[32, 34, 36, 38]):
    """지정된 디렉토리에서 폰트 파일들을 로드합니다."""
    loaded_fonts = []
    try:
        font_files = [f for f in os.listdir(font_dir) if f.lower().endswith(('.ttf', '.otf'))]
        if not font_files:
            raise FileNotFoundError("폰트 파일을 찾을 수 없습니다.")
        for font_file in font_files:
            for size in font_sizes:
                loaded_fonts.append(ImageFont.truetype(os.path.join(font_dir, font_file), size))
        print(f"'{font_dir}'에서 {len(font_files)}개 폰트 파일을 로드하여 {len(loaded_fonts)}개의 폰트 객체를 생성했습니다.")
    except FileNotFoundError as e:
        print(f"경고: '{font_dir}' 폴더에 폰트 파일이 없습니다. 기본 폰트를 사용합니다. ({e})")
        loaded_fonts.append(ImageFont.load_default())
    return loaded_fonts

def parse_layout_from_color_template(template_path, color_map):
    """색상 템플릿 이미지에서 각 라벨의 위치(bbox)를 파싱합니다."""
    try:
        image = cv2.imread(template_path)
        if image is None: return None
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception:
        return None
    layout = {label: [] for label in color_map.values()}
    for color, label in color_map.items():
        mask = cv2.inRange(image_rgb, np.array(color), np.array(color))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            layout[label].append((x, y, x + w, y + h))
    for label in layout:
        layout[label].sort(key=lambda bbox: (bbox[1], bbox[0]))
    return layout

def load_merchant_list(data_path):
    """거래처명 데이터 파일을 로드합니다."""
    try:
        merchant_df = pd.read_csv(data_path)
        merchant_list = merchant_df['상호명_Regulated'].dropna().unique().tolist()
        print(f"'{data_path}'에서 {len(merchant_list)}개의 고유 거래처명을 불러왔습니다.")
    except FileNotFoundError:
        print(f"경고: 거래처 데이터 파일('{data_path}')을 찾을 수 없습니다. 가상 거래처명을 사용합니다.")
        merchant_list = [f'가상상점_{i}' for i in range(200)]
        merchant_list.extend([f'가상회사_{i}(주)' for i in range(50)])
    return merchant_list

# --- 3. 데이터 생성 함수 ---

def generate_transaction_data(num_items, merchants, config, parsed_layout, allowed_date_formats):
    """한 이미지에 들어갈 거래 데이터(텍스트)를 생성합니다."""
    drawable_items = []
    memos = ['#체크카드', '#용돈', '#월급', '#이체', '#송금', '#카드결제', '#오픈뱅킹이체', '이자', '대체', '타행IB', '모바일']
    current_date = datetime.now() - timedelta(days=random.randint(0, 30))
    balance = random.randint(500000, 2000000)
    
    last_item_date_str = ""
    date_style = random.choice(config['date_style_options'])
    chosen_date_format = random.choice(allowed_date_formats)
    transaction_dates = []

    for i in range(num_items):
        day_delta = random.choices([0, 1, 2, 3], weights=[0.4, 0.3, 0.2, 0.1])[0]
        current_date -= timedelta(days=day_delta, hours=random.randint(1, 5))
        transaction_dates.append(current_date)
        date_str, time_str = chosen_date_format(current_date), current_date.strftime('%H:%M')

        if date_style == 'per_item' or (date_style == 'toss_like' and date_str != last_item_date_str):
            drawable_items.append({'label': 'DATE', 'text': date_str, 'item_index': i})
        last_item_date_str = date_str
        drawable_items.append({'label': 'TIME', 'text': time_str, 'item_index': i})

        merchant_name = random.choice(merchants)
        is_income = random.random() < 0.15
        amount_val = random.randint(500, 4000) * 100 if is_income else random.randint(10, 200) * 100
        
        if config['amount_style'] == 'text':
            label, text = ('AMOUNT_IN', f"입금 {amount_val:,}원") if is_income else ('AMOUNT_OUT', f"출금 {amount_val:,}원")
        else:
            amount_val_signed = amount_val if is_income else -amount_val
            label, text = ('AMOUNT_IN', f"{amount_val:,}원") if is_income else ('AMOUNT_OUT', f"{amount_val_signed:,}원")
        
        balance += amount_val if is_income else -amount_val
        
        # ★★★ 변경점: 잔액(BALANCE) 생성 방식 다양화 ★★★
        if random.random() < 0.5: # 50% 확률로 '잔액' 키워드 포함
            balance_text = f"잔액 {balance:,}원"
        else: # 50% 확률로 금액만 표시
            balance_text = f"{balance:,}원"

        drawable_items.extend([
            {'label': 'MERCHANT', 'text': merchant_name, 'item_index': i},
            {'label': label, 'text': text, 'item_index': i},
            {'label': 'BALANCE', 'text': balance_text, 'item_index': i}
        ])
        if random.random() < 0.7:
            drawable_items.append({'label': 'MEMO', 'text': random.choice(memos), 'item_index': i})

    if transaction_dates and parsed_layout.get('DATE_HEADER'):
        start_date, end_date = min(transaction_dates), max(transaction_dates)
        header_format_options = [
            lambda s, e, n: f"{e.strftime('%Y.%m.%d')} ~ {s.strftime('%Y.%m.%d')} ({n}건)",
            lambda s, e, n: f"{e.strftime('%Y년 %m월')}"
        ]
        header_text = random.choice(header_format_options)(start_date, end_date, num_items)
        drawable_items.append({'label': 'DATE_HEADER', 'text': header_text, 'item_index': 0})
    return drawable_items

# --- 4. 메인 생성 함수 ---

def generate_synthetic_images(template_configs, merchant_list, loaded_fonts, num_images, output_dir):
    """설정된 모든 템플릿을 사용하여 지정된 개수의 학습 이미지를 생성합니다."""
    if not loaded_fonts or not template_configs:
        print("폰트 또는 템플릿 설정이 없어 생성을 중단합니다.")
        return

    all_labels_data = []
    amount_color_schemes = {
        'blue_red':   {'AMOUNT_IN': (50, 100, 255), 'AMOUNT_OUT': (200, 30, 30)},
        'blue_black': {'AMOUNT_IN': (50, 100, 255), 'AMOUNT_OUT': (20, 20, 20)}
    }

    # ★★★ 변경점: 메모 색상 옵션 추가 ★★★
    memo_color_options = [
        (120, 120, 120), # 회색
        (44, 160, 44),   # 녹색
        (31, 119, 180)   # 연한 파란색 (토스 스타일)
    ]
    
    print(f"\n총 {num_images}개의 다양한 은행 명세서 이미지 생성을 시작합니다...")

    for i in tqdm(range(num_images), desc="이미지 생성 중"):
        chosen_config = random.choice(template_configs)
        parsed_layout = parse_layout_from_color_template(chosen_config['colored_path'], COLOR_TO_LABEL_MAP)
        if not parsed_layout or not parsed_layout.get('MERCHANT'):
            print(f"경고: '{chosen_config['name']}' 템플릿 파싱 실패. 건너뜁니다.")
            continue
            
        allowed_date_formats = [
            lambda d: d.strftime('%Y.%m.%d'), lambda d: d.strftime('%m.%d'), lambda d: f"{d.month}월 {d.day}일"
        ]
        if 'DATE' in parsed_layout and parsed_layout['DATE']:
            date_box_width = parsed_layout['DATE'][0][2] - parsed_layout['DATE'][0][0]
            if date_box_width < DATE_WIDTH_THRESHOLD:
                allowed_date_formats = [lambda d: d.strftime('%m.%d'), lambda d: f"{d.month}월 {d.day}일"]

        items_per_image = len(parsed_layout.get('MERCHANT', []))
        img = Image.open(chosen_config['clean_path']).convert("RGB")
        draw = ImageDraw.Draw(img)
        
        font = random.choice(loaded_fonts)
        memo_font_size = max(18, int(font.size * 0.85))
        memo_font = ImageFont.truetype(font.path, memo_font_size)
        
        image_data = generate_transaction_data(items_per_image, merchant_list, chosen_config, parsed_layout, allowed_date_formats)
        
        chosen_color_scheme = random.choice(list(amount_color_schemes.values()))
        # ★★★ 변경점: 이미지마다 메모 색상 랜덤 선택 ★★★
        chosen_memo_color = random.choice(memo_color_options)
        image_filename = f'synth_{i+1:05d}.png'

        for item in image_data:
            label, text, item_idx = item['label'], item['text'], item['item_index']
            layout_label = 'AMOUNT' if 'AMOUNT' in label else label
            current_item_idx = 0 if label == 'DATE_HEADER' else item_idx

            if layout_label in parsed_layout and current_item_idx < len(parsed_layout[layout_label]):
                x_min, y_min, x_max, y_max = parsed_layout[layout_label][current_item_idx]
                
                current_font = memo_font if label == 'MEMO' else font
                
                is_multiline = False
                if label in ['AMOUNT_IN', 'AMOUNT_OUT'] and chosen_config['amount_style'] == 'text':
                    if (y_max - y_min) > MULTILINE_HEIGHT_THRESHOLD:
                        text = text.replace(' ', '\n', 1)
                        is_multiline = True

                align_right = label in ['AMOUNT_OUT', 'AMOUNT_IN', 'BALANCE', 'TIME']
                
                if align_right:
                    max_line_width = max(draw.textlength(line, font=current_font) for line in text.split('\n'))
                    draw_x = x_max - max_line_width
                else:
                    draw_x = x_min

                if is_multiline:
                    color = chosen_color_scheme.get(label, (20,20,20))
                    draw.multiline_text((draw_x, y_min), text, font=current_font, fill=color, align="right" if align_right else "left")
                    final_bbox = draw.multiline_textbbox((draw_x, y_min), text, font=current_font)
                else:
                    text_bbox = draw.textbbox((0,0), text, font=current_font)
                    draw_y = y_min + (y_max - y_min - (text_bbox[3] - text_bbox[1])) / 2
                    
                    # ★★★ 변경점: 메모 색상 적용 ★★★
                    if label == 'MEMO':
                        color = chosen_memo_color
                    elif label == 'BALANCE':
                        color = (150, 150, 150)  # 잔액은 연한 회색으로 고정
                    elif label in ['DATE_HEADER', 'DATE', 'TIME']:
                        color = (100, 100, 100)
                    elif label in chosen_color_scheme:
                        color = chosen_color_scheme[label]
                    else:
                        color = (20, 20, 20)

                    draw.text((draw_x, draw_y), text, font=current_font, fill=color)
                    final_bbox = draw.textbbox((draw_x, draw_y), text, font=current_font)

                all_labels_data.append({'image_id': image_filename, 'text': text.replace('\n', ' '), 'x_min': final_bbox[0], 'y_min': final_bbox[1], 'x_max': final_bbox[2], 'y_max': final_bbox[3], 'label': label})

        img.save(os.path.join(output_dir, image_filename), "PNG")

    label_output_file = os.path.join(output_dir, '_labels.csv')
    labels_df = pd.DataFrame(all_labels_data)
    labels_df.to_csv(label_output_file, index=False, encoding='utf-8-sig')
    print(f"\n이미지 및 라벨 생성이 완료되었습니다. '{output_dir}' 폴더를 확인하세요.")
    print(f"총 {len(labels_df['image_id'].unique())}개의 이미지가 생성되었습니다.")

# --- 5. 스크립트 실행 부분 ---
# usage : python .\customOCR\generate.py --num_images 1000

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="합성 은행 거래내역 이미지 데이터셋 생성기")
    parser.add_argument('--num_images', type=int, default=100, help='생성할 총 이미지 개수')
    parser.add_argument('--template_dir', type=str, default='./bank_templates', help='템플릿 이미지가 있는 디렉토리')
    parser.add_argument('--font_dir', type=str, default='./fonts', help='TTF 폰트 파일이 있는 디렉토리')
    parser.add_argument('--merchant_data', type=str, default='./processed_data/region_all_processed_data_remap.csv', help='거래처명 데이터 CSV 파일 경로')
    parser.add_argument('--output_dir', type=str, default='./customOCR/generated_dataset', help='생성된 이미지와 라벨을 저장할 디렉토리')
    
    args = parser.parse_args()

    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 필요 데이터 로드
    fonts = load_fonts(args.font_dir)
    merchants = load_merchant_list(args.merchant_data)

    # 2. 템플릿 설정 로드
    template_files = glob.glob(os.path.join(args.template_dir, '*_clean.png'))
    template_configs = []
    for clean_path in template_files:
        colored_path = clean_path.replace('_clean.png', '_colored.png')
        if os.path.exists(colored_path):
            name = os.path.basename(clean_path).replace('_clean.png', '')
            amount_style = 'text' if 'shinhan' in name.lower() or 'kookmin' in name.lower() else 'sign'
            date_style_options = ['per_item', 'toss_like'] if 'kakao' in name.lower() else ['per_item']
            template_configs.append({
                'name': name, 'clean_path': clean_path, 'colored_path': colored_path,
                'amount_style': amount_style, 'date_style_options': date_style_options
            })
    print(f"{len(template_configs)}개의 템플릿 설정을 로드했습니다.")

    # 3. 이미지 생성 실행
    if merchants and template_configs and fonts:
        generate_synthetic_images(
            template_configs=template_configs,
            merchant_list=merchants,
            loaded_fonts=fonts,
            num_images=args.num_images,
            output_dir=args.output_dir
        )
    else:
        print("데이터 또는 템플릿, 폰트가 부족하여 이미지 생성을 시작할 수 없습니다.")