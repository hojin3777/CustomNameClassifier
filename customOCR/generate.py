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

# --- 1. 전역 설정 및 하이퍼파라미터 ---

# 라벨링을 위한 색상-라벨 매핑
COLOR_TO_LABEL_MAP = {
    (255, 0, 255): 'DATE',
    (0, 255, 255): 'DATE_HEADER',
    (0, 255, 0):   'MERCHANT',
    (0, 0, 255):   'MEMO',
    (255, 0, 0):   'AMOUNT',
    (255, 255, 0): 'BALANCE',
    (255, 127, 255): 'TIME',
}

# 폰트 크기 범위
FONT_SIZES = [34, 35, 36, 37, 38]

# 생성 로직을 위한 임계값
MULTILINE_HEIGHT_THRESHOLD = 55
DATE_WIDTH_THRESHOLD = 150
TIME_WIDTH_THRESHOLD = 80

# 생성될 텍스트 및 색상 옵션
MEMO_OPTIONS = ['#체크카드', '#용돈', '#월급', '#이체', '#송금', '#카드결제', '#오픈뱅킹이체', '이자', '대체', '타행IB', '모바일']
AMOUNT_STYLE_OPTIONS = ['text', 'sign', 'split_color'] # ★★★ 금액 스타일 옵션화
AMOUNT_COLOR_SCHEMES = {
    'blue_red':   {'AMOUNT_IN': (50, 100, 255), 'AMOUNT_OUT': (200, 30, 30)},
    'blue_black': {'AMOUNT_IN': (50, 100, 255), 'AMOUNT_OUT': (20, 20, 20)}
}
MEMO_COLOR_OPTIONS = [(120, 120, 120), (44, 160, 44), (31, 119, 180)]
DEFAULT_TEXT_COLOR = (20, 20, 20)
DATE_TIME_COLOR = (100, 100, 100)
BALANCE_COLOR = (150, 150, 150)


# --- 2. 유틸리티 함수 ---

def load_fonts(font_dir, font_sizes=FONT_SIZES):
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

def generate_transaction_data(num_items, merchants, config, use_balance_keyword, use_seconds, allowed_date_formats, amount_style, parsed_layout):
    """한 이미지에 들어갈 거래 데이터(텍스트)를 생성합니다."""
    drawable_items = []
    current_date = datetime.now() - timedelta(days=random.randint(0, 30))
    balance = random.randint(500000, 2000000)
    
    last_item_date_str = ""
    date_style = random.choice(config['date_style_options'])
    chosen_date_format = random.choice(allowed_date_formats)
    time_format = '%H:%M:%S' if use_seconds else '%H:%M'

    for i in range(num_items):
        day_delta = random.choices([0, 1, 2, 3], weights=[0.4, 0.3, 0.2, 0.1])[0]
        current_date -= timedelta(days=day_delta, hours=random.randint(1, 5))
        
        date_str, time_str = chosen_date_format(current_date), current_date.strftime(time_format)

        if date_style == 'per_item' or (date_style == 'toss_like' and date_str != last_item_date_str):
            drawable_items.append({'label': 'DATE', 'text': date_str, 'item_index': i})
        last_item_date_str = date_str
        drawable_items.append({'label': 'TIME', 'text': time_str, 'item_index': i})

        merchant_name = random.choice(merchants)
        is_income = random.random() < 0.15
        amount_val = random.randint(500, 4000) * 100 if is_income else random.randint(10, 200) * 100
        
        if amount_style == 'text' or amount_style == 'split_color':
            label, text = ('AMOUNT_IN', f"입금 {amount_val:,}원") if is_income else ('AMOUNT_OUT', f"출금 {amount_val:,}원")
        else: # 'sign' style
            amount_val_signed = amount_val if is_income else -amount_val
            label, text = ('AMOUNT_IN', f"{amount_val:,}원") if is_income else ('AMOUNT_OUT', f"{amount_val_signed:,}원")
        
        balance += amount_val if is_income else -amount_val
        balance_text = f"잔액 {balance:,}원" if use_balance_keyword else f"{balance:,}원"

        drawable_items.extend([
            {'label': 'MERCHANT', 'text': merchant_name, 'item_index': i},
            {'label': label, 'text': text, 'item_index': i},
            {'label': 'BALANCE', 'text': balance_text, 'item_index': i}
        ])
        if random.random() < 0.7:
            drawable_items.append({'label': 'MEMO', 'text': random.choice(MEMO_OPTIONS), 'item_index': i})

    # ★★★ 템플릿에 DATE_HEADER 영역이 존재할 때만 헤더 텍스트를 생성 ★★★
    num_headers = len(parsed_layout.get('DATE_HEADER', []))
    if num_headers > 0:
        # 첫 번째 헤더: 전체 기간
        start_date, end_date = current_date, datetime.now()
        header_text = f"{end_date.strftime('%Y.%m.%d')} ~ {start_date.strftime('%Y.%m.%d')} ({num_items}건)"
        drawable_items.append({'label': 'DATE_HEADER', 'text': header_text, 'item_index': 0})
        
        # ★★★ 나머지 헤더: "YYYY년 n월" 형식으로 랜덤하게 채움 ★★★
        if num_headers > 1:
            # 사용된 월 기록 (중복 방지)
            used_months = {current_date.month}
            year = current_date.year

            for header_idx in range(1, num_headers):
                random_month = random.randint(1, 12)
                # 이전에 사용되지 않은 월이 나올 때까지 반복
                while random_month in used_months:
                    random_month = random.randint(1, 12)
                used_months.add(random_month)
                
                # 헤더 텍스트 생성 및 추가
                month_header_text = f"{year}년 {random_month}월"
                drawable_items.append({'label': 'DATE_HEADER', 'text': month_header_text, 'item_index': header_idx})
        
    return drawable_items

# --- 4. 메인 생성 함수 ---

def generate_synthetic_images(template_configs, merchant_list, loaded_fonts, num_images, output_dir):
    """설정된 모든 템플릿을 사용하여 지정된 개수의 학습 이미지를 생성합니다."""
    if not loaded_fonts or not template_configs:
        print("폰트 또는 템플릿 설정이 없어 생성을 중단합니다.")
        return

    all_labels_data = []
    print(f"\n총 {num_images}개의 다양한 은행 명세서 이미지 생성을 시작합니다...")

    for i in tqdm(range(num_images), desc="이미지 생성 중"):
        image_filename = f'img_{i+1:05d}.png'
        chosen_config = random.choice(template_configs)
        parsed_layout = parse_layout_from_color_template(chosen_config['colored_path'], COLOR_TO_LABEL_MAP)
        if not parsed_layout or not parsed_layout.get('MERCHANT'):
            continue
            
        allowed_date_formats = [lambda d: d.strftime('%Y.%m.%d'), lambda d: d.strftime('%m.%d'), lambda d: f"{d.month}월 {d.day}일"]
        if 'DATE' in parsed_layout and parsed_layout['DATE']:
            if (parsed_layout['DATE'][0][2] - parsed_layout['DATE'][0][0]) < DATE_WIDTH_THRESHOLD:
                allowed_date_formats = [lambda d: d.strftime('%m.%d'), lambda d: f"{d.month}월 {d.day}일"]

        use_long_format_possible = False
        if 'TIME' in parsed_layout and parsed_layout['TIME']:
            if (parsed_layout['TIME'][0][2] - parsed_layout['TIME'][0][0]) > TIME_WIDTH_THRESHOLD:
                use_long_format_possible = True
        use_seconds_for_this_image = use_long_format_possible and random.random() < 0.5

        items_per_image = len(parsed_layout.get('MERCHANT', []))
        img = Image.open(chosen_config['clean_path']).convert("RGB")
        draw = ImageDraw.Draw(img)
        
        font = random.choice(loaded_fonts)
        memo_font = ImageFont.truetype(font.path, max(18, int(font.size * 0.85)))
        datetime_font = ImageFont.truetype(font.path, max(18, font.size - 2))
        
        use_balance_keyword = random.random() < 0.5
        chosen_amount_style = random.choice(AMOUNT_STYLE_OPTIONS) # ★★★ 금액 스타일 랜덤 선택
        
        image_data = generate_transaction_data(items_per_image, merchant_list, chosen_config, use_balance_keyword, use_seconds_for_this_image, allowed_date_formats, chosen_amount_style, parsed_layout)
        
        chosen_color_scheme = random.choice(list(AMOUNT_COLOR_SCHEMES.values()))
        chosen_memo_color = random.choice(MEMO_COLOR_OPTIONS)
        

        for item in image_data:
            label, text, item_idx = item['label'], item['text'], item['item_index']
            layout_label = 'AMOUNT' if 'AMOUNT' in label else label
            current_item_idx = item_idx

            if layout_label not in parsed_layout or current_item_idx >= len(parsed_layout[layout_label]):
                continue
            
            x_min, y_min, x_max, y_max = parsed_layout[layout_label][current_item_idx]
            
            if label == 'MEMO': current_font = memo_font
            elif label in ['DATE', 'TIME', 'DATE_HEADER']: current_font = datetime_font
            else: current_font = font
            
            align_right = label in ['AMOUNT_OUT', 'AMOUNT_IN', 'BALANCE']
            
            final_bbox = (0,0,0,0)
            is_multiline_amount = 'AMOUNT' in label and (y_max - y_min) > MULTILINE_HEIGHT_THRESHOLD

            if is_multiline_amount:
                # 두 줄짜리 금액은 항상 'text' 스타일(입금/출금)을 강제 사용
                parts = text.split(' ', 1)
                if len(parts) == 2:
                    prefix, number_part = parts
                else: # 혹시 모를 예외 처리 (sign 스타일로 생성된 경우)
                    prefix = "입금" if float(text.replace(',', '').replace('원','')) > 0 else "출금"
                    number_part = text.lstrip('-+')

                color = chosen_color_scheme[label]
                
                # 1. 첫 번째 줄 (접두사: "입금" 또는 "출금") 그리기
                prefix_bbox = draw.textbbox((0,0), prefix, font=current_font)
                prefix_h = prefix_bbox[3] - prefix_bbox[1]
                prefix_x = x_max - draw.textlength(prefix, font=current_font) if align_right else x_min
                prefix_y = y_min + (y_max - y_min) / 2 - prefix_h - 2 # 중앙보다 살짝 위로
                draw.text((prefix_x, prefix_y), prefix, font=current_font, fill=color)

                # 2. 두 번째 줄 (숫자 금액) 그리기
                number_x = x_max - draw.textlength(number_part, font=current_font) if align_right else x_min
                number_y = y_min + (y_max - y_min) / 2 + 2 # 중앙보다 살짝 아래로
                draw.text((number_x, number_y), number_part, font=current_font, fill=color)

                # 전체 영역에 대한 바운딩 박스 계산
                final_bbox = (min(prefix_x, number_x), prefix_y, x_max, number_y + prefix_h)

            elif chosen_amount_style == 'split_color' and 'AMOUNT' in label:
                # 기존 'split_color' 로직 (한 줄)
                parts = text.split(' ', 1)
                prefix, number_part = (parts[0] + ' ', parts[1]) if len(parts) > 1 else ('', parts[0])
                
                prefix_w = draw.textlength(prefix, font=current_font)
                draw_x_start = x_max - draw.textlength(text, font=current_font) if align_right else x_min
                draw_y = y_min + (y_max - y_min - current_font.getbbox("A")[3]) / 2

                draw.text((draw_x_start, draw_y), prefix, font=current_font, fill=DEFAULT_TEXT_COLOR)
                draw.text((draw_x_start + prefix_w, draw_y), number_part, font=current_font, fill=chosen_color_scheme[label])
                
                final_bbox = draw.textbbox((draw_x_start, draw_y), text, font=current_font)

            else:
                draw_x = x_max - draw.textlength(text, font=current_font) if align_right else x_min
                text_bbox = draw.textbbox((0,0), text, font=current_font)
                draw_y = y_min + (y_max - y_min - (text_bbox[3] - text_bbox[1])) / 2
                
                if label == 'MEMO': color = chosen_memo_color
                elif label == 'BALANCE': color = BALANCE_COLOR
                elif label in ['DATE_HEADER', 'DATE', 'TIME']: color = DATE_TIME_COLOR
                elif 'AMOUNT' in label: color = chosen_color_scheme[label]
                else: color = DEFAULT_TEXT_COLOR

                draw.text((draw_x, draw_y), text, font=current_font, fill=color)
                final_bbox = draw.textbbox((draw_x, draw_y), text, font=current_font)

            all_labels_data.append({'image_id': image_filename, 'text': text, 'x_min': final_bbox[0], 'y_min': final_bbox[1], 'x_max': final_bbox[2], 'y_max': final_bbox[3], 'label': label})

        img.save(os.path.join(output_dir, image_filename), "PNG")

    label_output_file = os.path.join(output_dir, '_labels.csv')
    labels_df = pd.DataFrame(all_labels_data)
    labels_df.to_csv(label_output_file, index=False, encoding='utf-8-sig')
    print(f"\n이미지 및 라벨 생성이 완료되었습니다. '{output_dir}' 폴더를 확인하세요.")
    print(f"총 {len(labels_df['image_id'].unique())}개의 이미지가 생성되었습니다.")

# --- 5. 스크립트 실행 부분 ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="합성 은행 거래내역 이미지 데이터셋 생성기")
    parser.add_argument('--num_images', type=int, default=50000, help='생성할 총 이미지 개수')
    parser.add_argument('--template_dir', type=str, default='./bank_templates', help='템플릿 이미지가 있는 디렉토리')
    parser.add_argument('--font_dir', type=str, default='./fonts', help='TTF 폰트 파일이 있는 디렉토리')
    parser.add_argument('--merchant_data', type=str, default='./processed_data/region_all_processed_data_remap.csv', help='거래처명 데이터 CSV 파일 경로')
    parser.add_argument('--output_dir', type=str, default='./customOCR/generated_dataset', help='생성된 이미지와 라벨을 저장할 디렉토리')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    fonts = load_fonts(args.font_dir)
    merchants = load_merchant_list(args.merchant_data)

    template_files = glob.glob(os.path.join(args.template_dir, '*_clean.png'))
    template_configs = []
    for clean_path in template_files:
        colored_path = clean_path.replace('_clean.png', '_colored.png')
        if not os.path.exists(colored_path): continue
        
        name = os.path.basename(clean_path).replace('_clean.png', '').lower()
        # ★★★ has_date_header 설정을 제거하여 로직 단순화 ★★★
        config = {
            'name': name, 'clean_path': clean_path, 'colored_path': colored_path,
            'date_style_options': ['per_item'],
        }
        
        if 'kakao' in name or 'toss' in name:
            config['date_style_options'] = ['per_item', 'toss_like']
            
        template_configs.append(config)
        
    print(f"{len(template_configs)}개의 템플릿 설정을 로드했습니다.")
    for cfg in template_configs:
        # ★★★ 출력문에서 불필요한 정보 제거 ★★★
        print(f" - {cfg['name']}: date_styles={cfg['date_style_options']}")

    if merchants and template_configs and fonts:
        generate_synthetic_images(template_configs, merchants, fonts, args.num_images, args.output_dir)
    else:
        print("데이터 또는 템플릿, 폰트가 부족하여 이미지 생성을 시작할 수 없습니다.")