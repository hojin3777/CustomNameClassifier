import cv2
import numpy as np

def convert_coord(coord_list):
    x_min, x_max, y_min, y_max = coord_list
    return [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]

# https://www.life2coding.com/cropping-polygon-or-non-rectangular-region-from-image-using-opencv-python/
# https://stackoverflow.com/questions/48301186/cropping-concave-polygon-from-image-using-opencv-python
def crop(image, points):
    # ★★★ 변경점: 입력된 points가 비어있을 경우에 대한 예외 처리 추가 ★★★
    # 비어있는 배열에 .min() 연산을 시도하여 발생하는 오류를 원천적으로 방지합니다.
    if len(points) == 0:
        # 빈 배열이 들어오면, 원본 이미지의 아주 작은 (1x1) 영역을 반환하여 오류를 막습니다.
        return np.zeros((1, 1, image.shape[2]), dtype=image.dtype)
    
    # ★★★ 변경점 1: pts를 (N, 1, 2) 형태의 3차원 배열로 생성 ★★★
    # drawContours가 요구하는 정확한 차원으로 맞춰줍니다.
    pts = np.array(points, np.int32).reshape((-1, 1, 2))

    # Crop the bounding rect
    rect = cv2.boundingRect(pts)
    x,y,w,h = rect
    # ★★★ 변경점: 음수 좌표가 슬라이싱 오류를 일으키는 것을 방지 ★★★
    # boundingRect가 음수 좌표를 반환하는 경우, 이를 0으로 클리핑합니다.
    x, y, w, h = max(0, x), max(0, y), w, h
    
    croped = image[y:y+h, x:x+w].copy()
    
    # make mask
    pts = pts - pts.min(axis=0)
    """
    pts = np.array(points, np.int32)

    # Crop the bounding rect
    rect = cv2.boundingRect(pts)
    x,y,w,h = rect
    croped = image[y:y+h, x:x+w].copy()

    # make mask
    pts = pts - pts.min(axis=0)
    """

    mask = np.zeros(croped.shape[:2], np.uint8)
    # cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
    # ★★★ 변경점: 3채널 색상 (255, 255, 255)를 1채널 스칼라 값 255로 변경 ★★★
    cv2.drawContours(mask, [pts], -1, 255, -1, cv2.LINE_AA) # <<<

    # do bit-op
    dst = cv2.bitwise_and(croped, croped, mask=mask)

    # add the white background
    bg = np.ones_like(croped, np.uint8) * 255
    cv2.bitwise_not(bg,bg, mask=mask)
    result = bg + dst

    return result
