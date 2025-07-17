import torch
from torch.utils.data import DataLoader, random_split
from transformers import LayoutLMv3Processor
from torch.optim import AdamW  # AdamW를 torch.optim에서 직접 가져옵니다.
from tqdm import tqdm
import os

# 이전에 만든 모듈들을 가져옵니다.
from config import (
    PRETRAINED_MODEL_NAME, IMAGE_DIR, LABEL_FILE, DEVICE,
    NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, TRAIN_RATIO, OUTPUT_DIR
)
from dataset import BankStatementDataset
from model import get_model

def main():
    """
    모델 학습의 전체 과정을 실행하는 메인 함수
    """
    print(f"사용 장치: {DEVICE}")
    
    # 1. 프로세서와 모델 로딩
    print("프로세서와 모델을 로딩합니다...")
    try:
        processor = LayoutLMv3Processor.from_pretrained(PRETRAINED_MODEL_NAME, apply_ocr=False)
        model = get_model()
        model.to(DEVICE)
    except Exception as e:
        print(f"모델 또는 프로세서 로딩 중 오류 발생: {e}")
        print("인터넷 연결 및 Hugging Face 모델 이름을 확인해주세요.")
        return

    # 2. 데이터셋 준비
    print("데이터셋을 준비합니다...")
    try:
        full_dataset = BankStatementDataset(
            image_dir=IMAGE_DIR,
            annotations_file=LABEL_FILE,
            processor=processor
        )
    except FileNotFoundError:
        print(f"오류: '{LABEL_FILE}' 또는 이미지 파일을 찾을 수 없습니다.")
        print("데이터 생성 스크립트(generate.py)를 먼저 실행하여 데이터셋을 생성해주세요.")
        return
    
    # 데이터셋을 학습용과 검증용으로 분리
    train_size = int(TRAIN_RATIO * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"전체 데이터: {len(full_dataset)}개")
    print(f"학습 데이터: {len(train_dataset)}개, 검증 데이터: {len(val_dataset)}개")

    # 데이터 로더 생성
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # 3. 옵티마이저 설정
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # 4. 학습 루프 시작
    print("\n--- 모델 학습을 시작합니다 ---")
    for epoch in range(NUM_EPOCHS):
        # --- 학습 단계 ---
        model.train()
        train_loss = 0
        
        # tqdm을 사용하여 진행 상황을 시각적으로 표시
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [학습]")
        for batch in progress_bar:
            # 데이터를 DEVICE(GPU 또는 CPU)로 이동
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            
            # 순전파 및 손실 계산
            outputs = model(**batch)
            loss = outputs.loss
            
            # 역전파
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch + 1} - 평균 학습 손실: {avg_train_loss:.4f}")

        # --- 검증 단계 ---
        model.eval()
        val_loss = 0
        
        with torch.no_grad(): # 기울기 계산 비활성화
            progress_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [검증]")
            for batch in progress_bar:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                val_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1} - 평균 검증 손실: {avg_val_loss:.4f}")
        print("-" * 50)

    # 5. 학습 완료 후 모델 저장
    print("학습이 완료되었습니다. 모델을 저장합니다...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 모델의 가중치와 프로세서를 함께 저장해야 나중에 불러와서 사용할 수 있습니다.
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    
    print(f"모델이 '{OUTPUT_DIR}' 경로에 성공적으로 저장되었습니다.")


if __name__ == '__main__':
    main()