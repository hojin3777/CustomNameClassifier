import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
import random
import os
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast, BertModel
from torch.utils.data import Dataset, DataLoader

# --- 전역 설정 ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TOKENIZER_PRETRAINED = 'kykim/bert-kor-base'
MODEL_NAME = 'bert-kor-cnn3_2' # 짧은 상호명 증강만 켜기
TIME_DATE = '250621_0240'
MAX_LEN = 60
BATCH_SIZE = 128
LEARNING_RATE = 1e-5
N_EPOCHS = 10
REPEAT_COND = 4 # 짧은 상호명 조건 추가

# --- 데이터 로딩 및 분할 함수 ---
def load_processed_data(file_path):
    df = pd.read_csv(file_path)
    result_df = df[['상호명_Regulated', '클래스']].copy()
    result_df.columns = ['store', 'class']
    return result_df

def split_dataset(df, test_size=0.2, random_state=42):
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df['class']
    )
    return train_df, test_df

# --- 데이터셋 클래스 (데이터 증강 포함) ---
class TokenDataset(Dataset):
    # 증강 확률 0.2로 설정 - OCR 오타 증강만 사용
    def __init__(self, dataframe, tokenizer_pretrained, max_len, augment_prob=0.2, is_train=True):
        self.data = dataframe
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_pretrained)
        self.max_len = max_len
        self.augment_prob = augment_prob
        self.is_train = is_train

        # 한글 자모 리스트
        self.CHOSUNG_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
        self.JUNGSUNG_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
        self.JONGSUNG_LIST = [None, 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

        # OCR 오타 맵 (현실성 낮은 'ㅡ','ㅣ' 매핑 제거)
        self.ocr_typo_map = {
            'ㅏ': 'ㅑ', 'ㅑ': 'ㅏ', 'ㅓ': 'ㅕ', 'ㅕ': 'ㅓ', 'ㅗ': 'ㅛ', 'ㅛ': 'ㅗ', 'ㅜ': 'ㅠ', 'ㅠ': 'ㅜ',
            'ㅐ': 'ㅔ', 'ㅔ': 'ㅐ', 'ㄱ': 'ㅋ', 'ㅋ': 'ㄱ', 'ㄷ': 'ㅌ', 'ㅌ': 'ㄷ',
            'ㅂ': 'ㅍ', 'ㅍ': 'ㅂ', 'ㅅ': 'ㅆ', 'ㅆ': 'ㅅ', 'ㅈ': 'ㅊ', 'ㅊ': 'ㅈ', 'ㅇ': 'ㅎ', 'ㅎ': 'ㅇ'
        }

    def __len__(self):
        return len(self.data)

    def _decompose_hangul(self, char):
        """한글 음절을 초성, 중성, 종성으로 분해"""
        if '가' <= char <= '힣':
            char_code = ord(char) - ord('가')
            chosung_idx = char_code // (21 * 28)
            jungsung_idx = (char_code % (21 * 28)) // 28
            jongsung_idx = char_code % 28
            return self.CHOSUNG_LIST[chosung_idx], self.JUNGSUNG_LIST[jungsung_idx], self.JONGSUNG_LIST[jongsung_idx]
        return char, None, None

    def _compose_hangul(self, chosung, jungsung, jongsung):
        """초성, 중성, 종성을 한글 음절로 조립"""
        try:
            chosung_idx = self.CHOSUNG_LIST.index(chosung)
            jungsung_idx = self.JUNGSUNG_LIST.index(jungsung)
            jongsung_idx = self.JONGSUNG_LIST.index(jongsung) if jongsung else 0
            return chr(ord('가') + chosung_idx * 21 * 28 + jungsung_idx * 28 + jongsung_idx)
        except (ValueError, IndexError):
            if chosung and not jungsung and not jongsung:
                 return chosung
            return ''

    def _introduce_ocr_errors(self, sentence, error_rate=0.05):
        """문장에 OCR과 유사한 오탈자를 삽입합니다."""
        result_chars = []
        for char in sentence:
            cho, jung, jong = self._decompose_hangul(char)
            if jung: # 한글인 경우
                if cho in self.ocr_typo_map and random.random() < error_rate:
                    cho = self.ocr_typo_map[cho]
                if jung in self.ocr_typo_map and random.random() < error_rate:
                    jung = self.ocr_typo_map[jung]
                if jong and jong in self.ocr_typo_map and random.random() < error_rate:
                    jong = self.ocr_typo_map[jong]
                result_chars.append(self._compose_hangul(cho, jung, jong))
            else: # 한글이 아닌 경우
                result_chars.append(char)
        return "".join(result_chars)

    def __getitem__(self, idx):
        sentence = str(self.data.iloc[idx]['store'])
        label = self.data.iloc[idx]['class']

        # --- 데이터 증강: OCR 오타 적용 ---
        if self.is_train and random.random() < self.augment_prob:
            sentence = self._introduce_ocr_errors(sentence)

        tokens = self.tokenizer(
            sentence, return_tensors='pt', truncation=True, padding='max_length',
            max_length=self.max_len, add_special_tokens=True
        )
        input_ids = tokens['input_ids'].squeeze(0)
        attention_mask = tokens['attention_mask'].squeeze(0)
        token_type_ids = torch.zeros_like(attention_mask)
        return {
            'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids,
        }, torch.tensor(label)

# --- 모델 클래스 ---
class BertCNNModel(nn.Module):
    def __init__(self, bert_pretrained, num_classes, dropout_rate=0.5, kernel_sizes=[2,3,4], num_filters=128):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_pretrained)
        self.convs = nn.ModuleList([nn.Conv1d(768, num_filters, k) for k in kernel_sizes])
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        x = outputs['last_hidden_state'].transpose(1, 2)
        conv_outs = [F.relu(conv(x)) for conv in self.convs]
        pooled = [F.max_pool1d(c, c.size(2)).squeeze(2) for c in conv_outs]
        cat = torch.cat(pooled, dim=1)
        cat = self.dropout(cat)
        logits = self.fc(cat)
        return logits

# --- 학습 및 평가 함수 (train_bert_all_cnn2.py 스타일 적용) ---
def model_train(model, data_loader, loss_fn, optimizer, device):
    model.train()
    running_loss = 0
    corr = 0
    counts = 0
    prograss_bar = tqdm(data_loader, unit='batch', total=len(data_loader), mininterval=1, desc="Training")
    for inputs, labels in prograss_bar:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)
        optimizer.zero_grad()
        output = model(**inputs)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()
        _, pred = output.max(dim=1)
        corr += pred.eq(labels).sum().item()
        counts += len(labels)
        running_loss += loss.item() * labels.size(0)
        prograss_bar.set_description(f"Loss: {running_loss/counts:.4f}, Acc: {corr/counts:.4f}")
    acc = corr / counts
    epoch_loss = running_loss / counts
    return epoch_loss, acc

def model_evaluate(model, data_loader, loss_fn, device):
    model.eval()
    running_loss = 0
    corr = 0
    counts = 0
    with torch.no_grad():
        prograss_bar = tqdm(data_loader, unit='batch', total=len(data_loader), mininterval=1, desc="Evaluating")
        for inputs, labels in prograss_bar:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            output = model(**inputs)
            _, pred = output.max(dim=1)
            corr += pred.eq(labels).sum().item()
            counts += len(labels)
            running_loss += loss_fn(output, labels).item() * labels.size(0)
            prograss_bar.set_description(f"Val_Loss: {running_loss/counts:.4f}, Val_Acc: {corr/counts:.4f}")
    acc = corr / counts
    epoch_loss = running_loss / counts
    return epoch_loss, acc

# --- 메인 실행 블록 ---
if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    save_path = f'./saved_model/{MODEL_NAME}_{TIME_DATE}'
    os.makedirs(save_path, exist_ok=True)
    print(f"Directory created or exists: {save_path}")

    # 1. 데이터 로드 및 분할
    data_df = load_processed_data('./processed_data/seoul_gyeonggi_combined_data.csv')
    train_df, test_df = split_dataset(data_df)

    # 2. 클래스 수 확인 (데이터의 최대 클래스 값 + 1)
    num_classes = data_df['class'].max() + 1
    print(f"Data-driven number of classes: {num_classes}")

    # 3. 데이터셋 및 데이터로더 생성
    train_data = TokenDataset(train_df, TOKENIZER_PRETRAINED, MAX_LEN, is_train=True)
    test_data = TokenDataset(test_df, TOKENIZER_PRETRAINED, MAX_LEN, is_train=False)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 4. 모델, 손실 함수, 옵티마이저 초기화
    model = BertCNNModel(TOKENIZER_PRETRAINED, num_classes=num_classes).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. 학습 루프 실행
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print("\n--- Training Start ---")
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        train_loss, train_acc = model_train(model, train_loader, loss_fn, optimizer, DEVICE)
        val_loss, val_acc = model_evaluate(model, test_loader, loss_fn, DEVICE)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
        
        print(f'Epoch: {epoch+1:02} | Time: {int(epoch_mins)}m {int(epoch_secs)}s')
        print(f'\tTrain Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {val_loss:.4f} |  Val. Acc: {val_acc*100:.2f}%')
        
        if val_acc > best_acc:
            print(f'[INFO] val_accuracy has been improved from {best_acc*100:.2f}% to {val_acc*100:.2f}%. Saving Model!')
            best_acc = val_acc
            torch.save(model.state_dict(), f'{save_path}/{MODEL_NAME}_{TIME_DATE}.pth')

    # 6. 학습 기록 저장
    with open(f'{save_path}/{MODEL_NAME}_{TIME_DATE}_history.pkl', 'wb') as f:
        pickle.dump(history, f)
        
    print("--- Training Complete ---")
    print(f"Best Validation Accuracy: {best_acc*100:.2f}%")