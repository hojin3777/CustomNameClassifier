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
MODEL_NAME = 'bert-kor-cnn3_3' # 랜덤 띄어쓰기 삽입
TIME_DATE = '250625_0255'
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

def insert_random_space(text, prob=0.15):
    if len(text) < 2:
        return text
    chars = list(text)
    new_chars = [chars[0]]
    for c in chars[1:]:
        if random.random() < prob:
            new_chars.append(' ')
        new_chars.append(c)
    return ''.join(new_chars)

# --- 데이터셋 클래스 (데이터 증강 포함) ---
class TokenDataset(Dataset):
    def __init__(self, dataframe, tokenizer_pretrained, max_len, augment_prob=0.5, is_train=True, min_len_for_space=6):
        self.data = dataframe
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_pretrained)
        self.max_len = max_len
        self.augment_prob = augment_prob
        self.is_train = is_train
        self.min_len_for_space = min_len_for_space  # 예: 5글자 이상만 적용

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = str(self.data.iloc[idx]['store'])
        label = self.data.iloc[idx]['class']
        # 일정 확률로, 지정 길이 이상이면 랜덤 띄어쓰기 증강
        if self.is_train and len(sentence) >= self.min_len_for_space and random.random() < self.augment_prob:
            sentence = insert_random_space(sentence, prob=0.15)
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