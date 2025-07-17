import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast, BertModel
from torch.utils.data import Dataset, DataLoader

# 전역에서 토크나이저 준비
tokenizer_pretrained = 'kykim/bert-kor-base'
bert_tokenizer = BertTokenizerFast.from_pretrained(tokenizer_pretrained)

#모델 이름 및 날짜 세팅
model_name = 'bert-kor_all' #weighted loss/오버샘플링 제거, dropout 증가
time_date = '2505100105'

class TokenDataset(Dataset):
    def __init__(self, dataframe, tokenizer_pretrained):
        self.data = dataframe
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_pretrained)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        sentence = self.data.iloc[idx]['store']
        label = self.data.iloc[idx]['class']
        tokens = self.tokenizer(
            str(sentence),
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=60,
            add_special_tokens=True
        )
        input_ids = tokens['input_ids'].squeeze(0)
        attention_mask = tokens['attention_mask'].squeeze(0)
        token_type_ids = torch.zeros_like(attention_mask)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
        }, torch.tensor(label)

class ImprovedBertModel(nn.Module):
    def __init__(self, bert_pretrained, num_classes=247, dropout_rate=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_pretrained)
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs['last_hidden_state'][:, 0, :]
        logits = self.classifier(pooled_output)
        return logits

def model_train(model, data_loader, loss_fn, optimizer, device):
    model.train()
    running_loss = 0
    corr = 0
    counts = 0
    prograss_bar = tqdm(data_loader, unit='batch', total=len(data_loader), mininterval=1)
    for idx, (inputs, labels) in enumerate(prograss_bar):
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
        prograss_bar.set_description(f"training loss: {running_loss/(counts):.5f}, training accuracy: {corr / counts:.5f}")
    acc = corr / len(data_loader.dataset)
    return running_loss / len(data_loader.dataset), acc

def model_evaluate(model, data_loader, loss_fn, device):
    model.eval()
    with torch.no_grad():
        corr = 0
        running_loss = 0
        for inputs, labels in data_loader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            output = model(**inputs)
            _, pred = output.max(dim=1)
            corr += torch.sum(pred.eq(labels)).item()
            running_loss += loss_fn(output, labels).item() * labels.size(0)
        acc = corr / len(data_loader.dataset)
        return running_loss / len(data_loader.dataset), acc

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    # 데이터 로드
    def load_processed_data(file_path):
        df = pd.read_csv(file_path)
        result_df = df[['상호명_Regulated', '클래스']].copy()
        result_df.columns = ['store', 'class']
        return result_df

    # 전체 데이터셋 경로로 변경
    data_df = load_processed_data('./processed_data/region_all_processed_data.csv')
    train_df, test_df = train_test_split(data_df, test_size=0.2, random_state=42, stratify=data_df['class'])

    train_data = TokenDataset(train_df, tokenizer_pretrained)
    test_data = TokenDataset(test_df, tokenizer_pretrained)

    # batch_size는 GPU 메모리에 맞게 조절 (16~128 추천, 64~128이 가장 무난)
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

    bert = ImprovedBertModel(tokenizer_pretrained).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(bert.parameters(), lr=1e-5)

    # 저장 경로 및 학습 기록
    save_path = f'./saved_model/{model_name}_{time_date}'
    os.makedirs(save_path, exist_ok=True)

    num_epochs = 10
    min_loss = np.inf
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        train_loss, train_acc = model_train(bert, train_loader, loss_fn, optimizer, device)
        val_loss, val_acc = model_evaluate(bert, test_loader, loss_fn, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        if val_loss < min_loss:
            print(f'[INFO] val_loss has been improved from {min_loss:.5f} to {val_loss:.5f}. Saving Model!')
            min_loss = val_loss
            torch.save(bert.state_dict(), f'./saved_model/{model_name}_{time_date}/{model_name}_{time_date}.pth')
        print(f'epoch {epoch+1:02d}, loss: {train_loss:.5f}, acc: {train_acc:.5f}, val_loss: {val_loss:.5f}, val_accuracy: {val_acc:.5f}')
    
    pickle.dump(history, open(os.path.join(save_path, f'{model_name}_{time_date}_history.pkl'), 'wb'))