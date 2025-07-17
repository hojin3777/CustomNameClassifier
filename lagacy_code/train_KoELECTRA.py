import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
# --- 수정: Electra 모델 및 토크나이저 import ---
from transformers import ElectraTokenizerFast, ElectraModel
from torch.utils.data import Dataset, DataLoader

# --- 수정: 사전학습 모델 이름 변경 ---
tokenizer_pretrained = 'monologg/koelectra-base-v3-discriminator'

#모델 이름 및 날짜 세팅
model_name = 'koelectra-cls_all'
time_date = '250623_0200' # 시간은 적절히 수정

class TokenDataset(Dataset):
    def __init__(self, dataframe, tokenizer_pretrained):
        self.data = dataframe
        # --- 수정: 토크나이저 클래스 변경 ---
        self.tokenizer = ElectraTokenizerFast.from_pretrained(tokenizer_pretrained)
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
        token_type_ids = tokens['token_type_ids'].squeeze(0) # Electra는 token_type_ids를 사용
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
        }, torch.tensor(label)

class ElectraCLSModel(nn.Module):
    def __init__(self, electra_pretrained, num_classes, dropout_rate=0.5):
        super().__init__()
        # --- 수정: 모델 클래스 변경 ---
        self.electra = ElectraModel.from_pretrained(electra_pretrained)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(self.electra.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.electra(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids
        )
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.fc(cls_output)
        return logits

def model_train(model, data_loader, loss_fn, optimizer, device):
    model.train()
    running_loss = 0
    corr = 0
    
    prograss_bar = tqdm(data_loader, unit='batch', total=len(data_loader), mininterval=1)
    
    for data, label in prograss_bar:
        # KoELECTRA의 TokenDataset은 딕셔너리를 반환하므로, 각 항목을 device로 보냅니다.
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        token_type_ids = data['token_type_ids'].to(device)
        label = label.to(device)
        
        optimizer.zero_grad()
        
        output = model(input_ids, attention_mask, token_type_ids)
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()
        
        _, pred = torch.max(output, dim=1)
        corr += (pred == label).sum().item()
        running_loss += loss.item() * input_ids.size(0)
        
    acc = corr / len(data_loader.dataset)
    return running_loss / len(data_loader.dataset), acc

def model_evaluate(model, data_loader, loss_fn, device):
    model.eval()
    with torch.no_grad():
        corr = 0
        running_loss = 0
        
        for data, label in data_loader:
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            token_type_ids = data['token_type_ids'].to(device)
            label = label.to(device)
            
            output = model(input_ids, attention_mask, token_type_ids)
            _, pred = torch.max(output, dim=1)
            
            corr += (pred == label).sum().item()
            running_loss += loss_fn(output, label).item() * input_ids.size(0)
            
        acc = corr / len(data_loader.dataset)
        return running_loss / len(data_loader.dataset), acc

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    def load_processed_data(file_path):
        df = pd.read_csv(file_path)
        result_df = df[['상호명_Regulated', '클래스']].copy()
        result_df.columns = ['store', 'class']
        return result_df

    data_df = load_processed_data('./processed_data/region_all_processed_data.csv')
    train_df, test_df = train_test_split(data_df, test_size=0.2, random_state=42, stratify=data_df['class'])
    
    num_classes = data_df['class'].nunique()

    train_data = TokenDataset(train_df, tokenizer_pretrained)
    test_data = TokenDataset(test_df, tokenizer_pretrained)

    train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

    # --- 수정: Electra 모델 생성 ---
    model = ElectraCLSModel(tokenizer_pretrained, num_classes=num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    save_path = f'./saved_model/{model_name}_{time_date}'
    os.makedirs(save_path, exist_ok=True)

    num_epochs = 10
    min_loss = np.inf
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        # model_train, model_evaluate 호출 부분은 동일
        train_loss, train_acc = model_train(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_acc = model_evaluate(model, test_loader, loss_fn, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        if val_loss < min_loss:
            print(f'[INFO] val_loss has been improved from {min_loss:.5f} to {val_loss:.5f}. Saving Model!')
            min_loss = val_loss
            torch.save(model.state_dict(), f'./saved_model/{model_name}_{time_date}/{model_name}_{time_date}.pth')
        print(f'epoch {epoch+1:02d}, loss: {train_loss:.5f}, acc: {train_acc:.5f}, val_loss: {val_loss:.5f}, val_accuracy: {val_acc:.5f}')
    
    pickle.dump(history, open(os.path.join(save_path, f'{model_name}_{time_date}_history.pkl'), 'wb'))