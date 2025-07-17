import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pickle
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel

# MeCab 토크나이즈된 pickle 데이터셋
class TokenDataset_Mecab(Dataset):
    def __init__(self, pickle_path):
        with open(pickle_path, 'rb') as f:
            self.data = pickle.load(f)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        tokens, label = self.data[idx]
        return tokens, label

# 모델 정의
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

# 학습/평가 함수
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

    # 데이터셋 로드
    train_data = TokenDataset_Mecab('./processed_data/train_mecab_tokenized.pkl')
    test_data = TokenDataset_Mecab('./processed_data/test_mecab_tokenized.pkl')

    train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

    # 모델/손실/옵티마이저
    bert = ImprovedBertModel('kykim/bert-kor-base').to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(bert.parameters(), lr=1e-5)

    # 저장 경로 및 학습 기록
    model_name = 'bert-kor-mecab_all'
    time_date = '2505071730'
    save_path = f'./saved_model/{model_name}_{time_date}'
    os.makedirs(save_path, exist_ok=True)

    num_epochs = 10
    min_loss = np.inf
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    # 학습 루프
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
            torch.save(bert.state_dict(), os.path.join(save_path, f'{model_name}_{time_date}.pth'))

        print(f'epoch {epoch+1:02d}, loss: {train_loss:.5f}, acc: {train_acc:.5f}, val_loss: {val_loss:.5f}, val_accuracy: {val_acc:.5f}')

    # 학습 기록 저장
    pickle.dump(history, open(os.path.join(save_path, f'{model_name}{time_date}_history.pkl'), 'wb'))