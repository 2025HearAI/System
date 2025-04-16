import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from features import extract_features
from model import CNNEmotionClassifier

class EmotionDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        feature = extract_features(self.file_paths[idx])  # (40, 100)
        feature = torch.tensor(feature, dtype=torch.float32).unsqueeze(0)  # (1, 40, 100)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feature, label

def train():
    # 예시용 오디오 파일 경로와 레이블
    file_paths = ["data/happy.wav", "data/sad.wav", "data/angry.wav"]
    labels = [0, 1, 2]  # 감정 클래스 인덱스

    dataset = EmotionDataset(file_paths, labels)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = CNNEmotionClassifier(num_classes=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
        total_loss = 0
        for x, y in dataloader:
            outputs = model(x)
            loss = criterion(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}")

    # 학습된 모델 저장
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/emotion_cnn.pth")

if __name__ == "__main__":
    train()
