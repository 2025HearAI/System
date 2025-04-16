import torch
from model import CNNEmotionClassifier
from features import extract_features

EMOTION_LABELS = ["Happy", "Sad", "Angry"]  # 필요에 따라 확장 가능

def predict_emotion(file_path, model_path="models/emotion_cnn.pth"):
    # 모델 불러오기
    model = CNNEmotionClassifier(num_classes=len(EMOTION_LABELS))
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 특징 추출
    feature = extract_features(file_path)  # (40, 100)
    feature_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 40, 100)

    with torch.no_grad():
        output = model(feature_tensor)
        predicted_idx = torch.argmax(output, dim=1).item()
        return EMOTION_LABELS[predicted_idx]

# 예측 테스트
if __name__ == "__main__":
    test_file = "data/happy_test.wav"
    result = predict_emotion(test_file)
    print(f"🔊 감정 예측 결과: {result}")
