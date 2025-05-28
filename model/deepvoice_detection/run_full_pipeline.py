# run_full_pipeline_deepvoice.py

import os
from pathlib import Path
import json
from mp3TOwav import convert_mp3_to_wav
from make_melspectrogram import wav_to_melspectrogram
from ModelLoader import load_model
import torch

# ✅ 사용자 입력: 분석할 MP3 파일 경로
input_mp3_path = Path("D:/2025_work/2025_VoicePhshing_Detection_Model/dataset/phishing_dataset/sample/딥보이스_예시.mp3")

# ✅ 1단계: MP3 → WAV 변환
print("\n🔊 [1단계] MP3 → WAV 변환 시작")
input_wav_path = input_mp3_path.with_suffix(".wav")
convert_mp3_to_wav(str(input_mp3_path), str(input_wav_path))
print(f"✔️ 변환 완료: {input_wav_path}")

# ✅ 2단계: Mel-spectrogram 변환
print("\n🎧 [2단계] Mel-spectrogram 변환")
mel_spec = wav_to_melspectrogram(str(input_wav_path)).unsqueeze(0)  # [1, 80, 300]
print(f"✔️ Mel-spectrogram shape: {mel_spec.shape}")

# ✅ 3단계: 딥보이스 탐지 모델 로드 및 추론
print("\n🧠 [3단계] 딥보이스 탐지 모델 추론")
THRESHOLD = 0.8
model_path = os.path.join(os.path.dirname(__file__), "modelWithThreshold.pt")
model = load_model(model_path)

with torch.no_grad():
    prob = model(mel_spec).item()

label = "fake" if prob >= THRESHOLD else "real"
print(f"\n✅ [최종 결과] 딥보이스 판별: {label.upper()} (확률: {prob:.4f})")

# ✅ 결과 JSON 저장 (선택사항)
result = {
    "label": label,
    "probability": round(prob, 4),
    "input_file": input_mp3_path.name
}
json_path = input_mp3_path.with_suffix(".json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print(f"📄 결과 JSON 저장 완료: {json_path}")
