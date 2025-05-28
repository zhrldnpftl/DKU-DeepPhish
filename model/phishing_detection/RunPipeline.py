<<<<<<< HEAD
import torch
import json
import os

# 변환/추론 관련 함수들 import
from mp3TOwav import convert_mp3_to_wav
from make_melspectrogram import wav_to_melspectrogram
from ModelLoader import load_model

# 판별 기준이 되는 threshold
THRESHOLD = 0.8

# 전체 파이프라인을 실행하는 함수 (mp3 → label + 확률 반환)
def run_inference(mp3_path):
    # mp3 파일을 같은 이름의 .wav로 저장
    wav_path = mp3_path.replace(".mp3", ".wav")
    convert_mp3_to_wav(mp3_path, wav_path)

    # wav 파일을 Mel-spectrogram으로 변환
    mel_spec = wav_to_melspectrogram(wav_path).unsqueeze(0)  # [1, 128, T]

    # 모델 로드
    model_path = os.path.join(os.path.dirname(__file__), "modelWithThreshold.pt")
    model = load_model(model_path)

    # 모델 추론
    with torch.no_grad():
        prob = model(mel_spec).item()  # 확률 값 추출

    # 라벨 지정 및 JSON 포맷으로 반환
    result = {
        "label": "fake" if prob >= THRESHOLD else "real",
        "probability": round(prob, 4)
    }

    return json.dumps(result)
=======
import torch
import json
import os

# 변환/추론 관련 함수들 import
from deepvoice_detection.mp3_to_wav_converter import convert_mp3_to_wav
from deepvoice_detection.mel_spectrogram_converter import wav_to_melspectrogram
from deepvoice_detection.model_loader import load_model

# 판별 기준이 되는 threshold
THRESHOLD = 0.8

# 전체 파이프라인을 실행하는 함수 (mp3 → label + 확률 반환)
def run_inference(mp3_path):
    # mp3 파일을 같은 이름의 .wav로 저장
    wav_path = mp3_path.replace(".mp3", ".wav")
    convert_mp3_to_wav(mp3_path, wav_path)

    # wav 파일을 Mel-spectrogram으로 변환
    mel_spec = wav_to_melspectrogram(wav_path).unsqueeze(0)  # [1, 128, T]

    # 모델 로드
    model_path = os.path.join(os.path.dirname(__file__), "modelWithThreshold.pt")
    model = load_model(model_path)

    # 모델 추론
    with torch.no_grad():
        prob = model(mel_spec).item()  # 확률 값 추출

    # 라벨 지정 및 JSON 포맷으로 반환
    result = {
        "label": "fake" if prob >= THRESHOLD else "real",
        "probability": round(prob, 4)
    }

    return json.dumps(result)
>>>>>>> 01ceff8d88bddf0ca764c4b65581df6234d25b8d
