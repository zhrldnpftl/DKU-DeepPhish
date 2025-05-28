import torchaudio
import torchaudio.transforms as T

# wav 파일을 Mel-spectrogram으로 변환하는 함수
def wav_to_melspectrogram(wav_path):
    # wav 파일 로딩
    waveform, sample_rate = torchaudio.load(wav_path)

    # Mel-spectrogram 변환기 설정 (128 Mel 벡터)
    mel_spec_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=128,
        n_fft=1024,
        hop_length=512
    )

    # Mel-spectrogram 생성
    mel_spec = mel_spec_transform(waveform)

    # 첫 번째 채널만 사용하고 반환 (shape: [128, 시간])
    return mel_spec.squeeze(0)
