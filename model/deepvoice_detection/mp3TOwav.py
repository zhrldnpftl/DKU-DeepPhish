from pydub import AudioSegment
import os

# mp3 파일을 wav 파일로 변환하는 함수
def convert_mp3_to_wav(mp3_path, wav_path, target_rate=16000):
    # mp3 파일을 로드
    audio = AudioSegment.from_mp3(mp3_path)
    
    # 샘플레이트를 16kHz로 변경하고 mono 채널로 설정
    audio = audio.set_frame_rate(target_rate).set_channels(1)
    
    # wav 형식으로 저장
    audio.export(wav_path, format="wav")
