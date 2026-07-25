import pyaudio
import pyaudio
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
import numpy as np
import wave 
from io import BytesIO

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 16000

class VoiceActivityDectector:
    def __init__(self):
        self.format = FORMAT
        self.rate = RATE
        self.chunk = CHUNK
        self.channels = 1
        self.model = load_silero_vad()
        self.pyaudio_instance = pyaudio.PyAudio()
        self.stream = self.pyaudio_instance.open(format=self.format,
        input=True,
        rate=self.rate,
        frames_per_buffer = self.chunk,
        channels = self.channels
        )
        self.buffer = BytesIO()
        print("Microphone Access successful")

    def read_chunk(self) -> bool:
        self.raw = self.stream.read(self.chunk)
        self.buffer.seek(0)
        self.buffer.truncate(0)

        with wave.open(self.buffer, "wb") as wave_file:
            wave_file.setframerate(self.rate)
            wave_file.setnchannels(self.channels)
            wave_file.setsampwidth(2)
            wave_file.writeframes(self.raw)
        self.buffer.seek(0)
        wav = read_audio(self.buffer)
        timestamp = get_speech_timestamps(wav, self.model, return_seconds=True)

        return bool(timestamp)
    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pyaudio_instance.terminate()

