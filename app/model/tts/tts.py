from kokoro import KPipeline
import soundfile as sf
import numpy as np
import io

pipeline = KPipeline(repo_id='hexgrad/Kokoro-82M',lang_code='a')

def generate_audio(text: str, voice: str = 'af_heart') -> bytes:
    audio_chunks = []

    generator = pipeline(text, voice=voice)
    for _, _, audio in generator:
        audio_chunks.append(audio)

    combined = np.concatenate(audio_chunks)
    buf = io.BytesIO()
    sf.write(buf, combined, 24000, format='WAV')
    buf.seek(0)
    return buf.read()