from pydub import AudioSegment
import io

def convert_audio_format(audio_bytes, from_format, to_format="wav"):
    """Конвертация между аудио форматами"""
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=from_format)
    buffer = io.BytesIO()
    audio.export(buffer, format=to_format)
    return buffer.getvalue()

def adjust_audio_speed(audio_bytes, speed=1.0):
    """Изменение скорости аудио"""
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    adjusted = audio.speedup(playback_speed=speed)
    buffer = io.BytesIO()
    adjusted.export(buffer, format="wav")
    return buffer.getvalue()
