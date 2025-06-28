import speech_recognition as sr
import pyttsx3
from fastapi import FastAPI, UploadFile, File
from pydub import AudioSegment
import io
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()

# Инициализация модели и TTS
model = AutoModelForCausalLM.from_pretrained("ваша_модель")
tokenizer = AutoTokenizer.from_pretrained("ваша_модель")
tts_engine = pyttsx3.init()

# Настройки голоса (можно кастомизировать)
tts_engine.setProperty('rate', 150)
tts_engine.setProperty('volume', 0.9)

def text_to_speech(text):
    """Синтез речи из текста"""
    tts_engine.say(text)
    tts_engine.runAndWait()

def speech_to_text(audio_data):
    """Распознавание речи из аудио"""
    r = sr.Recognizer()
    with sr.AudioFile(audio_data) as source:
        audio = r.record(source)
    try:
        return r.recognize_google(audio, language="ru-RU")
    except sr.UnknownValueError:
        return "Не удалось распознать речь"
    except sr.RequestError:
        return "Ошибка сервиса распознавания"

@app.post("/voice_query")
async def handle_voice_query(file: UploadFile = File(...)):
    # Конвертация в WAV
    audio = AudioSegment.from_file(io.BytesIO(await file.read()))
    audio.export("temp.wav", format="wav")
    
    # Распознавание
    user_text = speech_to_text("temp.wav")
    if "не удалось" in user_text.lower():
        return {"error": user_text}
    
    # Генерация ответа
    inputs = tokenizer(user_text, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=200)
    bot_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Озвучка ответа
    text_to_speech(bot_text)
    
    return {
        "user_text": user_text,
        "bot_response": bot_text,
        "audio_response": "Озвучено"
    }

@app.get("/interactive_voice")
async def interactive_voice_mode():
    """Интерактивный голосовой режим через консоль"""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Говорите...")
        audio = r.listen(source)
        
    try:
        user_text = r.recognize_google(audio, language="ru-RU")
        print(f"Вы сказали: {user_text}")
        
        # Генерация ответа
        inputs = tokenizer(user_text, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=200)
        bot_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Озвучка
        text_to_speech(bot_text)
        return {"response": bot_text}
    
    except Exception as e:
        return {"error": str(e)}
