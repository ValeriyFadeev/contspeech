import os
import tempfile
import torch
import whisper
import spacy
from pyannote.audio import Pipeline
from pydub import AudioSegment

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

# Глобальные переменные для моделей
asr_model = None
diar_pipeline = None
nlp = None

device = "cuda" if torch.cuda.is_available() else "cpu"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Загрузка моделей при старте приложения.
    """
    global asr_model, diar_pipeline, nlp

    # 1) Whisper – распознавание речи
    asr_model = whisper.load_model("base", device=device)

    # 2) pyannote.audio – диаризация (используем HF_TOKEN, если он задан)
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        diar_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=hf_token)
    else:
        diar_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

    # 3) spaCy – NER для поиска имён (тип "PER")
    nlp = spacy.load("ru_core_news_sm")

    yield
    # Здесь можно добавить освобождение ресурсов при завершении

app = FastAPI(lifespan=lifespan)

def convert_to_wav(input_file_path, sr=16000):
    """
    Конвертирует входной аудиофайл в формат WAV с частотой дискретизации sr.
    """
    audio = AudioSegment.from_file(input_file_path)
    audio = audio.set_channels(1).set_frame_rate(sr)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_name = tmp.name
    audio.export(tmp_name, format="wav")
    return tmp_name

def assign_names_to_speakers(segments, matched_speakers, nlp):
    """
    Сопоставляет имена спикерам через spaCy NER (тип "PER").
    Если имя не найдено – присваивает имя по схеме Speaker_X.
    """
    speaker_to_name = {}
    for seg, spk_label in zip(segments, matched_speakers):
        text = seg["text"]
        # Если для данного спикера уже найдено имя – пропускаем
        if spk_label in speaker_to_name:
            continue

        doc = nlp(text)
        person_names = [ent.text for ent in doc.ents if ent.label_ == "PER"]
        if person_names:
            speaker_to_name[spk_label] = person_names[0]
        else:
            speaker_to_name[spk_label] = None

    unnamed_count = 1
    for spk_label in set(matched_speakers):
        if speaker_to_name.get(spk_label) is None:
            speaker_to_name[spk_label] = f"Speaker_{unnamed_count}"
            unnamed_count += 1

    return speaker_to_name

def process_audio(input_file_path):
    """
    Обрабатывает аудиофайл:
    1. Конвертирует в WAV.
    2. Распознаёт речь с помощью Whisper.
    3. Выполняет диаризацию через pyannote.audio.
    4. Сопоставляет сегменты с идентификаторами спикеров.
    5. Присваивает имена спикерам через spaCy NER.
    Возвращает список строк с финальной транскрипцией.
    """
    wav_path = convert_to_wav(input_file_path, sr=16000)
    try:
        # Шаг 1: Распознавание речи (Whisper)
        stt_result = asr_model.transcribe(wav_path, language='ru')
        segments = stt_result.get("segments", [])  # Каждый сегмент: {start, end, text}

        # Шаг 2: Диаризация (pyannote.audio)
        diarization = diar_pipeline(wav_path)
        diar_results = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            diar_results.append((turn.start, turn.end, speaker))
        diar_results.sort(key=lambda x: x[0])

        # Шаг 3: Сопоставление сегментов с временными интервалами диаризации
        matched_speakers = []
        for seg in segments:
            seg_start = seg["start"]
            spk_label = "UNK"
            for (dst, den, spk) in diar_results:
                if dst <= seg_start < den:
                    spk_label = spk
                    break
            matched_speakers.append(spk_label)

        # Шаг 4: Присвоение имён спикерам
        speaker_name_map = assign_names_to_speakers(segments, matched_speakers, nlp)

        # Шаг 5: Формирование итогового текста
        output_lines = []
        for seg, spk_label in zip(segments, matched_speakers):
            assigned_name = speaker_name_map.get(spk_label, f"Speaker_{spk_label}")
            text = seg["text"]
            output_lines.append(f"{assigned_name}: {text}")

        return output_lines

    finally:
        # Удаляем временный WAV-файл
        os.unlink(wav_path)

@app.websocket("/ws/record")
async def websocket_record(websocket: WebSocket):
    """
    WebSocket-эндпоинт для записи с микрофона.
    Клиент начинает отправлять бинарные аудиоданные сразу после установления соединения.
    При отправке текстового сообщения "stop" запись завершается, производится обработка и
    результат (транскрипция с разделением по спикерам) отправляется обратно.
    """
    await websocket.accept()

    # Создаём временный файл для записи аудиоданных
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".audio")
    tmp_file_path = tmp_file.name
    tmp_file.close()

    try:
        while True:
            data = await websocket.receive()
            if "text" in data:
                # При получении команды "stop" завершаем запись
                if data["text"].lower() == "stop":
                    break
                # Можно обрабатывать и другие текстовые команды, если нужно
            elif "bytes" in data:
                # При получении бинарных данных дописываем их в файл
                with open(tmp_file_path, "ab") as f:
                    f.write(data["bytes"])
    except WebSocketDisconnect:
        # Если соединение оборвалось, можно обработать это по необходимости
        pass

    # Обработка записанного аудио
    transcription_lines = process_audio(tmp_file_path)
    os.unlink(tmp_file_path)

    # Отправляем результат обратно клиенту в формате JSON
    await websocket.send_json({"transcription": transcription_lines})

@app.post("/transcribe")
def transcribe_file(file: UploadFile = File(...)):
    """
    HTTP-эндпоинт для загрузки аудиофайла.
    Принимает аудиофайл через form-data (ключ 'file') и возвращает JSON с транскрипцией.
    """
    if not file or not file.filename:
        return JSONResponse(content={"error": "No file uploaded"}, status_code=400)

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_name = tmp.name
        tmp.write(file.file.read())

    lines = process_audio(tmp_name)
    os.unlink(tmp_name)

    return {"transcription": lines}
