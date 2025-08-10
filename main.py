import os
import queue
import threading
import time
import numpy as np
import sounddevice as sd
import torch
import argparse
import random
import keyboard # Nuevo
from dotenv import load_dotenv
from openai import OpenAI
from kokoro import KPipeline

# Cargar variables de entorno
load_dotenv()

# --- Configuración ---
VAD_THRESHOLD = 0.5
VAD_SAMPLING_RATE = 16000
VAD_MIN_SILENCE_DURATION_MS = 800

MIC_SAMPLING_RATE = 16000
MIC_CHANNELS = 1
MIC_DEVICE = None

WHISPER_MODEL = "small"
WHISPER_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER_COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "int8"

CHUTES_API_KEY = os.getenv("CHUTES_API_KEY")
LLM_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"

KOKORO_LANG = "e"
TTS_SAMPLING_RATE = 24000

# Configuración de Muletillas
FILLER_PHRASES = ["o sea", "tipo", "en plan"]
FILLER_SPEED = 1.2

# Configuración de Buffering para modo inmediato
TTS_BUFFER_SENTENCES = 2
TTS_BUFFER_CHARS = 150

# --- Colas y Eventos de Sincronización ---
audio_queue = queue.Queue()
text_queue = queue.Queue()
llm_response_queue = queue.Queue()
tts_is_speaking = threading.Event()
stop_playback_event = threading.Event() # Nuevo evento para detener la reproducción

# --- Modelos Globales ---
vad_model, utils, whisper_model, kokoro_pipeline = None, None, None, None
pregenerated_fillers = []

def initialize_models(selected_voice):
    global vad_model, utils, whisper_model, kokoro_pipeline, pregenerated_fillers
    from faster_whisper import WhisperModel
    try:
        print("Inicializando modelos...")
        torch.set_num_threads(1)
        vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, onnx=False)
        whisper_model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)
        kokoro_pipeline = KPipeline(lang_code=KOKORO_LANG)
        
        # Generar muletillas
        print("Generando muletillas de audio...")
        print(f"Usando voz para muletillas: {selected_voice}")
        for phrase in FILLER_PHRASES:
            filler_generator = kokoro_pipeline(phrase, voice=selected_voice, speed=FILLER_SPEED)
            filler_audio_chunks = []
            for _, _, audio_chunk in filler_generator:
                filler_audio_chunks.append(audio_chunk.numpy())
            pregenerated_fillers.append(np.concatenate(filler_audio_chunks))
        print("Muletillas generadas.")

        print("--- Modelos inicializados correctamente ---")
        return True
    except Exception as e:
        print(f"Error fatal durante la inicialización: {e}")
        return False

def audio_capture_and_vad(q_audio, speaking_event):
    (get_speech_timestamps, _, _, _, _) = utils
    print("\n--- Sistema listo. ¡Habla ahora! ---")
    speech_buffer = []
    is_speaking = False
    silence_start_time = None
    with sd.InputStream(samplerate=MIC_SAMPLING_RATE, channels=MIC_CHANNELS, device=MIC_DEVICE, dtype='float32', blocksize=512) as stream:
        while True:
            if speaking_event.is_set():
                time.sleep(0.1)
                continue
            chunk, _ = stream.read(512)
            speech_prob = vad_model(torch.from_numpy(chunk.flatten()), MIC_SAMPLING_RATE).item()
            if speech_prob > VAD_THRESHOLD:
                if not is_speaking:
                    print("Habla detectada...", end="", flush=True)
                    is_speaking = True
                speech_buffer.append(chunk)
                silence_start_time = None
            else:
                if is_speaking:
                    if silence_start_time is None: silence_start_time = time.time()
                    speech_buffer.append(chunk)
                    if time.time() - silence_start_time > VAD_MIN_SILENCE_DURATION_MS / 1000.0:
                        print(" Fin. Transcribiendo.", flush=True)
                        q_audio.put(np.concatenate(speech_buffer).tobytes())
                        is_speaking = False
                        speech_buffer = []
                        silence_start_time = None

def process_transcription(q_audio, q_text):
    while True:
        try:
            audio_bytes = q_audio.get(timeout=1)
            audio_float32 = np.frombuffer(audio_bytes, dtype=np.float32)
            segments, _ = whisper_model.transcribe(audio_float32, beam_size=5, language="es")
            text = "".join(s.text for s in segments).strip()
            if text:
                print(f"Tú: {text}", flush=True)
                q_text.put(text)
        except queue.Empty:
            continue

def query_llm(q_text, q_llm_res):
    system_prompt = "Eres un asistente de voz conversacional. Tu objetivo es entender la intención del usuario por contexto, incluso si hay errores en la transcripción. Responde de forma corta, concisa y en texto plano, sin usar Markdown ni otros formatos. Utiliza el historial de la conversación para mantener el contexto."
    history = [{"role": "system", "content": system_prompt}]
    client = OpenAI(base_url="https://llm.chutes.ai/v1", api_key=CHUTES_API_KEY)
    while True:
        try:
            user_text = q_text.get(timeout=1)
            history.append({"role": "user", "content": user_text})
            print("IA: ", end="", flush=True)
            stream = client.chat.completions.create(model=LLM_MODEL, messages=history, stream=True)
            full_res = ""
            for chunk in stream:
                if chunk.choices:
                    content = chunk.choices[0].delta.content or ""
                    if content:
                        print(content, end="", flush=True)
                        q_llm_res.put(content)
                        full_res += content
            print("", flush=True)
            if full_res: history.append({"role": "assistant", "content": full_res})
            q_llm_res.put("<END_OF_RESPONSE>")
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error en hilo LLM: {e}")
            time.sleep(1)

def tts_and_playback(q_llm_res, speaking_event, immediate_playback_mode, selected_voice, stop_event):
    global kokoro_pipeline, pregenerated_fillers
    sentence_buffer = ""
    sentence_enders = [".", "?", "!", ":", "\n"]
    sentence_count = 0
    
    while True:
        try:
            if immediate_playback_mode:
                # Modo inmediato (buffering híbrido y muletillas)
                token = q_llm_res.get(timeout=1)
                
                if not sentence_buffer and token != "<END_OF_RESPONSE>":
                    speaking_event.set() # Inicia el habla

                if token == "<END_OF_RESPONSE>":
                    if sentence_buffer.strip():
                        generator = kokoro_pipeline(sentence_buffer.strip(), voice=selected_voice, speed=1.2)
                        with sd.RawOutputStream(samplerate=TTS_SAMPLING_RATE, channels=1, dtype='float32') as stream:
                            for _, _, audio_chunk in generator:
                                if stop_event.is_set(): stream.abort(); break # Detener si se pulsa espacio
                                stream.write(audio_chunk.numpy())
                    sentence_buffer = ""
                    sentence_count = 0
                    speaking_event.clear() # Finaliza el habla
                    stop_event.clear() # Limpiar el evento de parada
                    continue
                
                sentence_buffer += token
                if any(ender in token for ender in sentence_enders):
                    sentence_count += 1
                
                # Condición para sintetizar y reproducir un chunk
                if sentence_count >= TTS_BUFFER_SENTENCES or len(sentence_buffer) >= TTS_BUFFER_CHARS:
                    generator = kokoro_pipeline(sentence_buffer.strip(), voice=selected_voice, speed=1.2)
                    with sd.RawOutputStream(samplerate=TTS_SAMPLING_RATE, channels=1, dtype='float32') as stream:
                        for _, _, audio_chunk in generator:
                            if stop_event.is_set(): stream.abort(); break # Detener si se pulsa espacio
                            stream.write(audio_chunk.numpy())
                    
                    # Reproducir muletilla si no es el final de la respuesta
                    if q_llm_res.qsize() > 0 or q_llm_res.full(): # Si hay más tokens en la cola
                        filler_audio = random.choice(pregenerated_fillers)
                        with sd.RawOutputStream(samplerate=TTS_SAMPLING_RATE, channels=1, dtype='float32') as stream:
                            if stop_event.is_set(): stream.abort(); break # Detener si se pulsa espacio
                            stream.write(filler_audio)

                    sentence_buffer = ""
                    sentence_count = 0

            else:
                # Modo fluido (respuesta completa)
                full_response_text = ""
                while True:
                    token = q_llm_res.get()
                    if token == "<END_OF_RESPONSE>":
                        break
                    full_response_text += token
                
                if full_response_text.strip():
                    speaking_event.set()
                    print("Generando audio para la respuesta completa...", flush=True)
                    generator = kokoro_pipeline(full_response_text.strip(), voice=selected_voice, speed=1.2)
                    with sd.RawOutputStream(samplerate=TTS_SAMPLING_RATE, channels=1, dtype='float32') as stream:
                        for _, _, audio_chunk in generator:
                            if stop_event.is_set(): stream.abort(); break # Detener si se pulsa espacio
                            stream.write(audio_chunk.numpy())
                    speaking_event.clear()
                    stop_event.clear() # Limpiar el evento de parada
                    print("Reproducción finalizada.", flush=True)

            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error en TTS/Playback: {e}")
            speaking_event.clear()
            stop_event.clear() # Asegurarse de limpiar el evento en caso de error

def keyboard_monitor(stop_event):
    print("Monitor de teclado iniciado. Presiona ESPACIO para interrumpir.")
    keyboard.add_hotkey('space', lambda: stop_event.set())
    keyboard.wait() # Mantiene el hilo vivo

def main():
    parser = argparse.ArgumentParser(description="Sistema de conversación en tiempo real.")
    parser.add_argument("-i", "--immediate", action="store_true", 
                        help="Activa el modo de reproducción inmediata (frase a frase).")
    parser.add_argument("-v", "--voice", type=str, default="ef_dora", 
                        help="Selecciona la voz de Kokoro (ej: ef_dora, em_alex, em_santa).")
    args = parser.parse_args()

    if not initialize_models(args.voice): return

    threads = [
        threading.Thread(target=audio_capture_and_vad, args=(audio_queue, tts_is_speaking,), daemon=True),
        threading.Thread(target=process_transcription, args=(audio_queue, text_queue,), daemon=True),
        threading.Thread(target=query_llm, args=(text_queue, llm_response_queue,), daemon=True),
        threading.Thread(target=tts_and_playback, args=(llm_response_queue, tts_is_speaking, args.immediate, args.voice, stop_playback_event,), daemon=True),
        threading.Thread(target=keyboard_monitor, args=(stop_playback_event,), daemon=True) # Nuevo hilo para el teclado
    ]

    for t in threads: t.start()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        print("\nDeteniendo el sistema.")

if __name__ == "__main__":
    main()