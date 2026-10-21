import torch
import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando: {device}")

model = ChatterboxMultilingualTTS.from_pretrained(device=device)

texto = "Hola, este es un taller de Inteligencia Artificial Generativa usando Chatterbox TTS."
wav = model.generate(texto, language_id="es")
ta.save("salida_es.wav", wav, model.sr)
print("Audio generado: salida_es.wav")
