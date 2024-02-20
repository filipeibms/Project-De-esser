import numpy as np
import librosa
import soundfile as sf

def deesser(audio_file, frequency, threshold):
    # Carrega o arquivo de áudio
    y, sr = librosa.load(audio_file, sr=None, mono=True)

    # Aplica um filtro passa-baixa para enfatizar as frequências sibilantes
    y_lowpass = librosa.effects.preemphasis(y)

    # Calcula o envelope de amplitude
    envelope = librosa.onset.onset_strength(y=y_lowpass, sr=sr, aggregate=np.median)

    # Interpola o envelope para a mesma dimensão dos dados de áudio
    envelope_interp = np.interp(np.arange(len(y)), np.linspace(0, len(y), len(envelope)), envelope)

    # Aplica a compressão seletiva com base no envelope e no threshold
    gain = np.where(envelope_interp > threshold, 1.0, 0.0)
    y_deessed = y * gain

    # Salva o áudio processado em um novo arquivo
    sf.write("audio_deessed.wav", y_deessed, sr)

# Exemplo de uso
audio_file = "audio_input.wav"
# Escolha da frequência de corte desejada
frequency = float(input("Escolha a frequência de corte desejada: "))  # Frequência de corte para a sibilância
threshold = float(input("Escolha o threshold desejado: "))   # Threshold para ativação da compressão
threshold = np.float32(threshold)  # Converter para o tipo de dados float32

deesser(audio_file, frequency, threshold)
