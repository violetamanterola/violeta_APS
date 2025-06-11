# -*- coding: utf-8 -*-
"""
Created on Thu May 15 11:47:09 2025

@author: viole
"""

#TS5

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.io as sio
from scipy.io.wavfile import write

#%% funcion BT para estimar densidad espectral de potencia (funcion q hizo Mariano)
def blackman_tukey(x, M=None):
    x = x.reshape(-1)
    N = len(x)
    if M is None:
        M = N // 5 #en caso de no indicar toma M=5 x default
    r_len = 2 * M - 1
    xx = x[:r_len]
    r = np.correlate(xx, xx, mode='same') / r_len
    windowed_r = r * sig.windows.blackman(r_len)
    Px = np.abs(np.fft.fft(windowed_r, n=N))
    return Px
    
# función para estimar ancho de banda (BW)
def estimar_ancho_de_banda(Pxx, f, low=0.95, high=0.98):
    Pxx_half = Pxx[:len(Pxx)//2] #para q me tome solo la aprte positiva
    f_half = f[:len(Pxx)//2]
    energia_total = np.sum(Pxx_half)
    energia_acum = np.cumsum(Pxx_half)
    energia_norm = energia_acum / energia_total

    #ubica el indice donde se concentra el 95% y el 98% de la energía total => dentro de ese rango se encuentra la frecuencia de ancho de banda segun este criterio
    idx_low = np.where(energia_norm >= 0.95)[0][0]
    idx_high = np.where(energia_norm >= 0.98)[0][0]
    
    f_low = f_half[idx_low]
    f_high = f_half[idx_high]
    bw = f_high - f_low
    return f_low, f_high, bw, energia_norm, f_half,Pxx_half

# función general para procesar la señal
def analizar_senal(nombre_archivo, fs=None, nombre='Señal'):
    print(f"\nProcesando: {nombre}")
    
    # Cargar y normalizar
    if nombre_archivo.endswith('.npy'):
        x = np.load(nombre_archivo).ravel() 

        # print(f"Muestras: {N}, Fs: {fs} Hz")
    elif nombre_archivo.endswith('.wav'): #el archivo de audio se lee dsintinto xq es .wav
        fs, x = sio.wavfile.read(nombre_archivo)
    else:
        raise ValueError("Formato de archivo no compatible (.npy o .wav)")
    
    x = x - np.mean(x) 
    #normalizo x varianza 
    x = x / np.std(x)
    N = len(x)
    print(N)

    # PSD Blackman-Tukey
    psd_bt = blackman_tukey(x)
    frecs = np.fft.fftfreq(N, d=1/fs)

    # Ancho de banda
    f95, f98, bw, energia_norm, f_half,Pxx_half = estimar_ancho_de_banda(psd_bt, frecs)
 #%%plots   
    # Mostrar gráficos
    

    # Resultados (hacer tabla !!!)
    print(f"Frecuencia donde se acumula el 95% de la energía: {f95:.2f} Hz")
    print(f"Frecuencia donde se acumula el 98% de la energía: {f98:.2f} Hz")
    print(f"Ancho de banda efectivo (95%-98%): {f98 - f95:.2f} Hz")

    # Plots
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(frecs[:N//2], 10 * np.log10(psd_bt[:N//2] + 1e-12))
    plt.title(f'PSD (Blackman-Tukey) - {nombre}')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('PSD [dB]')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(f_half, 10 * np.log10(Pxx_half + 1e-12), label='PSD (dB)', color='navy')
    plt.axvline(f95, color='red', linestyle='--', label=f'f95 = {f95:.2f} Hz')
    plt.axvline(f98, color='green', linestyle='--', label=f'f98 = {f98:.2f} Hz')    
    # plt.xlim(x_min, x_max)
    plt.title(f'PSD (Blackman-Tukey) - Zoom en BW - {nombre}')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('PSD [dB]')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    

    # plt.subplot(1, 2, 2)
    plt.figure(figsize=(10, 5))
    plt.plot(f_half, energia_norm)
    plt.axhline(0.95, color='r', linestyle='--')
    plt.axhline(0.98, color='g', linestyle='--')
    plt.axvline(f95, color='r', linestyle=':', label=f'f95 = {f95:.2f} Hz')
    plt.axvline(f98, color='g', linestyle=':', label=f'f98 = {f98:.2f} Hz')
    plt.title('Energía acumulada (normalizada)')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Energía acumulada')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#%%llamo a la señal a analizar!
# analizar_senal('ppg_sin_ruido.npy', fs=400, nombre='PPG')
analizar_senal('ecg_sin_ruido.npy', fs=1000, nombre='ECG')
# analizar_senal('la cucaracha.wav',nombre='Audio 1')
# analizar_senal('prueba psd.wav',nombre='Audio 2')
# analizar_senal('silbido.wav',nombre='Audio 3')
# analizar_senal('audio2.npy', fs=48000, nombre='Audio 2')



