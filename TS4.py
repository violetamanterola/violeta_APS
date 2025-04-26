# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 17:38:52 2025

@author: viole
"""
# en este caso es con 10dB
# no hago padding, entonces tengo problema de reosulicón espectral, veo de medio bit para la derecha y medio bir hacia la izquierda

#%% Módulos
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal 
from scipy.fft import fft
import pandas as pd

fs = 1000 # [Hz]
N = 1000
df = fs / N
# kn= 1/10
ts = 1/fs
nn=N
#%%Datos simulación
w0=fs/4 #mitad de banda digital
SNR=10 #dB
R=200 #cantidad de repeticiones
a1=np.sqrt(2) #amplitud de la señal, para que la potencia de la senoidal sea de 1W.
a1_real= np.mean(a1)

#%%
#freceuncia aleatoria: 
fr = np.random.uniform(-1/2,1/2,size=(1,R)) #tamaño 1xR

tt = np.linspace(0,(N-1)*ts,N).reshape((N,1)) # VEctor flat,[0*100] tengo q cambiarle las dimensiones con reshape de 1000x1000 o 1000x200 es NxR
vtt = np.tile(tt,(1,R)) #para q quede la matriz de de NxR, q me lo repite en la segunda dimension , osea me repite R veces algo que era N.


w1 = w0 + fr*(df)

#%%argumento matriz S
Xk = a1*np.sin(2*np.pi*w1*vtt) #debe quedar con dimensiones de [N,R]

#%%Matriz de Ruido 
#generación ruido aleatorio 
pot_ruido = 10**(-SNR /10)
na = np.random.normal(0, np.sqrt(pot_ruido), size=(N,R)) #genera señal ruido analógico gaussiano.

#%% FFT´S
Xk_fft = np.fft.fft(Xk, axis=0) /N #hace la fft por columnas 
# grilla de sampleo frecuencial
ff = np.linspace(0, (N-1)*df, N)
bfrec = ff <= fs/2

#%% sumo ambas señales!

sr = Xk + na #mi señal mas el ruido 

#%% ventaneo señal 
ventana1 = signal.windows.blackmanharris(N).reshape(N,1)
signal_1 = sr * ventana1
signal_1_fft = (1/N)*np.fft.fft(signal_1,axis=0)
signal_1_abs = np.abs(signal_1_fft)

ventana2 = signal.windows.flattop(N).reshape(N,1)
signal_2 = sr * ventana2
signal_2_fft = (1/N)*np.fft.fft(signal_2,axis=0)
signal_2_abs = np.abs(signal_2_fft)

ventana3 = signal.windows.hamming(N).reshape(N,1)
signal_3 = sr * ventana3
signal_3_fft = (1/N)*np.fft.fft(signal_3,axis=0)
signal_3_abs = np.abs(signal_3_fft)

#%% señal rectangular sin ventana
signal_0 = sr * np.ones((N,1))
signal_0_fft = (1/N)*np.fft.fft(signal_0,axis=0)
signal_0_abs = np.abs(signal_0_fft)
#%% plot de ventanas 
def plot_ventana (fft_matriz,fs,N,ventana):
    df = fs / N
    ff = np.linspace(0, (N - 1) * df, N)
    bfrec = ff <= fs / 2
    plt.figure(figsize=(12, 6))
    plt.plot(ff[bfrec], 10 * np.log10(2 * np.abs(fft_matriz[bfrec, :])**2+ 1e-12))
    plt.title(f'Espectros individuales - Window {ventana}')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Densidad espectral [dB]')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
 
plot_ventana (fft_matriz = signal_1_fft, fs=1000,N=1000, ventana='Blackmanharris' )   
plot_ventana (fft_matriz = signal_2_fft, fs=1000,N=1000, ventana='Flattop' )   
plot_ventana (fft_matriz = signal_3_fft, fs=1000,N=1000, ventana='Hamming' )


#%%estimador de amplitud
est_amp_0=np.abs(signal_0_fft[N//4, :]) #en N//4 xq es la fercuencia central de la senoidal, donde se espera la mayor amplitud
est_amp_1=np.abs(signal_1_fft[N//4, :])
est_amp_2=np.abs(signal_2_fft[N//4, :])
est_amp_3=np.abs(signal_3_fft[N//4, :])


#%%estimador de freceuncia 
#ahora busco donde la amplitud del espectro es maximo 
k_0=np.argmax(signal_0_abs[:N//2, :],axis =0 ) #pongo N//2, tomo solo los positivos, puedo hacerlo ya que la fft es simétrica
k_1=np.argmax(signal_1_abs[:N//2, :],axis =0 )
k_2=np.argmax(signal_2_abs[:N//2, :],axis =0 )
k_3=np.argmax(signal_3_abs[:N//2, :],axis =0 )

o0 = k_0*df
o1 = k_1*df
o2 = k_2*df
o3 = k_3*df

#%%calculo sesgo y varianza 

# Valor verdadero de la frecuencia promedio
f_real = np.mean(w1)  # frecuencia promedio real de la señal

# Rectangular
esperanza_f0 = np.mean(o0)
sesgo_f0 = esperanza_f0 - f_real
varianza_f0 = np.var(o0)

# Blackman-Harris
esperanza_f1 = np.mean(o1)
sesgo_f1 = esperanza_f1 - f_real
varianza_f1 = np.var(o1)

# Flattop
esperanza_f2 = np.mean(o2)
sesgo_f2 = esperanza_f2 - f_real
varianza_f2 = np.var(o2)

# Hamming
esperanza_f3 = np.mean(o3)
sesgo_f3 = esperanza_f3 - f_real
varianza_f3 = np.var(o3)

#lo mismo para la amplitud 
esperanza_a0 =np.mean(est_amp_0)
sesgo_a0 = esperanza_a0 - a1_real
varianza_a0 = np.var(est_amp_0)

esperanza_a1 =np.mean(est_amp_1)
sesgo_a1 = esperanza_a1 - a1_real
varianza_a1 = np.var(est_amp_1)

esperanza_a2 =np.mean(est_amp_2)
sesgo_a2 = esperanza_a2 - a1_real
varianza_a2 = np.var(est_amp_2)

esperanza_a3 =np.mean(est_amp_3)
sesgo_a3 = esperanza_a3 - a1_real
varianza_a3 = np.var(est_amp_3)


#%%ploteos histogramas 
plt.figure(figsize=(10, 6))
plt.hist(o0, bins=30, alpha=0.5, color = 'green',label='Rectangular-sin ventana')
# plt.hist(o1, bins=30, alpha=0.5, color = 'purple',label='Blackman-Harris')
# plt.hist(o2, bins=30, alpha=0.5,color='orange', label='Flattop')
# plt.hist(k_3, bins=30, alpha=0.5, color='blue',label='Hamming')

plt.title("Histograma de máximos espectrales por ventana")
plt.xlabel("Magnitud máxima [dB]")
plt.ylabel("Frecuencia")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(est_amp_0, bins=30, alpha=0.5, color = 'green',label='Rectangular-sin ventana')
plt.hist(est_amp_1, bins=30, alpha=0.5, color = 'purple',label='Blackman-Harris')
plt.hist(est_amp_2, bins=30, alpha=0.5,color='orange', label='Flattop')
plt.hist(est_amp_3, bins=30, alpha=0.5, color='blue',label='Hamming')

plt.title("Histograma de máximos espectrales por ventana")
plt.xlabel(".")
plt.ylabel("Frecuencia")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"Sesgo frecuencia Rectangular: {sesgo_f0:.6f} Hz, Varianza: {varianza_f0:.6f}")
print(f"Sesgo frecuencia Blackman-Harris: {sesgo_f1:.6f} Hz, Varianza: {varianza_f1:.6f}")
print(f"Sesgo frecuencia Flattop: {sesgo_f2:.6f} Hz, Varianza: {varianza_f2:.6f}")
print(f"Sesgo frecuencia Hamming: {sesgo_f3:.6f} Hz, Varianza: {varianza_f3:.6f}")

#%% plot de ventanas 
# def plot_ventana (fft_matriz,fs,N,ventana):
#     df = fs / N
#     ff = np.linspace(0, (N - 1) * df, N)
#     bfrec = ff <= fs / 2
#     plt.figure(figsize=(12, 6))
#     plt.plot(ff[bfrec], 10 * np.log10(2 * np.abs(fft_matriz[bfrec, :])**2+ 1e-12))
#     plt.title(f'Espectros individuales - Window {ventana}')
#     plt.xlabel('Frecuencia [Hz]')
#     plt.ylabel('Densidad espectral [dB]')
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()
# plot_ventana (fft_matriz = signal_0_fft, fs=1000,N=1000, ventana='Rectangular' ) 
# plot_ventana (fft_matriz = signal_1_fft, fs=1000,N=1000, ventana='Blackman' )   
# plot_ventana (fft_matriz = signal_2_fft, fs=1000,N=1000, ventana='Flattop' )   
# plot_ventana (fft_matriz = signal_3_fft, fs=1000,N=1000, ventana='Hamming' )

#%% tabla de datos

ventanas = ['Rectangular', 'Blackman-Harris', 'Flattop', 'Hamming']

# Datos para SNR = 10 dB (de tu código anterior)
sesgos10 = [sesgo_f0, sesgo_f1, sesgo_f2, sesgo_f3]
varianzas10 = [varianza_f0, varianza_f1, varianza_f2, varianza_f3]

df = pd.DataFrame({
    'Ventana': ventanas,
    'Sesgo 10 dB (Hz)': sesgos10,
    'Varianza 10 dB (Hz²)': varianzas10,
})

print(df)

# Datos para SNR = 10 dB (de tu código anterior)
sesgos10_A = [sesgo_a0, sesgo_a1, sesgo_a2, sesgo_a3]
varianzas10_A = [varianza_a0, varianza_a1, varianza_a2, varianza_a3]

df1 = pd.DataFrame({
    'Ventana': ventanas,
    'Sesgo 10 dB (Hz)': sesgos10_A,
    'Varianza 10 dB (Hz²)': varianzas10_A,
})

print(df1)