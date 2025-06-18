# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 19:46:34 2025

@author: viole
"""

#TS7 FILTROS FIR
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from pytc2.sistemas_lineales import plot_plantilla
import scipy.io as sio

#%%importo señal ECG
mat_struct = sio.loadmat('./ECG_TP4.mat')
ecg_one_lead =(mat_struct['ecg_lead']).flatten()
N = len(ecg_one_lead)
cant_muestras = N
fs = 1000
nyq = fs / 2

#%%función para mostrar la plantilla tanto para los filtros FIR como IIR
def plot_plantilla_filtros(w, h, label, fpass, fstop, ripple, attenuation, fs, title=''):
    """Grafica la respuesta en frecuencia y superpone la plantilla del filtro FIR"""
    # plt.figure(figsize=(10, 4))
    plt.figure()
    plt.plot(w, 20 * np.log10(np.abs(h) + 1e-12), label=label)
    plot_plantilla(
        filter_type='bandpass',
        fpass=fpass,
        fstop=fstop,
        ripple=ripple,
        attenuation=attenuation,
        fs=fs
    )
    plt.title(title)
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Magnitud [dB]')
    plt.legend()
    plt.grid(True)
    plt.show()

#%%funcion para que muestre el filtrado del ecg en distintas regiones, algunas sin ruido: para mostrar que no modifica la señal de no..
# y otras con rudio donde la señal debe ser filtrada 
# def mostrar_filtrado_ecg(ecg_original, ecg_filtrado, fs, etiqueta='ECG Filtrado'):
#     """
#     Muestra el ECG original y uno filtrado en regiones con y sin ruido.
#     Parámetros:
#     - ecg_original: señal ECG original
#     - ecg_filtrado: señal ECG luego del filtrado
#     - fs: frecuencia de muestreo
#     - etiqueta: nombre del filtro usado (para la leyenda)
#     """
#     # --- Regiones con ruido (muestras)
#     regiones_con_ruido = [(4000, 5500), (10000, 11000)]
    
#     for inicio, fin in regiones_con_ruido:
#         zoom_region = np.arange(max(0, int(inicio)), min(len(ecg_original), int(fin)))
#         plt.figure(figsize=(10, 4))
#         plt.plot(zoom_region, ecg_original[zoom_region], label='ECG Original', linewidth=2)
#         plt.plot(zoom_region, ecg_filtrado[zoom_region], label=etiqueta)
#         plt.title(f'Filtrado en región CON ruido [{inicio}, {fin}]')
#         plt.xlabel('Muestras')
#         plt.ylabel('Amplitud')
#         plt.legend()
#         plt.grid(True)
#         plt.show()

#     # --- Regiones sin ruido (en minutos → muestras)
#     regiones_sin_ruido = [
#         (5 * 60 * fs, 5.2 * 60 * fs),
#         (12 * 60 * fs, 12.4 * 60 * fs),
#         (15 * 60 * fs, 15.2 * 60 * fs)
#     ]

#     for inicio, fin in regiones_sin_ruido:
#         zoom_region = np.arange(max(0, int(inicio)), min(len(ecg_original), int(fin)))
#         plt.figure(figsize=(10, 4))
#         plt.plot(zoom_region, ecg_original[zoom_region], label='ECG Original', linewidth=2)
#         plt.plot(zoom_region, ecg_filtrado[zoom_region], label=etiqueta)
#         plt.title(f'Filtrado en región SIN ruido [{int(inicio)}, {int(fin)}]')
#         plt.xlabel('Muestras')
#         plt.ylabel('Amplitud')
#         plt.legend()
#         plt.grid(True)
#         plt.show()
def mostrar_filtrado_ecg(ecg_original, ecg_filtrado_o_filtro, fs, etiqueta='ECG Filtrado', tipo_filtro=None):
    """
    Muestra el ECG original y uno filtrado en regiones con y sin ruido.

    Parámetros:
    - ecg_original: señal ECG original
    - ecg_filtrado_o_filtro: o la señal ECG ya filtrada, o los coeficientes del filtro
    - fs: frecuencia de muestreo
    - etiqueta: nombre del filtro usado (para la leyenda)
    - tipo_filtro: 'FIR' o 'IIR' si se están pasando coeficientes del filtro
    """

    # --- Si se pasa un filtro y no una señal filtrada
    if tipo_filtro is not None:
        if tipo_filtro == 'FIR':
            ecg_filtrado = sig.lfilter(ecg_filtrado_o_filtro, 1, ecg_original)
        elif tipo_filtro == 'IIR':
            b, a = ecg_filtrado_o_filtro
            ecg_filtrado = sig.filtfilt(b, a, ecg_original)
        else:
            raise ValueError("tipo_filtro debe ser 'FIR' o 'IIR'")
    else:
        # Ya se pasó la señal filtrada
        ecg_filtrado = ecg_filtrado_o_filtro

    # --- Regiones con ruido
    regiones_con_ruido = [(4000, 5500), (10000, 11000)]
    for inicio, fin in regiones_con_ruido:
        zoom_region = np.arange(int(inicio), int(fin))
        plt.figure(figsize=(10, 4))
        plt.plot(zoom_region, ecg_original[zoom_region], label='ECG Original', linewidth=2)
        plt.plot(zoom_region, ecg_filtrado[zoom_region], label=etiqueta)
        plt.title(f'Filtrado en región CON ruido [{inicio}, {fin}]')
        plt.xlabel('Muestras')
        plt.ylabel('Amplitud')
        plt.legend()
        plt.grid(True)
        plt.show()

    # --- Regiones sin ruido
    regiones_sin_ruido = [
        (5 * 60 * fs, 5.2 * 60 * fs),
        (12 * 60 * fs, 12.4 * 60 * fs),
        (15 * 60 * fs, 15.2 * 60 * fs)
    ]
    for inicio, fin in regiones_sin_ruido:
        zoom_region = np.arange(int(inicio), int(fin))
        plt.figure(figsize=(10, 4))
        plt.plot(zoom_region, ecg_original[zoom_region], label='ECG Original', linewidth=2)
        plt.plot(zoom_region, ecg_filtrado[zoom_region], label=etiqueta)
        plt.title(f'Filtrado en región SIN ruido [{int(inicio)}, {int(fin)}]')
        plt.xlabel('Muestras')
        plt.ylabel('Amplitud')
        plt.legend()
        plt.grid(True)
        plt.show()

#%%metodo ventanas
# combinación de un filtro pasa-bajos con un filtro pasa-altos (por convolución)

# --- PASAALTOS (ganancia 1 desde 0.5 Hz, atenuación 0 desde 0.2 Hz)
fstop_hp = 0.1
fpass_hp = 0.5
cant_coef_hp = 5101 # más largo, transición muy angosta
frecs_hp = np.array([0, fstop_hp, fpass_hp, nyq]) / nyq
gains_hp_db = [-np.inf, -40, -0.1, 0]  # en dB → 0 dB en stop, transición, luego 0 dB en paso (lineal será 1)
gains_hp = 10**(np.array(gains_hp_db) / 20)

num_hp = sig.firwin2(cant_coef_hp, frecs_hp, gains_hp, window='hamming')

# --- PASABAJOS (ganancia 1 hasta 30 Hz, atenuación desde 40 Hz)
fpass_lp = 35
fstop_lp = 50
cant_coef_lp = 2001 # menos exigente
frecs_lp = np.array([0, fpass_lp, fstop_lp, nyq]) / nyq
gains_lp_db = [0, -0.1, -40, -80]  # en dB → pasa, transición, luego stop
gains_lp = 10**(np.array(gains_lp_db) / 20)

num_lp = sig.firwin2(cant_coef_lp, frecs_lp, gains_lp, window='hamming')

# --- COMBINACIÓN: FIR pasabanda por convolución
FIR_bp_ventana= np.convolve(num_hp, num_lp)
##
w1, h1 = sig.freqz(FIR_bp_ventana, worN=2048, fs=fs)
plot_plantilla_filtros(w1, h1, label='FIR Ventana', fpass=(fpass_hp, fpass_lp), fstop=(fstop_hp, fstop_lp), ripple=1, attenuation=40, fs=fs, title='Filtro FIR (Ventanas)')

mostrar_filtrado_ecg(ecg_one_lead, FIR_bp_ventana, fs, etiqueta='FIR Ventana', tipo_filtro='FIR')

# --- VISUALIZACIÓN RESPUESTA EN FRECUENCIA-------------------#
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(w1, np.unwrap(np.angle(h1)) * 180 / np.pi, color='darkred',label='Fase')
plt.ylabel('Fase [grados]')
plt.xlabel('Frecuencia [Hz]')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.subplot(1, 2, 2)
plt.plot(w1, 20 * np.log10(np.abs(h1) + 1e-12), color='teal', label='Módulo')
plt.ylabel('Módulo [dB]')
plt.xlabel('Frecuencia [Hz]')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.suptitle('Respuesta en frecuencia del filtro FIR (ventana Hamming)')
plt.tight_layout()
plt.show()

#%%método cuadrados mínimoss

ripple = 1         # dB
attenuation = 40   # dB

fpass = ( [1.0,35.0])
fstop = ([.1,50.0])
bands = np.array([0, fstop[0], fpass[0], fpass[1], fpass[1]+1, nyq]) #sumandole asi 1 lo hago simetrico -->transición de 1Hz en ambas bandas
desired = np.array([0, 0, 1, 1, 0, 0])

FIR_lsq = sig.firls(numtaps = 1501, bands= bands,desired=desired, fs=fs) #lsq = lowsquare = cuadrados mínimos

w2, h2 = sig.freqz(FIR_lsq, worN=2048, fs=fs)
plot_plantilla_filtros(w2, h2, label='FIR Ventana', fpass=(fpass[0], fpass[1]), fstop=(fstop[0], fstop[1]), ripple=1, attenuation=40, fs=fs, title='Filtro FIR (Cuadrados mínimos)')

mostrar_filtrado_ecg(ecg_one_lead, FIR_lsq, fs, etiqueta='FIR Cuadrados mínimos', tipo_filtro='FIR')

# --- VISUALIZACIÓN RESPUESTA EN FREUENCIA -----------------------------------------------------------------------
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(w2, np.unwrap(np.angle(h2)) * 180 / np.pi, color='darkred',label='Fase')
plt.ylabel('Fase [grados]')
plt.xlabel('Frecuencia [Hz]')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.subplot(1, 2, 2)
plt.plot(w2, 20 * np.log10(np.abs(h2) + 1e-12), color='teal', label='Módulo')
plt.ylabel('Módulo [dB]')
plt.xlabel('Frecuencia [Hz]')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.suptitle('Respuesta en frecuencia del filtro FIR (cuadrados mínimos)')
plt.tight_layout()
plt.show()

#%%# método remez
fpass = ( [1.0,35.0])
fstop = ([.1,50.0])
bands_1 = np.array([0, fstop[0], fpass[0], fpass[1], fpass[1]+1, nyq])
desired_1 = np.array([0, 0, 1, 1, 0, 0])

FIR_remez = sig.remez(2501, bands_1, desired_1[::2], fs=fs)

w3, h3 = sig.freqz(FIR_remez, worN=2048, fs=fs)
plot_plantilla_filtros(w3, h3, label='FIR Ventana', fpass=(fpass[0], fpass[1]), fstop=(fstop[0], fstop[1]), ripple=1, attenuation=40, fs=fs, title='Filtro FIR (Remez)')

mostrar_filtrado_ecg(ecg_one_lead, FIR_remez, fs, etiqueta='FIR Remez', tipo_filtro='FIR')

# --- VISUALIZACIÓN RESPUESTA EN FRECUENCIA-------------------#
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(w3, np.unwrap(np.angle(h3)) * 180 / np.pi, color='darkred',label='Fase')
plt.ylabel('Fase [grados]')
plt.xlabel('Frecuencia [Hz]')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.subplot(1, 2, 2)
plt.plot(w3, 20 * np.log10(np.abs(h3) + 1e-12), color='teal', label='Módulo')
plt.ylabel('Módulo [dB]')
plt.xlabel('Frecuencia [Hz]')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.suptitle('Respuesta en frecuencia del filtro FIR (método remez)')
plt.tight_layout()
plt.show()

#%%
