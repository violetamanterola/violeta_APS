# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 15:43:23 2025

@author: viole
"""

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
from pytc2.sistemas_lineales import plot_plantilla#, group_delay
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
        
        if tipo_filtro == 'FIR': #los FIR VAN CON lfilter PARA DEMOSTRAR Q LA FASE ES LINEAL SOLO PRESENTA UN RETARDO DE GRUPO
            ecg_filtrado = sig.lfilter(ecg_filtrado_o_filtro, [1], ecg_original)
        
        elif tipo_filtro == 'IIR': # LOS IIR VAN CON FILTFILT PARA CORREGIR LA DISTORSIÓN DE FASE QUE TIENEN 
            ecg_filtrado = sig.sosfiltfilt(ecg_filtrado_o_filtro, ecg_original)
        else:
            raise ValueError("tipo_filtro debe ser 'FIR' o 'IIR'")
    else:
        # Ya se pasó la señal filtrada
        ecg_filtrado = ecg_filtrado_o_filtro

    # --- Regiones SIN ruido
    regiones_sin_ruido = [(4000, 5500), (10000, 11000),(5 * 60 * fs, 5.2 * 60 * fs)]
    for inicio, fin in regiones_sin_ruido:
        zoom_region = np.arange(int(inicio), int(fin))
        plt.figure(figsize=(10, 4))
        plt.plot(zoom_region, ecg_original[zoom_region], label='ECG Original', linewidth=2)
        plt.plot(zoom_region, ecg_filtrado[zoom_region], color='magenta',label=etiqueta)
        plt.title(f'Filtrado en región sin ruido [{inicio}, {fin}]')
        plt.xlabel('Muestras')
        plt.ylabel('Amplitud')
        plt.legend()
        plt.grid(True)
        plt.show()

    # --- Regiones con ruido
    regiones_ruido = [
        (12 * 60 * fs, 12.4 * 60 * fs),
        (15 * 60 * fs, 15.2 * 60 * fs)
    ]
    for inicio, fin in regiones_ruido:
        zoom_region = np.arange(int(inicio), int(fin))
        plt.figure(figsize=(10, 4))
        plt.plot(zoom_region, ecg_original[zoom_region], label='ECG Original', linewidth=2)
        plt.plot(zoom_region, ecg_filtrado[zoom_region], color='magenta',label=etiqueta)
        plt.title(f'Filtrado en región con ruido [{int(inicio)}, {int(fin)}]')
        plt.xlabel('Muestras')
        plt.ylabel('Amplitud')
        plt.legend()
        plt.grid(True)
        plt.show()
#%%
aprox_name = 'butter'
fs = 1000
nyq_frec = fs/2
fpass= np.array([1.0,35.0]) #banda de paso wp
ripple = 1 #dB
fstop= ([.1,50.]) #comienzo banda de atenuación, hasta 50 (interferencia de la red eléctrica)
attenuation = 40 #dB

mi_sos = sig.iirdesign(fpass, fstop, ripple, attenuation, ftype=aprox_name, output='sos',fs=fs)
# las columnas 3,4y5 son los coeficiente a0,a1,a2 y de la 0 a la 2 son los coef b

orden_iir = mi_sos.shape[0] * 2  # cada sección SOS es un biquadro (orden 2)
print(f"Orden del filtro IIR: {orden_iir}") #para verificar de que orden es cada filtro y así poder comparar 

#%%plantilla de diseño, para analizarlo
npoints = 1000 #asi evalua equiespaciado

#para obtner mayor resuloción antesd e la bandad de paso, necesito un muestreo log => a freqz le puedo pasar un vector.
w_rad = np.append(np.logspace(-2,0.8,250), np.logspace(0.9,1.6,250))
w_rad = np.append(w_rad, np.linspace(40, nyq_frec, 500, endpoint=True) )/nyq_frec * np.pi

w, hh = sig.sosfreqz(mi_sos, worN=w_rad) #worN le puedo pasar un entero o un vector 
plt.plot(w/np.pi*nyq_frec, 20*np.log10(np.abs(hh)+1e-15), label='mi_sos')

plot_plantilla(filter_type = 'bandpass' , fpass = (fpass[0],fpass[1]), ripple = ripple , fstop = (fstop[0],fstop[1]), attenuation = attenuation, fs = fs)
plt.title(f'Plantilla filtro IIR orden {orden_iir}')
plt.legend()
plt.show()

mostrar_filtrado_ecg(ecg_one_lead, mi_sos, fs, etiqueta='IIR SOS ' + aprox_name.upper(), tipo_filtro='IIR')

##-------------------- VISUALIZACIÓN RESPUESTA EN FRECUENCIA-----------------##
plt.figure(figsize=(12, 4))

# Respuesta en módulo
plt.subplot(1, 2, 1)
plt.plot( w / np.pi * nyq_frec, 20 * np.log10(np.abs(hh) + 1e-12), color='purple',label='Respuesta en módulo')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Módulo [dB]')
plt.grid(True)
plt.legend()

# Respuesta en fase (esto es antes de filtrar, para verificar que NO TIENEN FASE LINEAL LOS FILTROS IIR)
plt.subplot(1, 2, 2)
plt.plot( w / np.pi * nyq_frec, np.unwrap(np.angle(hh)), label='Fase', color='cyan')
plt.ylabel('Fase [rad]')
plt.xlabel('Frecuencia [Hz]')
plt.grid(True)
plt.legend()

plt.suptitle(f'Respuesta en frecuencia del filtro IIR ({aprox_name})')
plt.tight_layout()
plt.show()

#%%
aprox_name = 'cheby1' #ripple en la banda pasante
mi_sos_2 = sig.iirdesign(fpass, fstop, ripple, attenuation, ftype=aprox_name, output='sos',fs=fs)

orden_iir_2 = mi_sos.shape[0] * 2  # cada sección SOS es un biquadro (orden 2)
print(f"Orden del filtro IIR: {orden_iir_2}")

#%%plantilla de diseño, para analizarlo
#npoints = 1000 #asi evalua equiespaciado
#para obtner mayor resuloción antesd e la bandad de paso, necesito un muestreo log => a freqz le puedo pasar un vector.
w_rad_2 = np.append(np.logspace(-2,0.8,250), np.logspace(0.9,1.6,250))
w_rad_2 = np.append(w_rad, np.linspace(40, nyq_frec, 500, endpoint=True) )/nyq_frec * np.pi

w2, hh2 = sig.sosfreqz(mi_sos_2, worN=w_rad) #worN le puedo pasar un entero o un vector 

#orden_iir = mi_sos.shape[0] * 2  # cada sección SOS es un biquadro (orden 2)
#print(f"Orden del filtro IIR: {orden_iir}")

plt.plot(w2/np.pi*nyq_frec, 20*np.log10(np.abs(hh2)+1e-15), label='mi_sos')
plot_plantilla(filter_type = 'bandpass' , fpass = (fpass[0],fpass[1]), ripple = ripple , fstop = (fstop[0],fstop[1]), attenuation = attenuation, fs = fs)
plt.title(f'Plantilla filtro IIR orden {orden_iir}')
plt.legend()
plt.show()

mostrar_filtrado_ecg(ecg_one_lead, mi_sos_2, fs, etiqueta='ECG filtro IIR ' + aprox_name.upper(), tipo_filtro='IIR')

##-------------------- VISUALIZACIÓN RESPUESTA EN FRECUENCIA-----------------##
plt.figure(figsize=(12, 4))

# Respuesta en módulo
plt.subplot(1, 2, 1)
plt.plot( w2 / np.pi * nyq_frec, 20 * np.log10(np.abs(hh2) + 1e-12), color='purple',label='Respuesta en módulo')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Módulo [dB]')
plt.grid(True)
plt.legend()

# Respuesta en fase (esto es antes de filtrar, para verificar que NO TIENEN FASE LINEAL LOS FILTROS IIR)
plt.subplot(1, 2, 2)
plt.plot( w2 / np.pi * nyq_frec, np.unwrap(np.angle(hh2)), label='Fase', color='cyan')
plt.ylabel('Fase [rad]')
plt.xlabel('Frecuencia [Hz]')
plt.grid(True)
plt.legend()

plt.suptitle(f'Respuesta en frecuencia del filtro IIR ({aprox_name})')
plt.tight_layout()
plt.show()

#%%
aprox_name = 'ellip'
mi_sos_3 = sig.iirdesign(fpass, fstop, ripple, attenuation, ftype=aprox_name, output='sos',fs=fs)
# las columnas 3,4y5 son los coeficiente a0,a1,a2 y de la 0 a la 2 son los coef b

orden_iir_3 = mi_sos.shape[0] * 2  # cada sección SOS es un biquadro (orden 2)
print(f"Orden del filtro IIR: {orden_iir_3}")

#%%plantilla de diseño, para analizarlo
npoints = 1000 #asi evalua equiespaciado

#para obtner mayor resuloción antesd e la bandad de paso, necesito un muestreo log => a freqz le puedo pasar un vector.
w_rad_3 = np.append(np.logspace(-2,0.8,250), np.logspace(0.9,1.6,250))
w_rad_3 = np.append(w_rad, np.linspace(40, nyq_frec, 500, endpoint=True) )/nyq_frec * np.pi

w3, hh3 = sig.sosfreqz(mi_sos_3, worN=w_rad) #worN le puedo pasar un entero o un vector 
plt.plot(w/np.pi*nyq_frec, 20*np.log10(np.abs(hh)+1e-15), label='mi_sos')

plot_plantilla(filter_type = 'bandpass' , fpass = (fpass[0],fpass[1]), ripple = ripple , fstop = (fstop[0],fstop[1]), attenuation = attenuation, fs = fs)
plt.title(f'Plantilla filtro IIR orden {orden_iir}')
plt.legend()
plt.show()

mostrar_filtrado_ecg(ecg_one_lead, mi_sos, fs, etiqueta='ECG filtro IIR ' + aprox_name.upper(), tipo_filtro='IIR')

##-------------------- VISUALIZACIÓN RESPUESTA EN FRECUENCIA-----------------##
plt.figure(figsize=(12, 4))

# Respuesta en módulo
plt.subplot(1, 2, 1)
plt.plot( w3 / np.pi * nyq_frec, 20 * np.log10(np.abs(hh3) + 1e-12), color='purple',label='Respuesta en módulo')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Módulo [dB]')
plt.grid(True)
plt.legend()

# Respuesta en fase (esto es antes de filtrar, para verificar que NO TIENEN FASE LINEAL LOS FILTROS IIR)
plt.subplot(1, 2, 2)
plt.plot( w3 / np.pi * nyq_frec, np.unwrap(np.angle(hh3)), label='Fase', color='cyan')
plt.ylabel('Fase [rad]')
plt.xlabel('Frecuencia [Hz]')
plt.grid(True)
plt.legend()

plt.suptitle(f'Respuesta en frecuencia del filtro IIR ({aprox_name})')
plt.tight_layout()
plt.show()


