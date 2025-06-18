# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 21:29:36 2025

@author: viole
"""

# filtros de mediana
# ecg filtrado con metodo de ventana mediana
import sympy as sp
import numpy as np
import scipy.signal as sig
from scipy.signal.windows import hamming, kaiser, blackmanharris
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.signal import find_peaks

#%%señal ECG
fs_ecg = 1000
mat_struct = sio.loadmat('./ECG_TP4.mat')
ecg_one_lead =(mat_struct['ecg_lead']).flatten()

# segmento de interes
ecg_segment = ecg_one_lead[700000:745000]
N = len(ecg_one_lead)

t_segment = np.linspace(0,len(ecg_segment)/fs_ecg, len(ecg_segment) )
#%%
# Normalización tipo z-score: (x - media) / std
# def normalize(signal):
#     return (signal - np.mean(signal)) / np.std(signal)

# # Aplicar normalización
# ecg = normalize(ecg_one_lead)


# # Crear un vector de tiempo para la señal ECG completa
# t_ecg = np.arange(len(ecg)) / fs_ecg

# # Crear vectores de tiempo para los patrones (usualmente de menor duración)
# t_qrs = np.arange(len(qrs_pattern)) / fs_ecg

#%% filtro no lineal, FILTRO DE MEDIANA

# from scipy.signal import medfilt

# Ventanas
win1_samples = 200 
win2_samples = 600 #probar tambien con 1200 

# Aseguro q sea impar --> si es par le sumo 1
if win1_samples % 2 == 0:
    win1_samples += 1
if win2_samples % 2==0:
    win2_samples +=1
    
print(win1_samples,win2_samples)
#filtros de mediana --> linea de base 
#primer filtro de mediana (200ms)
ecg_med1 = sig.medfilt(ecg_segment,kernel_size=win1_samples)

#segundo filtro de mediana (600ms) -->al reusltado del filtro anterior le aplico otro con ventana de 600ms
ecg_med2 = sig.medfilt(ecg_med1,kernel_size=win2_samples) # lo conceto en casacada con el q acabo de hacer

#entonces ecg registrado sin interferencias quedaría: 
ecg_sin_interferencias = ecg_segment - ecg_med2
#%% Visualizacion 
plt.figure(figsize=(12,5))
# plt.plot(ecg_one_lead)

plt.plot(t_segment, ecg_segment, label='Señal original',alpha=0.6)
plt.plot(t_segment, ecg_med2, label='Linea de base - Filtrado (200ms + 600ms)', linewidth=2)
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.title('ECG filtrado con etapas de mediana')
plt.legend
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()

#le sumo al grafico entonces como quedaria el ecg filtrado por filtro de mediana
plt.figure(figsize=(12,5))
plt.plot(t_segment, ecg_segment, label='ECG con interferencias', alpha=0.5)
plt.plot(t_segment, ecg_med2, label='Línea de base estimada', linewidth=2)
plt.plot(t_segment, ecg_sin_interferencias, label='ECG sin interferencias (filtrado)', linewidth=1.5)
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.title('Filtrado no lineal del ECG (mediana 200ms + 600ms)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%
from scipy.interpolate import CubicSpline

#crear el spline cubico 
#cubicspline: interpolará exactamente los puntos que le das como entrada, generando una función suave, compuesta de segmentos cúbicos, con continuidad de primera y segunda derivada.
#tomar cierto intervalo  antes del complejo QRS --> xq es dodne la señal suele estar mas estable
#asi evito la distorsion del complejo QRS (segmento Pq)
# spline = CubicSpline(t_segment, y)
qrs = (mat_struct['qrs_detections']).flatten() #utilizo vector qrs_detections para identificar los segmentos isoélectricos entre la onda P y el complejo QRS
#xq mat_struct es un diccionario :)

# luego del grafico --> estimo alrededor de 90ms pre complejo QRS en una ventana de 20 ms para evitar agarra tanta varianza

punto_previo = 90 #a ojo aprox para estimar el nivel isoeléctrico (evitando la onda p y complejo qrs, x eso lo ubico antes asi lo puedo restar)
ventana = 20 # Tomar una ventana corta (por ejemplo, 20 ms) para calcular un valor de referencia
segmentos = []
t_base = []

for idx in qrs:
    i_ini = idx - punto_previo 
    i_fin = i_ini + ventana
    if i_ini >= 0 and i_fin < len(ecg_one_lead):
        segmento = ecg_one_lead[i_ini:i_fin]
        segmentos.append(segmento)
        t_base.append(i_ini + ventana // 2)  # punto central del segmento para usar como soporte temporal del spline

segmentos_array = np.array(segmentos)
puntos_base = np.mean(segmentos_array, axis=1)  # media por latido
t_base = np.array(t_base)

# spline sobre toda la señal -->Interpolar esos puntos usando un spline cúbico para obtener la estimación de la línea de base en toda la señal.
cs_total = CubicSpline(t_base, puntos_base)
spline_total = cs_total(np.arange(len(ecg_one_lead)))

# señal filtrada
ecg_filtrado = ecg_one_lead - spline_total

# graficar en el segmento de interés con la linea de base
plt.figure(figsize=(10,4))
plt.plot(t_segment, ecg_segment, label='Original')
plt.plot(t_segment, spline_total[700000:745000], label='Línea base spline', linewidth=2)
plt.xlabel('Tiempo [s]')
plt.title('Filtrado de línea base con spline cúbico')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#grafico ecg_filtrado 
plt.figure(figsize=(10,4))
plt.plot(t_segment, ecg_segment, label='ECG Original')
plt.plot(t_segment, ecg_filtrado[700000:745000], label='ECG filtrado', alpha=0.8)
plt.xlabel('Tiempo [s]')
plt.title('Filtrado de línea base con spline cúbico')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#%%FILTRO ADAPTADO 

# --> es un filtr odiseñado para detectar una forma de onda conocida (en este caso, el complejo QRS) dentro de una señal ruidosa. 

# la idea aca es: 
    #Correlacionar la señal ECG con un patrón típico de QRS (qrs_pattern1).
    # El filtro adaptado responde fuertemente cuando la forma del patrón aparece en la señal.
    # Esto genera picos en la salida que corresponden al momento del latido.
    

#%% Cargar señal y patrón QRS

#¿como puedo hacer un dectector de latidos a partir de la señal resultante del filtro adaptado. 
fs = 1000
mat_struct = sio.loadmat('ECG_TP4.mat')
ecg = mat_struct['ecg_lead'].flatten()
qrs_pattern = mat_struct['qrs_pattern1'].flatten()
qrs_true = mat_struct['qrs_detections'].flatten()

# qrs_pattern = normalize(qrs_pattern)
# qrs_true = normalize(qrs_true)
#  Realice la detección de los latidos, comparando las detecciones obtenidas con las que se incluyen en la variable qrs_detections.
#%% Filtro adaptado = correlación con el patrón (invertido y desplazado)
matched_filter = np.correlate(ecg, qrs_pattern[::-1], mode='same') #para que sea una correlación cruzada

#%%
# Detectar picos
# Parámetros recomendados:
# - height: umbral mínimo (por ejemplo, 50% del máximo)
# - distance: separación mínima entre picos (en muestras)
peaks, properties = find_peaks(matched_filter,height=30, distance=250)  # ~250 ms si fs = 1000 Hz

# Crear vector de tiempo para la correlación
fs = 1000  # Hz
t_corr = np.arange(len(matched_filter)) / fs

# Graficar correlación con los picos
plt.figure(figsize=(10, 4))
plt.plot(t_corr, matched_filter, label='Correlación ECG - QRS')
plt.plot(t_corr, ecg, label='ECG')
plt.plot(t_corr[peaks], matched_filter[peaks], 'rx', label='Picos detectados')
plt.title('Picos de correlación (posibles latidos)')
plt.xlabel('Tiempo [s]')
plt.ylabel('Correlación')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#%%
# -- Tolerancia en muestras para considerar un match (ej: ±50 ms si fs = 1000 Hz)
fs = 1000
tol = int(0.05 * fs)  # 50 ms

# -- Convertir a arrays enteros
qrs_true = np.array(qrs_true, dtype=int)
peaks = np.array(peaks, dtype=int)

# -- Inicializar indicadores
TP = 0  # Verdaderos positivos
FP = 0  # Falsos positivos
FN = 0  # Falsos negativos

# -- Etiquetas para los picos detectados
detected_flags = np.zeros_like(qrs_true, dtype=bool)

# --- Comparar cada pico detectado con los latidos reales
for peak in peaks:
    # Buscar si hay un latido verdadero cerca del pico detectado
    match = np.any(np.abs(qrs_true - peak) <= tol)
    if match:
        TP += 1
        idx = np.argmin(np.abs(qrs_true - peak))
        detected_flags[idx] = True
    else:
        FP += 1

# -- Falsos negativos: latidos reales no detectados
FN = np.sum(~detected_flags)

# -- Métricas
sensibilidad = TP / (TP + FN)
valor_predictivo_positivo = TP / (TP + FP)

# -- Mostrar resultados
print(f'Sensibilidad (Recall): {sensibilidad:.3f}')
print(f'Valor Predictivo Positivo (Precision): {valor_predictivo_positivo:.3f}')
print(f'Latidos reales detectados correctamente (TP): {TP}')
print(f'Falsos positivos (FP): {FP}')
print(f'Falsos negativos (FN): {FN}')

#%%
qrs_true = mat_struct['qrs_detections'].flatten()
# Parámetros del tramo de interés (en segundos)
t_inicio = 700
t_fin = 705

# Convertir a índices
i_inicio = int(t_inicio * fs)
i_fin = int(t_fin * fs)

# Crear vector de tiempo para ese tramo
t_ecg = np.arange(len(ecg_one_lead)) / fs
t_segmento = t_ecg[i_inicio:i_fin]
ecg_segmento = ecg_one_lead[i_inicio:i_fin]

# Filtrar detecciones verdaderas (qrs_true) dentro del rango
qrs_true_segmento = qrs_true[(qrs_true >= i_inicio) & (qrs_true < i_fin)]

# Filtrar picos detectados por filtro adaptado (peaks)
peaks_segmento = peaks[(peaks >= i_inicio) & (peaks < i_fin)]

# Graficar
plt.figure(figsize=(12, 5))
plt.plot(t_segmento, ecg_segmento, label='ECG', color='orange', alpha=0.8)

# Detecciones verdaderas (por ejemplo en rojo)
plt.plot(t_ecg[qrs_true_segmento], ecg_one_lead[qrs_true_segmento], 'rx', label='QRS verdaderos')

# Picos detectados con filtro adaptado (por ejemplo en azul)
plt.plot(t_ecg[peaks_segmento], ecg_one_lead[peaks_segmento], 'b*', label='Picos detectados (filtro adaptado)')

plt.title('ECG con detecciones verdaderas y filtro adaptado')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))

# Señal ECG
plt.plot(t_segmento, ecg_segmento, label='ECG', color='orange', alpha=0.8)

# Detecciones verdaderas (QRS)
plt.plot(t_ecg[qrs_true_segmento], ecg_one_lead[qrs_true_segmento], 'rx', label='QRS verdaderos')

# Detecciones por filtro adaptado (picos)
plt.plot(t_ecg[peaks_segmento], ecg_one_lead[peaks_segmento], 'b*', label='Picos detectados (filtro adaptado)')

# Líneas verticales para marcar los latidos verdaderos
for t_qrs in t_ecg[qrs_true_segmento]:
    plt.axvline(x=t_qrs, color='red', linestyle='--', alpha=0.4)

# Líneas verticales para marcar los picos detectados
for t_peak in t_ecg[peaks_segmento]:
    plt.axvline(x=t_peak, color='blue', linestyle=':', alpha=0.4)

plt.title('ECG con detecciones y marcas de latidos')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()