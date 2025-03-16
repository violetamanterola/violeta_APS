# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 15:27:25 2025

@author: Violeta Manterola
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def my_funcion_sen( vmax =1, dc=0, ff=1, ph=0, nn=1000, fs=1000 ):
    #donde ff es la frecuencia de la señal. 
  
    ts = 1/fs # tiempo de muestreo
    #df = fs/N # resolución espectral
    t_simulacion = nn * ts # segundo

    # grilla de sampleo temporal
    tt = np.arange(start = 0, stop = t_simulacion, step = ts)
    xx = dc + vmax * np.sin( 2 * np.pi * ff * tt + ph )#aplico la funcion numpysin
    
    return tt,xx


# Presentación gráfica de la señal (es la msima para todas las frecuencias)
def configuracion_plot(xx,tt,frec):
    plt.figure(1)
    plt.plot(tt, xx, label=f'{frec} Hz')
    #plt.title(f'Señal de {frec} Hz')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud [V]')
    plt.legend(loc='upper right')
    

# ejemplo con 500 Hz la señal. 
N = 1000
fs = 1000  # frecuencia de muestreo en Hz

frec1 = 999
tt1, xx1 = my_funcion_sen(vmax=1, dc=0, ff=frec1, ph=0, nn=N, fs=fs)

frec2 = 1001
tt2, xx2 = my_funcion_sen(vmax=1, dc=0, ff=frec2, ph=0, nn=N, fs=fs)

frec3 = 2001
tt3, xx3 = my_funcion_sen(vmax=1, dc=0, ff=frec3, ph=0, nn=N, fs=fs)

frec4 = 500
tt4, xx4 = my_funcion_sen(vmax=1, dc=0, ff=frec4, ph=0, nn=N, fs=fs)

# configuracion_plot(xx1,tt1,frec1)
# configuracion_plot(xx2,tt2,frec2)
# configuracion_plot(xx3,tt3,frec3)
# configuracion_plot(xx4,tt4,frec4)
# plt.show()

#ahora con alguna otra señal propia de un generador de señales, como una CUADRADA
def my_funcion_cuad(vmax =1, dc=0, ff=1, ph=0, nn=1000, fs=1000):
    ts = 1/fs # tiempo de muestreo
    t_simulacion = nn * ts # segundo

    # grilla de sampleo temporal
    tt = np.arange(start = 0, stop = t_simulacion, step = ts)
    xx = dc + vmax * signal.square(2 * np.pi * ff * tt + ph)
    
    return tt,xx
N = 1000
fs = 1000  # frecuencia de muestreo en Hz
frec1 = 999
tt1, xx1 = my_funcion_cuad(vmax=1, dc=0, ff=frec1, ph=0, nn=N, fs=fs)

frec2 = 1001
tt2,xx2 = my_funcion_cuad(vmax=1, dc=0, ff=frec2, ph=0, nn=N, fs=fs)

frec3 = 2001
tt3,xx3 = my_funcion_cuad(vmax=1, dc=0, ff=frec3, ph=0, nn=N, fs=fs)

frec4 = 500
tt4,xx4 = my_funcion_cuad(vmax=1, dc=0, ff=frec4, ph=0, nn=N, fs=fs)

configuracion_plot(xx1,tt1,frec1)
configuracion_plot(xx2,tt2,frec2)
configuracion_plot(xx3,tt3,frec3)
configuracion_plot(xx4,tt4,frec4)
plt.show()
