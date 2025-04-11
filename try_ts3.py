# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 20:38:57 2025

@author: viole
"""

import numpy as np
import matplotlib.pyplot as plt

def my_funcion_sen(vmax=1, dc=0, ff=1, ph=0, nn=1000, fs=1000):
    ts = 1/fs
    t_simulacion = nn * ts
    tt = np.arange(start=0, stop=t_simulacion, step=ts)
    xx = dc + vmax * np.sin(2 * np.pi * ff * tt + ph)
    return tt, xx
def simular_adc(B=4, kn=1.0, Vf=2.0, N=1000, fs=1000,aliasing=False): #armo una función para poder así llamarla y variar el numero de bits y kn.
    #paso de cuantización
    q = Vf / 2**(B - 1)
    df = fs / N
    fa = 0.75*fs if aliasing else df #para incluir luego el caso con aliasing. 
    #señal sinusoidal sin ruido 
    tt, xx = my_funcion_sen(vmax=1, dc=0, ff=fa, ph=0, nn=N, fs=fs)
    xn = xx / np.std(xx)  # normalización para una varianza = 1

    #ruido analógico
    pot_ruido = (q**(2) / 12) * kn
    nn = np.random.normal(0, np.sqrt(pot_ruido), size=N) #genera señal ruido analógico gaussiano.
    # señal con ruido (aditivo)
    sr = xn + nn

    #cuantización de la señal 
    srq = np.round(sr / q) * q

    #ruido
    nq = srq - sr
    #nn = na

    #FFTs
    ft_SR = 1/N * np.fft.fft(sr)
    ft_Srq = 1/N * np.fft.fft(srq)
    ft_xn = 1/N * np.fft.fft(xn)
    ft_Nq = 1/N * np.fft.fft(nq)
    ft_Nn = 1/N * np.fft.fft(nn)

    #frecuencias
    ff = np.linspace(0, (N-1)*df, N)
    bfrec = ff <= fs/2

    #potencias promedio
    nNn_mean = np.mean(np.abs(ft_Nn)**2)
    Nnq_mean = np.mean(np.abs(ft_Nq)**2)

    # Gráfico temporal
    if aliasing:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(tt, xn, color='orange', label='$s$ (analógica)')
        plt.xlim(0, 0.4)
        plt.title('Señal analógica original (sin ruido, sin cuantizar)')
        plt.xlabel('Tiempo [s]')
        plt.ylabel('Amplitud [V]')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(tt, srq, linestyle='-', color='blue',alpha=0.7,label='$s_Q$ (ADC out)')
        plt.plot(tt, sr, 'go:', markersize=2, label='$s_R = s + n$ (ADC in)')
        plt.title(f'Señal ADC - {B} bits, ±$V_R$={Vf} V, q={q:.3f} V')
        plt.xlim(0, 0.4)
        plt.xlabel('Tiempo [s]')
        plt.ylabel('Amplitud [V]')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()    
    elif (B == 8 and kn == 1) or (B == 16 and kn == 10):
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(tt, srq, linestyle='-', color='blue',alpha=0.7,label='$s_Q$ (ADC out)')
            plt.plot(tt, sr, 'go:', markersize=2, alpha=0.7,label='$s_R = s + n$ (ADC in)')
            plt.plot(tt, xn, linestyle=':', color='orange',alpha=0.7, label='$s$ (analógica)')
            plt.xlabel('Tiempo [s]')
            plt.ylabel('Amplitud [V]')
            plt.legend()
            plt.grid(True)
    
            plt.subplot(1, 2, 2)
            plt.plot(tt, srq, linestyle='-', color='blue', label='$s_Q$ (ADC out)')
            plt.plot(tt, sr, 'go:', markersize=2, label='$s_R = s + n$ (ADC in)')
            plt.plot(tt, xn, linestyle=':', color='orange', label='$s$ (analógica)')
            plt.xlim(0, 0.06)
            plt.ylim(0, 0.6)
            plt.title('Zoom (0 - 0.06 s)')
            plt.xlabel('Tiempo [s]')
            plt.ylabel('Amplitud [V]')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
    else:
            plt.figure(figsize=(10, 6))
            #Señal cuantizada s_Q, sale cuantizada del ADC (out)
            plt.plot(tt, srq, linestyle='-', color='blue',alpha=0.7,label='$s_Q$ (ADC out)')
            #Señal con ruido s_R, es la que entra al ADC (in)
            plt.plot(tt, sr, 'go:',alpha=0.7, markersize=2, label='$s_R = s + n$ (ADC in)')
            #Señal analógica s
            plt.plot(tt, xn, linestyle=':', color='orange',alpha=0.7, label='$s$ (analógica)')
            plt.title(f'Señal ADC - {B} bits, ±$V_R$={Vf} V, q={q:.3f} V')
            plt.xlabel('Tiempo [s]')
            plt.ylabel('Amplitud [V]')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        
    # Gráfico espectral
    #hago esto para q se aprecie mejor en los gráficos la difrencia entre los pisos de ruido. 
    piso_analog_db = 10 * np.log10(2 * nNn_mean)
    piso_digital_db = 10 * np.log10(2 * Nnq_mean)
    piso_min = min(piso_analog_db, piso_digital_db)
    #margen (-50 dB más abajo del piso más bajo)
    margen_db = 50
    ymin = piso_min - margen_db
    plt.figure(figsize=(15, 8))
    plt.ylim(ymin, 1)
    plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Srq[bfrec])**2), lw=2, color='magenta',label='$ s_Q = Q_{B,V_F}\{s_R\}$ (ADC out)' )
    plt.plot(ff[bfrec], 10 * np.log10(2 * np.abs(ft_xn[bfrec])**2), ':b', label='$s$ (analógica)')
    plt.plot(ff[bfrec], 10 * np.log10(2 * np.abs(ft_SR[bfrec])**2), ':g', label='$s_R = s + n$ (ADC in)')
    plt.plot(ff[bfrec], 10 * np.log10(2 * np.abs(ft_Nn[bfrec])**2), ':r')
    plt.plot(ff[bfrec], 10 * np.log10(2 * np.abs(ft_Nq[bfrec])**2), ':c')
    plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([nNn_mean, nNn_mean]) ), '--r', label= '$ \overline{n} = $' + '{:3.1f} dB (piso analog.)'.format(10* np.log10(2* nNn_mean)) )
    plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([Nnq_mean, Nnq_mean]) ), '--c', label='$ \overline{n_Q} = $' + '{:3.1f} dB (piso digital)'.format(10* np.log10(2* Nnq_mean)) )
    plt.title('Espectro de señales y ruidos')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Densidad de Potencia [dB]')
    plt.legend(fontsize=14, loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Histograma del ruido de cuantización
    # con esto puedo ver cuantas veces el rudio de cuantixaci{on cae en ciertos rangos. 
    plt.figure(figsize=(6, 4))
    #plt.figure(3)
    bins = 10
    plt.hist(nq.flatten(), bins=bins)
    plt.plot( np.array([-q/2, -q/2, q/2, q/2]), np.array([0, N/bins, N/bins, 0]), '--r' )
    plt.title( 'Ruido de cuantización para {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q))
    plt.xlabel('Error de cuantización [V]')
    plt.ylabel('Frecuencia relativa')
    plt.show()

simular_adc(B=4, kn=1, Vf=2, fs=1000, N=1000,aliasing=False)
simular_adc(B=4, kn=1, Vf=2, fs=1000, N=1000,aliasing=True)
