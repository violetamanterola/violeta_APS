# -*- coding: utf-8 -*-
"""
Created on Thu May 22 14:13:41 2025

@author: viole
"""

import numpy as np
import matplotlib.pyplot as plt

# Eje de frecuencia normalizada (de 0 a π)
w = np.linspace(0, np.pi, 1024)
z = np.exp(1j * w)

# Definición de filtros (transformadas Z evaluadas en e^{jω})
Hz = {
    'Filtro a': z**-3 + z**-2 + z**-1 + 1,
    'Filtro b': z**-4 + z**-3 + z**-2 + z**-1 + 1,
    'Filtro c': 1 - z**-1,
    'Filtro d': 1 - z**-2
}

# --- Respuesta de módulo (sin dB) ---
plt.figure(figsize=(10, 4))
for label, H in Hz.items():
    plt.plot(w / np.pi, np.abs(H), label=label)
plt.title('Respuesta en Magnitud |H(e^{jω})|')
plt.xlabel('Frecuencia normalizada (×π rad/muestra)')
plt.ylabel('Magnitud (valor absoluto)')
plt.grid()
plt.legend()
plt.tight_layout()

#%% ---Respuesta de módulo en dB---
plt.figure(figsize=(10, 4))
for label, H in Hz.items():
    plt.plot(w/np.pi, 20*np.log10(np.abs(H)), label=label)
plt.title('Respuesta en Magnitud |H(e^{jω})|')
plt.xlabel('Frecuencia normalizada (×π rad/muestra)')
plt.ylabel('Magnitud [dB]')
plt.grid()
plt.legend()
plt.tight_layout()
# Crear figura con 4 subplots (uno por filtro)
fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

# Recorrer y graficar cada filtro
# for i, (label, H) in enumerate(Hz.items()):
#     mag_db = 20 * np.log10(np.maximum(np.abs(H), 1e-8))  # evitar log(0)
#     axs[i].plot(w / np.pi, mag_db)
#     axs[i].set_title(f'{label} - Magnitud [dB]')
#     axs[i].set_ylabel('dB')
#     axs[i].grid()

# # Etiqueta común en el eje x
# axs[-1].set_xlabel('Frecuencia normalizada (×π rad/muestra)')

# plt.tight_layout()
# plt.show()


#%% Respuesta fase
plt.figure(figsize=(10, 4))
for label, H in Hz.items():
    plt.plot(w/np.pi, np.angle(H), label=label)
plt.title('Respuesta en Fase ∠H(e^{jω})')
plt.xlabel('Frecuencia normalizada (×π rad/muestra)')
plt.ylabel('Fase [rad]')
plt.grid()
plt.legend()
plt.tight_layout()

plt.show()

