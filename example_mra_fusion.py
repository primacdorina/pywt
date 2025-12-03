#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Esempio Semplice di MRA Data Fusion
====================================

Script di esempio rapido per testare l'analisi MRA con una singola immagine.
"""

import numpy as np
import pywt
from pywt import data
import matplotlib.pyplot as plt


def mra_fusion_simple(image, wavelet='db4', level=3):
    """
    Applica MRA fusion in modo semplificato.

    Returns
    -------
    rgb_image : ndarray
        Immagine RGB con R=fine, G=medio, B=grosso
    """
    # Applica MRA
    coeffs = pywt.mra2(image, wavelet=wavelet, level=level,
                      transform='swt2', mode='periodization')

    # Estrai i dettagli
    # NOTA: mra2() restituisce i livelli dal più grossolano al più fine
    # coeffs[0] = approssimazione (non usata)
    # coeffs[1] = (H1, V1, D1) - dettagli GROSSI (basse frequenze)
    # coeffs[2] = (H2, V2, D2) - dettagli MEDI (medie frequenze)
    # coeffs[3] = (H3, V3, D3) - dettagli FINI (alte frequenze)

    # Somma dettagli per ogni livello (CORRETTO: coeffs[1]=GROSSO, coeffs[3]=FINE)
    H1, V1, D1 = coeffs[1]
    B = H1 + V1 + D1  # Canale Blu = dettagli GROSSI

    H2, V2, D2 = coeffs[2]
    G = H2 + V2 + D2  # Canale Verde = dettagli MEDI

    H3, V3, D3 = coeffs[3]
    R = H3 + V3 + D3  # Canale Rosso = dettagli FINI

    # Normalizza nel range [0, 255]
    def norm(x):
        x = np.abs(x)
        return ((x - x.min()) / (x.max() - x.min() + 1e-10) * 255).astype(np.uint8)

    rgb_image = np.stack([norm(R), norm(G), norm(B)], axis=-1)

    return rgb_image, (R, G, B)


# Esempio di utilizzo
if __name__ == '__main__':
    print("Caricamento immagine di esempio...")
    # Usa l'immagine camera di PyWavelets
    image = data.camera()
    print(f"Shape immagine: {image.shape}")

    print("\nApplicando MRA fusion...")
    rgb_fused, (R, G, B) = mra_fusion_simple(image, wavelet='db4', level=3)

    print(f"Shape RGB fuso: {rgb_fused.shape}")
    print(f"\nStatistiche canali:")
    print(f"  R (fine):   min={R.min():.2f}, max={R.max():.2f}, mean={R.mean():.2f}")
    print(f"  G (medio):  min={G.min():.2f}, max={G.max():.2f}, mean={G.mean():.2f}")
    print(f"  B (grosso): min={B.min():.2f}, max={B.max():.2f}, mean={B.mean():.2f}")

    # Visualizza
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Originale')
    axes[0].axis('off')

    axes[1].imshow(rgb_fused)
    axes[1].set_title('RGB Fuso')
    axes[1].axis('off')

    axes[2].imshow(rgb_fused[:, :, 0], cmap='Reds')
    axes[2].set_title('R: Dettagli FINI')
    axes[2].axis('off')

    axes[3].imshow(rgb_fused[:, :, 1], cmap='Greens')
    axes[3].set_title('G: Dettagli MEDI')
    axes[3].axis('off')

    axes[4].imshow(rgb_fused[:, :, 2], cmap='Blues')
    axes[4].set_title('B: Dettagli GROSSI')
    axes[4].axis('off')

    plt.suptitle('MRA Data Fusion - Esempio', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/tmp/mra_fusion_example.png', dpi=150, bbox_inches='tight')
    print("\nFigura salvata in: /tmp/mra_fusion_example.png")
    plt.show()

    print("\n✓ Esempio completato!")
