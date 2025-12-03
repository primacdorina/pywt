#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analisi Multi-Risoluzione (MRA) con Data Fusion per Scala
==========================================================

Questo script esegue un'analisi multi-risoluzione usando PyWavelets per
rilevare danni a diverse scale in immagini:
- R (Rosso) = H1 + V1 + D1 → Dettagli FINI (crepe piccole)
- G (Verde) = H2 + V2 + D2 → Dettagli MEDI (danni medi)
- B (Blu)   = H3 + V3 + D3 → Dettagli GROSSI (crolli)

Dove:
- H = dettagli orizzontali
- V = dettagli verticali
- D = dettagli diagonali
- 1, 2, 3 = livelli di scala wavelet
"""

import numpy as np
import pywt
import matplotlib.pyplot as plt
from pathlib import Path
import warnings


def normalize_to_uint8(data):
    """
    Normalizza un array nel range [0, 255] per visualizzazione.

    Parameters
    ----------
    data : ndarray
        Array da normalizzare

    Returns
    -------
    ndarray
        Array normalizzato in uint8
    """
    # Normalizza nel range [0, 1]
    data_min = np.min(data)
    data_max = np.max(data)

    if data_max - data_min > 1e-10:  # evita divisione per zero
        normalized = (data - data_min) / (data_max - data_min)
    else:
        normalized = np.zeros_like(data)

    # Converti in uint8 [0, 255]
    return (normalized * 255).astype(np.uint8)


def apply_mra_fusion(image, wavelet='db4', level=3):
    """
    Applica la decomposizione MRA e crea l'immagine RGB fusa.

    Parameters
    ----------
    image : ndarray
        Immagine in input (2D per grayscale)
    wavelet : str, optional
        Tipo di wavelet da usare (default: 'db4')
    level : int, optional
        Numero di livelli di decomposizione (default: 3)

    Returns
    -------
    rgb_fused : ndarray
        Immagine RGB con data fusion (H, W, 3)
    mra_coeffs : list
        Coefficienti MRA completi
    detail_maps : dict
        Dizionario con le mappe di dettaglio per ogni scala
    """
    # Assicurati che l'immagine sia 2D
    if image.ndim > 2:
        # Se è RGB, converti in grayscale
        if image.shape[-1] == 3:
            image = np.mean(image, axis=-1)
        else:
            image = image.squeeze()

    # Applica la decomposizione MRA 2D
    print(f"Applicando MRA con wavelet '{wavelet}' e {level} livelli...")
    mra_coeffs = pywt.mra2(image, wavelet=wavelet, level=level,
                           transform='swt2', mode='periodization')

    print(f"Decomposizione completata. Numero di livelli: {len(mra_coeffs)}")
    print(f"  - coeffs[0]: Approssimazione, shape = {mra_coeffs[0].shape}")

    # Verifica che abbiamo almeno 3 livelli di dettaglio
    num_detail_levels = len(mra_coeffs) - 1  # -1 perché il primo è l'approssimazione

    if num_detail_levels < 3:
        warnings.warn(
            f"Richiesti 3 livelli ma disponibili solo {num_detail_levels}. "
            f"Alcuni canali potrebbero essere vuoti."
        )

    # Inizializza i canali RGB
    height, width = image.shape
    R_channel = np.zeros((height, width), dtype=np.float64)
    G_channel = np.zeros((height, width), dtype=np.float64)
    B_channel = np.zeros((height, width), dtype=np.float64)

    # Dizionario per memorizzare le mappe di dettaglio
    # NOTA: mra2() restituisce i livelli dal più grossolano al più fine
    # coeffs[1] = dettagli GROSSI (basse frequenze)
    # coeffs[2] = dettagli MEDI (medie frequenze)
    # coeffs[3] = dettagli FINI (alte frequenze)
    detail_maps = {
        'fine': {'H': None, 'V': None, 'D': None, 'sum': None},       # coeffs[3]
        'medium': {'H': None, 'V': None, 'D': None, 'sum': None},     # coeffs[2]
        'coarse': {'H': None, 'V': None, 'D': None, 'sum': None}      # coeffs[1]
    }

    # Estrai e combina i dettagli per ogni livello
    # IMPORTANTE: L'ordine è dal GROSSOLANO al FINE
    for i in range(1, min(4, len(mra_coeffs))):  # livelli 1, 2, 3
        H, V, D = mra_coeffs[i]  # (Horizontal, Vertical, Diagonal)

        print(f"  - coeffs[{i}]: Livello {i}, shapes = H:{H.shape}, V:{V.shape}, D:{D.shape}")

        # Somma dei dettagli per questo livello
        detail_sum = H + V + D

        # Assegna ai canali RGB (CORRETTO: coeffs[1]=GROSSO, coeffs[3]=FINE)
        if i == 1:
            # Livello 1 → Canale B (dettagli GROSSI - crolli)
            B_channel = detail_sum
            detail_maps['coarse']['H'] = H
            detail_maps['coarse']['V'] = V
            detail_maps['coarse']['D'] = D
            detail_maps['coarse']['sum'] = detail_sum
            print(f"    → B channel (dettagli GROSSI): range [{B_channel.min():.3f}, {B_channel.max():.3f}]")
        elif i == 2:
            # Livello 2 → Canale G (dettagli MEDI - danni medi)
            G_channel = detail_sum
            detail_maps['medium']['H'] = H
            detail_maps['medium']['V'] = V
            detail_maps['medium']['D'] = D
            detail_maps['medium']['sum'] = detail_sum
            print(f"    → G channel (dettagli MEDI): range [{G_channel.min():.3f}, {G_channel.max():.3f}]")
        elif i == 3:
            # Livello 3 → Canale R (dettagli FINI - crepe piccole)
            R_channel = detail_sum
            detail_maps['fine']['H'] = H
            detail_maps['fine']['V'] = V
            detail_maps['fine']['D'] = D
            detail_maps['fine']['sum'] = detail_sum
            print(f"    → R channel (dettagli FINI): range [{R_channel.min():.3f}, {R_channel.max():.3f}]")

    # Normalizza ogni canale indipendentemente
    R_norm = normalize_to_uint8(np.abs(R_channel))
    G_norm = normalize_to_uint8(np.abs(G_channel))
    B_norm = normalize_to_uint8(np.abs(B_channel))

    # Crea l'immagine RGB fusa
    rgb_fused = np.stack([R_norm, G_norm, B_norm], axis=-1)

    print(f"\nImmagine RGB fusa creata con shape: {rgb_fused.shape}")

    return rgb_fused, mra_coeffs, detail_maps


def visualize_results(original_image, rgb_fused, detail_maps, save_path=None):
    """
    Visualizza i risultati dell'analisi MRA.

    Parameters
    ----------
    original_image : ndarray
        Immagine originale
    rgb_fused : ndarray
        Immagine RGB con data fusion
    detail_maps : dict
        Dizionario con le mappe di dettaglio
    save_path : str, optional
        Path dove salvare la figura
    """
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))

    # Riga 0: Immagine originale e RGB fuso
    axes[0, 0].imshow(original_image, cmap='gray')
    axes[0, 0].set_title('Immagine Originale', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(rgb_fused)
    axes[0, 1].set_title('RGB Fuso\n(R=Fine, G=Medio, B=Grosso)',
                        fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    # Canali separati
    axes[0, 2].imshow(rgb_fused[:, :, 0], cmap='Reds')
    axes[0, 2].set_title('R: Dettagli FINI\n(Crepe piccole)',
                        fontsize=11, fontweight='bold', color='red')
    axes[0, 2].axis('off')

    axes[0, 3].imshow(rgb_fused[:, :, 1], cmap='Greens')
    axes[0, 3].set_title('G: Dettagli MEDI\n(Danni medi)',
                        fontsize=11, fontweight='bold', color='green')
    axes[0, 3].axis('off')

    axes[0, 4].imshow(rgb_fused[:, :, 2], cmap='Blues')
    axes[0, 4].set_title('B: Dettagli GROSSI\n(Crolli)',
                        fontsize=11, fontweight='bold', color='blue')
    axes[0, 4].axis('off')

    # Righe 1-3: Dettagli per ogni livello (H, V, D, Somma)
    # ORDINE CORRETTO: Fine (R), Medio (G), Grosso (B)
    levels = ['fine', 'medium', 'coarse']
    level_names = ['FINE - coeffs[3] (R)', 'MEDIO - coeffs[2] (G)', 'GROSSO - coeffs[1] (B)']

    for row, (level, level_name) in enumerate(zip(levels, level_names), start=1):
        if detail_maps[level]['H'] is not None:
            # H - Orizzontale
            axes[row, 0].imshow(np.abs(detail_maps[level]['H']), cmap='gray')
            axes[row, 0].set_title(f'{level_name}\nH (Orizzontale)', fontsize=10)
            axes[row, 0].axis('off')

            # V - Verticale
            axes[row, 1].imshow(np.abs(detail_maps[level]['V']), cmap='gray')
            axes[row, 1].set_title(f'{level_name}\nV (Verticale)', fontsize=10)
            axes[row, 1].axis('off')

            # D - Diagonale
            axes[row, 2].imshow(np.abs(detail_maps[level]['D']), cmap='gray')
            axes[row, 2].set_title(f'{level_name}\nD (Diagonale)', fontsize=10)
            axes[row, 2].axis('off')

            # Somma H+V+D
            axes[row, 3].imshow(np.abs(detail_maps[level]['sum']), cmap='hot')
            axes[row, 3].set_title(f'{level_name}\nH+V+D (Somma)',
                                  fontsize=10, fontweight='bold')
            axes[row, 3].axis('off')

            # Istogramma
            axes[row, 4].hist(detail_maps[level]['sum'].flatten(), bins=50,
                            color='steelblue', alpha=0.7)
            axes[row, 4].set_title(f'{level_name}\nDistribuzione', fontsize=10)
            axes[row, 4].set_xlabel('Valore coefficiente')
            axes[row, 4].set_ylabel('Frequenza')
            axes[row, 4].grid(alpha=0.3)
        else:
            for col in range(5):
                axes[row, col].axis('off')
                axes[row, col].text(0.5, 0.5, 'N/A',
                                   ha='center', va='center', fontsize=14)

    plt.suptitle('Analisi Multi-Risoluzione (MRA) - Data Fusion per Scala',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nFigura salvata in: {save_path}")

    plt.show()


def process_dataset(data_folder, output_folder=None, wavelet='db4',
                   level=3, num_samples=5):
    """
    Processa un dataset di immagini applicando l'analisi MRA.

    Parameters
    ----------
    data_folder : str
        Path alla cartella con i dati
    output_folder : str, optional
        Path dove salvare i risultati
    wavelet : str, optional
        Tipo di wavelet (default: 'db4')
    level : int, optional
        Numero di livelli (default: 3)
    num_samples : int, optional
        Numero di campioni da processare (default: 5)
        Se None, processa tutto il dataset
    """
    # Carica il dataset
    data_path = Path(data_folder) / 'X_train_augmented.npy'
    print(f"Caricamento dataset da: {data_path}")

    try:
        X = np.load(data_path)
        print(f"Dataset caricato: shape = {X.shape}, dtype = {X.dtype}")
    except FileNotFoundError:
        print(f"\nERRORE: File non trovato: {data_path}")
        print("Utilizzo dati di esempio invece...")
        # Usa dati di esempio se il file non esiste
        from pywt import data
        X = np.array([data.camera() for _ in range(5)])
        print(f"Dati di esempio creati: shape = {X.shape}")

    # Crea cartella di output
    if output_folder is None:
        output_folder = Path(data_folder) / 'mra_results'
    else:
        output_folder = Path(output_folder)

    output_folder.mkdir(parents=True, exist_ok=True)
    print(f"Risultati saranno salvati in: {output_folder}")

    # Determina quanti campioni processare
    if num_samples is None:
        num_samples = len(X)
    else:
        num_samples = min(num_samples, len(X))

    print(f"\nProcessando {num_samples} immagini...")
    print("=" * 80)

    # Array per memorizzare tutte le immagini RGB fuse
    all_rgb_fused = []

    # Processa ogni immagine
    for idx in range(num_samples):
        print(f"\n[{idx+1}/{num_samples}] Processando immagine {idx}...")

        image = X[idx]

        # Applica MRA fusion
        rgb_fused, mra_coeffs, detail_maps = apply_mra_fusion(
            image, wavelet=wavelet, level=level
        )

        all_rgb_fused.append(rgb_fused)

        # Visualizza e salva i risultati
        save_path = output_folder / f'mra_analysis_image_{idx:04d}.png'
        visualize_results(image, rgb_fused, detail_maps, save_path=save_path)

        # Salva anche l'immagine RGB fusa singola
        rgb_path = output_folder / f'rgb_fused_image_{idx:04d}.png'
        plt.imsave(rgb_path, rgb_fused)
        print(f"Immagine RGB fusa salvata in: {rgb_path}")

    # Salva tutte le immagini RGB fuse come array numpy
    all_rgb_fused = np.array(all_rgb_fused)
    fused_array_path = output_folder / 'all_rgb_fused.npy'
    np.save(fused_array_path, all_rgb_fused)
    print(f"\n{'='*80}")
    print(f"Tutte le immagini RGB fuse salvate in: {fused_array_path}")
    print(f"Shape del dataset fuso: {all_rgb_fused.shape}")
    print(f"\nAnalisi completata! Risultati in: {output_folder}")

    return all_rgb_fused


def main():
    """
    Funzione principale per eseguire l'analisi.
    """
    # Configurazione
    data_folder = '/content/drive/MyDrive/Tesi_magistrale/Data/'
    output_folder = '/content/drive/MyDrive/Tesi_magistrale/Data/mra_results/'

    # Parametri wavelet
    wavelet = 'db4'  # Daubechies 4 - buona per analisi danni
    # Altri wavelet consigliati: 'sym4', 'coif3', 'bior3.5'
    level = 3  # 3 livelli per catturare fine, medio, grosso

    # Numero di campioni da processare (None = tutti)
    num_samples = 5

    print("=" * 80)
    print("ANALISI MULTI-RISOLUZIONE (MRA) - DATA FUSION PER SCALA")
    print("=" * 80)
    print(f"\nConfigurazione:")
    print(f"  - Wavelet: {wavelet}")
    print(f"  - Livelli: {level}")
    print(f"  - Data folder: {data_folder}")
    print(f"  - Output folder: {output_folder}")
    print(f"  - Campioni da processare: {num_samples if num_samples else 'tutti'}")
    print()

    # Esegui l'analisi
    rgb_fused_dataset = process_dataset(
        data_folder=data_folder,
        output_folder=output_folder,
        wavelet=wavelet,
        level=level,
        num_samples=num_samples
    )

    print("\n" + "=" * 80)
    print("ANALISI COMPLETATA CON SUCCESSO!")
    print("=" * 80)

    return rgb_fused_dataset


if __name__ == '__main__':
    # Esegui l'analisi
    rgb_fused_dataset = main()
