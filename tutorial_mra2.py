"""
Tutorial Completo: Come usare e interpretare pywt.mra2()
=========================================================

Questo tutorial spiega in dettaglio:
1. Come applicare la funzione mra2()
2. Come leggere gli output
3. Cosa rappresentano i campioni
4. Come interpretare l'output parte per parte
"""

import numpy as np
import pywt
import matplotlib.pyplot as plt

# ============================================================================
# PARTE 1: CREARE UN'IMMAGINE DI TEST
# ============================================================================
print("=" * 70)
print("PARTE 1: CREARE UN'IMMAGINE DI TEST")
print("=" * 70)

# Creiamo un'immagine semplice 64x64 per capire meglio
# Usiamo una combinazione di pattern per vedere diversi dettagli
rows, cols = 64, 64
x, y = np.meshgrid(np.linspace(0, 10, cols), np.linspace(0, 10, rows))

# Immagine con diversi componenti di frequenza
image = np.sin(x) + 0.5 * np.sin(5 * y) + 0.3 * np.random.randn(rows, cols)

print(f"Dimensioni immagine: {image.shape}")
print(f"Valore minimo: {image.min():.4f}")
print(f"Valore massimo: {image.max():.4f}")
print(f"Media: {image.mean():.4f}")
print()

# ============================================================================
# PARTE 2: APPLICARE pywt.mra2() CON PARAMETRI SPECIFICATI
# ============================================================================
print("=" * 70)
print("PARTE 2: APPLICARE pywt.mra2()")
print("=" * 70)

# Parametri richiesti
wavelet = 'db2'  # Daubechies 2
level = 3        # 3 livelli di decomposizione

print(f"Wavelet: {wavelet}")
print(f"Level: {level}")
print()

# Applicare la MRA2 (Multiresolution Analysis 2D)
coeffs_mra = pywt.mra2(image, wavelet=wavelet, level=level)

print("Decomposizione completata!")
print()

# ============================================================================
# PARTE 3: STRUTTURA DELL'OUTPUT - CAPIRE COSA VIENE RESTITUITO
# ============================================================================
print("=" * 70)
print("PARTE 3: STRUTTURA DELL'OUTPUT")
print("=" * 70)

print(f"Tipo dell'output: {type(coeffs_mra)}")
print(f"Numero di elementi nella lista: {len(coeffs_mra)}")
print()

print("IMPORTANTE: L'output è una lista con struttura specifica:")
print("-" * 70)
print(f"Numero totale di livelli: {level}")
print(f"Numero di elementi nella lista: {len(coeffs_mra)} = 1 (approssimazione) + {level} (livelli di dettaglio)")
print()

# ============================================================================
# PARTE 4: ANALISI DETTAGLIATA ELEMENTO PER ELEMENTO
# ============================================================================
print("=" * 70)
print("PARTE 4: ANALISI DETTAGLIATA ELEMENTO PER ELEMENTO")
print("=" * 70)

print("\n>>> ELEMENTO 0: APPROSSIMAZIONE (cA) AL LIVELLO PIÙ BASSO <<<")
print("-" * 70)
print(f"Tipo: {type(coeffs_mra[0])}")
print(f"Shape: {coeffs_mra[0].shape}")
print(f"Dtype: {coeffs_mra[0].dtype}")
print()
print("SIGNIFICATO:")
print("  - Questo è il componente di APPROSSIMAZIONE al livello {level}")
print("  - Rappresenta le frequenze BASSE dell'immagine")
print("  - È la versione 'sfocata' dell'immagine originale")
print("  - Cattura la struttura generale senza i dettagli fini")
print(f"  - Ha la stessa dimensione dell'immagine originale: {coeffs_mra[0].shape}")
print()

# Analizzare i dettagli per ogni livello
for i in range(1, len(coeffs_mra)):
    livello = level - i + 1  # I livelli sono numerati al contrario
    print(f"\n>>> ELEMENTO {i}: DETTAGLI AL LIVELLO {livello} <<<")
    print("-" * 70)

    # coeffs_mra[i] è una tupla con 3 elementi
    print(f"Tipo: {type(coeffs_mra[i])}")
    print(f"Numero di componenti: {len(coeffs_mra[i])}")
    print()

    # Ogni tupla contiene 3 array: (cH, cV, cD)
    cH, cV, cD = coeffs_mra[i]

    print(f"  [0] Dettagli ORIZZONTALI (cH{livello}):")
    print(f"      - Shape: {cH.shape}")
    print(f"      - Cattura i bordi e le variazioni ORIZZONTALI")
    print(f"      - Mostra dove l'immagine cambia da sinistra a destra")
    print()

    print(f"  [1] Dettagli VERTICALI (cV{livello}):")
    print(f"      - Shape: {cV.shape}")
    print(f"      - Cattura i bordi e le variazioni VERTICALI")
    print(f"      - Mostra dove l'immagine cambia dall'alto verso il basso")
    print()

    print(f"  [2] Dettagli DIAGONALI (cD{livello}):")
    print(f"      - Shape: {cD.shape}")
    print(f"      - Cattura i bordi e le variazioni DIAGONALI")
    print(f"      - Mostra le variazioni in entrambe le direzioni contemporaneamente")
    print()

    print(f"  FREQUENZE AL LIVELLO {livello}:")
    if livello == 3:
        print(f"      - Livello più basso: frequenze medio-basse")
        print(f"      - Scale più grandi, dettagli più grossolani")
    elif livello == 2:
        print(f"      - Livello intermedio: frequenze medie")
        print(f"      - Scale intermedie")
    elif livello == 1:
        print(f"      - Livello più alto: frequenze alte")
        print(f"      - Scale più piccole, dettagli più fini")
    print()

# ============================================================================
# PARTE 5: SCHEMA VISIVO DELLA STRUTTURA
# ============================================================================
print("=" * 70)
print("PARTE 5: SCHEMA VISIVO DELLA STRUTTURA")
print("=" * 70)
print()
print("coeffs_mra = [")
print(f"    coeffs_mra[0] = cA{level},                    # Array {coeffs_mra[0].shape}")
print(f"    coeffs_mra[1] = (cH{level}, cV{level}, cD{level}),    # Tupla di 3 array")
print(f"    coeffs_mra[2] = (cH{level-1}, cV{level-1}, cD{level-1}),    # Tupla di 3 array")
print(f"    coeffs_mra[3] = (cH{level-2}, cV{level-2}, cD{level-2})     # Tupla di 3 array")
print("]")
print()
print("Dove:")
print(f"  cA{level}  = Approssimazione al livello {level} (frequenze BASSE)")
print(f"  cH{level}  = Dettagli Orizzontali al livello {level}")
print(f"  cV{level}  = Dettagli Verticali al livello {level}")
print(f"  cD{level}  = Dettagli Diagonali al livello {level}")
print("  ... e così via per gli altri livelli")
print()

# ============================================================================
# PARTE 6: COME ACCEDERE AI DATI - ESEMPI PRATICI
# ============================================================================
print("=" * 70)
print("PARTE 6: COME ACCEDERE AI DATI - ESEMPI PRATICI")
print("=" * 70)
print()

# Accesso all'approssimazione
print("# Per accedere all'approssimazione (frequenze basse):")
print("approssimazione = coeffs_mra[0]")
approssimazione = coeffs_mra[0]
print(f"  Shape: {approssimazione.shape}")
print(f"  Range: [{approssimazione.min():.4f}, {approssimazione.max():.4f}]")
print()

# Accesso ai dettagli livello 3
print(f"# Per accedere ai dettagli al livello {level}:")
print("cH3, cV3, cD3 = coeffs_mra[1]")
cH3, cV3, cD3 = coeffs_mra[1]
print(f"  cH{level} shape: {cH3.shape}")
print(f"  cV{level} shape: {cV3.shape}")
print(f"  cD{level} shape: {cD3.shape}")
print()

# Accesso ai dettagli livello 2
print(f"# Per accedere ai dettagli al livello {level-1}:")
print("cH2, cV2, cD2 = coeffs_mra[2]")
cH2, cV2, cD2 = coeffs_mra[2]
print(f"  cH{level-1} shape: {cH2.shape}")
print()

# Accesso ai dettagli livello 1
print(f"# Per accedere ai dettagli al livello {level-2}:")
print("cH1, cV1, cD1 = coeffs_mra[3]")
cH1, cV1, cD1 = coeffs_mra[3]
print(f"  cH{level-2} shape: {cH1.shape}")
print()

# Accesso a un singolo dettaglio
print("# Per accedere solo ai dettagli orizzontali del livello 2:")
print("dettagli_orizzontali_lv2 = coeffs_mra[2][0]")
dettagli_orizzontali_lv2 = coeffs_mra[2][0]
print(f"  Shape: {dettagli_orizzontali_lv2.shape}")
print()

# ============================================================================
# PARTE 7: INTERPRETAZIONE DEI VALORI
# ============================================================================
print("=" * 70)
print("PARTE 7: INTERPRETAZIONE DEI VALORI")
print("=" * 70)
print()

print("COME LEGGERE I COEFFICIENTI:")
print("-" * 70)
print()
print("1. APPROSSIMAZIONE (cA):")
print("   - Valori vicini alla media dell'immagine originale")
print("   - Rappresenta il 'contenuto generale' dell'immagine")
print("   - Valori alti = zone chiare, valori bassi = zone scure")
print()

print("2. DETTAGLI ORIZZONTALI (cH):")
print("   - Valori vicini a 0 = nessuna variazione orizzontale")
print("   - Valori positivi/negativi = bordi orizzontali")
print("   - Grandi valori assoluti = forti cambiamenti orizzontali")
print()

print("3. DETTAGLI VERTICALI (cV):")
print("   - Valori vicini a 0 = nessuna variazione verticale")
print("   - Valori positivi/negativi = bordi verticali")
print("   - Grandi valori assoluti = forti cambiamenti verticali")
print()

print("4. DETTAGLI DIAGONALI (cD):")
print("   - Valori vicini a 0 = nessuna variazione diagonale")
print("   - Valori positivi/negativi = bordi diagonali")
print("   - Grandi valori assoluti = forti cambiamenti diagonali")
print()

# Esempi numerici
print("ESEMPI DI VALORI:")
print("-" * 70)
print(f"\nApprossimazione (cA{level}):")
print(f"  Valore nel pixel (10, 10): {approssimazione[10, 10]:.4f}")
print(f"  Media: {approssimazione.mean():.4f}")
print(f"  Deviazione standard: {approssimazione.std():.4f}")
print()

print(f"Dettagli Orizzontali (cH{level}):")
print(f"  Valore nel pixel (10, 10): {cH3[10, 10]:.4f}")
print(f"  Media: {cH3.mean():.6f} (dovrebbe essere vicina a 0)")
print(f"  Max valore assoluto: {np.abs(cH3).max():.4f}")
print()

# ============================================================================
# PARTE 8: PROPRIETÀ IMPORTANTE - RICOSTRUZIONE
# ============================================================================
print("=" * 70)
print("PARTE 8: PROPRIETÀ IMPORTANTE - RICOSTRUZIONE")
print("=" * 70)
print()

print("PROPRIETÀ ADDITIVA:")
print("-" * 70)
print("Una caratteristica fondamentale di mra2() è che la somma di")
print("tutti i coefficienti restituisce l'immagine originale!")
print()

# Ricostruzione usando imra2
image_ricostruita = pywt.imra2(coeffs_mra)

# Verifica
errore = np.abs(image - image_ricostruita).max()
print(f"Errore massimo di ricostruzione: {errore:.2e}")
print(f"Le immagini sono identiche: {np.allclose(image, image_ricostruita)}")
print()

print("FORMULA DI RICOSTRUZIONE:")
print("-" * 70)
print("immagine_originale = cA3 + cH3 + cV3 + cD3 + cH2 + cV2 + cD2 + cH1 + cV1 + cD1")
print()
print("In Python:")
print("image_ricostruita = coeffs_mra[0]  # cA3")
print("for i in range(1, len(coeffs_mra)):")
print("    for detail in coeffs_mra[i]:  # (cH, cV, cD)")
print("        image_ricostruita += detail")
print()

# Verifica manuale
image_ricostruita_manuale = coeffs_mra[0].copy()
for i in range(1, len(coeffs_mra)):
    for detail in coeffs_mra[i]:
        image_ricostruita_manuale += detail

errore_manuale = np.abs(image - image_ricostruita_manuale).max()
print(f"Errore con ricostruzione manuale: {errore_manuale:.2e}")
print()

# ============================================================================
# PARTE 9: VISUALIZZAZIONE
# ============================================================================
print("=" * 70)
print("PARTE 9: VISUALIZZAZIONE")
print("=" * 70)
print()

print("Creazione di visualizzazioni per capire meglio...")
print()

# Creare una figura con tutte le componenti
fig, axes = plt.subplots(4, 4, figsize=(16, 16))
fig.suptitle('Analisi Multiresolution 2D (MRA2) - db2, level=3', fontsize=16, fontweight='bold')

# Immagine originale
axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title('Immagine Originale', fontweight='bold')
axes[0, 0].axis('off')

# Approssimazione
axes[0, 1].imshow(coeffs_mra[0], cmap='gray')
axes[0, 1].set_title(f'cA{level}\n(Approssimazione)', fontweight='bold')
axes[0, 1].axis('off')

# Ricostruzione
axes[0, 2].imshow(image_ricostruita, cmap='gray')
axes[0, 2].set_title('Ricostruzione', fontweight='bold')
axes[0, 2].axis('off')

# Errore di ricostruzione
errore_vis = image - image_ricostruita
im_err = axes[0, 3].imshow(errore_vis, cmap='seismic',
                           vmin=-np.abs(errore_vis).max(),
                           vmax=np.abs(errore_vis).max())
axes[0, 3].set_title(f'Errore\n(max={errore:.2e})', fontweight='bold')
axes[0, 3].axis('off')
plt.colorbar(im_err, ax=axes[0, 3], fraction=0.046)

# Dettagli per ogni livello
livelli_info = [
    (1, 3, 'Livello 3\n(frequenze medio-basse)'),
    (2, 2, 'Livello 2\n(frequenze medie)'),
    (3, 1, 'Livello 1\n(frequenze alte)')
]

for row_idx, (idx, livello_num, titolo) in enumerate(livelli_info, start=1):
    cH, cV, cD = coeffs_mra[idx]

    # Etichetta della riga
    axes[row_idx, 0].text(0.5, 0.5, titolo,
                          ha='center', va='center',
                          fontsize=12, fontweight='bold',
                          transform=axes[row_idx, 0].transAxes)
    axes[row_idx, 0].axis('off')

    # Dettagli Orizzontali
    vmax_h = np.abs(cH).max()
    im_h = axes[row_idx, 1].imshow(cH, cmap='seismic', vmin=-vmax_h, vmax=vmax_h)
    axes[row_idx, 1].set_title(f'cH{livello_num}\n(Orizzontali)')
    axes[row_idx, 1].axis('off')
    plt.colorbar(im_h, ax=axes[row_idx, 1], fraction=0.046)

    # Dettagli Verticali
    vmax_v = np.abs(cV).max()
    im_v = axes[row_idx, 2].imshow(cV, cmap='seismic', vmin=-vmax_v, vmax=vmax_v)
    axes[row_idx, 2].set_title(f'cV{livello_num}\n(Verticali)')
    axes[row_idx, 2].axis('off')
    plt.colorbar(im_v, ax=axes[row_idx, 2], fraction=0.046)

    # Dettagli Diagonali
    vmax_d = np.abs(cD).max()
    im_d = axes[row_idx, 3].imshow(cD, cmap='seismic', vmin=-vmax_d, vmax=vmax_d)
    axes[row_idx, 3].set_title(f'cD{livello_num}\n(Diagonali)')
    axes[row_idx, 3].axis('off')
    plt.colorbar(im_d, ax=axes[row_idx, 3], fraction=0.046)

plt.tight_layout()
plt.savefig('/home/user/pywt/mra2_tutorial_visualization.png', dpi=150, bbox_inches='tight')
print("Visualizzazione salvata come 'mra2_tutorial_visualization.png'")
print()

# ============================================================================
# PARTE 10: RIEPILOGO E BEST PRACTICES
# ============================================================================
print("=" * 70)
print("PARTE 10: RIEPILOGO E BEST PRACTICES")
print("=" * 70)
print()

print("RIEPILOGO:")
print("-" * 70)
print(f"1. mra2() restituisce una lista con {1 + level} elementi")
print(f"2. Elemento 0: approssimazione (cA{level})")
print(f"3. Elementi 1-{level}: tuple con dettagli (cH, cV, cD) per ogni livello")
print("4. Tutti gli array hanno la stessa dimensione dell'input")
print("5. La somma di tutti i coefficienti = immagine originale")
print()

print("COME ACCEDERE AI DATI:")
print("-" * 70)
print("# Approssimazione")
print("cA = coeffs_mra[0]")
print()
print("# Dettagli livello 3 (dal più basso)")
print("cH3, cV3, cD3 = coeffs_mra[1]")
print()
print("# Dettagli livello 2")
print("cH2, cV2, cD2 = coeffs_mra[2]")
print()
print("# Dettagli livello 1 (il più alto)")
print("cH1, cV1, cD1 = coeffs_mra[3]")
print()
print("# Oppure in un loop:")
print("for i in range(1, len(coeffs_mra)):")
print("    livello = level - i + 1")
print("    cH, cV, cD = coeffs_mra[i]")
print("    print(f'Livello {livello}: cH, cV, cD')")
print()

print("INTERPRETAZIONE:")
print("-" * 70)
print("• cA  = contenuto a bassa frequenza (struttura generale)")
print("• cH  = bordi e variazioni orizzontali")
print("• cV  = bordi e variazioni verticali")
print("• cD  = bordi e variazioni diagonali")
print("• Livelli bassi (3) = dettagli grossolani")
print("• Livelli alti (1) = dettagli fini")
print()

print("USI COMUNI:")
print("-" * 70)
print("1. Compressione: eliminare dettagli con valori piccoli")
print("2. Denoising: rimuovere rumore dai dettagli ad alta frequenza")
print("3. Feature extraction: usare coefficienti come features")
print("4. Analisi multi-scala: studiare pattern a diverse scale")
print()

print("=" * 70)
print("TUTORIAL COMPLETATO!")
print("=" * 70)
print()
print(f"File di visualizzazione salvato: mra2_tutorial_visualization.png")
print()
