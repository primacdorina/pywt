# Analisi Multi-Risoluzione (MRA) - Data Fusion per Scala

## 📋 Descrizione

Questo progetto implementa un'analisi multi-risoluzione (MRA) usando PyWavelets per il rilevamento di danni a diverse scale in immagini. La tecnica utilizza la decomposizione wavelet 2D per creare una rappresentazione RGB dove ogni canale colore rappresenta una scala di dettaglio diversa.

### Principio

L'idea è sostituire i tradizionali canali RGB con informazioni di scala wavelet:

- **R (Rosso)** = H1 + V1 + D1 → **Dettagli FINI** (crepe piccole, alte frequenze) - da `coeffs[1]`
- **G (Verde)** = H2 + V2 + D2 → **Dettagli MEDI** (danni medi, medie frequenze) - da `coeffs[2]`
- **B (Blu)** = H3 + V3 + D3 → **Dettagli GROSSI** (crolli, basse frequenze) - da `coeffs[3]`

Dove:
- **H** = dettagli orizzontali
- **V** = dettagli verticali
- **D** = dettagli diagonali
- **1, 2, 3** = livelli di decomposizione wavelet

**NOTA IMPORTANTE**: Nella trasformata wavelet stazionaria (SWT), ogni livello successivo cattura frequenze più basse:
- `coeffs[0]` = approssimazione (componente a bassa frequenza)
- `coeffs[1]` = **Livello 1** = dettagli FINI (alte frequenze)
- `coeffs[2]` = **Livello 2** = dettagli MEDI (medie frequenze)
- `coeffs[3]` = **Livello 3** = dettagli GROSSI (basse frequenze)

## 📁 File Forniti

1. **`mra_data_fusion_analysis.py`** - Script completo per processare dataset
2. **`example_mra_fusion.py`** - Esempio semplice e rapido
3. **`MRA_FUSION_README.md`** - Questa guida

## 🚀 Utilizzo

### Esempio Rapido

```python
python example_mra_fusion.py
```

Questo script usa l'immagine di esempio di PyWavelets e mostra i risultati.

### Processare il Tuo Dataset

```python
from mra_data_fusion_analysis import process_dataset

# Configura i path
data_folder = '/content/drive/MyDrive/Tesi_magistrale/Data/'
output_folder = '/content/drive/MyDrive/Tesi_magistrale/Data/mra_results/'

# Esegui l'analisi
rgb_fused = process_dataset(
    data_folder=data_folder,
    output_folder=output_folder,
    wavelet='db4',      # Tipo di wavelet
    level=3,            # Numero di livelli
    num_samples=10      # Quante immagini processare (None = tutte)
)
```

### Uso Programmatico

```python
import numpy as np
from mra_data_fusion_analysis import apply_mra_fusion

# Carica la tua immagine
image = np.load('your_image.npy')

# Applica MRA fusion
rgb_fused, mra_coeffs, detail_maps = apply_mra_fusion(
    image,
    wavelet='db4',
    level=3
)

# rgb_fused è l'immagine RGB risultante (H, W, 3)
# - rgb_fused[:,:,0] = canale R (dettagli fini)
# - rgb_fused[:,:,1] = canale G (dettagli medi)
# - rgb_fused[:,:,2] = canale B (dettagli grossi)
```

## 🔧 Parametri Configurabili

### Wavelet

Il tipo di wavelet influenza la qualità dell'analisi:

- **`'db4'`** (default) - Daubechies 4, buon compromesso generale
- **`'sym4'`** - Symlet 4, simmetrico, buono per bordi
- **`'coif3'`** - Coiflet 3, buona localizzazione tempo-frequenza
- **`'bior3.5'`** - Biorthogonal, buona per compressione
- **`'haar'`** - Più semplice, veloce ma meno preciso

### Level

Numero di livelli di decomposizione (default: 3):

- **level=2**: Solo 2 scale (fine e medio)
- **level=3**: 3 scale (fine, medio, grosso) ✓ consigliato
- **level=4**: 4 scale (più dettagliato ma più lento)

Il livello massimo dipende dalle dimensioni dell'immagine.

### Transform

- **`'swt2'`** (default) - Stationary Wavelet Transform, non decimato
  - ✓ Preserva la dimensione originale
  - ✓ Invariante alle traslazioni
  - ✓ Migliore per rilevamento features

- **`'dwt2'`** - Discrete Wavelet Transform, decimato
  - ✓ Più veloce
  - ✓ Meno ridondante
  - ✗ Riduce le dimensioni

## 📊 Output

### File Generati

Per ogni immagine processata:

1. **`mra_analysis_image_XXXX.png`** - Visualizzazione completa dell'analisi
   - Immagine originale
   - RGB fuso
   - Canali R, G, B separati
   - Dettagli H, V, D per ogni livello
   - Istogrammi

2. **`rgb_fused_image_XXXX.png`** - Solo l'immagine RGB fusa

3. **`all_rgb_fused.npy`** - Array NumPy con tutte le immagini RGB fuse

### Interpretazione

Nell'immagine RGB fusa:

- **Rosso intenso** → Presenza di dettagli fini (crepe, texture ad alta frequenza)
- **Verde intenso** → Presenza di dettagli medi (danni di dimensione media)
- **Blu intenso** → Presenza di dettagli grossi (grandi variazioni, crolli)
- **Bianco** → Presenza di dettagli a tutte le scale
- **Nero** → Assenza di dettagli (regioni omogenee)

## 💡 Esempio Completo

```python
import numpy as np
import matplotlib.pyplot as plt
from mra_data_fusion_analysis import apply_mra_fusion, visualize_results

# 1. Carica i tuoi dati
X = np.load('/path/to/X_train_augmented.npy')
print(f"Dataset shape: {X.shape}")

# 2. Processa una singola immagine
image = X[0]
rgb_fused, mra_coeffs, detail_maps = apply_mra_fusion(
    image,
    wavelet='db4',
    level=3
)

# 3. Visualizza i risultati
visualize_results(
    original_image=image,
    rgb_fused=rgb_fused,
    detail_maps=detail_maps,
    save_path='my_analysis.png'
)

# 4. Accedi ai dettagli per ogni scala
fine_details = detail_maps['level_1']['sum']     # R channel
medium_details = detail_maps['level_2']['sum']   # G channel
coarse_details = detail_maps['level_3']['sum']   # B channel

# 5. Analizza dove ci sono i danni
threshold = np.percentile(fine_details, 90)  # Top 10% dei dettagli
fine_damage_mask = fine_details > threshold

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(image, cmap='gray')
plt.title('Originale')
plt.subplot(122)
plt.imshow(fine_damage_mask, cmap='Reds')
plt.title('Maschera Crepe Fini')
plt.show()
```

## 📚 Riferimenti Teorici

La decomposizione MRA è basata su:

1. **Wavelet Transform 2D**: Decompone l'immagine in sotto-bande di frequenza
2. **Multi-Resolution Analysis**: Analizza l'immagine a diverse scale
3. **Additive Decomposition**: La somma di tutti i coefficienti ricostruisce l'originale

Formula matematica:
```
f(x,y) = A_n + Σ(H_j + V_j + D_j)  per j=1 to n
```

Dove:
- `f(x,y)` = immagine originale
- `A_n` = approssimazione al livello n
- `H_j, V_j, D_j` = dettagli orizzontali, verticali, diagonali al livello j

## 🔬 Applicazioni

Questa tecnica è particolarmente utile per:

- ✓ Rilevamento di crepe in strutture
- ✓ Analisi di danni in immagini di edifici
- ✓ Ispezione di superfici
- ✓ Segmentazione multi-scala
- ✓ Feature extraction per deep learning
- ✓ Preprocessing per modelli di classificazione danni

## 🐛 Troubleshooting

### Errore: "File not found"

Se il file `X_train_augmented.npy` non è trovato, lo script usa automaticamente dati di esempio.

### Warning: "Richiesti 3 livelli ma disponibili solo X"

L'immagine è troppo piccola per 3 livelli. Riduci il parametro `level`.

### Memoria insufficiente

Riduci `num_samples` o processa le immagini una alla volta.

### Risultati non chiari

Prova wavelet diversi (`'sym4'`, `'coif3'`) o regola i livelli.

## 📞 Supporto

Per problemi o domande:
- Controlla la documentazione PyWavelets: https://pywavelets.readthedocs.io/
- Vedi esempi in `pywt/tests/test_mra.py`

## 📄 Licenza

Questo codice usa PyWavelets che è distribuito sotto licenza MIT.

---

**Autore**: Generato per analisi tesi magistrale
**Data**: 2025
**Versione**: 1.0
