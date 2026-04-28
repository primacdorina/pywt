"""
================================================================================
DEMO FDD - Frequency Domain Decomposition per Operational Modal Analysis (OMA)
================================================================================

DESCRIZIONE GENERALE:
La FDD (Frequency Domain Decomposition) è una tecnica di analisi modale che
estrae frequenze proprie e forme modali da segnali misurati (accelerazioni)
in condizioni di esercizio (output-only). Non serve la forzante applicata.

PRINCIPIO DI FUNZIONAMENTO:
1. Si calcolano le densità spettrali di potenza incrociate (CPSD) tra tutte
   le coppie sensore-sensore, costruendo una matrice S_xx(f) 3D.
2. A ogni frequenza f_k, la matrice S_xx[k, :, :] ha forma (n_sensori x n_sensori).
3. Si esegue la Decomposizione ai Valori Singolari (SVD) su ogni matrice.
4. I modi della struttura causano picchi nei valori singolari: il primo SV
   ha un "boom" in corrispondenza delle frequenze proprie.
5. Dal vettore singolare sinistro si estrae la forma modale (spatial pattern).

SIMULAZIONE IN QUESTO DEMO:
- Generiamo artificialmente 3 sinusoidi a frequenze diverse (2.5, 7.8, 15.3 Hz).
- Le combiniamo su 7 sensori via "forme modali finte" (come se fossero oscillazioni
  di una struttura reale con quegli n modi propri).
- Aggiungiamo rumore gaussiano bianco.
- Sappiamo GIA' quali frequenze aspettarci: la FDD dovrebbe trovarle.
- Nessun salvataggio file: solo grafici diretti.

STEP DELLA PIPELINE:
A) Generare il segnale simulato multi-canale (7 sensori x 600 s a 200 Hz)
B) Calcolare S_xx (matrice CPSD 3D) via metodo di Welch (scipy.signal.csd)
C) SVD frequenza per frequenza
D) Peak picking sul primo valore singolare
E) Estrazione delle forme modali
F) Plot dei risultati
================================================================================
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

print("\n" + "="*80)
print(" FDD - FREQUENCY DOMAIN DECOMPOSITION DEMO")
print("="*80 + "\n")


# ==============================================================================
# STEP 0: PARAMETRI DI CONFIGURAZIONE
# ==============================================================================
print("[STEP 0] Parametri di configurazione\n")

# Frequenza di campionamento [Hz].
# Rappresenta quanti campioni per secondo vengono acquisiti dai sensori.
# Con FS_ACCEL=200 Hz possiamo rappresentare frequenze fino a FS_ACCEL/2 = 100 Hz
# (limite di Nyquist). Qualsiasi frequenza > 100 Hz nel nostro segnale
# verrà aliasata (ripiegata).
FS_ACCEL = 200

# Lunghezza della "finestra" temporale da analizzare [secondi].
# In un caso reale: "analizza gli ultimi 10 minuti di vibrazioni".
# Qui generiamo esattamente questo intervallo.
FDD_DURATION_SEC = 600

# Ogni quanto (idealmente) rieseguiremmo la FDD [secondi].
# In questo demo è solo informativo: facciamo una sola analisi.
FDD_INTERVAL_SEC = 600

# Lunghezza del segmento per il metodo di Welch [campioni].
# Il metodo di Welch divide i dati in segmenti, applica una finestra (Hann),
# fa la FFT su ciascuno e media i risultati.
# - nperseg grande -> buona risoluzione frequenziale, meno mediazioni.
# - nperseg piccolo -> molte mediazioni (riduce il rumore), risoluzione scarsa.
# Risoluzione frequenziale: Δf = FS_ACCEL / nperseg = 200 / 1024 ≈ 0.195 Hz.
nperseg = 1024

# Numero di sensori (canali) che simuliamo.
# Ogni sensore misura le vibrazioni in un punto della struttura.
# La CPSD si calcola tra TUTTE le coppie (i, j) con i,j ∈ [0, n_sensori-1].
n_sensori = 7

# Numero di modi propri che vogliamo identificare.
# Nel nostro segnale sintetico sappiamo che ce ne sono esattamente 3.
# L'algoritmo cercherà i top n_modes picchi nel primo valore singolare.
n_modes = 3

# Frequenze [Hz] delle sinusoidi che generiamo.
# Sappiamo in anticipo che la FDD dovrebbe trovarle come picchi.
# Sono le "frequenze proprie" della nostra struttura fittizia.
freq_modali = [2.5, 7.8, 15.3]

# Ampiezza di ciascuna sinusoide [adimensionale].
# Non sono le ampiezze vere che misureremo sui sensori: le 7 ampiezze
# su ciascun sensore dipendono dalla forma modale e dalla posizione
# del sensore rispetto al modo.
amp_modali = [1.0, 0.7, 0.4]

# Deviazione standard del rumore gaussiano bianco aggiunto.
# Lo teniamo piccolo rispetto alle ampiezze delle sinusoidi
# cosi' i picchi restano ben visibili nella FDD.
sigma_rumore = 0.3

# Stampa dei parametri
print(f"  FS_ACCEL           = {FS_ACCEL} Hz")
print(f"  FDD_DURATION_SEC   = {FDD_DURATION_SEC} s")
print(f"  FDD_INTERVAL_SEC   = {FDD_INTERVAL_SEC} s")
print(f"  nperseg (Welch)    = {nperseg} campioni")
print(f"  n_sensori          = {n_sensori}")
print(f"  n_modes            = {n_modes}")
print(f"  freq_modali [Hz]   = {freq_modali}")
print(f"  amp_modali         = {amp_modali}")
print(f"  sigma_rumore       = {sigma_rumore}")
print()


# ==============================================================================
# STEP 1: GENERAZIONE DEL SEGNALE MULTI-CANALE SIMULATO
# ==============================================================================
print("[STEP 1] Generazione del segnale simulato\n")

# Numero totale di campioni generati nel tempo.
# n_campioni = n_secondi * frequenza_campionamento.
# Con 600 s a 200 Hz -> 600 * 200 = 120.000 campioni.
n_campioni = int(FDD_DURATION_SEC * FS_ACCEL)

# Vettore tempo in secondi.
# t = [0, 1/FS_ACCEL, 2/FS_ACCEL, ..., (n_campioni-1)/FS_ACCEL]
# Rappresenta l'istante temporale di ciascun campione.
t = np.arange(n_campioni) / FS_ACCEL

# Genero le 3 sinusoidi pure (una riga per ogni modo).
# Per ogni modo m calcolo: A_m * sin(2π * f_m * t)
# Risultato: sinusoidi.shape = (n_modes, n_campioni) = (3, 120000)
# Ogni riga è una sinusoide pura, senza rumore, a una frequenza specifica.
sinusoidi = np.array([
    amp * np.sin(2 * np.pi * freq * t)
    for freq, amp in zip(freq_modali, amp_modali)
])

print(f"  sinusoidi.shape = {sinusoidi.shape}")
print(f"    -> {sinusoidi.shape[0]} sinusoidi x {sinusoidi.shape[1]} campioni temporali")
print()

# FORME MODALI: simulazione di come ciascun sensore "sente" ogni modo.
# Se avessimo una trave con 7 sensori posizionati lungo la trave,
# ogni modo proprio causerebbe un pattern spaziale diverso sui sensori.
# Qui usiamo forme modali sinusoidali (analitiche), come se avessimo
# una trave semplicemente appoggiata con 7 punti di misura.
#
# forme_modali[i, m] = (ampiezza del modo m al sensore i)
# Shape: (n_sensori, n_modes) = (7, 3)
#
# Usiamo il pattern di una trave semplicemente appoggiata:
# phi_m(x) = sin(m * π * x / L) con x ∈ [0, L]
# Per i sensori spaziati uniformemente: x_i = (i+1) / (n_sensori+1)
forme_modali = np.zeros((n_sensori, n_modes))
for i in range(n_sensori):
    for m in range(n_modes):
        # Posizione normalizzata del sensore i (da 0 a 1 lungo la "trave")
        x_i = (i + 1) / (n_sensori + 1)
        # Modo m+1 di una trave semplicemente appoggiata
        # (il modo 0 non esiste, partiamo da modo 1)
        forme_modali[i, m] = np.sin((m + 1) * np.pi * x_i)

print(f"  forme_modali.shape = {forme_modali.shape}")
print(f"    -> matrice (n_sensori={n_sensori}) x (n_modes={n_modes})")
print(f"    -> ogni riga = come il modo corrispettivo si "
      f"manifesta su quel sensore")
print(f"\n  forme_modali =")
print(f"  {forme_modali}")
print()

# Combino le 3 sinusoidi sui 7 sensori usando le forme modali.
# Algebra lineare:
# segnali_puri = forme_modali @ sinusoidi
#   shape: (n_sensori, n_modes) @ (n_modes, n_campioni)
#        = (n_sensori, n_campioni)
#
# Risultato: segnali_puri[i, t] = somma su m di: forme_modali[i, m] * sinusoidi[m, t]
# Cioè: il segnale al sensore i è la combinazione pesata delle 3 sinusoidi,
# con pesi dati dalla forma modale.
segnali_puri = forme_modali @ sinusoidi

print(f"  segnali_puri.shape = {segnali_puri.shape}")
print(f"    -> (n_sensori={n_sensori}, n_campioni={n_campioni})")
print()

# Aggiungo rumore gaussiano bianco indipendente su ciascun sensore.
# Per ogni campione t e sensore i: y[i, t] = segnale_puro[i, t] + rumore[i, t]
# rumore ~ N(0, sigma_rumore^2), generato indipendente per ogni (i, t).
rumore = np.random.normal(loc=0.0, scale=sigma_rumore,
                          size=segnali_puri.shape)

# data_matrix è la matrice finale dei segnali: righe=campioni, colonne=sensori.
# Nota il .T (trasposizione): scipy.signal.csd e np.linalg.svd
# lavorano più comodamente con righe=campioni, colonne=canali.
data_matrix = (segnali_puri + rumore).T  # shape: (n_campioni, n_sensori) = (120000, 7)

print(f"  rumore.shape = {rumore.shape}")
print(f"  data_matrix.shape = {data_matrix.shape}")
print(f"    -> (n_campioni={n_campioni}, n_sensori={n_sensori})")
print(f"    -> ogni riga è uno istante temporale, ogni colonna è un sensore")
print()


# ==============================================================================
# STEP 2: CALCOLO DELLA MATRICE CPSD S_xx (Cross Power Spectral Density)
# ==============================================================================
print("[STEP 2] Calcolo della matrice CPSD S_xx via Welch\n")

# Calcolo il vettore delle frequenze nel dominio FFT.
# np.fft.rfftfreq restituisce i bin di frequenza per la trasformata di Fourier
# reale (rfft) di un segnale di lunghezza nperseg.
# Questi bin vanno da 0 Hz (DC) a FS_ACCEL/2 Hz (Nyquist).
f = np.fft.rfftfreq(nperseg, d=1.0 / FS_ACCEL)
n_freq = len(f)

print(f"  f.shape = {f.shape}")
print(f"  n_freq = {n_freq}")
print(f"  f[0] = {f[0]:.3f} Hz (DC)")
print(f"  f[-1] = {f[-1]:.3f} Hz (Nyquist = FS_ACCEL/2)")
print(f"  Δf = {f[1] - f[0]:.6f} Hz")
print()

# Alloco la matrice CPSD S_xx, dimensione (n_freq, n_sensori, n_sensori).
# È un "cubo" 3D di matrici complesse, una per ogni frequenza.
# S_xx[k, i, j] = cross-PSD tra sensore i e sensore j alla frequenza f[k]
#
# - La "fetta" S_xx[k, :, :] è una matrice complessa (n_sensori x n_sensori)
#   alla frequenza f[k].
# - Diagonale S_xx[k, i, i] = auto-PSD del sensore i (numero reale >= 0)
# - Fuori diagonale S_xx[k, i, j] con i≠j = cross-PSD (numero complesso)
# - Proprietà: S_xx[k, i, j] = conj(S_xx[k, j, i]) (matrice Hermitiana)
#
# Utilizzo dtype=complex perché le cross-PSD fuori diagonale sono complesse.
S_xx = np.zeros((n_freq, n_sensori, n_sensori), dtype=complex)

print(f"  S_xx.shape = {S_xx.shape}")
print(f"    -> un 'cubo' di {n_freq} matrici {n_sensori}x{n_sensori}, "
      f"una per ogni frequenza")
print(f"    -> memoria: ~{S_xx.nbytes / 1e6:.1f} MB")
print()

# Riempio S_xx con le CPSD calcolate via scipy.signal.csd.
# signal.csd(x, y) implementa il metodo di Welch:
# 1. Divide x e y in segmenti sovrapposti di nperseg campioni.
# 2. Applica una finestra (Hann di default) a ciascun segmento.
# 3. Calcola la FFT di ogni segmento → X_k, Y_k (k = indice frequenza)
# 4. Per ogni segmento: calcola X_k * conj(Y_k) (cross-spettro).
# 5. Media su tutti i segmenti → stima della cross-PSD.
# Quando i==j il passo 4 diventa |X_k|^2, dando l'auto-PSD (Welch classico).
#
# signal.csd restituisce:
# - freqs: (n_freq,) vettore delle frequenze [Hz]
# - Pxy: (n_freq,) vettore complesso della cross-PSD per ogni bin
#
# Assegno il vettore Pxy[:] lungo l'asse 0 (frequenza) della matrice S_xx:
# S_xx[:, i, j] = Pxy[:] dalla csd tra sensore i e sensore j.

print("  Riempimento di S_xx[k, i, j] via signal.csd...\n")
for i in range(n_sensori):
    for j in range(n_sensori):
        # Prendo la colonna i (sensore i, tutti i n_campioni) e colonna j
        # da data_matrix, che ha shape (n_campioni, n_sensori).
        sensore_i = data_matrix[:, i]  # shape (n_campioni,)
        sensore_j = data_matrix[:, j]  # shape (n_campioni,)

        # signal.csd applica Welch, ritorna:
        # freqs (scartato con _): le frequenze, uguali per tutte le coppie
        # Pxy: (n_freq,) complex <- cross-PSD per ogni bin
        _, Pxy = signal.csd(sensore_i, sensore_j, fs=FS_ACCEL, nperseg=nperseg)

        # Assegno Pxy (di lunghezza n_freq) al "tubo" [k, i, j]
        # per tutte le frequenze k.
        S_xx[:, i, j] = Pxy

        if (i % 2 == 0) and (j % 2 == 0):
            print(f"    S_xx[:, {i}, {j}] <- CSD(sensore {i}, sensore {j})")

print(f"\n  S_xx riempita. Dimensione: {S_xx.shape}\n")


# ==============================================================================
# STEP 3: DECOMPOSIZIONE AI VALORI SINGOLARI (SVD) PER OGNI FREQUENZA
# ==============================================================================
print("[STEP 3] SVD frequenza per frequenza\n")

# Per ogni bin di frequenza k, faccio la SVD della matrice S_xx[k, :, :]
# di shape (n_sensori, n_sensori).
#
# np.linalg.svd(A) restituisce: U, sigma, Vh
# - A = U @ diag(sigma) @ Vh (decomposizione)
# - U.shape = (n_sensori, n_sensori) -> colonne = vettori singolari sinistri
# - sigma.shape = (min(m, n),) = (n_sensori,) -> valori singolari, REALI, >= 0
# - Vh.shape = (n_sensori, n_sensori) -> Vh = V^H (coniugato trasposto)
#
# I valori singolari sigma sono ORDINATI DECRESCENTI:
# sigma[0] >= sigma[1] >= ... >= sigma[n_sensori-1]
#
# Idea FDD: in corrispondenza di un modo proprio della struttura, i sensori
# vibrano "in sincronia" secondo la forma modale phi. Quindi S_xx[k_modo] ≈ phi * phi^H * S_q
# che è una matrice di RANGO 1. La SVD lo cattura: sigma[0] >> sigma[1] >> sigma[2],...
# Lontano dai modi: tutti i valori singolari sono simili (rumore).

# Alloco due array per salvare i risultati:
# sigma_1: solo il primo (dominante) valore singolare per ogni k -> shape (n_freq,)
# sigma_all: tutti i valori singolari per ogni k -> shape (n_freq, n_sensori)
# U_all: tutte le matrici U per ogni k -> shape (n_freq, n_sensori, n_sensori)
sigma_1 = np.zeros(n_freq)
sigma_all = np.zeros((n_freq, n_sensori))
U_all = np.zeros((n_freq, n_sensori, n_sensori), dtype=complex)

print(f"  sigma_1.shape = {sigma_1.shape}")
print(f"  sigma_all.shape = {sigma_all.shape}")
print(f"  U_all.shape = {U_all.shape}")
print()

print("  Esecuzione SVD per k in [0, n_freq)...\n")
for k in range(n_freq):
    # Estraggo la "fetta" k-esima della matrice CPSD.
    # S_k è una matrice quadrata complessa di shape (n_sensori, n_sensori).
    S_k = S_xx[k, :, :]  # shape (7, 7)

    # SVD della matrice S_k.
    # U: (7, 7) complessa, colonne = vettori singolari sinistri
    # sigma_k: (7,) reale >= 0, ordinati decrescenti
    # Vh: (7, 7) complessa
    U, sigma_k, Vh = np.linalg.svd(S_k)

    # Salvo tutti i risultati per il bin k.
    sigma_all[k, :] = sigma_k       # sigma_all[k, :] = [sigma_0, sigma_1, ..., sigma_6]
    sigma_1[k] = sigma_k[0]          # sigma_1[k] = il valore singolare dominante
    U_all[k, :, :] = U               # U_all[k, :, :] = matrice U completa

print(f"  SVD completata per tutti i {n_freq} bin di frequenza.\n")

# Stampa diagnostica: primi 5 bin di frequenza, per debug.
print("  Diagnostica sui primi 5 bin:")
for k in range(min(5, n_freq)):
    print(f"    k={k}, f={f[k]:.3f} Hz: sigma=[{sigma_all[k, 0]:.3e}, "
          f"{sigma_all[k, 1]:.3e}, {sigma_all[k, 2]:.3e}, ...]")
print()


# ==============================================================================
# STEP 4: PEAK PICKING SUL PRIMO VALORE SINGOLARE
# ==============================================================================
print("[STEP 4] Peak picking su sigma_1(f)\n")

# Cerco i picchi (massimi locali) della curva sigma_1(f).
# In corrispondenza dei modi propri della struttura, sigma_1 ha dei picchi netti
# perché la matrice CPSD diventa di rango 1 (vedi spiegazione sopra).
#
# scipy.signal.find_peaks(x, height=..., distance=...) ritorna gli indici
# dei massimi locali di x, con filtri su altezza minima e distanza minima.
#
# - height=np.max(sigma_1)*0.1: prendo solo picchi che superano il 10% del massimo
#   di sigma_1. Filtra il rumore di fondo e gli artefatti.
# - distance=10: distanza minima fra picchi successivi (in numero di bin).
#   Con Δf ≈ 0.195 Hz, distance=10 corrisponde a ~2 Hz: evita di prendere
#   due volte lo stesso picco dovuto a oscillazioni.

# Calcolo i parametri di filtro
altezza_min = np.max(sigma_1) * 0.1
distanza_min = 10

picchi, _ = signal.find_peaks(sigma_1, height=altezza_min, distance=distanza_min)

print(f"  Parametri find_peaks:")
print(f"    height (10% di max)  = {altezza_min:.3e}")
print(f"    distance (bin)       = {distanza_min}")
print(f"  Numero di picchi trovati: {len(picchi)}")
print(f"  Indici picchi: {picchi}")
print(f"  Frequenze picchi: {f[picchi]}")
print()

# Selezione dei top n_modes modi (ordino per ampiezza decrescente di sigma_1,
# prendo i top 3, poi li riordino per frequenza crescente).
if len(picchi) > 0:
    # Ordino gli indici dei picchi in base al valore di sigma_1 (decrescente).
    # np.argsort(sigma_1[picchi])[::-1] mi da l'ordine di decrescenza.
    # Prendo i primi n_modes elementi -> i picchi piu' alti.
    top_idx_per_ampiezza = np.argsort(sigma_1[picchi])[::-1]
    top = picchi[top_idx_per_ampiezza[:n_modes]]

    # Riordino per frequenza crescente (per comodità di stampa).
    top = np.sort(top)

    # Estraggo le frequenze corrispondenti.
    fn = f[top]  # shape (n_modes,)
else:
    top = np.array([], dtype=int)
    fn = np.array([])

# Estraggo le singole frequenze per la stampa (uno per uno).
fn1 = float(fn[0]) if len(fn) > 0 else 0.0
fn2 = float(fn[1]) if len(fn) > 1 else 0.0
fn3 = float(fn[2]) if len(fn) > 2 else 0.0

print(f"  Top {n_modes} modi (ordinati per frequenza crescente):")
print(f"    fn1 = {fn1:.3f} Hz" if fn1 > 0 else "    fn1 = non identificata")
print(f"    fn2 = {fn2:.3f} Hz" if fn2 > 0 else "    fn2 = non identificata")
print(f"    fn3 = {fn3:.3f} Hz" if fn3 > 0 else "    fn3 = non identificata")
print()

# Stampa di confronto con le frequenze imposte.
print(f"  Confronto con frequenze IMPOSTE nel segnale:")
print(f"    freq_modali (imposte) = {freq_modali}")
print(f"    fn (identificate)      = {np.round(fn, 3).tolist()}")
print(f"    Errori [Hz]:           = {np.round(fn - freq_modali[:len(fn)], 3).tolist()}")
print(f"    (Errori attesi ~ Δf = {f[1] - f[0]:.6f} Hz)")
print()


# ==============================================================================
# STEP 5: ESTRAZIONE DELLE FORME MODALI
# ==============================================================================
print("[STEP 5] Estrazione delle forme modali dai vettori singolari\n")

# Per ogni modo (ogni picco), estraggo il vettore singolare sinistro
# U_all[k, :, 0] alla frequenza del picco. Questo vettore è la forma modale
# stimata (up to phase e normalizzazione).

forme_modali_stimate = np.zeros((n_sensori, n_modes), dtype=complex)

for p, k in enumerate(top):
    # Vettore singolare sinistro della matrice U_all[k, :, :]
    # associato al primo (dominante) valore singolare.
    # phi ha shape (n_sensori,) = (7,)
    phi = U_all[k, :, 0]

    # Allineo la fase: divido per la fase della componente con max modulo.
    # Così phi diventa (quasi) reale, e lo posso plottare facilmente.
    idx_max = np.argmax(np.abs(phi))
    fase_ref = np.angle(phi[idx_max])
    phi = phi * np.exp(-1j * fase_ref)

    # Prendo la parte reale (fase quasi nulla, parte immaginaria ~ 0).
    phi = np.real(phi)

    # Normalizzo a max(|phi|)=1 per facilitare il confronto tra modi.
    phi = phi / np.max(np.abs(phi))

    forme_modali_stimate[:, p] = phi

    print(f"  Modo {p+1} a f={f[k]:.3f} Hz:")
    print(f"    phi (normalizzato) = {np.round(phi, 3)}")
    print()


# ==============================================================================
# STEP 6: PLOT DEI RISULTATI
# ==============================================================================
print("[STEP 6] Creazione dei plot\n")

# Finestra temporale per il plot dei segnali: primi 10 secondi
# per vedere chiaramente le oscillazioni (600 s sarebbero illeggibili).
finestra_plot = int(10 * FS_ACCEL)

# ============ Plot 1: Le 3 sinusoidi + segnale totale ============
# Mostro le 3 sinusoidi pure singolarmente, poi il segnale totale
# (combinazione + rumore) da uno dei sensori.
fig1, ax1 = plt.subplots(4, 1, figsize=(12, 8), sharex=True)

# Prima sottofigura: sinusoide 1
ax1[0].plot(t[:finestra_plot], sinusoidi[0, :finestra_plot],
            color='steelblue', linewidth=1.0)
ax1[0].set_ylabel("Sinusoide 1")
ax1[0].set_title(f"Sinusoide 1: f={freq_modali[0]} Hz, A={amp_modali[0]}")
ax1[0].grid(True, alpha=0.3)

# Seconda sottofigura: sinusoide 2
ax1[1].plot(t[:finestra_plot], sinusoidi[1, :finestra_plot],
            color='darkorange', linewidth=1.0)
ax1[1].set_ylabel("Sinusoide 2")
ax1[1].set_title(f"Sinusoide 2: f={freq_modali[1]} Hz, A={amp_modali[1]}")
ax1[1].grid(True, alpha=0.3)

# Terza sottofigura: sinusoide 3
ax1[2].plot(t[:finestra_plot], sinusoidi[2, :finestra_plot],
            color='crimson', linewidth=1.0)
ax1[2].set_ylabel("Sinusoide 3")
ax1[2].set_title(f"Sinusoide 3: f={freq_modali[2]} Hz, A={amp_modali[2]}")
ax1[2].grid(True, alpha=0.3)

# Quarta sottofigura: segnale totale da un sensore (es. sensore 0)
# Mostra la combinazione delle 3 sinusoidi (via forme modali) + rumore.
ax1[3].plot(t[:finestra_plot], data_matrix[:finestra_plot, 0],
            color='purple', linewidth=1.0)
ax1[3].set_ylabel("Sensore 0")
ax1[3].set_xlabel("Tempo [s]")
ax1[3].set_title("Segnale totale su sensore 0 = combinazione delle 3 sinusoidi + rumore")
ax1[3].grid(True, alpha=0.3)

fig1.suptitle("Step 1: Le 3 sinusoidi e il segnale totale (primi 10 secondi)")
fig1.tight_layout()

print("  Plot 1: sinusoidi + segnale totale ... OK")

# ============ Plot 2: I 7 valori singolari vs frequenza (scala LINEARE) ============
# Mostro tutte le 7 curve sigma_j(f) per j=0, 1, ..., 6 in scala lineare.
# Questo permette di capire come il primo SV "domina" sui secondi,
# soprattutto in corrispondenza dei modi.
fig2, ax2 = plt.subplots(figsize=(12, 5))

for j in range(n_sensori):
    # Plotto sigma_all[:, j] che contiene il j-esimo valore singolare
    # per tutte le frequenze.
    ax2.plot(f, sigma_all[:, j], linewidth=1.0, label=f"sigma_{j+1}(f)")

ax2.set_xlim(0, FS_ACCEL / 2)  # zoom fino a Nyquist (100 Hz)
ax2.set_xlabel("Frequenza [Hz]")
ax2.set_ylabel("Valore singolare [scala lineare]")
ax2.set_title(f"Step 2: I {n_sensori} valori singolari della CPSD vs frequenza (LINEARE)")
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right', fontsize=9)
fig2.tight_layout()

print("  Plot 2: 7 valori singolari (lineare) ... OK")

# ============ Plot 3: Primo SV con picchi marcati ============
# Mostro sigma_1(f) con i picchi identificati marcati con pallini rossi
# e linee verticali. Questo è il classico "FDD plot".
fig3, ax3 = plt.subplots(figsize=(12, 5))

# Curva del primo valore singolare
ax3.plot(f, sigma_1, color='steelblue', linewidth=1.5, label='sigma_1(f)')

# Picchi trovati: pallini rossi sulla curva
if len(top) > 0:
    ax3.plot(f[top], sigma_1[top], 'o', color='red', markersize=12,
             label='Picchi selezionati (modalità)')

# Linee verticali tratteggiate rosse in corrispondenza dei modi identificati
for idx, (fn_i, k) in enumerate(zip(fn, top)):
    ax3.axvline(x=fn_i, color='red', linestyle='--', alpha=0.6, linewidth=1.0)
    # Etichetta sopra ogni linea
    ax3.text(fn_i, ax3.get_ylim()[1] * 0.95, f'fn{idx+1}={fn_i:.2f}Hz',
             rotation=90, verticalalignment='top', fontsize=9, color='red')

# Linee verticali tratteggiate grigie per le frequenze imposte (per confronto)
for fm_i in freq_modali:
    ax3.axvline(x=fm_i, color='gray', linestyle='--', alpha=0.4, linewidth=1.0)

ax3.set_xlim(0, FS_ACCEL / 2)
ax3.set_xlabel("Frequenza [Hz]")
ax3.set_ylabel("sigma_1(f) [scala lineare]")
ax3.set_title(f"Step 3: Primo valore singolare con picchi (modalità identificate)")
ax3.grid(True, alpha=0.3)
ax3.legend(loc='upper right')
fig3.tight_layout()

print("  Plot 3: primo SV con picchi marcati ... OK")

# ============ Plot 4: Forme modali identificate ============
# Mostro le n_modes forme modali stimate, una al lato dell'altra.
fig4, ax4 = plt.subplots(1, n_modes, figsize=(4 * n_modes, 4), sharey=True)

# Se c'è un solo modo, ax4 non è un array: lo forziamo per coerenza.
if n_modes == 1:
    ax4 = [ax4]

for p, (k, phi) in enumerate(zip(top, forme_modali_stimate.T)):
    # Disegno il pattern modale come uno "stem plot" (bastone + pallino).
    ax4[p].stem(np.arange(n_sensori), phi, basefmt=' ', linefmt='steelblue',
                markerfmt='o')
    ax4[p].axhline(y=0, color='k', linewidth=0.5, alpha=0.5)
    ax4[p].set_xlabel("Numero sensore")
    ax4[p].set_title(f"Modo {p+1}: f={f[k]:.2f} Hz")
    ax4[p].grid(True, alpha=0.3, axis='y')

# Solo la prima sottofigura ha la label dell'asse y
ax4[0].set_ylabel("Ampiezza (normalizzato)")

fig4.suptitle("Step 4: Forme modali identificate (pattern spaziale su sensori)")
fig4.tight_layout()

print("  Plot 4: forme modali ... OK")
print()

# Mostra tutti i grafici
plt.show()

print("\n" + "="*80)
print(" FDD DEMO COMPLETATO")
print("="*80 + "\n")
