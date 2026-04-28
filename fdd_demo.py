"""
Demo della Frequency Domain Decomposition (FDD).

Idea generale:
    La FDD e' una tecnica di "Operational Modal Analysis" (OMA).
    A partire da segnali multi-canale (tipicamente accelerazioni misurate
    su una struttura) stima le frequenze proprie e le forme modali
    sfruttando solo la risposta in esercizio (output-only).

    Il cuore della FDD e' questo:
      1) Si calcola la matrice di densita' spettrale di potenza
         incrociata (Cross Power Spectral Density, CPSD) tra tutti
         i canali. E' una matrice complessa di dimensione
         (n_canali x n_canali x n_frequenze).
      2) Per ogni frequenza si esegue la SVD della CPSD.
      3) I valori singolari (in particolare il primo, il piu' grande)
         hanno dei picchi in corrispondenza delle frequenze proprie
         della struttura.
      4) I vettori singolari sinistri associati al primo valore
         singolare in corrispondenza dei picchi sono una stima della
         forma modale a quella frequenza.

    In questo demo NON c'e' una struttura vera: simuliamo direttamente
    i segnali come somma di tre sinusoidi a frequenze note, con
    aggiunta di rumore gaussiano bianco. In questo modo sappiamo
    in anticipo che la FDD dovra' produrre 3 picchi in corrispondenza
    delle 3 frequenze imposte.
"""

# -----------------------------------------------------------------------------
# Importazioni delle librerie necessarie.
# numpy: calcolo numerico (vettori, FFT, algebra lineare).
# scipy.signal: contiene csd() per la cross power spectral density.
# matplotlib.pyplot: per disegnare i grafici.
# -----------------------------------------------------------------------------
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


# =============================================================================
# 1) PARAMETRI DI CONFIGURAZIONE
# =============================================================================

# Frequenza di campionamento [Hz]: numero di campioni acquisiti al secondo.
# Con fs=200 Hz possiamo rappresentare frequenze fino a fs/2 = 100 Hz
# (limite di Nyquist). Le nostre sinusoidi devono stare sotto questo limite.
fs = 200

# Lunghezza del segmento usato dalla FFT all'interno del metodo di Welch [campioni].
# n_seg piu' grande = miglior risoluzione in frequenza ma meno mediazioni.
# Risoluzione in frequenza: df = fs / n_seg = 200/1024 ~= 0.195 Hz.
n_seg = 1024

# Durata della "finestra" di dati su cui eseguire la FDD [secondi].
# In un caso reale ad esempio "analizza gli ultimi 10 minuti".
FDD_DURATION_SEC = 600   # 10 minuti

# Ogni quanto, idealmente, rieseguiamo la FDD [secondi].
# In questo demo lo usiamo solo come parametro indicativo: facciamo
# una sola analisi, ma stampiamo a video il valore.
FDD_INTERVAL_SEC = 600

# Numero di canali (sensori) simulati. La FDD ha senso con piu' canali:
# usiamo 4 canali per avere una matrice CPSD non banale.
n_canali = 4

# Frequenze [Hz] delle 3 sinusoidi che inseriamo nel segnale.
# Sono i "modi" attesi: la FDD dovra' identificarli come picchi.
# Le scegliamo distanti tra loro e ben sotto fs/2 = 100 Hz.
freq_modali = [2.5, 7.8, 15.3]

# Ampiezze delle 3 sinusoidi su ciascun canale.
# Forme modali "finte": ogni riga (lunga 3) dice quanto pesa ognuno
# dei 3 modi su quel canale. In una struttura reale le forme modali
# descrivono come oscillano i diversi punti del sistema.
forme_modali = np.array([
    [1.0,  1.0,  1.0],   # canale 0
    [0.8, -0.5,  1.0],   # canale 1
    [0.5, -1.0, -0.7],   # canale 2
    [0.2,  0.6, -1.0],   # canale 3
])

# Deviazione standard del rumore gaussiano bianco aggiunto a ciascun canale.
# Tenuto piccolo rispetto alle ampiezze delle sinusoidi cosi' i picchi
# restano ben visibili nella FDD.
sigma_rumore = 0.3


# =============================================================================
# 2) GENERAZIONE DEL SEGNALE SIMULATO
# =============================================================================

# Numero totale di campioni da generare. Lunghezza in secondi * fs.
n_campioni = int(FDD_DURATION_SEC * fs)

# Vettore tempo in secondi: [0, 1/fs, 2/fs, ..., (N-1)/fs].
t = np.arange(n_campioni) / fs

# Imposto un seed per il generatore casuale: cosi' i risultati
# (rumore e fasi casuali) sono riproducibili tra esecuzioni diverse.
rng = np.random.default_rng(seed=42)

# Per rendere il segnale piu' "realistico" diamo a ciascuna sinusoide
# una fase iniziale casuale: non cambia la frequenza, ma rompe la
# perfetta sincronia tra canali (anche se in un modo puro la fase
# relativa tra canali e' fissa: qui la teniamo uguale per tutti i
# canali sullo stesso modo, come accadrebbe in un modo reale).
fasi_iniziali = rng.uniform(0, 2 * np.pi, size=len(freq_modali))

# Costruisco la matrice "fonti": una riga per ogni modo, una colonna
# per ogni istante temporale. Ogni riga e' una sinusoide pura.
# fonti.shape = (n_modi, n_campioni)
fonti = np.vstack([
    np.sin(2 * np.pi * f * t + phi)
    for f, phi in zip(freq_modali, fasi_iniziali)
])

# Combino le sinusoidi sui canali tramite le forme modali.
# forme_modali ha shape (n_canali, n_modi),
# fonti ha shape (n_modi, n_campioni),
# quindi il prodotto e' (n_canali, n_campioni).
segnali = forme_modali @ fonti

# Aggiungo rumore gaussiano bianco indipendente su ciascun canale.
# rumore.shape = (n_canali, n_campioni).
rumore = rng.normal(loc=0.0, scale=sigma_rumore, size=segnali.shape)
segnali = segnali + rumore

# A scopo informativo stampo la lunghezza del segnale e l'intervallo FDD.
print(f"Generati {n_canali} canali x {n_campioni} campioni "
      f"({FDD_DURATION_SEC} s a {fs} Hz).")
print(f"FDD_INTERVAL_SEC = {FDD_INTERVAL_SEC} s "
      f"(in questo demo eseguiamo l'analisi una sola volta).")


# =============================================================================
# 3) CALCOLO DELLA MATRICE CPSD (Cross Power Spectral Density)
# =============================================================================

# La matrice CPSD G(f) e' di dimensione (n_canali, n_canali, n_freq).
# - Sulla diagonale ci sono le auto-PSD di ciascun canale.
# - Fuori diagonale ci sono le PSD incrociate tra canali diversi
#   (numeri complessi: ampiezza + fase relativa).
# Usiamo scipy.signal.csd con il metodo di Welch:
#   - divide il segnale in segmenti di n_seg campioni con overlap 50%,
#   - applica una finestra (Hann di default),
#   - calcola FFT su ciascun segmento e media i risultati.
# Questo riduce la varianza della stima spettrale.

# Prima chiamata "di prova" solo per scoprire il vettore delle frequenze
# e la sua lunghezza n_freq, cosi' possiamo dimensionare la matrice CPSD.
freqs, _ = signal.csd(segnali[0], segnali[0],
                      fs=fs, nperseg=n_seg)
n_freq = freqs.size

# Inizializzo la matrice CPSD a zero, di tipo complesso.
CPSD = np.zeros((n_canali, n_canali, n_freq), dtype=complex)

# Ciclo doppio sui canali per riempire tutta la matrice.
for i in range(n_canali):
    for j in range(n_canali):
        # csd(x, y) restituisce Gxy(f). Per i==j coincide con la PSD.
        _, Gij = signal.csd(segnali[i], segnali[j],
                            fs=fs, nperseg=n_seg)
        CPSD[i, j, :] = Gij


# =============================================================================
# 4) DECOMPOSIZIONE AI VALORI SINGOLARI (SVD) FREQUENZA PER FREQUENZA
# =============================================================================

# Per ogni frequenza f_k facciamo la SVD della matrice
# G(f_k) di shape (n_canali, n_canali).
#   G = U * diag(s) * V^H
# - s e' il vettore dei valori singolari ordinati in modo decrescente.
# - Le colonne di U sono i vettori singolari sinistri.
# Salviamo:
#   sv: tutti i valori singolari (n_freq, n_canali).
#   U_all: tutte le matrici U (n_freq, n_canali, n_canali).
sv = np.zeros((n_freq, n_canali))
U_all = np.zeros((n_freq, n_canali, n_canali), dtype=complex)

for k in range(n_freq):
    # np.linalg.svd su matrice quadrata complessa: ritorna U, s, Vh.
    U, s, _ = np.linalg.svd(CPSD[:, :, k])
    sv[k, :] = s
    U_all[k, :, :] = U


# =============================================================================
# 5) IDENTIFICAZIONE DEI PICCHI SUL PRIMO VALORE SINGOLARE
# =============================================================================

# Il "primo valore singolare" e' la curva sv[:, 0] in funzione della
# frequenza. I picchi di questa curva corrispondono ai modi della
# struttura: nel nostro caso ci aspettiamo 3 picchi vicino a
# freq_modali = [2.5, 7.8, 15.3] Hz.

# Lavoriamo in scala dB per rendere i picchi piu' evidenti.
# Aggiungiamo un piccolissimo eps per evitare log(0).
eps = 1e-20
sv1_db = 10.0 * np.log10(sv[:, 0] + eps)

# find_peaks individua i massimi locali. distance=5 forza una
# distanza minima tra picchi (in numero di bin di frequenza)
# per evitare di prendere doppi picchi ravvicinati dovuti al rumore.
# prominence filtra i picchi troppo "piatti".
picchi_idx, _ = signal.find_peaks(sv1_db, distance=5, prominence=3)

# Frequenze identificate (in Hz) corrispondenti ai picchi trovati.
freq_identificate = freqs[picchi_idx]

# Stampa di confronto tra frequenze imposte e frequenze stimate.
print("\n--- Risultati FDD ---")
print(f"Frequenze imposte    : {freq_modali}")
print(f"Frequenze identificate dai picchi del 1o SV: "
      f"{np.round(freq_identificate, 3).tolist()}")


# =============================================================================
# 6) GRAFICI
# =============================================================================

# --- Figura 1: i segnali nel tempo (solo i primi 10 secondi per chiarezza) ---
# Plottare 600 secondi a 200 Hz e' troppo: si vedrebbe solo una "macchia".
finestra_plot = int(10 * fs)  # 10 secondi
fig1, ax1 = plt.subplots(n_canali, 1, figsize=(10, 6), sharex=True)
for i in range(n_canali):
    ax1[i].plot(t[:finestra_plot], segnali[i, :finestra_plot], lw=0.8)
    ax1[i].set_ylabel(f"Canale {i}")
    ax1[i].grid(True, alpha=0.3)
ax1[-1].set_xlabel("Tempo [s]")
fig1.suptitle("Segnali simulati nel tempo (primi 10 s)")
fig1.tight_layout()


# --- Figura 2: i valori singolari della CPSD vs frequenza ---
# Questa e' la "FDD plot" classica.
# - La curva del 1o SV ha picchi alle frequenze modali.
# - Le curve degli altri SV restano basse: se due modi sono vicini
#   o accoppiati, anche il 2o SV puo' avere un picco; nel nostro
#   caso semplice ci aspettiamo che resti basso.
fig2, ax2 = plt.subplots(figsize=(10, 5))
for j in range(n_canali):
    ax2.plot(freqs, 10 * np.log10(sv[:, j] + eps),
             label=f"SV {j+1}", lw=1.0)

# Marco i picchi identificati con un pallino rosso.
ax2.plot(freq_identificate, sv1_db[picchi_idx],
         "ro", label="Picchi del 1o SV")

# Linee verticali tratteggiate sulle frequenze "vere" (per confronto).
for fm in freq_modali:
    ax2.axvline(fm, color="k", ls="--", alpha=0.4)

ax2.set_xlim(0, fs / 2)             # zoom fino alla Nyquist
ax2.set_xlabel("Frequenza [Hz]")
ax2.set_ylabel("Valori singolari [dB]")
ax2.set_title("FDD: valori singolari della CPSD")
ax2.grid(True, alpha=0.3)
ax2.legend(loc="upper right")
fig2.tight_layout()


# --- Figura 3: forme modali identificate ---
# Per ciascun picco prendo il 1o vettore singolare sinistro U[:, 0]
# alla frequenza del picco. La sua componente piu' grande in modulo
# viene usata come riferimento di fase per renderlo reale e
# normalizzato (cosi' lo si puo' confrontare con la forma modale
# imposta).
fig3, ax3 = plt.subplots(1, len(picchi_idx),
                         figsize=(4 * len(picchi_idx), 4),
                         sharey=True)
# Se c'e' un solo picco ax3 non sarebbe un array: forziamolo.
if len(picchi_idx) == 1:
    ax3 = [ax3]

for p, k in enumerate(picchi_idx):
    # Vettore singolare sinistro associato al 1o SV alla freq. del picco.
    phi = U_all[k, :, 0]

    # Allineo la fase: divido per la fase della componente max in modulo,
    # cosi' phi diventa (quasi) reale.
    idx_max = np.argmax(np.abs(phi))
    phi = phi * np.exp(-1j * np.angle(phi[idx_max]))
    phi = np.real(phi)

    # Normalizzo a max(|phi|)=1 per facilitare il confronto.
    phi = phi / np.max(np.abs(phi))

    ax3[p].stem(np.arange(n_canali), phi, basefmt=" ")
    ax3[p].set_title(f"Modo a {freqs[k]:.2f} Hz")
    ax3[p].set_xlabel("Canale")
    ax3[p].axhline(0, color="k", lw=0.5)
    ax3[p].grid(True, alpha=0.3)
ax3[0].set_ylabel("Ampiezza modale (norm.)")
fig3.suptitle("Forme modali identificate dai picchi del 1o SV")
fig3.tight_layout()


# =============================================================================
# 7) COME LEGGERE I GRAFICI
# =============================================================================
# - Figura 1: vedo i 4 segnali nel tempo. Sono somma di 3 sinusoidi
#   con pesi diversi (forme_modali) + rumore. A occhio si intuisce
#   che oscillano, ma non si distinguono le singole frequenze.
#
# - Figura 2 (la piu' importante): la curva blu (1o SV) deve avere
#   3 picchi netti in corrispondenza di 2.5, 7.8 e 15.3 Hz.
#   Le linee tratteggiate verticali nere mostrano dove "dovrebbero"
#   stare i picchi: i pallini rossi sono quelli effettivamente
#   trovati dall'algoritmo. La discrepanza dipende dalla risoluzione
#   df = fs/n_seg ~ 0.195 Hz.
#   Le altre curve (SV2, SV3, SV4) sono il "rumore": se restano
#   sotto al primo SV vuol dire che ogni picco e' dominato da un
#   solo modo (modi ben separati e indipendenti).
#
# - Figura 3: per ogni picco si stima la forma modale come 1o vettore
#   singolare sinistro. Confrontandola con le colonne di
#   forme_modali, dovrebbero (a meno del segno e di una costante
#   moltiplicativa) coincidere. Questa e' la conferma che la FDD
#   ha identificato correttamente i modi.
# =============================================================================

# Mostra tutte le figure.
plt.show()
