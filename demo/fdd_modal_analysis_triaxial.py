"""
Demo della Frequency Domain Decomposition (FDD) - sensori TRIASSIALI.

Idea generale:
    La FDD e' una tecnica di "Operational Modal Analysis" (OMA).
    A partire da segnali multi-canale (accelerazioni misurate
    su un ponte) stima le frequenze proprie e le forme modali
    sfruttando solo la risposta in esercizio (output-only).

    Il cuore della FDD:
      1) Si calcola la matrice di densita' spettrale di potenza
         incrociata (Cross Power Spectral Density, CPSD) tra tutte
         le coppie sensore-sensore, costruendo una matrice S_xx(f) 3D
         di forma (n_sensori x n_sensori x n_frequenze).
      2) Per ogni frequenza si esegue la SVD della CPSD.
      3) I modi della struttura causano picchi nel primo valore
         singolare sigma_1(f).
      4) I vettori singolari sinistri associati a sigma_1 in
         corrispondenza dei picchi sono una stima della forma
         modale a quella frequenza.

    *** VERSIONE TRIASSIALE ***
    Ogni sensore misura l'accelerazione lungo 3 assi (X, Y, Z),
    quindi abbiamo n_sensori * 3 = 7 * 3 = 21 canali totali.
    In questo demo la FDD viene eseguita SEPARATAMENTE per
    ciascun asse: per ogni asse si costruisce la propria matrice
    CPSD (n_sensori x n_sensori x n_freq) e si identificano
    frequenze proprie e forme modali specifiche di quella direzione.
    Si ottengono cosi' 3 analisi indipendenti (una per asse).

    Le forme modali identificate per i tre assi descrivono come
    oscilla la struttura lungo X, Y, Z al variare del modo:
    un modo flessionale verticale dara' grandi componenti su Z,
    un modo torsionale dara' componenti accoppiate su X e Y, ecc.

    Come nella versione monoassiale, il segnale e' simulato come
    somma di tre sinusoidi a frequenze note + rumore gaussiano:
    sappiamo in anticipo che la FDD dovra' produrre 3 picchi su
    ogni asse, vicino alle stesse 3 frequenze imposte.
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


# =============================================================================
# 1) PARAMETRI DI CONFIGURAZIONE
# =============================================================================
# Frequenza di campionamento [Hz].
fs = 200

# Lunghezza del segmento usato dalla FFT all'interno del metodo di Welch.
n_seg = 1024

# Durata della "finestra" di dati su cui eseguire la FDD [secondi].
FDD_DURATION_SEC = 600   # 10 minuti

# Periodo nominale di esecuzione della FDD [secondi].
FDD_INTERVAL_SEC = 600

# Numero di sensori TRIASSIALI.
n_sensori = 7

# Numero di assi per sensore (X, Y, Z).
n_assi = 3
assi_nomi = ['X', 'Y', 'Z']

# Frequenze [Hz] delle 3 sinusoidi che inseriamo nel segnale.
# Sono le "frequenze proprie" attese: la FDD dovra' identificarle
# come picchi su tutti e 3 gli assi (con ampiezze diverse, perche'
# ogni modo si manifesta in modo diverso lungo X, Y, Z).
freq_modali = [2.5, 7.8, 15.3]
amp_modali = [1.0, 0.7, 0.4]

# Forme modali per ciascun asse: array di forma (n_assi, n_sensori, n_modi).
# - forme_modali[0] -> matrice (7 x 3) con la componente X di ogni modo.
# - forme_modali[1] -> matrice (7 x 3) con la componente Y di ogni modo.
# - forme_modali[2] -> matrice (7 x 3) con la componente Z di ogni modo.
# In una struttura reale ogni modo ha componenti X/Y/Z diverse;
# qui le scegliamo arbitrarie ma distinte per mostrare che la FDD
# e' in grado di identificarle separatamente su ogni asse.
forme_modali = np.array([
    # ----- Asse X -----
    [[ 1.0,  1.0,  1.0],   # sensore 0
     [ 0.8, -0.5,  1.0],   # sensore 1
     [ 0.5, -1.0, -0.7],   # sensore 2
     [ 0.2,  0.6, -1.0],   # sensore 3
     [-0.3,  0.9,  0.4],   # sensore 4
     [-0.7,  0.2, -0.6],   # sensore 5
     [-1.0, -0.8,  0.8]],  # sensore 6
    # ----- Asse Y -----
    [[ 0.9, -0.6,  0.7],   # sensore 0
     [ 0.6,  0.8,  0.5],   # sensore 1
     [ 0.3, -0.4, -0.9],   # sensore 2
     [-0.1,  0.7,  0.6],   # sensore 3
     [-0.5,  0.3, -0.5],   # sensore 4
     [-0.8, -0.6,  0.9],   # sensore 5
     [-0.9,  0.5, -0.7]],  # sensore 6
    # ----- Asse Z -----
    [[ 0.4,  0.5,  0.6],   # sensore 0
     [ 0.7,  0.9, -0.3],   # sensore 1
     [ 1.0, -0.2,  0.8],   # sensore 2
     [ 0.6,  0.4,  1.0],   # sensore 3
     [-0.4,  0.7,  0.5],   # sensore 4
     [-0.6, -0.3,  0.4],   # sensore 5
     [-0.8,  0.6, -0.9]],  # sensore 6
])

# Deviazione standard del rumore gaussiano bianco aggiunto a ciascun
# canale (un canale = un sensore su un asse).
sigma_rumore = 0.3


# =============================================================================
# 2) GENERAZIONE DEL SEGNALE SIMULATO (TRIASSIALE)
# =============================================================================
n_campioni = int(FDD_DURATION_SEC * fs)
t = np.arange(n_campioni) / fs
rng = np.random.default_rng()

# Fasi iniziali casuali per ogni modo (uguali sui 3 assi: e' lo stesso
# modo della struttura, le 3 componenti X/Y/Z oscillano in fase).
fasi_iniziali = rng.uniform(0, 2 * np.pi, size=len(freq_modali))

# Sinusoidi pure (una per modo). fonti.shape = (n_modi, n_campioni).
fonti = np.array([
    amp * np.sin(2 * np.pi * f * t + phi)
    for f, amp, phi in zip(freq_modali, amp_modali, fasi_iniziali)
])

# Genero i segnali per ogni asse e per ogni sensore.
# segnali.shape = (n_assi, n_sensori, n_campioni).
# Per ciascun asse a:
#   forme_modali[a] (n_sensori, n_modi) @ fonti (n_modi, n_campioni)
#   = (n_sensori, n_campioni)
# Poi sommo rumore gaussiano bianco indipendente su ogni canale.
segnali = np.zeros((n_assi, n_sensori, n_campioni))
for a in range(n_assi):
    segnali[a] = forme_modali[a] @ fonti
    rumore = rng.normal(loc=0.0, scale=sigma_rumore, size=segnali[a].shape)
    segnali[a] += rumore

print(f"Generati {n_sensori} sensori triassiali x {n_assi} assi "
      f"x {n_campioni} campioni ({FDD_DURATION_SEC} s a {fs} Hz).")
print(f"Canali totali = {n_sensori * n_assi}")
print(f"FDD_INTERVAL_SEC = {FDD_INTERVAL_SEC} s "
      f"(eseguiamo l'analisi una sola volta, separata per asse).")


# =============================================================================
# 3) FUNZIONE DI ANALISI FDD PER UN SINGOLO ASSE
# =============================================================================
# La incapsuliamo in una funzione cosi' la possiamo richiamare
# tre volte (una per X, una per Y, una per Z) senza duplicare codice.
def fdd_su_asse(segnali_asse, fs, n_seg):
    """
    Esegue la FDD su un set di n_sensori segnali (un solo asse).

    Parametri
    ---------
    segnali_asse : ndarray, shape (n_sensori, n_campioni)
        Segnali di tutti i sensori per UN solo asse.
    fs : float
        Frequenza di campionamento [Hz].
    n_seg : int
        Lunghezza dei segmenti per il metodo di Welch.

    Ritorna
    -------
    dict con chiavi:
        'f'                 : vettore frequenze [Hz], shape (n_freq,)
        'sigma_all'         : tutti i valori singolari, shape (n_freq, n_sens)
        'sigma_1'           : primo valore singolare, shape (n_freq,)
        'U_all'             : vettori singolari sinistri,
                              shape (n_freq, n_sens, n_sens)
        'picchi_idx'        : indici dei picchi su sigma_1
        'freq_identificate' : frequenze [Hz] dei picchi
    """
    n_sens = segnali_asse.shape[0]

    # --- 3a) Vettore frequenze e CPSD S_xx(f) ---
    f = np.fft.rfftfreq(n_seg, d=1./fs)
    n_freq = len(f)

    S_xx = np.zeros((n_sens, n_sens, n_freq), dtype=complex)
    for i in range(n_sens):
        for j in range(n_sens):
            _, S_xx[i, j, :] = signal.csd(segnali_asse[i],
                                          segnali_asse[j],
                                          fs=fs, nperseg=n_seg)

    # --- 3b) SVD di S_xx ad ogni frequenza ---
    sigma_1 = np.zeros(n_freq)
    sigma_all = np.zeros((n_freq, n_sens))
    U_all = np.zeros((n_freq, n_sens, n_sens), dtype=complex)
    for k in range(n_freq):
        U, sigma_k, _ = np.linalg.svd(S_xx[:, :, k])
        sigma_all[k, :] = sigma_k
        sigma_1[k] = sigma_k[0]
        U_all[k, :, :] = U

    # --- 3c) Identificazione dei picchi su sigma_1 ---
    soglia = 0.05 * np.max(sigma_1)
    picchi_idx, _ = signal.find_peaks(sigma_1, distance=5,
                                      height=soglia, prominence=soglia)
    freq_identificate = f[picchi_idx]

    return {
        'f': f,
        'sigma_all': sigma_all,
        'sigma_1': sigma_1,
        'U_all': U_all,
        'picchi_idx': picchi_idx,
        'freq_identificate': freq_identificate,
    }


# =============================================================================
# 4) ANALISI FDD: UNA PER OGNI ASSE
# =============================================================================
# Eseguo 3 analisi FDD indipendenti, una per X, una per Y, una per Z.
# I risultati di ciascuna sono salvati nel dizionario `risultati`.
risultati = {}
print("\n--- Risultati FDD per asse ---")
for a, asse in enumerate(assi_nomi):
    risultati[asse] = fdd_su_asse(segnali[a], fs, n_seg)
    print(f"[Asse {asse}] frequenze imposte    : {freq_modali}")
    print(f"[Asse {asse}] frequenze identificate: "
          f"{np.round(risultati[asse]['freq_identificate'], 3).tolist()}")


# =============================================================================
# 5) GRAFICI
# =============================================================================

# --- Figura 1: segnali nel tempo (primi 10 s) ---
# Griglia n_sensori (righe) x n_assi (colonne).
finestra_plot = int(10 * fs)
fig1, ax1 = plt.subplots(n_sensori, n_assi,
                         figsize=(12, 9), sharex=True)
for i in range(n_sensori):
    for a in range(n_assi):
        ax1[i, a].plot(t[:finestra_plot],
                       segnali[a, i, :finestra_plot], lw=0.7)
        ax1[i, a].grid(True, alpha=0.3)
        if i == 0:
            ax1[i, a].set_title(f"Asse {assi_nomi[a]}")
        if a == 0:
            ax1[i, a].set_ylabel(f"Sens {i}")
for a in range(n_assi):
    ax1[-1, a].set_xlabel("Tempo [s]")
fig1.suptitle("Segnali simulati nel tempo (primi 10 s) - "
              "7 sensori triassiali")
fig1.tight_layout()


# --- Figura 2: TUTTI i 7 valori singolari in scala LINEARE, per asse ---
fig2, ax2 = plt.subplots(1, n_assi, figsize=(15, 5), sharey=True)
for a, asse in enumerate(assi_nomi):
    res = risultati[asse]
    for j in range(n_sensori):
        ax2[a].plot(res['f'], res['sigma_all'][:, j],
                    label=f"SV {j+1}", lw=1.0)
    ax2[a].set_xlim(0, fs / 2)
    ax2[a].set_xlabel("Frequenza [Hz]")
    ax2[a].set_title(f"Asse {asse}")
    ax2[a].grid(True, alpha=0.3)
    if a == 0:
        ax2[a].set_ylabel("Valori singolari (lineare)")
ax2[-1].legend(loc="upper right", ncol=2, fontsize=8)
fig2.suptitle(f"Andamento dei {n_sensori} valori singolari per asse "
              f"- scala lineare")
fig2.tight_layout()


# --- Figura 3: TUTTI i 7 valori singolari in scala dB, per asse ---
fig3, ax3 = plt.subplots(1, n_assi, figsize=(15, 5), sharey=True)
for a, asse in enumerate(assi_nomi):
    res = risultati[asse]
    sigma_all_dB = 10 * np.log10(res['sigma_all'] + 1e-20)
    for j in range(n_sensori):
        ax3[a].plot(res['f'], sigma_all_dB[:, j],
                    label=f"SV {j+1}", lw=1.0)
    ax3[a].set_xlim(0, fs / 2)
    ax3[a].set_xlabel("Frequenza [Hz]")
    ax3[a].set_title(f"Asse {asse}")
    ax3[a].grid(True, alpha=0.3)
    if a == 0:
        ax3[a].set_ylabel("Valori singolari [dB]")
ax3[-1].legend(loc="upper right", ncol=2, fontsize=8)
fig3.suptitle(f"Andamento dei {n_sensori} valori singolari per asse "
              f"- scala dB")
fig3.tight_layout()


# --- Figura 4: sigma_1 con picchi in scala LINEARE, per asse ---
fig4, ax4 = plt.subplots(1, n_assi, figsize=(15, 5), sharey=True)
for a, asse in enumerate(assi_nomi):
    res = risultati[asse]
    ax4[a].plot(res['f'], res['sigma_1'], color="C0", lw=1.2,
                label="1o SV (sigma_1)")
    ax4[a].plot(res['freq_identificate'],
                res['sigma_1'][res['picchi_idx']],
                "ro", markersize=8, label="Picchi identificati")
    for fk, sk in zip(res['freq_identificate'],
                      res['sigma_1'][res['picchi_idx']]):
        ax4[a].annotate(f"{fk:.2f} Hz", xy=(fk, sk),
                        xytext=(5, 5), textcoords="offset points")
    for fm in freq_modali:
        ax4[a].axvline(fm, color="k", ls="--", alpha=0.4)
    ax4[a].set_xlim(0, fs / 2)
    ax4[a].set_xlabel("Frequenza [Hz]")
    ax4[a].set_title(f"Asse {asse}")
    ax4[a].grid(True, alpha=0.3)
    if a == 0:
        ax4[a].set_ylabel("sigma_1 (lineare)")
ax4[-1].legend(loc="upper right", fontsize=8)
fig4.suptitle("FDD: 1o valore singolare con picchi modali per asse "
              "- scala lineare")
fig4.tight_layout()


# --- Figura 5: sigma_1 con picchi in scala dB, per asse ---
fig5, ax5 = plt.subplots(1, n_assi, figsize=(15, 5), sharey=True)
for a, asse in enumerate(assi_nomi):
    res = risultati[asse]
    sigma_1_dB = 10 * np.log10(res['sigma_1'] + 1e-20)
    ax5[a].plot(res['f'], sigma_1_dB, color="C0", lw=1.2,
                label="1o SV (sigma_1)")
    ax5[a].plot(res['freq_identificate'], sigma_1_dB[res['picchi_idx']],
                "ro", markersize=8, label="Picchi identificati")
    for fk, sk_dB in zip(res['freq_identificate'],
                         sigma_1_dB[res['picchi_idx']]):
        ax5[a].annotate(f"{fk:.2f} Hz", xy=(fk, sk_dB),
                        xytext=(5, 5), textcoords="offset points")
    for fm in freq_modali:
        ax5[a].axvline(fm, color="k", ls="--", alpha=0.4)
    ax5[a].set_xlim(0, fs / 2)
    ax5[a].set_xlabel("Frequenza [Hz]")
    ax5[a].set_title(f"Asse {asse}")
    ax5[a].grid(True, alpha=0.3)
    if a == 0:
        ax5[a].set_ylabel("sigma_1 [dB]")
ax5[-1].legend(loc="upper right", fontsize=8)
fig5.suptitle("FDD: 1o valore singolare con picchi modali per asse "
              "- scala dB")
fig5.tight_layout()


# --- Figura 6: forme modali identificate, una riga per asse ---
# Per ciascun asse e per ciascun picco prendo il 1o vettore singolare
# sinistro U[:, 0] alla frequenza del picco. La componente piu' grande
# in modulo viene usata come riferimento di fase per renderlo reale e
# normalizzato (cosi' lo si puo' confrontare con la forma modale imposta
# in forme_modali[a]).
n_picchi_max = max(len(risultati[asse]['picchi_idx']) for asse in assi_nomi)
fig6, ax6 = plt.subplots(n_assi, n_picchi_max,
                         figsize=(4 * n_picchi_max, 3 * n_assi),
                         sharey=True, squeeze=False)
for a, asse in enumerate(assi_nomi):
    res = risultati[asse]
    for p, k in enumerate(res['picchi_idx']):
        # 1o vettore singolare sinistro alla freq. del picco.
        phi = res['U_all'][k, :, 0]

        # Allineo la fase: divido per la fase della componente
        # max in modulo, cosi' phi diventa (quasi) reale.
        idx_max = np.argmax(np.abs(phi))
        phi = phi * np.exp(-1j * np.angle(phi[idx_max]))
        phi = np.real(phi)

        # Normalizzo a max(|phi|)=1 per facilitare il confronto.
        phi = phi / np.max(np.abs(phi))

        ax6[a, p].stem(np.arange(n_sensori), phi, basefmt=" ")
        ax6[a, p].set_title(f"Asse {asse} - {res['f'][k]:.2f} Hz")
        ax6[a, p].set_xlabel("Sensore")
        ax6[a, p].axhline(0, color="k", lw=0.5)
        ax6[a, p].grid(True, alpha=0.3)
    ax6[a, 0].set_ylabel("Ampiezza modale (norm.)")
fig6.suptitle("Forme modali identificate dai picchi del 1o SV "
              "per ciascun asse")
fig6.tight_layout()


# =============================================================================
# 6) COME LEGGERE I GRAFICI
# =============================================================================
# - Figura 1: griglia 7x3 dei segnali nel tempo. Colonne = assi X/Y/Z,
#   righe = sensori. Ogni colonna mostra come un asse oscilla per i 7
#   sensori (somma di 3 sinusoidi pesate dalle forme modali di
#   quell'asse + rumore).
#
# - Figura 2: i 7 valori singolari in scala LINEARE, separati per asse.
#   SV1 esplode ai picchi modali e gli altri SV si schiacciano contro
#   lo zero (problema della scala lineare).
#
# - Figura 3: gli stessi SV in scala dB, separati per asse. SV1 stacca
#   dagli altri ai modi: ogni picco e' dominato da UN solo modo.
#
# - Figura 4: sigma_1 in LINEARE per ciascun asse, con pallini rossi
#   sui picchi e linee tratteggiate sulle frequenze modali vere.
#
# - Figura 5: sigma_1 in dB per ciascun asse: i picchi diventano
#   "campane" larghe e ben distinguibili dal fondo. I pallini rossi
#   devono cadere alle 3 frequenze imposte SU OGNI ASSE.
#
# - Figura 6: forme modali stimate. Ogni riga e' un asse (X, Y, Z),
#   ogni colonna un modo. Confrontando con le colonne di
#   forme_modali[a] devono coincidere a meno del segno e di una
#   costante moltiplicativa: e' la conferma finale che la FDD ha
#   identificato correttamente i modi su tutti e tre gli assi.
# =============================================================================

plt.show()
