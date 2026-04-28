"""
Frequency Domain Decomposition (FDD) - implementazione pratica.

L'FDD e' una tecnica di Operational Modal Analysis (OMA) introdotta da
Brincker et al. (2000). Permette di stimare frequenze naturali e forme
modali di una struttura a partire dalle sole risposte dei sensori
(output-only), assumendo che l'eccitazione sia rumore bianco non
correlato e che lo smorzamento sia basso.

Pipeline:
    1. CPSD (Welch) -> Gyy(f) di forma (Ns, Ns, Nf)
       Gyy[i, j, k] = densita' spettrale incrociata fra sensore i e j
       alla frequenza f[k].
    2. SVD per ogni frequenza -> sigma_i(f), U(f)
       Vicino a un modo, Gyy ~ rank-1 e il primo valore singolare
       contiene "tutta" l'energia di quel modo.
    3. Picchi del primo valore singolare -> frequenze naturali
       (i picchi di sigma_1(f) corrispondono ai modi della struttura).
    4. Primo vettore singolare al picco -> forma modale
       (U[:, 0, k_picco] e' una stima di phi del modo).
"""

import numpy as np
from scipy.signal import csd, find_peaks
import matplotlib.pyplot as plt


def compute_cpsd_matrix(Y, fs, nperseg, noverlap=None, window="hann"):
    """
    Calcola la matrice CPSD Gyy(f) di forma (Ns, Ns, Nf).

    Parametri
    ---------
    Y       : array (Ns, Nt) - segnali dei sensori (Ns canali, Nt campioni)
    fs      : frequenza di campionamento [Hz]
    nperseg : lunghezza del segmento per il metodo di Welch
              (piu' lungo = miglior risoluzione in frequenza ma piu' varianza)
    noverlap: campioni di sovrapposizione fra segmenti (default: 50%)
    window  : finestra applicata a ogni segmento (default Hann)

    Ritorna
    -------
    f   : (Nf,) vettore delle frequenze [Hz]
    Gyy : (Ns, Ns, Nf) matrice CPSD complessa, hermitiana per ogni f
    """
    Ns, Nt = Y.shape
    # se non specificato, uso il classico 50% di overlap
    if noverlap is None:
        noverlap = nperseg // 2

    # prima chiamata "fittizia" solo per ottenere il vettore delle frequenze
    # (cosi' so quanti bin Nf devo allocare nella matrice Gyy)
    f, _ = csd(Y[0], Y[0], fs=fs, nperseg=nperseg,
               noverlap=noverlap, window=window)
    Nf = len(f)
    Gyy = np.zeros((Ns, Ns, Nf), dtype=complex)

    # riempio solo il triangolo superiore (i <= j) e poi rifletto in basso
    # con il coniugato perche' Gyy(f) e' hermitiana: Gji(f) = conj(Gij(f)).
    # Questo dimezza il numero di csd() calcolate.
    for i in range(Ns):
        for j in range(i, Ns):
            _, Gij = csd(Y[i], Y[j], fs=fs, nperseg=nperseg,
                         noverlap=noverlap, window=window)
            Gyy[i, j, :] = Gij
            if i != j:
                Gyy[j, i, :] = np.conj(Gij)
    return f, Gyy


def fdd_svd(Gyy):
    """
    SVD di Gyy(f) per ogni frequenza.

    Per ogni bin di frequenza decompongo Gyy(f) = U(f) * S(f) * U(f)^H
    (e' hermitiana semidefinita positiva quindi V = U). I valori singolari
    sigma_i(f) sono ordinati in modo decrescente: sigma_1 >= sigma_2 >= ...

    Ritorna
    -------
    S : (Ns, Nf)     valori singolari ordinati decrescenti
    U : (Ns, Ns, Nf) vettori singolari sinistri (per colonna)
    """
    Ns, _, Nf = Gyy.shape
    S = np.zeros((Ns, Nf))
    U = np.zeros((Ns, Ns, Nf), dtype=complex)
    # ciclo bin-per-bin: SVD su matrice piccola Ns x Ns
    for k in range(Nf):
        Uk, sk, _ = np.linalg.svd(Gyy[:, :, k])
        S[:, k] = sk
        U[:, :, k] = Uk
    return S, U


def identify_modes(f, S, U, fmin=0.5, fmax=None,
                   prominence_db=10, min_distance_hz=0.3):
    """
    Trova picchi del primo valore singolare ed estrae le forme modali.

    Parametri
    ---------
    f               : vettore frequenze
    S, U            : output di fdd_svd()
    fmin, fmax      : banda di interesse [Hz] (taglia DC e alte freq.)
    prominence_db   : prominenza minima del picco in dB (filtra rumore)
    min_distance_hz : distanza minima fra picchi [Hz] (evita doppi picchi
                      sullo stesso modo)

    Ritorna
    -------
    f_n   : (n_modi,) frequenze naturali identificate [Hz]
    phi_n : (Ns, n_modi) forme modali normalizzate (parte reale, max=1)
    idx   : indici dei picchi nel vettore f
    """
    if fmax is None:
        fmax = f[-1]

    # passo in dB cosi' la prominenza ha un significato fisico chiaro
    # (un modo svetta tipicamente di 10-30 dB sopra il fondo)
    s1_db = 10 * np.log10(S[0] + 1e-30)
    # maschera la banda d'interesse: i picchi vicino a 0 Hz (drift, DC) e
    # vicino a fs/2 sono quasi sempre artefatti
    mask = (f >= fmin) & (f <= fmax)

    # converto la distanza minima da Hz a numero di bin per find_peaks
    df = f[1] - f[0]
    distance = max(int(min_distance_hz / df), 1)

    # find_peaks lavora sull'array masherato; recupero l'indice "globale"
    # nell'array f originale tramite np.where(mask)
    peaks, _ = find_peaks(s1_db[mask], prominence=prominence_db,
                          distance=distance)
    idx = np.where(mask)[0][peaks]

    f_n = f[idx]
    # forma modale = primo vettore singolare alla frequenza del picco
    phi_n = U[:, 0, idx]                           # primo vett. singolare

    # I vettori singolari sono definiti a meno di una fase complessa
    # globale exp(j*alpha): li ruoto perche' la componente di modulo
    # massimo sia reale positiva, e li riscalo perche' max(|phi|) = 1.
    # Risultato: forme modali "leggibili" e confrontabili fra loro.
    for k in range(phi_n.shape[1]):
        v = phi_n[:, k]
        v = v * np.exp(-1j * np.angle(v[np.argmax(np.abs(v))]))
        phi_n[:, k] = v / np.max(np.abs(v))
    return f_n, phi_n, idx


def mac(phi1, phi2):
    """
    Modal Assurance Criterion fra due vettori modali.

    MAC = |phi1^H phi2|^2 / (phi1^H phi1 * phi2^H phi2)

    Vale 1 se phi1 e phi2 sono proporzionali (stessa forma modale),
    0 se sono ortogonali. E' invariante a scala e fase complessa,
    quindi e' la metrica standard per confrontare forme modali.
    """
    num = np.abs(np.vdot(phi1, phi2)) ** 2
    den = np.vdot(phi1, phi1).real * np.vdot(phi2, phi2).real
    return num / den


# ---------------------------------------------------------------------------
# DEMO: 7 sensori, struttura "shear-type" simulata con 3 modi, fs=200, nperseg=1024
#
# Idea: simulo una struttura a 7 piani con 3 modi noti (frequenze e forme
# modali assegnate a mano), eccitati da rumore bianco indipendente. Le
# risposte sono la sovrapposizione modale phi_m * x_m(t). Applico FDD e
# verifico che le frequenze e le forme modali identificate coincidano
# con quelle "vere" (MAC ~ 1).
# ---------------------------------------------------------------------------
def demo():
    # seed fisso => risultati riproducibili
    rng = np.random.default_rng(0)

    fs = 200.0                                      # campionamento [Hz]
    nperseg = 1024                                  # ~5 s a 200 Hz, df ~0.2 Hz
    T = 120.0                                       # durata acquisizione: 2 minuti
    Nt = int(T * fs)
    t = np.arange(Nt) / fs

    Ns = 7                                          # numero di sensori (= piani)
    f_true = np.array([3.2, 8.7, 15.4])             # frequenze "vere" [Hz]
    zeta   = np.array([0.01, 0.015, 0.02])          # smorzamenti modali (1-2%)

    # forme modali "vere" (Ns x n_modi): seni stazionari, tipici di una
    # mensola continua. Il modo m ha m semi-onde lungo l'altezza.
    phi_true = np.array([
        [np.sin(np.pi*(i+1)/(Ns+1) * (m+1)) for m in range(len(f_true))]
        for i in range(Ns)
    ])

    # genero ogni modo come oscillatore SDOF eccitato da rumore bianco
    # gaussiano, e sommo i contributi modali nelle risposte dei sensori:
    #   y(t) = sum_m phi_m * x_m(t)
    Y = np.zeros((Ns, Nt))
    for m, (fm, zm) in enumerate(zip(f_true, zeta)):
        wn = 2*np.pi*fm                             # pulsazione naturale
        wd = wn*np.sqrt(1-zm**2)                    # pulsazione smorzata (qui non usata)
        # integratore di Eulero esplicito: per fs=200 Hz e modi <20 Hz
        # e' abbastanza accurato; per casi piu' rigidi usare Newmark/RK4.
        x = np.zeros(Nt); v = 0.0
        dt = 1/fs
        force = rng.standard_normal(Nt)             # rumore bianco di eccitazione
        for k in range(1, Nt):
            # equazione SDOF: x'' + 2*zeta*wn*x' + wn^2*x = force
            a = force[k] - 2*zm*wn*v - wn**2*x[k-1]
            v += a*dt
            x[k] = x[k-1] + v*dt
        # contributo del modo m a tutti i sensori
        Y += np.outer(phi_true[:, m], x)

    # aggiungo un po' di rumore di misura indipendente sui sensori (2% rms)
    Y += 0.02*np.std(Y)*rng.standard_normal(Y.shape)

    # ---- FDD ----
    # 1) CPSD via Welch
    f, Gyy = compute_cpsd_matrix(Y, fs=fs, nperseg=nperseg)
    # 2) SVD bin-per-bin
    S, U = fdd_svd(Gyy)
    # 3+4) picchi su sigma_1 e forme modali ai picchi
    f_n, phi_n, idx = identify_modes(f, S, U, fmin=1.0, fmax=fs/2,
                                     prominence_db=8, min_distance_hz=1.0)

    print("Frequenze naturali identificate [Hz]:", np.round(f_n, 3))
    print("Frequenze vere [Hz]:                 ", f_true)
    for m in range(min(len(f_n), len(f_true))):
        # confronto MAC con la forma modale vera piu' vicina in frequenza
        diffs = np.abs(f_true - f_n[m])
        j = np.argmin(diffs)
        print(f" modo {m+1}: f={f_n[m]:6.3f} Hz  "
              f"MAC vs modo vero {j+1} = {mac(phi_n[:, m], phi_true[:, j]):.3f}")

    # ---- plot ----
    # Pannello 1: spettro dei valori singolari (in dB) con i picchi marcati.
    # Pannello 2: forme modali identificate (parte reale, sensore per sensore).
    fig, ax = plt.subplots(2, 1, figsize=(9, 7))
    s_db = 10*np.log10(S + 1e-30)
    for i in range(Ns):
        ax[0].plot(f, s_db[i], lw=0.8,
                   label=f"$\\sigma_{i+1}$" if i < 3 else None)
    ax[0].plot(f[idx], s_db[0, idx], "rv", ms=8, label="picchi")
    ax[0].set_xlim(0, 30); ax[0].set_xlabel("frequenza [Hz]")
    ax[0].set_ylabel("valori singolari [dB]")
    ax[0].set_title("FDD - SVD della CPSD")
    ax[0].legend(); ax[0].grid(True, alpha=0.3)

    for m in range(phi_n.shape[1]):
        ax[1].plot(np.arange(1, Ns+1), np.real(phi_n[:, m]),
                   "-o", label=f"modo {m+1} @ {f_n[m]:.2f} Hz")
    ax[1].set_xlabel("# sensore"); ax[1].set_ylabel("ampiezza modale")
    ax[1].set_title("Forme modali identificate")
    ax[1].legend(); ax[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("fdd_demo.png", dpi=120)
    print("salvato fdd_demo.png")


if __name__ == "__main__":
    demo()
