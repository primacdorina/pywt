"""
Frequency Domain Decomposition (FDD) - implementazione pratica.

Pipeline:
    1. CPSD (Welch) -> Gyy(f) di forma (Ns, Ns, Nf)
    2. SVD per ogni frequenza -> sigma_i(f), U(f)
    3. Picchi del primo valore singolare -> frequenze naturali
    4. Primo vettore singolare al picco -> forma modale
"""

import numpy as np
from scipy.signal import csd, find_peaks
import matplotlib.pyplot as plt


def compute_cpsd_matrix(Y, fs, nperseg, noverlap=None, window="hann"):
    """
    Calcola la matrice CPSD Gyy(f) di forma (Ns, Ns, Nf).

    Y       : array (Ns, Nt) - segnali dei sensori
    fs      : frequenza di campionamento [Hz]
    nperseg : lunghezza del segmento per Welch
    """
    Ns, Nt = Y.shape
    if noverlap is None:
        noverlap = nperseg // 2

    f, _ = csd(Y[0], Y[0], fs=fs, nperseg=nperseg,
               noverlap=noverlap, window=window)
    Nf = len(f)
    Gyy = np.zeros((Ns, Ns, Nf), dtype=complex)

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
    Ritorna:
        S : (Ns, Nf)   valori singolari ordinati decrescenti
        U : (Ns, Ns, Nf) vettori singolari sinistri
    """
    Ns, _, Nf = Gyy.shape
    S = np.zeros((Ns, Nf))
    U = np.zeros((Ns, Ns, Nf), dtype=complex)
    for k in range(Nf):
        Uk, sk, _ = np.linalg.svd(Gyy[:, :, k])
        S[:, k] = sk
        U[:, :, k] = Uk
    return S, U


def identify_modes(f, S, U, fmin=0.5, fmax=None,
                   prominence_db=10, min_distance_hz=0.3):
    """
    Trova picchi del primo valore singolare ed estrae le forme modali.

    Ritorna:
        f_n   : frequenze naturali [Hz]
        phi_n : forme modali normalizzate (Ns, n_modi)
        idx   : indici dei picchi nel vettore frequenze
    """
    if fmax is None:
        fmax = f[-1]

    s1_db = 10 * np.log10(S[0] + 1e-30)
    mask = (f >= fmin) & (f <= fmax)

    df = f[1] - f[0]
    distance = max(int(min_distance_hz / df), 1)

    peaks, _ = find_peaks(s1_db[mask], prominence=prominence_db,
                          distance=distance)
    idx = np.where(mask)[0][peaks]

    f_n = f[idx]
    phi_n = U[:, 0, idx]                           # primo vett. singolare
    # normalizzazione a parte reale + max unitario
    for k in range(phi_n.shape[1]):
        v = phi_n[:, k]
        v = v * np.exp(-1j * np.angle(v[np.argmax(np.abs(v))]))
        phi_n[:, k] = v / np.max(np.abs(v))
    return f_n, phi_n, idx


def mac(phi1, phi2):
    """Modal Assurance Criterion fra due vettori modali."""
    num = np.abs(np.vdot(phi1, phi2)) ** 2
    den = np.vdot(phi1, phi1).real * np.vdot(phi2, phi2).real
    return num / den


# ---------------------------------------------------------------------------
# DEMO: 7 sensori, struttura "shear-type" simulata con 3 modi, fs=200, nperseg=1024
# ---------------------------------------------------------------------------
def demo():
    rng = np.random.default_rng(0)

    fs = 200.0
    nperseg = 1024
    T = 120.0                                       # 2 minuti
    Nt = int(T * fs)
    t = np.arange(Nt) / fs

    Ns = 7
    f_true = np.array([3.2, 8.7, 15.4])             # frequenze "vere" [Hz]
    zeta   = np.array([0.01, 0.015, 0.02])          # smorzamenti

    # forme modali "vere" (Ns x n_modi) - tipica struttura a mensola
    phi_true = np.array([
        [np.sin(np.pi*(i+1)/(Ns+1) * (m+1)) for m in range(len(f_true))]
        for i in range(Ns)
    ])

    # genero ogni modo come oscillatore eccitato da rumore bianco
    Y = np.zeros((Ns, Nt))
    for m, (fm, zm) in enumerate(zip(f_true, zeta)):
        wn = 2*np.pi*fm
        wd = wn*np.sqrt(1-zm**2)
        # risposta SDOF a rumore bianco -> integrazione semplice
        x = np.zeros(Nt); v = 0.0
        dt = 1/fs
        force = rng.standard_normal(Nt)
        for k in range(1, Nt):
            a = force[k] - 2*zm*wn*v - wn**2*x[k-1]
            v += a*dt
            x[k] = x[k-1] + v*dt
        Y += np.outer(phi_true[:, m], x)

    # rumore di misura sui sensori
    Y += 0.02*np.std(Y)*rng.standard_normal(Y.shape)

    # ---- FDD ----
    f, Gyy = compute_cpsd_matrix(Y, fs=fs, nperseg=nperseg)
    S, U = fdd_svd(Gyy)
    f_n, phi_n, idx = identify_modes(f, S, U, fmin=1.0, fmax=fs/2,
                                     prominence_db=8, min_distance_hz=1.0)

    print("Frequenze naturali identificate [Hz]:", np.round(f_n, 3))
    print("Frequenze vere [Hz]:                 ", f_true)
    for m in range(min(len(f_n), len(f_true))):
        # confronto MAC con la forma modale vera piu' vicina
        diffs = np.abs(f_true - f_n[m])
        j = np.argmin(diffs)
        print(f" modo {m+1}: f={f_n[m]:6.3f} Hz  "
              f"MAC vs modo vero {j+1} = {mac(phi_n[:, m], phi_true[:, j]):.3f}")

    # ---- plot ----
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
