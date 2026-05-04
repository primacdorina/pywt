"""FDD triassiale - VERSIONE DIDATTICA per estrazione e plot delle forme modali.

Questo script chiarisce esattamente i 2 punti che spesso confondono:

   1) COME SI TIRANO FUORI I NUMERI DALLA MATRICE U
   2) COME SI PLOTTANO LE FORME MODALI

------------------------------------------------------------------------
TEORIA IN BREVE
------------------------------------------------------------------------
A ogni frequenza f_k il CPSD S_xx(f_k) e' una matrice n_sens x n_sens
hermitiana semidefinita positiva. La SVD la decompone come:

    S_xx(f_k) = U(f_k) @ diag(sigma) @ V(f_k)^*

con sigma_1 >= sigma_2 >= ... >= 0. ALLA RISONANZA un solo modo domina
la risposta: sigma_1 >> sigma_2, e la PRIMA COLONNA di U(f_k) e' la
forma modale del modo dominante (a meno di scala/segno/fase).

Estrazione "carta e penna":

    phi_complesso = U_all[k, :, 0]      # 6 numeri complessi
    # ruoto la fase in modo che la componente max diventi reale positiva
    phi_complesso *= exp(-i * angle(phi[idx_max]))
    phi_reale = real(phi_complesso)
    phi_normalizzato = phi_reale / max(|phi_reale|)

------------------------------------------------------------------------
DISPOSIZIONE SENSORI (vista dall'alto, ponte lungo 35 m):

       y=1 (DESTRA) :       A2          A4          A6
                            |           |           |
                          x=8.75      x=17.5     x=26.25
                            |           |           |
       y=0 (SINISTRA):      A1          A3          A5

Asse del ponte (x) ----- 0 ------------- 17.5 m ------------- 35 m

CONVENZIONE ASSI ACCELEROMETRO:
       X = longitudinale (lungo l'asse del ponte)
       Y = trasversale   (laterale)
       Z = verticale
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# =============================================================================
# 1) PARAMETRI E SIMULAZIONE (uguali al progetto originale)
# =============================================================================
fs = 200
n_seg = 1024
FDD_DURATION_SEC = 600
n_sensori = 6
ASSI = ["X", "Y", "Z"]
freq_modali = [2.5, 7.8, 15.3]
amp_modali = [1.0, 0.7, 0.5]
sigma_rumore = 0.15

x_sezioni = np.array([8.75, 17.5, 26.25])
idx_sin = [0, 2, 4]                                # A1, A3, A5
idx_des = [1, 3, 5]                                # A2, A4, A6
NOMI = [f"A{i + 1}" for i in range(n_sensori)]
X_SENS = np.array([8.75, 8.75, 17.5, 17.5, 26.25, 26.25])  # x lungo il ponte
Y_SENS = np.array([0.0,  1.0,  0.0,  1.0,  0.0,   1.0])    # 0=sin, 1=des
L_PONTE = 35.0

forme_modali_per_asse = {
    "X": np.array([
        [1.0,  0.1,  0.0],
        [1.0,  0.1,  0.0],
        [1.0,  0.0,  0.0],
        [1.0,  0.0,  0.0],
        [1.0, -0.1,  0.0],
        [1.0, -0.1,  0.0],
    ]),
    "Y": np.array([
        [0.7,  0.0,  0.8],
        [0.7,  0.0,  0.8],
        [1.0,  0.0,  0.0],
        [1.0,  0.0,  0.0],
        [0.7,  0.0, -0.8],
        [0.7,  0.0, -0.8],
    ]),
    "Z": np.array([
        [0.7,  0.5,  0.8],
        [0.7, -0.5,  0.8],
        [1.0,  0.0,  0.0],
        [1.0,  0.0,  0.0],
        [0.7,  0.5, -0.8],
        [0.7, -0.5, -0.8],
    ]),
}

n_camp = int(FDD_DURATION_SEC * fs)
t = np.arange(n_camp) / fs
rng = np.random.default_rng(0)                     # seed fisso = risultati stabili
fasi = rng.uniform(0.0, 2 * np.pi, size=len(freq_modali))
sorgenti = np.array([
    A * np.sin(2 * np.pi * f0 * t + ph)
    for f0, A, ph in zip(freq_modali, amp_modali, fasi)
])
segnali_per_asse = {
    asse: forme_modali_per_asse[asse] @ sorgenti
          + rng.normal(0.0, sigma_rumore, size=(n_sensori, n_camp))
    for asse in ASSI
}


# =============================================================================
# 2) FUNZIONI FDD
# =============================================================================
def calcola_cpsd(segnali, fs, n_seg):
    """CPSD S_xx(f) di shape (n_sens, n_sens, n_freq).

    S_xx[i, j, k] = densita' spettrale incrociata fra il sensore i e il
    sensore j al bin di frequenza k. Sulla diagonale (i==j) ho lo
    spettro di potenza del singolo sensore.
    """
    n_sens = segnali.shape[0]
    f, _ = signal.csd(segnali[0], segnali[0], fs=fs, nperseg=n_seg)
    S_xx = np.zeros((n_sens, n_sens, len(f)), dtype=complex)
    for i in range(n_sens):
        for j in range(n_sens):
            _, S_xx[i, j] = signal.csd(segnali[i], segnali[j], fs=fs, nperseg=n_seg)
    return f, S_xx


def svd_per_freq(S_xx):
    """SVD bin per bin.

    Restituisce:
       sigma  -> shape (n_freq, n_sens). sigma[k] sono i valori singolari
                 al bin k, ordinati: sigma[k, 0] e' il piu' grande.
       U_all  -> shape (n_freq, n_sens, n_sens). U_all[k] e' la matrice U
                 al bin k. Le COLONNE di U_all[k] sono i vettori singolari
                 sinistri.
    """
    n_sens, _, n_freq = S_xx.shape
    sigma = np.zeros((n_freq, n_sens))
    U_all = np.zeros((n_freq, n_sens, n_sens), dtype=complex)
    for k in range(n_freq):
        U_all[k], sigma[k], _ = np.linalg.svd(S_xx[:, :, k])
    return sigma, U_all


def trova_picchi(sigma_1, distanza_min=5, soglia_x_mediana=5.0):
    soglia = soglia_x_mediana * np.median(sigma_1)
    picchi, _ = signal.find_peaks(
        sigma_1, distance=distanza_min, height=soglia, prominence=soglia,
    )
    return picchi


# =============================================================================
# 3) ESTRAZIONE FORMA MODALE  --  IL PUNTO CHIAVE
# =============================================================================
def estrai_forma_modale(U_all, k):
    """Restituisce phi reale di lunghezza n_sens, normalizzato in [-1, +1].

    Indici di U_all[k, :, 0]:
        k  -> bin di frequenza (qui: il picco identificato)
        :  -> tutti i sensori (riga)
        0  -> PRIMA colonna di U  =  primo vettore singolare sinistro

    PASSO 1) Si prende la prima colonna:
                phi = U_all[k, :, 0]                      # complesso

    PASSO 2) Si rende reale:
                idx_max = argmax(|phi|)
                phi *= exp(-i * angle(phi[idx_max]))
                phi  = real(phi)

             Idea: una vera forma modale deflessiva ha solo fasi 0 o 180
             gradi (i sensori si muovono in fase o in opposizione di fase).
             La SVD numerica restituisce un vettore complesso con una fase
             globale arbitraria; la "raddrizziamo" mettendo la componente
             di modulo massimo come riferimento reale positivo.

    PASSO 3) Si normalizza:
                phi /= max(|phi|)
             Cosi' il sensore "leader" vale +1 e gli altri stanno fra -1 e +1.
    """
    phi = U_all[k, :, 0]                                            # 1)
    idx_max = int(np.argmax(np.abs(phi)))
    phi = phi * np.exp(-1j * np.angle(phi[idx_max]))                # 2a-b
    phi = np.real(phi)                                              # 2c
    phi = phi / np.max(np.abs(phi))                                 # 3
    return phi


def mac(a, b):
    """Modal Assurance Criterion: 1 = forme uguali (a meno di scala), 0 = ortogonali.

    Se uno dei due vettori e' nullo (modo che fisicamente non esiste su quell'asse)
    restituisce 0 invece di NaN.
    """
    den = np.dot(a, a) * np.dot(b, b)
    if den == 0.0:
        return 0.0
    return float(np.abs(np.dot(a, b)) ** 2 / den)


def normalizza_segno(phi):
    """Normalizza un vettore reale alla stessa convenzione di estrai_forma_modale.

    Se phi e' tutto zero lo restituisce com'e' (non c'e' niente da normalizzare).
    """
    if not np.any(phi):
        return phi
    idx_max = int(np.argmax(np.abs(phi)))
    return phi / phi[idx_max]


# =============================================================================
# 4) PLOT  --  3 PANNELLI COMPLEMENTARI PER OGNI FORMA MODALE
# =============================================================================
def plot_forma_modale(asse, f_k, phi, phi_vero, mac_val):
    """3 pannelli che mostrano la stessa forma modale da angolazioni diverse.

       (a) Bar chart con ETICHETTE NUMERICHE -> leggi i numeri
       (b) Vista in pianta sin vs des in funzione della posizione
            -> riconosci flessione / torsione / S-shape
       (c) Confronto identificata vs teorica con MAC -> validazione
    """
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

    # -------- (a) bar chart per sensore --------------------------------------
    colori = ["tab:blue" if Y_SENS[i] == 0 else "tab:red" for i in range(n_sensori)]
    bars = ax[0].bar(NOMI, phi, color=colori, edgecolor="k")
    for bar, val in zip(bars, phi):
        h = bar.get_height()
        offset = 0.04 if h >= 0 else -0.04
        ax[0].text(bar.get_x() + bar.get_width() / 2, h + offset,
                   f"{val:+.2f}", ha="center",
                   va="bottom" if h >= 0 else "top", fontsize=9)
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].set_ylim(-1.35, 1.35)
    ax[0].set_ylabel(r"$\varphi$  (normalizzato)")
    ax[0].set_title("(a) phi sensore-per-sensore")
    ax[0].grid(True, axis="y", alpha=0.3)
    ax[0].legend(handles=[Patch(color="tab:blue", label="Sinistra"),
                          Patch(color="tab:red",  label="Destra")],
                 loc="upper right", fontsize=9)

    # -------- (b) vista in pianta sinistra/destra ----------------------------
    ax[1].plot(x_sezioni, phi[idx_sin], "-ob", ms=10, label="Sinistra (A1,A3,A5)")
    ax[1].plot(x_sezioni, phi[idx_des], "--sr", ms=10, label="Destra (A2,A4,A6)")
    for xs, p in zip(x_sezioni, phi[idx_sin]):
        ax[1].annotate(f"{p:+.2f}", (xs, p), textcoords="offset points",
                       xytext=(0, 10), ha="center", fontsize=9, color="b")
    for xs, p in zip(x_sezioni, phi[idx_des]):
        ax[1].annotate(f"{p:+.2f}", (xs, p), textcoords="offset points",
                       xytext=(0, -16), ha="center", fontsize=9, color="r")
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_xlim(0, L_PONTE)
    ax[1].set_ylim(-1.35, 1.35)
    ax[1].set_xlabel("Posizione lungo il ponte [m]")
    ax[1].set_ylabel(r"$\varphi$  (normalizzato)")
    ax[1].set_title("(b) Vista in pianta")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend(fontsize=9)

    # -------- (c) confronto con la teorica + MAC -----------------------------
    i_sens = np.arange(n_sensori)
    w = 0.38
    ax[2].bar(i_sens - w / 2, phi, w, color="tab:blue", edgecolor="k",
              label="identificata")
    ax[2].bar(i_sens + w / 2, phi_vero, w, color="tab:orange", edgecolor="k",
              label="teorica")
    ax[2].set_xticks(i_sens)
    ax[2].set_xticklabels(NOMI)
    ax[2].axhline(0, color="k", lw=0.5)
    ax[2].set_ylim(-1.35, 1.35)
    ax[2].set_title(f"(c) Identificata vs teorica  -  MAC = {mac_val:.3f}")
    ax[2].grid(True, axis="y", alpha=0.3)
    ax[2].legend(fontsize=9)

    fig.suptitle(f"ASSE {asse}  -  forma modale a {f_k:.2f} Hz", fontweight="bold")
    fig.tight_layout()


def plot_sigma_1(asse, f, sigma_1, picchi):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.semilogy(f, sigma_1, lw=1.0)
    if len(picchi):
        ax.semilogy(f[picchi], sigma_1[picchi], "ro", ms=8)
        for k in picchi:
            ax.annotate(f"{f[k]:.2f} Hz", (f[k], sigma_1[k]),
                        textcoords="offset points", xytext=(6, 6))
    ax.set_xlim(0, fs / 2)
    ax.set_xlabel("Frequenza [Hz]")
    ax.set_ylabel(r"$\sigma_1$ (log)")
    ax.set_title(f"Asse {asse}: 1o valore singolare con picchi")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()


# =============================================================================
# 5) MAIN: una FDD per asse + estrazione esplicita
# =============================================================================
np.set_printoptions(precision=3, suppress=True, linewidth=120)

for asse in ASSI:
    print("\n" + "=" * 72 + f"\n  ASSE {asse}\n" + "=" * 72)

    seg = segnali_per_asse[asse]
    f, S_xx = calcola_cpsd(seg, fs, n_seg)
    sigma, U_all = svd_per_freq(S_xx)
    sigma_1 = sigma[:, 0]
    picchi = trova_picchi(sigma_1)

    print(f"  Picchi trovati:  {f[picchi].round(3)} Hz")
    plot_sigma_1(asse, f, sigma_1, picchi)

    forme_vere = forme_modali_per_asse[asse]

    for k in picchi:
        f_k = f[k]
        print(f"\n  --------- PICCO a {f_k:.3f} Hz  (bin k = {k}) ---------")

        # Mostro che U[k] e' una matrice 6x6 e che ci interessa solo la 1a colonna
        U_k = U_all[k]
        print(f"    U[k] e' una matrice {U_k.shape[0]}x{U_k.shape[1]} complessa.")
        print(f"    Mi serve solo la PRIMA COLONNA -> U[k, :, 0]:")
        print(f"      U[k, :, 0] = {U_k[:, 0]}")
        print(f"      |U[k, :, 0]| = {np.abs(U_k[:, 0])}")
        print(f"      angle(U[k, :, 0]) [deg] = {np.degrees(np.angle(U_k[:, 0]))}")
        print(f"    sigma[k] = {sigma[k]}")
        print(f"    -> sigma_1/sigma_2 = {sigma[k, 0] / sigma[k, 1]:.1f} (alto = un solo modo domina)")

        # Estrazione e stampa della forma modale finale
        phi = estrai_forma_modale(U_all, k)
        print(f"    phi (reale, normalizzato in [-1,+1]) = {phi}")

        # Confronto col modo teorico piu' simile
        macs = [mac(phi, forme_vere[:, j]) for j in range(forme_vere.shape[1])]
        j_best = int(np.argmax(macs))
        print(f"    MAC(phi, M1)={macs[0]:.3f}  MAC(phi, M2)={macs[1]:.3f}  "
              f"MAC(phi, M3)={macs[2]:.3f}   ->  miglior match: M{j_best + 1}")

        # Preparo la teorica per il plot (allineo segno per leggibilita')
        phi_vero = normalizza_segno(forme_vere[:, j_best].astype(float))
        if np.dot(phi, phi_vero) < 0:
            phi_vero = -phi_vero

        plot_forma_modale(asse, f_k, phi, phi_vero, macs[j_best])

plt.show()
