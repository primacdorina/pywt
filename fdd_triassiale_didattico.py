"""FDD triassiale: 6 sensori x 3 assi, una FDD indipendente per asse.

CONVENZIONE ASSI (dal progetto):
  X --> misura le accelerazioni LONGITUDINALI (lungo l'asse del ponte)
  Y --> misura le accelerazioni TRASVERSALI (laterali)
  Z --> misura le accelerazioni VERTICALI

DISPOSIZIONE SENSORI sulla campata:
  A1, A2 -> sezione a 8.75 m  (A1 lato sinistro, A2 lato destro)
  A3, A4 -> sezione a 17.5 m  (A3 lato sinistro, A4 lato destro)  -> mezzeria
  A5, A6 -> sezione a 26.25 m (A5 lato sinistro, A6 lato destro)

DISPOSIZIONE SENSORI (vista dall'alto, ponte lungo 35 m):

       y=1 (DESTRA) :       A2          A4          A6
                            |           |           |
                          x=8.75      x=17.5     x=26.25
                            |           |           |
       y=0 (SINISTRA):      A1          A3          A5

Asse del ponte (x) ----- 0 ------------- 17.5 m ------------- 35 m

SIMULAZIONE: per ogni asse abbiamo 3 modi a frequenze diverse:
  - Asse X (longitudinale): vede 2.5 Hz forte, 7.8 Hz debolissimo, 15.3 Hz assente
  - Asse Y (trasversale):   vede 2.5 Hz, NON vede 7.8 Hz, vede 15.3 Hz (forma a S)
  - Asse Z (verticale):     vede tutti e 3 i modi (parabola, torsione, forma a S)
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# mplcursors per le coordinate interattive (hover) - opzionale
try:
    import mplcursors
    HAS_MPLCURSORS = True
except ImportError:
    HAS_MPLCURSORS = False


# 1) PARAMETRI
fs = 200
n_seg = 1024
FDD_DURATION_SEC = 600
n_sensori = 6
ASSI = ["X", "Y", "Z"]
freq_modali = [2.5, 7.8, 15.3]
amp_modali = [1.0, 0.7, 0.5]
sigma_rumore = 0.15
x_sezioni = np.array([8.75, 17.5, 26.25])
idx_sin = [0, 2, 4]                                     # A1, A3, A5
idx_des = [1, 3, 5]                                     # A2, A4, A6
X_SENS = np.array([8.75, 8.75, 17.5, 17.5, 26.25, 26.25])  # x lungo il ponte
Y_SENS = np.array([0.0,  1.0,  0.0,  1.0,  0.0,   1.0])    # 0=sin, 1=des
L_PONTE = 35.0
NOMI = [f"A{i + 1}" for i in range(n_sensori)]

# Forme modali: definite secondo la fisica del ponte (6 sensori x 3 modi)
forme_modali_per_asse = {
    # ----- ASSE X (longitudinale) -----
    #          M1(2.5Hz)  M2(7.8Hz)  M3(15.3Hz)
    # A1-A6:   1.0        0.1->-0.1  0.0
    #
    # M1: tutti i sensori si muovono nella stessa direzione longitudinale
    #     -> traslazione quasi rigida del ponte lungo il suo asse
    # M2: piccola variazione (0.1 -> 0 -> -0.1) ma molto debole
    #     -> il picco a 7.8 Hz risultera' quasi invisibile su X
    # M3: colonna tutta zero -> 15.3 Hz NON appare su X
    #
    # Fisicamente ha senso che X veda pochi modi: le oscillazioni
    # longitudinali di un ponte sono poco energetiche.
    "X": np.array([
        [1.0,  0.1,  0.0],   # A1
        [1.0,  0.1,  0.0],   # A2
        [1.0,  0.0,  0.0],   # A3
        [1.0,  0.0,  0.0],   # A4
        [1.0, -0.1,  0.0],   # A5
        [1.0, -0.1,  0.0],   # A6
    ]),

    # ----- ASSE Y (trasversale) -----
    #          M1(2.5Hz)  M2(7.8Hz)  M3(15.3Hz)
    # A1-A6:   0.7        0.0        0.8->-0.8
    #
    # M1: tutti oscillano lateralmente insieme -> modo trasversale globale
    # M2: colonna tutta zero -> 7.8 Hz NON appare su Y
    # M3: A1/A2 vanno in un verso, A5/A6 nell'altro -> forma a S laterale
    "Y": np.array([
        [0.7,  0.0,  0.8],   # A1
        [0.7,  0.0,  0.8],   # A2
        [1.0,  0.0,  0.0],   # A3
        [1.0,  0.0,  0.0],   # A4
        [0.7,  0.0, -0.8],   # A5
        [0.7,  0.0, -0.8],   # A6
    ]),

    # ----- ASSE Z (verticale) -----
    #          M1(parabola)  M2(torsione)  M3(forma a S)
    # A1 sin:  [ 0.7,         +0.5,         +0.8 ]
    # A2 des:  [ 0.7,         -0.5,         +0.8 ]   <- M2: sin e des opposti = torsione
    # A3 sin:  [ 1.0,          0.0,          0.0 ]
    # A4 des:  [ 1.0,          0.0,          0.0 ]   <- M1: centro si muove di piu'
    # A5 sin:  [ 0.7,         +0.5,         -0.8 ]
    # A6 des:  [ 0.7,         -0.5,         -0.8 ]   <- M3: segno opposto al centro = S
    "Z": np.array([
        [0.7,  0.5,  0.8],   # A1
        [0.7, -0.5,  0.8],   # A2
        [1.0,  0.0,  0.0],   # A3
        [1.0,  0.0,  0.0],   # A4
        [0.7,  0.5, -0.8],   # A5
        [0.7, -0.5, -0.8],   # A6
    ]),
}

# QUADRO COMPLETO dei picchi attesi:
#         2.5 Hz       7.8 Hz       15.3 Hz
#   X     forte        debolissimo  assente
#   Y     presente     assente      presente
#   Z     presente     presente     presente

n_campioni = int(FDD_DURATION_SEC * fs)
t = np.arange(n_campioni) / fs
rng = np.random.default_rng()


# 2) generazione segnali (modi) (3 sorgenti modali condivise sui sensori)
fasi_iniziali = rng.uniform(0, 2 * np.pi, size=len(freq_modali))
sorgenti_modali = np.array([A_m * np.sin(2 * np.pi * f_m * t + phi_m)
    for f_m, A_m, phi_m in zip(freq_modali, amp_modali, fasi_iniziali)])

segnali_per_asse = {}
for asse in ASSI:
    segnali = forme_modali_per_asse[asse] @ sorgenti_modali  # ogni sensore misura una combinazione dei modi
    rumore = rng.normal(0.0, sigma_rumore, size=segnali.shape)
    segnali_per_asse[asse] = segnali + rumore


# 3) FDD
def calcola_cpsd(segnali, fs, n_seg):
    # CPSD S_xx(f) di shape (n_sensori, n_sensori, n_freq).
    n_sens, _ = segnali.shape
    f = np.fft.rfftfreq(n_seg, d=1./fs)
    S_xx = np.zeros((n_sens, n_sens, len(f)), dtype=complex)
    for i in range(n_sens):
        for j in range(n_sens):
            _, S_xx[i, j, :] = signal.csd(segnali[i], segnali[j], fs=fs, nperseg=n_seg)
    return f, S_xx


def svd_per_frequenza(S_xx):
    # SVD bin per bin -> sigma_all (n_freq, n_sens), U_all.
    n_sens, _, n_freq = S_xx.shape
    sigma_all = np.zeros((n_freq, n_sens))
    U_all = np.zeros((n_freq, n_sens, n_sens), dtype=complex)
    for k in range(n_freq):
        U, sk, _ = np.linalg.svd(S_xx[:, :, k])
        sigma_all[k, :] = sk
        U_all[k, :, :] = U
    return sigma_all, U_all


def trova_picchi(sigma_1, distanza_min=5, n_volte_mediana=5):
    # Peak picking sul 1o SV con soglia robusta.
    soglia = n_volte_mediana * np.median(sigma_1)
    picchi_idx, _ = signal.find_peaks(sigma_1, distance=distanza_min,
                                      height=soglia, prominence=soglia)
    return picchi_idx


def estrai_forma_modale(U_all, k):
    # Estrazione della forma modale (phi) dal 1o vettore singolare,
    # normalizzata in [-1, +1] e resa reale.
    #
    # Idea: una vera forma modale deflessiva ha solo fasi 0 o 180 gradi
    #       (i sensori si muovono in fase o in opposizione di fase).
    #       La SVD numerica restituisce un vettore complesso con una fase
    #       globale arbitraria: la "raddrizziamo" mettendo la componente
    #       di modulo massimo come riferimento reale positivo.
    #
    #   1) si prende la prima colonna di U
    #   2) si rende reale rendendo la componente di modulo massimo positiva
    #   3) si normalizza in [-1, +1] per comodita' di visualizzazione
    #      (non cambia la forma)
    phi = U_all[k, :, 0]
    idx_max = np.argmax(np.abs(phi))
    phi = phi * np.exp(-1j * np.angle(phi[idx_max]))
    phi = np.real(phi)                                    # si rende reale
    phi = phi / np.max(np.abs(phi))                       # normalizzazione
    return phi


def mac(a, b):
    # Modal Assurance Criterion: 1 = forme uguali (a meno di scala), 0 = ortogonali.
    # Robusto a vettori nulli (modi non eccitati su un asse).
    den = np.dot(a, a) * np.dot(b, b)
    if den == 0.0:
        return 0.0
    return float(np.abs(np.dot(a, b)) ** 2 / den)


# 4) PLOT
def plot_tutti_sv(risultati, fs, f_min=0):
    """UNA figura: tutti gli SV, lineare + dB, per tutti e tre gli assi (3x2)."""
    f_max = fs / 2
    fig, axes = plt.subplots(len(ASSI), 2, figsize=(14, 4 * len(ASSI)), sharex="col")
    fig.suptitle("Andamento di tutti i Valori Singolari per ogni asse (lineare e dB)",
                 fontweight="bold")
    for i, asse in enumerate(ASSI):
        f, sigma_all, _, _ = risultati[asse]
        n_sens = sigma_all.shape[1]
        sigma_all_dB = 10 * np.log10(sigma_all + 1e-20)
        # ----- scala lineare -----
        for j in range(n_sens):
            axes[i, 0].plot(f, sigma_all[:, j], label=f"SV {j + 1}", lw=1.0)
        axes[i, 0].set_xlim(f_min, f_max)
        axes[i, 0].set_xlabel("Frequenza [Hz]")
        axes[i, 0].set_ylabel("Valori singolari (lineare)")
        axes[i, 0].set_title(f"Asse {asse}: andamento dei {n_sens} SV - scala lineare")
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 0].legend(loc="upper right", ncol=2, fontsize=8)
        # ----- scala dB -----
        for j in range(n_sens):
            axes[i, 1].plot(f, sigma_all_dB[:, j], label=f"SV {j + 1}", lw=1.0)
        axes[i, 1].set_xlim(f_min, f_max)
        axes[i, 1].set_xlabel("Frequenza [Hz]")
        axes[i, 1].set_ylabel("Valori singolari [dB]")
        axes[i, 1].set_title(f"Asse {asse}: andamento dei {n_sens} SV - scala dB")
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].legend(loc="upper right", ncol=2, fontsize=8)
    fig.tight_layout()


def plot_sigma1_picchi(risultati, fs, f_min=0):
    """UNA figura: sigma_1 con picchi in rosso, lineare + dB, per tutti gli assi.

    Hover col mouse -> mostra (frequenza, ampiezza) sia in tooltip
    (mplcursors) sia nella status bar in basso (format_coord).
    """
    f_max = fs / 2
    fig, axes = plt.subplots(len(ASSI), 2, figsize=(14, 4 * len(ASSI)), sharex="col")
    fig.suptitle(
        "1o valore singolare con picchi - hover col cursore per leggere (f, ampiezza)",
        fontweight="bold",
    )
    artisti = []
    for i, asse in enumerate(ASSI):
        f, sigma_all, _, picchi_idx = risultati[asse]
        sigma_1 = sigma_all[:, 0]
        sigma_1_dB = 10 * np.log10(sigma_1 + 1e-20)
        freq_picchi = f[picchi_idx]

        # ----- scala lineare -----
        ln, = axes[i, 0].plot(f, sigma_1, color="C0", lw=1.2, label="1o SV (sigma_1)")
        artisti.append(ln)
        if len(picchi_idx):
            axes[i, 0].plot(freq_picchi, sigma_1[picchi_idx], "ro",
                            markersize=8, label="Picchi identificati")
            for fk, sk in zip(freq_picchi, sigma_1[picchi_idx]):
                axes[i, 0].annotate(f"{fk:.2f} Hz", xy=(fk, sk), xytext=(5, 5),
                                    textcoords="offset points")
        axes[i, 0].set_xlim(f_min, f_max)
        axes[i, 0].set_xlabel("Frequenza [Hz]")
        axes[i, 0].set_ylabel("sigma_1 (lineare)")
        axes[i, 0].set_title(f"Asse {asse}: 1o valore singolare con picchi - scala lineare")
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 0].legend(loc="upper right")
        axes[i, 0].format_coord = lambda x, y: f"f = {x:.3f} Hz   ampiezza = {y:.5f}"

        # ----- scala dB -----
        ln, = axes[i, 1].plot(f, sigma_1_dB, color="C0", lw=1.2, label="1o SV (sigma_1)")
        artisti.append(ln)
        if len(picchi_idx):
            axes[i, 1].plot(freq_picchi, sigma_1_dB[picchi_idx], "ro",
                            markersize=8, label="Picchi identificati")
            for fk, sk in zip(freq_picchi, sigma_1_dB[picchi_idx]):
                axes[i, 1].annotate(f"{fk:.2f} Hz", xy=(fk, sk), xytext=(5, 5),
                                    textcoords="offset points")
        axes[i, 1].set_xlim(f_min, f_max)
        axes[i, 1].set_xlabel("Frequenza [Hz]")
        axes[i, 1].set_ylabel("sigma_1 [dB]")
        axes[i, 1].set_title(f"Asse {asse}: 1o valore singolare con picchi - scala dB")
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].legend(loc="upper right")
        axes[i, 1].format_coord = lambda x, y: f"f = {x:.3f} Hz   ampiezza = {y:.2f} dB"

    # Tooltip al passaggio del mouse - se mplcursors e' installato
    if HAS_MPLCURSORS:
        cur = mplcursors.cursor(artisti, hover=True)

        @cur.connect("add")
        def _hover(sel):
            x, y = sel.target
            sel.annotation.set_text(f"f = {x:.3f} Hz\nA = {y:.4g}")
            sel.annotation.get_bbox_patch().set(fc="white", alpha=0.9)

    fig.tight_layout()


def plot_forma_modale(asse, f_k, phi):
    """Una figura per ogni picco: bar chart + vista dall'alto del ponte.

    Vista dall'alto: linee SOLIDE blu (sinistra) e rossa (destra),
    cerchietti come segnapunto sui sensori.
    """
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    fig.suptitle(f"Asse {asse}: forma modale a {f_k:.2f} Hz", fontweight="bold")

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

    # -------- (b) vista dall'alto del ponte (cerchietti, linee SOLIDE) -------
    ax[1].plot(x_sezioni, phi[idx_sin], "-o", color="tab:blue",
               ms=10, lw=1.6, label="Sinistra (A1, A3, A5)")
    ax[1].plot(x_sezioni, phi[idx_des], "-o", color="tab:red",
               ms=10, lw=1.6, label="Destra (A2, A4, A6)")
    for xs, p in zip(x_sezioni, phi[idx_sin]):
        ax[1].annotate(f"{p:+.2f}", (xs, p), textcoords="offset points",
                       xytext=(0, 10), ha="center", fontsize=9, color="tab:blue")
    for xs, p in zip(x_sezioni, phi[idx_des]):
        ax[1].annotate(f"{p:+.2f}", (xs, p), textcoords="offset points",
                       xytext=(0, -16), ha="center", fontsize=9, color="tab:red")
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_xlim(0, L_PONTE)
    ax[1].set_ylim(-1.35, 1.35)
    ax[1].set_xlabel("Posizione lungo il ponte [m]")
    ax[1].set_ylabel(r"$\varphi$  (normalizzato)")
    ax[1].set_title("(b) Vista dall'alto del ponte")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend(fontsize=9)

    fig.tight_layout()


# 5) MAIN: una FDD per asse
F_PLOT_MIN = 0
F_PLOT_MAX = fs / 2

risultati = {}
for asse in ASSI:
    segnali = segnali_per_asse[asse]
    f, S_xx = calcola_cpsd(segnali, fs, n_seg)
    sigma_all, U_all = svd_per_frequenza(S_xx)
    sigma_1 = sigma_all[:, 0]
    picchi_idx = trova_picchi(sigma_1)
    risultati[asse] = (f, sigma_all, U_all, picchi_idx)

# Figure aggregate (subplot per assi e scale)
plot_tutti_sv(risultati, fs, F_PLOT_MIN)
plot_sigma1_picchi(risultati, fs, F_PLOT_MIN)

# Una figura per ogni picco di ogni asse: bar + vista dall'alto
for asse in ASSI:
    f, _, U_all, picchi_idx = risultati[asse]
    for k in picchi_idx:
        phi = estrai_forma_modale(U_all, k)
        plot_forma_modale(asse, f[k], phi)

# Stampe come riassunto finale
np.set_printoptions(precision=3, suppress=True, linewidth=120)
for asse in ASSI:
    print("\n" + "=" * 72 + f"\n  ASSE {asse}\n" + "=" * 72)
    f, sigma_all, U_all, picchi_idx = risultati[asse]
    print(f"  Picchi trovati a:  {f[picchi_idx].round(3)} Hz")
    forme_vere = forme_modali_per_asse[asse]
    for k in picchi_idx:
        f_k = f[k]
        print(f"\n  --------- Picco a {f_k:.3f} Hz  (bin k = {k}) ---------")
        U_k = U_all[k]
        print(f"      U[k, :, 0]              = {U_k[:, 0]}")
        print(f"      |U[k, :, 0]|            = {np.abs(U_k[:, 0])}")
        print(f"      angle(U[k, :, 0]) [deg] = {np.degrees(np.angle(U_k[:, 0]))}")
        print(f"    sigma[k] = {sigma_all[k]}")
        phi = estrai_forma_modale(U_all, k)
        print(f"    phi (reale, normalizzato in [-1,+1]) = {phi}")
        macs = [mac(phi, forme_vere[:, j]) for j in range(forme_vere.shape[1])]
        j_best = int(np.argmax(macs))
        print(f"    MAC vs M1/M2/M3 = {[round(m, 3) for m in macs]}  "
              f"->  miglior match: M{j_best + 1}")

plt.show()
