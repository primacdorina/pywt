"""
FDD - parte didattica sulle FORME MODALI (vettori singolari sinistri U).

Obiettivo: capire cosa sono, come si estraggono e come si interpretano le
colonne di U che escono dalla SVD della matrice CPSD S_xx(f), bin per bin.

Schema del codice:
  1) Si genera un dataset triassiale sintetico (come nel tuo script principale)
     con FORME MODALI VERE note - cosi possiamo validare cio' che identifichiamo.
  2) Si calcola CPSD e SVD per un solo asse (esempio: X), passo per passo.
  3) Si estrae phi_identificato dalla 1a colonna di U ai picchi di sigma_1.
  4) Si confronta phi_identificato con la forma modale vera (MAC).
  5) Si producono plot mirati a leggere/interpretare U.
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------
# 1) PARAMETRI E DATI SINTETICI (stessa impostazione del tuo script)
# --------------------------------------------------------------------------
fs = 200
n_seg = 1024
DURATA_SEC = 600
n_sensori = 7
freq_modali = [2.5, 7.8, 15.3]      # frequenze "vere" dei modi
amp_modali  = [0.01, 0.7, 0.9]
sigma_rumore = 0.3

# Forme modali VERE: ogni colonna e' un modo (3 modi), ogni riga un sensore (7).
# Le useremo come "ground truth" per misurare quanto bene FDD le ricostruisce.
forme_modali_vere_X = np.array([
    [ 1.0,  0.3,  0.1 ],
    [ 0.8, -0.2,  0.1 ],
    [ 0.5, -0.4, -0.1 ],
    [ 0.2,  0.2, -0.1 ],
    [-0.3,  0.3,  0.05],
    [-0.7,  0.1, -0.05],
    [-1.0, -0.3,  0.1 ],
])  # shape (n_sensori, n_modi) = (7, 3)

# Generazione segnali: 3 sorgenti modali sinusoidali condivise sui sensori
n_campioni = int(DURATA_SEC * fs)
t = np.arange(n_campioni) / fs
rng = np.random.default_rng(0)        # seed per riproducibilita'
fasi = rng.uniform(0, 2*np.pi, size=len(freq_modali))
sorgenti = np.array([
    A * np.sin(2*np.pi*f*t + phi) for f, A, phi in zip(freq_modali, amp_modali, fasi)
])                                    # shape (n_modi, n_campioni)

# segnali asse X: combinazione lineare delle sorgenti pesata dalle forme modali
segnali = forme_modali_vere_X @ sorgenti + rng.normal(0, sigma_rumore,
                                                     size=(n_sensori, n_campioni))


# --------------------------------------------------------------------------
# 2) CPSD + SVD bin per bin
# --------------------------------------------------------------------------
def calcola_cpsd(x, fs, n_seg):
    """S_xx(f) di shape (n_sens, n_sens, n_freq), Hermitiana ad ogni f."""
    n_sens = x.shape[0]
    f = np.fft.rfftfreq(n_seg, d=1.0/fs)
    S = np.zeros((n_sens, n_sens, len(f)), dtype=complex)
    for i in range(n_sens):
        for j in range(n_sens):
            _, S[i, j, :] = signal.csd(x[i], x[j], fs=fs, nperseg=n_seg)
    return f, S


def svd_per_frequenza(S):
    """Per ogni f_k -> S(:,:,k) = U Sigma V*. Restituisce sigma_all e U_all."""
    n_sens, _, n_freq = S.shape
    sigma_all = np.zeros((n_freq, n_sens))           # valori singolari (reali)
    U_all = np.zeros((n_freq, n_sens, n_sens), dtype=complex)
    for k in range(n_freq):
        U, sk, _ = np.linalg.svd(S[:, :, k])
        sigma_all[k, :] = sk
        U_all[k, :, :] = U
    return sigma_all, U_all


f, S_xx = calcola_cpsd(segnali, fs, n_seg)
sigma_all, U_all = svd_per_frequenza(S_xx)
sigma_1 = sigma_all[:, 0]


# --------------------------------------------------------------------------
# 3) PEAK PICKING su sigma_1 e ESTRAZIONE FORME MODALI
# --------------------------------------------------------------------------
soglia = 5 * np.median(sigma_1)
picchi_idx, _ = signal.find_peaks(sigma_1, distance=5,
                                  height=soglia, prominence=soglia)
freq_picchi = f[picchi_idx]
print(f"\nPicchi trovati su sigma_1: {len(picchi_idx)}")
for fk in freq_picchi:
    print(f"  - {fk:.3f} Hz")


def estrai_forma_modale(U_all, k):
    """
    Forma modale dal 1o vettore singolare a bin k.

    Step:
      a) phi = U_all[k, :, 0]            -> vettore complesso (n_sens,)
      b) trova il sensore con |phi| max  -> idx_max
      c) ruota l'intero vettore in modo che phi[idx_max] sia REALE positivo
      d) prendi la parte reale (la parte immaginaria residua e' rumore/smorz.)
      e) normalizza max|phi| = 1
    """
    phi = U_all[k, :, 0]
    idx_max = np.argmax(np.abs(phi))
    phi = phi * np.exp(-1j * np.angle(phi[idx_max]))
    phi_im_residuo = np.max(np.abs(np.imag(phi)))    # diagnostica
    phi = np.real(phi)
    phi = phi / np.max(np.abs(phi))
    return phi, phi_im_residuo


# Estrai una forma modale per ogni picco e tieni anche la parte immaginaria
# residua (che ti dice quanto "puro" e' il modo: se e' grande sei in zona
# di modi accoppiati o forte smorzamento).
forme_identificate = []
imag_residue = []
for k in picchi_idx:
    phi, im_res = estrai_forma_modale(U_all, k)
    forme_identificate.append(phi)
    imag_residue.append(im_res)
forme_identificate = np.array(forme_identificate)    # shape (n_modi_id, n_sens)


# --------------------------------------------------------------------------
# 4) MODAL ASSURANCE CRITERION (MAC)
# --------------------------------------------------------------------------
def mac(phi_a, phi_b):
    """MAC(a, b) in [0, 1]: 1 = uguali, 0 = ortogonali. Insensibile a scala/segno."""
    num = np.abs(np.vdot(phi_a, phi_b))**2
    den = np.vdot(phi_a, phi_a).real * np.vdot(phi_b, phi_b).real
    return num / den


n_id = forme_identificate.shape[0]
n_vere = forme_modali_vere_X.shape[1]
MAC_matrix = np.zeros((n_id, n_vere))
for i in range(n_id):
    for j in range(n_vere):
        MAC_matrix[i, j] = mac(forme_identificate[i], forme_modali_vere_X[:, j])

print("\nMatrice MAC (righe = modi identificati, colonne = modi veri):")
print(np.round(MAC_matrix, 3))
print("\nInterpretazione: cerca un valore vicino a 1 in ogni riga: "
      "indica a quale modo vero corrisponde il modo identificato.")


# --------------------------------------------------------------------------
# 5) PLOT DIDATTICI SU U
# --------------------------------------------------------------------------

# (A) sigma_1 con i picchi - cosi' vedi DOVE estrai i modi
fig, ax = plt.subplots(figsize=(10, 4))
ax.semilogy(f, sigma_1, lw=1.0)
ax.semilogy(freq_picchi, sigma_1[picchi_idx], "ro", ms=8, label="Picchi")
for fk in freq_picchi:
    ax.axvline(fk, color="r", alpha=0.2, ls="--")
    ax.annotate(f"{fk:.2f} Hz", xy=(fk, sigma_1[picchi_idx[
        np.argmin(np.abs(freq_picchi - fk))]]),
        xytext=(5, 10), textcoords="offset points")
ax.set_xlim(0, 25)
ax.set_xlabel("Frequenza [Hz]")
ax.set_ylabel("sigma_1 (log)")
ax.set_title("Asse X - sigma_1: dove estraiamo le forme modali")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.tight_layout()


# (B) Le 7 colonne di U a UN picco specifico:
# vedi che SOLO la 1a e' "interpretabile", le altre sono rumore/ortogonali
k0 = picchi_idx[np.argmin(np.abs(freq_picchi - 2.5))]   # picco vicino a 2.5 Hz
fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), sharey=True)
for j in range(4):
    col = U_all[k0, :, j]
    col = col * np.exp(-1j * np.angle(col[np.argmax(np.abs(col))]))
    col = np.real(col); col = col / max(np.max(np.abs(col)), 1e-12)
    axes[j].stem(np.arange(n_sensori), col, basefmt=" ")
    axes[j].axhline(0, color="k", lw=0.5)
    axes[j].set_title(f"U[:, {j}]  (sigma={sigma_all[k0, j]:.2g})")
    axes[j].set_xlabel("Sensore")
    axes[j].grid(True, alpha=0.3)
axes[0].set_ylabel("Ampiezza (norm.)")
fig.suptitle(f"Asse X @ {f[k0]:.2f} Hz - prime 4 colonne di U "
             "(solo la 1a e' la forma modale)")
fig.tight_layout()


# (C) Confronto identificato vs vero per OGNI picco trovato + MAC sul titolo
fig, axes = plt.subplots(1, n_id, figsize=(4*n_id, 4), sharey=True)
if n_id == 1:
    axes = [axes]
for i, (phi_id, k) in enumerate(zip(forme_identificate, picchi_idx)):
    j_best = np.argmax(MAC_matrix[i, :])
    phi_vero = forme_modali_vere_X[:, j_best]
    phi_vero = phi_vero / np.max(np.abs(phi_vero))
    # se sono "anti-fase" (MAC e' simmetrico al segno), allinea il segno per il plot
    if np.dot(phi_id, phi_vero) < 0:
        phi_vero = -phi_vero
    x = np.arange(n_sensori)
    axes[i].stem(x - 0.1, phi_id,   linefmt="C0-", markerfmt="C0o",
                 basefmt=" ", label="identificato")
    axes[i].stem(x + 0.1, phi_vero, linefmt="C3--", markerfmt="C3s",
                 basefmt=" ", label="vero")
    axes[i].axhline(0, color="k", lw=0.5)
    axes[i].set_title(f"f={f[k]:.2f} Hz\nMAC={MAC_matrix[i, j_best]:.3f} "
                      f"(vs modo vero #{j_best+1})")
    axes[i].set_xlabel("Sensore")
    axes[i].grid(True, alpha=0.3)
    axes[i].legend(fontsize=8)
axes[0].set_ylabel("phi (norm.)")
fig.suptitle("Forme modali: identificate (FDD) vs vere (ground truth)")
fig.tight_layout()


# (D) Heatmap MAC: vista d'insieme della qualita' dell'identificazione
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(MAC_matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
for i in range(n_id):
    for j in range(n_vere):
        ax.text(j, i, f"{MAC_matrix[i,j]:.2f}", ha="center", va="center",
                color="w" if MAC_matrix[i,j] < 0.5 else "k", fontsize=10)
ax.set_xticks(range(n_vere))
ax.set_xticklabels([f"Modo vero {j+1}\n({freq_modali[j]} Hz)" for j in range(n_vere)])
ax.set_yticks(range(n_id))
ax.set_yticklabels([f"Id. {i+1}\n({freq_picchi[i]:.2f} Hz)" for i in range(n_id)])
plt.colorbar(im, ax=ax, label="MAC")
ax.set_title("Matrice MAC")
fig.tight_layout()


# --------------------------------------------------------------------------
# 6) STAMPA RIASSUNTO
# --------------------------------------------------------------------------
print("\n" + "="*70)
print("RIASSUNTO INTERPRETATIVO (asse X)")
print("="*70)
for i, (phi_id, k) in enumerate(zip(forme_identificate, picchi_idx)):
    j_best = np.argmax(MAC_matrix[i, :])
    print(f"\nModo identificato {i+1}: f = {f[k]:.3f} Hz")
    print(f"  -> matcha modo vero #{j_best+1} (atteso a {freq_modali[j_best]} Hz)"
          f"  con MAC = {MAC_matrix[i, j_best]:.3f}")
    print(f"  parte immaginaria residua su phi: {imag_residue[i]:.3g}  "
          "(piccola = modo 'pulito')")
    print(f"  phi_identificato = {np.round(phi_id, 2)}")
    print(f"  nodi (|phi|<0.15)  -> sensori {np.where(np.abs(phi_id)<0.15)[0].tolist()}")
    print(f"  antinodo principale -> sensore {int(np.argmax(np.abs(phi_id)))}")

plt.show()
