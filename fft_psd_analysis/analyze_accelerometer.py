#!/usr/bin/env python3
"""Analisi FFT e PSD di dati accelerometrici da prove di vibrazione su trave/ponte.

Pipeline:
    1. Caricamento dati da CSV
    2. Pre-processing: rimozione DC, detrend, filtro passa-banda
    3. FFT (rFFT) con finestra di Hann -> spettro di ampiezza a singolo lato
    4. PSD via metodo di Welch (segmenti finestrati con overlap)
    5. Peak-picking sulla PSD per identificare le frequenze proprie
    6. Plot multi-pannello (serie temporale | FFT | PSD) per ogni canale

Esempi d'uso:
    # Dati reali
    python analyze_accelerometer.py dati.csv --fs 1000 --channels ax ay az

    # Solo certi canali, banda 1-200 Hz
    python analyze_accelerometer.py dati.csv --fs 2048 \\
        --channels acc1 acc2 --hp 1 --lp 200

    # Modalita' demo: genera dati sintetici e fa girare la pipeline
    python analyze_accelerometer.py --demo
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal


def load_csv(path: Path, time_col: str | None, channels: list[str] | None):
    """Legge un CSV e ritorna (tempo, dati[N,n_ch], nomi_canali).

    Se ``time_col`` non e' specificato, l'asse temporale viene ricostruito da fs.
    Se ``channels`` non e' specificato, prende tutte le colonne numeriche
    diverse da ``time_col``.
    """
    df = pd.read_csv(path)
    t = df[time_col].to_numpy() if (time_col and time_col in df.columns) else None
    if channels is None:
        channels = [
            c for c in df.columns
            if c != time_col and np.issubdtype(df[c].dtype, np.number)
        ]
    data = df[channels].to_numpy(dtype=float)
    return t, data, list(channels)


def preprocess(
    x: np.ndarray,
    fs: float,
    hp_cutoff: float | None = 0.5,
    lp_cutoff: float | None = None,
) -> np.ndarray:
    """Rimuove DC, applica detrend lineare e filtro Butterworth zero-phase.

    Filtro applicato con ``sosfiltfilt`` (zero phase distortion), ordine 4.
    """
    x = x - np.mean(x, axis=0)
    x = signal.detrend(x, axis=0, type="linear")
    nyq = 0.5 * fs
    if hp_cutoff and hp_cutoff > 0:
        sos = signal.butter(4, hp_cutoff / nyq, btype="high", output="sos")
        x = signal.sosfiltfilt(sos, x, axis=0)
    if lp_cutoff and 0 < lp_cutoff < nyq:
        sos = signal.butter(4, lp_cutoff / nyq, btype="low", output="sos")
        x = signal.sosfiltfilt(sos, x, axis=0)
    return x


def compute_fft(x: np.ndarray, fs: float, window: str = "hann"):
    """Spettro di ampiezza a singolo lato. Compensa il guadagno coerente della finestra.

    Ritorna (freq[Hz], amp[m/s^2]) con shape (N//2+1,) o (N//2+1, n_ch).
    """
    if x.ndim == 1:
        x = x[:, None]
    n_samples = x.shape[0]
    w = signal.get_window(window, n_samples)
    coherent_gain = w.mean()
    Xf = np.fft.rfft(x * w[:, None], axis=0)
    freq = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    amp = np.abs(Xf) / (n_samples * coherent_gain) * 2.0
    # I bin DC e (se N pari) Nyquist non vanno raddoppiati
    amp[0] /= 2.0
    if n_samples % 2 == 0:
        amp[-1] /= 2.0
    return freq, amp.squeeze()


def compute_psd(
    x: np.ndarray,
    fs: float,
    nperseg: int | None = None,
    overlap: float = 0.5,
    window: str = "hann",
):
    """PSD a singolo lato con il metodo di Welch.

    ``nperseg`` di default e' ~4 secondi (compromesso risoluzione/varianza).
    Unita' di misura della PSD: (unita' di x)^2 / Hz.
    """
    if nperseg is None:
        nperseg = min(x.shape[0], int(fs * 4))
        # Forziamo potenza di 2 per FFT efficiente
        nperseg = 1 << (nperseg - 1).bit_length() >> 1
        nperseg = max(nperseg, 256)
    noverlap = int(nperseg * overlap)
    freq, Pxx = signal.welch(
        x, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap,
        axis=0, scaling="density", detrend=False,
    )
    return freq, Pxx


def find_modes(
    freq: np.ndarray,
    psd: np.ndarray,
    prominence_db: float = 10.0,
    fmin: float = 1.0,
    distance_hz: float = 0.5,
):
    """Peak picking sulla PSD in dB. Ritorna (frequenze_picco, valori_PSD)."""
    mask = freq >= fmin
    f_m = freq[mask]
    p_m = psd[mask]
    psd_db = 10.0 * np.log10(p_m + 1e-30)
    df = f_m[1] - f_m[0] if len(f_m) > 1 else 1.0
    peaks, _ = signal.find_peaks(
        psd_db, prominence=prominence_db, distance=max(int(distance_hz / df), 1)
    )
    return f_m[peaks], p_m[peaks]


def plot_results(t, x, fs, f_fft, amp, f_psd, psd, channels, modes_per_ch, out_path):
    if x.ndim == 1:
        x = x[:, None]
        amp = amp[:, None]
        psd = psd[:, None]
    if t is None:
        t = np.arange(x.shape[0]) / fs
    n_ch = x.shape[1]
    fig, axes = plt.subplots(n_ch, 3, figsize=(16, 3.2 * n_ch), squeeze=False)

    for i in range(n_ch):
        modes_f, modes_p = modes_per_ch[i]

        ax_t, ax_f, ax_p = axes[i]

        ax_t.plot(t, x[:, i], lw=0.6, color="steelblue")
        ax_t.set(xlabel="Tempo [s]", ylabel="Accel [m/s²]",
                 title=f"{channels[i]} — serie temporale")
        ax_t.grid(alpha=0.3)

        ax_f.semilogy(f_fft, amp[:, i], lw=0.7, color="darkorange")
        ax_f.set(xlabel="Frequenza [Hz]", ylabel="|A(f)| [m/s²]",
                 title=f"{channels[i]} — Spettro FFT (Hann)")
        ax_f.grid(alpha=0.3, which="both")
        ax_f.set_xlim(0, fs / 2)

        ax_p.semilogy(f_psd, psd[:, i], lw=0.8, color="seagreen")
        if len(modes_f) > 0:
            ax_p.plot(modes_f, modes_p, "rv", ms=8, label=f"Modi ({len(modes_f)})")
            for fm in modes_f:
                ax_p.axvline(fm, color="red", alpha=0.2, ls="--")
            ax_p.legend(loc="upper right")
        ax_p.set(xlabel="Frequenza [Hz]", ylabel="PSD [(m/s²)²/Hz]",
                 title=f"{channels[i]} — PSD (Welch)")
        ax_p.grid(alpha=0.3, which="both")
        ax_p.set_xlim(0, fs / 2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def make_demo_data(fs: float, duration: float, modes_hz=(7.3, 23.1, 48.5, 91.2)):
    """Genera un segnale sintetico con modi noti + rumore + drift, per test rapidi."""
    rng = np.random.default_rng(42)
    n = int(fs * duration)
    t = np.arange(n) / fs
    sig = np.zeros((n, 3))
    amps = [(1.0, 0.6, 0.3, 0.15), (0.8, 0.7, 0.4, 0.2), (0.4, 0.5, 0.6, 0.25)]
    damp = [0.005, 0.008, 0.012, 0.020]
    for ch in range(3):
        for f, a, d in zip(modes_hz, amps[ch], damp):
            phase = rng.uniform(0, 2 * np.pi)
            envelope = np.exp(-d * 2 * np.pi * f * (t - t[0]))  # leggero decadimento
            envelope = 0.7 + 0.3 * envelope  # mantiene quasi stazionario
            sig[:, ch] += a * envelope * np.sin(2 * np.pi * f * t + phase)
        sig[:, ch] += 0.05 * rng.standard_normal(n)             # rumore
        sig[:, ch] += 0.001 * t                                  # drift lineare
        sig[:, ch] += 0.5                                        # offset DC (gravita')
    return t, sig, ["acc_x", "acc_y", "acc_z"]


def main():
    p = argparse.ArgumentParser(
        description="Analisi FFT e PSD per dati accelerometrici (trave/ponte).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("csv", nargs="?", type=Path, help="File CSV con i dati (omettere se --demo)")
    p.add_argument("--fs", type=float, help="Frequenza di campionamento [Hz]")
    p.add_argument("--time-col", default=None, help="Nome colonna tempo (opzionale)")
    p.add_argument("--channels", nargs="+", default=None,
                   help="Colonne canale (default: tutte numeriche)")
    p.add_argument("--hp", type=float, default=0.5,
                   help="Cutoff passa-alto [Hz] (default 0.5; 0 per disabilitare)")
    p.add_argument("--lp", type=float, default=None,
                   help="Cutoff passa-basso [Hz] (default: nessuno)")
    p.add_argument("--nperseg", type=int, default=None,
                   help="Lunghezza segmento Welch (default ~4 s, potenza di 2)")
    p.add_argument("--prominence-db", type=float, default=10.0,
                   help="Prominenza minima dei picchi PSD in dB (default 10)")
    p.add_argument("--fmin", type=float, default=1.0,
                   help="Frequenza minima per il peak picking [Hz] (default 1)")
    p.add_argument("-o", "--out", type=Path, default=Path("fft_psd_results.png"),
                   help="File immagine di output")
    p.add_argument("--save-modes", type=Path, default=None,
                   help="Se specificato, salva la tabella dei modi in CSV")
    p.add_argument("--demo", action="store_true",
                   help="Genera dati sintetici e fa girare la pipeline (smoke test)")
    args = p.parse_args()

    if args.demo:
        fs = args.fs or 1000.0
        print(f"[DEMO] Genero dati sintetici a {fs} Hz, 20 s, 3 canali...")
        t, x, channels = make_demo_data(fs, duration=20.0)
    else:
        if args.csv is None or args.fs is None:
            p.error("Servono il file CSV e --fs (oppure usa --demo).")
        print(f"Caricamento {args.csv}...")
        t, x, channels = load_csv(args.csv, args.time_col, args.channels)
        fs = args.fs

    if x.ndim == 1:
        x = x[:, None]
    print(f"  Shape: {x.shape[0]} campioni x {x.shape[1]} canali  ({channels})")
    print(f"  Durata: {x.shape[0] / fs:.2f} s   Nyquist: {fs / 2:.1f} Hz")

    print("Pre-processing (DC removal + detrend + filtro)...")
    x_proc = preprocess(x, fs, hp_cutoff=args.hp, lp_cutoff=args.lp)

    print("Computing FFT...")
    f_fft, amp = compute_fft(x_proc, fs)

    print("Computing PSD (Welch)...")
    f_psd, psd = compute_psd(x_proc, fs, nperseg=args.nperseg)

    print("Peak picking (frequenze proprie candidate):")
    if psd.ndim == 1:
        psd = psd[:, None]
    if amp.ndim == 1:
        amp = amp[:, None]
    modes_rows = []
    modes_per_ch = []
    for i, ch in enumerate(channels):
        f_modes, p_modes = find_modes(
            f_psd, psd[:, i],
            prominence_db=args.prominence_db, fmin=args.fmin,
        )
        modes_per_ch.append((f_modes, p_modes))
        if len(f_modes):
            print(f"  {ch:>10s}: " + ", ".join(f"{f:7.2f} Hz" for f in f_modes))
        else:
            print(f"  {ch:>10s}: nessun picco con prominenza >= {args.prominence_db} dB")
        for f, p in zip(f_modes, p_modes):
            modes_rows.append({"channel": ch, "freq_hz": f, "psd": p})

    if args.save_modes and modes_rows:
        pd.DataFrame(modes_rows).to_csv(args.save_modes, index=False)
        print(f"Modi salvati in: {args.save_modes}")

    print(f"Generazione plot -> {args.out}...")
    plot_results(t, x_proc, fs, f_fft, amp, f_psd, psd, channels, modes_per_ch, args.out)
    print("Done.")


if __name__ == "__main__":
    main()
