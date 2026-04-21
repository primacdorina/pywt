import os
import queue
import uuid
import time
import threading
import numpy as np
import pandas as pd
from scipy import signal
from collections import defaultdict, deque

import matplotlib
matplotlib.use('Agg')  # Previene l'apertura di finestre GUI
import matplotlib.pyplot as plt

from lib.utils import get_logger, ensure_dir
from lib.database_pro import StatData, FFTData, PSDData, FDDData
from sensor_map import SENSOR_FREQUENCIES, SENSOR_TYPES

logger = get_logger("shm_analysis")

# ==========================================
# CONFIGURAZIONE ANALISI
# ==========================================
ENABLE_STATISTICS = True
ENABLE_FFT = True
ENABLE_PSD = True
ENABLE_FDD = True
ENABLE_PLOTS = True

# Analisi Standard (Stat, FFT, PSD)
ANALYSIS_INTERVAL_SEC = 60  # Ampiezza della finestra temporale (window) e intervallo di calcolo per analisi standard

# Analisi Modale (FDD)
FDD_DURATION_SEC = 600      # Lunghezza della finestra temporale da analizzare (es. ultimi 10 min)
FDD_INTERVAL_SEC = 600    # Ogni quanto eseguire la FDD (es. 3600s = ogni ora)

ACCEL_SENSORS = [f"accel_mems_{str(i).zfill(2)}" for i in range(1, 10)]
FS_ACCEL = 200 # Hz (frequenza campionamento accelerometri)
# ==========================================

# Buffer per FDD: fdd_history mantiene abbastanza dati per coprire la durata richiesta
fdd_sync_buffer = defaultdict(dict)
fdd_history = deque(maxlen=FDD_DURATION_SEC * FS_ACCEL)

# Variabile di controllo per l'intervallo FDD (basata sui secondi di dati processati)
seconds_since_last_fdd = 0

def calculate_fdd_async(data_matrix, timestamp, db_session_maker, q_mqtt, plot_dir):
    """
    FDD - Frequency Domain Decomposition (Brincker et al., 2000).

    Input:
        data_matrix: shape (n_campioni, n_sensori) — colonne = sensori accel sincroni

    Procedura:
    ┌─────────────────────────────────────────────────────────────────┐
    │ STEP 1 — Matrice spettrale Syy                                  │
    │   Syy e' una matrice (n_sensori x n_sensori) per ogni freq f.   │
    │   Ogni elemento Syy[i,j,f] e' la Cross-PSD tra sensore i e j:  │
    │     - se i == j  →  auto-PSD del sensore i  (come Welch)        │
    │     - se i != j  →  cross-PSD tra i e j  (correlazione spaziale)│
    │   La cross-PSD misura quanta energia i due sensori condividono   │
    │   a quella frequenza. E' calcolata con signal.csd, che usa lo   │
    │   stesso algoritmo di Welch (blocchi sovrapposti + media).      │
    │                                                                  │
    │   IMPORTANTE: signal.csd viene chiamato 9x9=81 volte (una per   │
    │   coppia), ciascuna restituisce l'intero array di frequenze.    │
    │   Cosi' si calcola una sola volta tutto, non frequenza per freq.│
    ├─────────────────────────────────────────────────────────────────┤
    │ STEP 2 — SVD per ogni frequenza                                  │
    │   Per ogni riga di frequenza k si fa:                           │
    │       Syy[:,:,k] = U * diag(sigma) * V^H                       │
    │   sigma[0] >= sigma[1] >= sigma[2] ...                          │
    │   sigma[0] (sigma1) rappresenta la "potenza del modo dominante" │
    │   a quella frequenza.                                            │
    ├─────────────────────────────────────────────────────────────────┤
    │ STEP 3 — Peak picking                                            │
    │   I picchi di sigma1(f) corrispondono alle frequenze naturali.  │
    │   Si normalizza sigma1 e si cercano i picchi piu' prominenti.   │
    └─────────────────────────────────────────────────────────────────┘
    """
    logger.info(f"Avvio calcolo FDD asincrono: blocco terminante al {timestamp}")
    session = db_session_maker()
    try:
        n_campioni, n_sensori = data_matrix.shape
        nperseg = 1024
        noverlap = nperseg // 2  # 50% overlap, standard Welch

        # ── STEP 1: Costruzione matrice spettrale Syy ──────────────────────
        # signal.csd(x, y) = Cross-PSD tra x e y, calcolata con metodo Welch:
        #   - divide i segnali in blocchi di nperseg campioni
        #   - applica una finestra di Hann su ogni blocco
        #   - calcola la FFT incrociata di ogni blocco
        #   - media i risultati → stima robusta della Cross-PSD
        # Restituisce: (frequenze, Sxy) entrambi array di lunghezza nperseg//2 + 1
        #
        # Chiamiamo csd UNA VOLTA per ogni coppia (i,j) e salviamo tutto Syy.
        # NON chiamare csd dentro il loop sulle frequenze: sarebbe 513x81 chiamate
        # invece di 81, ricalcolando la stessa cosa migliaia di volte.

        frequenze, _ = signal.csd(data_matrix[:, 0], data_matrix[:, 0],
                                  fs=FS_ACCEL, nperseg=nperseg, noverlap=noverlap)
        n_freq = len(frequenze)

        # Syy[i, j, k] = Cross-PSD tra sensore i e sensore j alla frequenza k
        Syy = np.zeros((n_sensori, n_sensori, n_freq), dtype=complex)
        for i in range(n_sensori):
            for j in range(n_sensori):
                _, Syy[i, j, :] = signal.csd(data_matrix[:, i], data_matrix[:, j],
                                              fs=FS_ACCEL, nperseg=nperseg, noverlap=noverlap)

        # ── STEP 2: SVD di Syy ad ogni frequenza ──────────────────────────
        # Per ogni frequenza k, Syy[:,:,k] e' una matrice 9x9 complessa.
        # np.linalg.svd restituisce U, sigma, Vh dove:
        #   - U: vettori singolari sinistri (approssimano le forme modali)
        #   - sigma: valori singolari in ordine decrescente
        #   - Vh: vettori singolari destri (coniugati trasposti)
        # sigma[0] e' il piu' grande: indica quanto "forte" e' il modo dominante
        # a quella frequenza. Se a f=2.3 Hz la struttura ha un modo proprio,
        # sigma[0] avra' un picco netto a 2.3 Hz.

        sigma1 = np.zeros(n_freq)  # primo valore singolare (modo dominante)
        sigma2 = np.zeros(n_freq)  # secondo (modo secondario)
        sigma3 = np.zeros(n_freq)  # terzo

        for k in range(n_freq):
            _, valori_singolari, _ = np.linalg.svd(Syy[:, :, k])
            sigma1[k] = valori_singolari[0]
            if n_sensori > 1:
                sigma2[k] = valori_singolari[1]
            if n_sensori > 2:
                sigma3[k] = valori_singolari[2]

        # ── STEP 3: Peak picking su sigma1 ────────────────────────────────
        # Banda strutturale di interesse: 0.1 Hz (esclude deriva DC) - 25 Hz
        banda_mask = (frequenze >= 0.1) & (frequenze <= 25.0)
        freq_banda  = frequenze[banda_mask]
        sigma1_banda = sigma1[banda_mask]

        # Normalizza sigma1 tra 0 e 1 per rendere la soglia indipendente
        # dall'ampiezza assoluta (che dipende dall'intensita' del rumore ambientale)
        sigma1_norm = sigma1_banda / (np.max(sigma1_banda) + 1e-10)

        # Cerca picchi: altezza minima 30% del massimo, almeno 10 bin di distanza
        # (a nperseg=1024 e fs=200 Hz, 1 bin = 200/1024 ≈ 0.2 Hz → 10 bin ≈ 2 Hz min tra modi)
        indici_picchi, _ = signal.find_peaks(sigma1_norm, height=0.3, distance=10)

        # Ordina i picchi per ampiezza di sigma1 (non normalizzata) decrescente
        # e prendi le prime 3 frequenze naturali fn1, fn2, fn3
        if len(indici_picchi) > 0:
            picchi_ordinati = indici_picchi[np.argsort(sigma1_banda[indici_picchi])[::-1]]
            fn1 = float(freq_banda[picchi_ordinati[0]]) if len(picchi_ordinati) > 0 else 0.0
            fn2 = float(freq_banda[picchi_ordinati[1]]) if len(picchi_ordinati) > 1 else 0.0
            fn3 = float(freq_banda[picchi_ordinati[2]]) if len(picchi_ordinati) > 2 else 0.0
        else:
            fn1, fn2, fn3 = 0.0, 0.0, 0.0

        # ── Salvataggio Database ───────────────────────────────────────────
        record = FDDData(
            timestamp=timestamp,
            mode_1_freq=fn1, mode_2_freq=fn2, mode_3_freq=fn3
        )
        session.add(record)
        session.commit()

        # ── Plot ───────────────────────────────────────────────────────────
        # Grafico in scala logaritmica (semilogy):
        #   - le curve sigma1, sigma2, sigma3 mostrano la "potenza modale"
        #   - i picchi netti di sigma1 sono le frequenze naturali
        #   - sigma2 e sigma3 aiutano a vedere se un picco e' un modo reale
        #     (sigma1 picco isolato) o una biforcazione (sigma1 e sigma2 picco insieme)
        #   - le linee verticali tratteggiate marcano fn1, fn2, fn3 identificate
        if ENABLE_PLOTS:
            fdd_plot_dir = os.path.join(plot_dir, "FDD_System")
            ensure_dir(fdd_plot_dir)
            plot_path = os.path.join(fdd_plot_dir, f"{timestamp}_fdd_{FDD_DURATION_SEC}s.png")

            plt.figure(figsize=(12, 6))
            plt.semilogy(freq_banda, sigma1_banda,          label='σ1 - modo dominante',   color='blue')
            plt.semilogy(freq_banda, sigma2[banda_mask],    label='σ2 - modo secondario',  color='orange', alpha=0.7)
            plt.semilogy(freq_banda, sigma3[banda_mask],    label='σ3 - modo terziario',   color='green',  alpha=0.5)

            for freq_nat, etichetta, colore in [
                (fn1, f'fn1 = {fn1:.2f} Hz', 'red'),
                (fn2, f'fn2 = {fn2:.2f} Hz', 'purple'),
                (fn3, f'fn3 = {fn3:.2f} Hz', 'brown'),
            ]:
                if freq_nat > 0:
                    plt.axvline(x=freq_nat, linestyle='--', color=colore, alpha=0.8, label=etichetta)

            plt.title(f"FDD — Valori Singolari di Syy | Window: {FDD_DURATION_SEC}s | {timestamp}")
            plt.xlabel("Frequenza [Hz]")
            plt.ylabel("Valori Singolari (scala log)")
            plt.legend()
            plt.grid(True, which='both', alpha=0.3)
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()

        q_mqtt.put({
            "type": "FDD",
            "timestamp": timestamp,
            "mode_1_hz": fn1, "mode_2_hz": fn2, "mode_3_hz": fn3
        })
        logger.info(f"FDD completata: fn1={fn1:.2f}Hz, fn2={fn2:.2f}Hz, fn3={fn3:.2f}Hz")

    except Exception as e:
        logger.error(f"Errore durante FDD: {e}")
        session.rollback()
    finally:
        session.close()


def shm_analysis_worker(q_files, q_mqtt, stop_event, plot_dir, db_session_maker):
    global seconds_since_last_fdd

    logger.info(f"Avvio thread Analisi (Standard: {ANALYSIS_INTERVAL_SEC}s, FDD: ogni {FDD_INTERVAL_SEC}s)")
    session = db_session_maker()
    sensor_buffers = {}

    while not stop_event.is_set():
        try:
            filepath = q_files.get(timeout=1)
        except queue.Empty:
            continue

        try:
            filename = os.path.basename(filepath)
            sensor_name, timestamp = filename.replace('.csv', '').rsplit('_', 1)
            fs = SENSOR_FREQUENCIES.get(sensor_name, 200)
            sensor_type = SENSOR_TYPES.get(sensor_name, 200)

            df = pd.read_csv(filepath, header=None)
            new_samples = df[0].values

            # --- 1. GESTIONE BUFFER STANDARD (Stat, FFT, PSD) ---
            if sensor_name not in sensor_buffers:
                max_samples = ANALYSIS_INTERVAL_SEC * fs
                sensor_buffers[sensor_name] = deque(maxlen=max_samples)

            sensor_buffers[sensor_name].extend(new_samples)
            current_buffer = np.array(sensor_buffers[sensor_name])

            if len(current_buffer) >= (ANALYSIS_INTERVAL_SEC * fs):
                # Esecuzione Analisi Statistiche
                if ENABLE_STATISTICS:
                    # Get stats
                    max_val=float(np.max(current_buffer))
                    min_val=float(np.min(current_buffer))
                    mean_val=float(np.mean(current_buffer))
                    median_val=float(np.median(current_buffer))

                    # Save in DB
                    session.add(StatData(
                        sensor_name=sensor_name,
                        timestamp=timestamp,
                        max_val=max_val,
                        min_val=min_val,
                        mean_val=mean_val,
                        median_val=median_val))

                    # Get mqtt body
                    msg_id = str(uuid.uuid4())
                    gateway_id = "GTW_DEMO"
                    sensor_id = sensor_name
                    ts_now = int(time.time() * 1000)
                    ts_start = ts_now - 60000 # Esempio: inizio lettura 1 secondo fa
                    time_window = 60000
                    frequency = fs
                    version = "1.0"
                    sensor_data = [
                        {"rowId": "max", "Value": [max_val]},
                        {"rowId": "min", "Value": [min_val]},
                        {"rowId": "mean", "Value": [mean_val]},
                        {"rowId": "median", "Value": [median_val]}]
                    q_mqtt.put({"MsgId": msg_id, "IDGateway": gateway_id, "SensorId": sensor_id, "SensorType": sensor_type, "Timestamp": ts_now, "TimestampStart": ts_start, "Timewindow": time_window, "Frequency": frequency, "Version": version, "SensorData": sensor_data})

                is_dynamic = "temp" not in sensor_name and "inclinometer" not in sensor_name

                # Esecuzione FFT
                if ENABLE_FFT and is_dynamic:
                    n = len(current_buffer)
                    fft_vals = np.fft.rfft(current_buffer)
                    fft_freq = np.fft.rfftfreq(n, d=1/fs)
                    amplitudes = np.abs(fft_vals) / n
                    p_idx = np.argmax(amplitudes[1:]) + 1

                    session.add(FFTData(
                        sensor_name=sensor_name, timestamp=timestamp,
                        peak_freq=float(fft_freq[p_idx]), peak_amplitude=float(amplitudes[p_idx])
                    ))

                    if ENABLE_PLOTS:
                        s_dir = os.path.join(plot_dir, sensor_name)
                        ensure_dir(s_dir)
                        plt.figure(figsize=(10, 4))
                        plt.plot(fft_freq, amplitudes, color='blue')
                        plt.title(f"FFT Spectrum - {sensor_name} ({timestamp})")
                        plt.grid(True)
                        plt.savefig(os.path.join(s_dir, f"{timestamp}_fft_{ANALYSIS_INTERVAL_SEC}s.png"))
                        plt.close()

                # Esecuzione PSD
                if ENABLE_PSD and is_dynamic:
                    f_psd, p_mag = signal.welch(current_buffer, fs, nperseg=fs*2)
                    p_idx = np.argmax(p_mag[1:]) + 1

                    session.add(PSDData(
                        sensor_name=sensor_name, timestamp=timestamp,
                        peak_freq=float(f_psd[p_idx]), peak_psd=float(p_mag[p_idx])
                    ))

                    if ENABLE_PLOTS:
                        s_dir = os.path.join(plot_dir, sensor_name)
                        ensure_dir(s_dir)
                        plt.figure(figsize=(10, 4))
                        plt.semilogy(f_psd, p_mag, color='red')
                        plt.title(f"PSD - {sensor_name} ({timestamp})")
                        plt.grid(True, which='both')
                        plt.savefig(os.path.join(s_dir, f"{timestamp}_psd_{ANALYSIS_INTERVAL_SEC}s.png"))
                        plt.close()

                session.commit()
                # q_mqtt.put({"type": "ANALYSIS_COMPLETE", "sensor": sensor_name, "ts": timestamp})
                sensor_buffers[sensor_name].clear()

            # --- 2. GESTIONE BUFFER FDD (Sincronizzato su N sensori) ---
            if ENABLE_FDD and sensor_name in ACCEL_SENSORS:
                fdd_sync_buffer[timestamp][sensor_name] = new_samples

                # Quando abbiamo il pacchetto di dati per tutti i 9 sensori per questo specifico timestamp
                if len(fdd_sync_buffer[timestamp]) == len(ACCEL_SENSORS):
                    matrice_min = np.column_stack([fdd_sync_buffer[timestamp][s] for s in ACCEL_SENSORS])
                    fdd_history.extend(matrice_min)
                    del fdd_sync_buffer[timestamp]

                    # Incrementiamo il contatore dei dati processati (basandoci sulla lunghezza del CSV appena letto)
                    # Assumendo che ogni file CSV contenga 1 secondo di dati:
                    seconds_since_last_fdd += 60

                    # Verifichiamo se è ora di far partire la FDD e se abbiamo abbastanza dati nel buffer
                    if seconds_since_last_fdd >= FDD_INTERVAL_SEC:
                        if len(fdd_history) >= (FDD_DURATION_SEC * FS_ACCEL):

                            # Estraiamo esattamente la finestra temporale richiesta (Duration)
                            dati_fdd = np.array(fdd_history)

                            threading.Thread(
                                target=calculate_fdd_async,
                                args=(dati_fdd, timestamp, db_session_maker, q_mqtt, plot_dir),
                                daemon=True
                            ).start()

                            # Reset del timer per il prossimo intervallo
                            seconds_since_last_fdd = 0

        except Exception as e:
            logger.error(f"Errore nel file {filepath}: {e}")
            session.rollback()

    session.close()
