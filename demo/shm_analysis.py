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
    FDD: Frequency Domain Decomposition.
    1. Calcola la CPSD matrix per ogni coppia di canali
    2. Esegue SVD sulla CPSD matrix ad ogni frequenza
    3. I picchi del primo singular value indicano le frequenze naturali
    """
    logger.info(f"Avvio calcolo FDD asincrono: blocco terminante al {timestamp}")
    session = db_session_maker()
    try:
        n_samples, n_channels = data_matrix.shape
        nperseg = 1024

        # 1. Calcolo delle frequenze (asse comune)
        f, _ = signal.csd(data_matrix[:, 0], data_matrix[:, 0], fs=FS_ACCEL, nperseg=nperseg)
        n_freqs = len(f)

        # 2. Costruzione della CPSD matrix G[i, j, k]:
        #    CSD tra canale i e canale j alla frequenza k
        G = np.zeros((n_channels, n_channels, n_freqs), dtype=complex)
        for i in range(n_channels):
            for j in range(n_channels):
                _, G[i, j, :] = signal.csd(data_matrix[:, i], data_matrix[:, j],
                                            fs=FS_ACCEL, nperseg=nperseg)

        # 3. SVD della CPSD matrix ad ogni frequenza
        #    I singular values S[0] >= S[1] >= S[2] ... per ogni frequenza
        s1 = np.zeros(n_freqs)
        s2 = np.zeros(n_freqs)
        s3 = np.zeros(n_freqs)
        for k in range(n_freqs):
            _, S, _ = np.linalg.svd(G[:, :, k])
            s1[k] = S[0]
            if n_channels > 1:
                s2[k] = S[1]
            if n_channels > 2:
                s3[k] = S[2]

        # 4. Peak picking sul primo singular value nella banda di interesse
        freq_min, freq_max = 0.1, 25.0
        band = (f >= freq_min) & (f <= freq_max)
        f_band = f[band]
        s1_band = s1[band]

        # Trova picchi: altezza minima = 10% del massimo, distanza minima = 5 bins
        peaks, _ = signal.find_peaks(s1_band,
                                     height=np.max(s1_band) * 0.1,
                                     distance=5)

        # Ordina i picchi per ampiezza (decrescente) e prendi i primi 3
        if len(peaks) > 0:
            peaks_sorted = peaks[np.argsort(s1_band[peaks])[::-1]]
            mode_1 = float(f_band[peaks_sorted[0]]) if len(peaks_sorted) > 0 else 0.0
            mode_2 = float(f_band[peaks_sorted[1]]) if len(peaks_sorted) > 1 else 0.0
            mode_3 = float(f_band[peaks_sorted[2]]) if len(peaks_sorted) > 2 else 0.0
        else:
            mode_1, mode_2, mode_3 = 0.0, 0.0, 0.0

        # 5. Salvataggio Database
        record = FDDData(
            timestamp=timestamp,
            mode_1_freq=mode_1, mode_2_freq=mode_2, mode_3_freq=mode_3
        )
        session.add(record)
        session.commit()

        # 6. Plotting FDD
        if ENABLE_PLOTS:
            fdd_plot_dir = os.path.join(plot_dir, "FDD_System")
            ensure_dir(fdd_plot_dir)
            plot_path = os.path.join(fdd_plot_dir, f"{timestamp}_fdd_{FDD_DURATION_SEC}s.png")

            plt.figure(figsize=(12, 6))
            plt.semilogy(f_band, s1_band, label='1st Singular Value (S1)', color='blue')
            if np.any(s2[band] > 0):
                plt.semilogy(f_band, s2[band], label='2nd Singular Value (S2)', color='orange', alpha=0.7)
            if np.any(s3[band] > 0):
                plt.semilogy(f_band, s3[band], label='3rd Singular Value (S3)', color='green', alpha=0.5)

            # Marca le frequenze naturali identificate
            colors_mode = ['red', 'purple', 'brown']
            for idx, (mode_label, mode_freq) in enumerate(
                [('Mode 1', mode_1), ('Mode 2', mode_2), ('Mode 3', mode_3)]
            ):
                if mode_freq > 0:
                    plt.axvline(x=mode_freq, linestyle='--', color=colors_mode[idx], alpha=0.8,
                                label=f'{mode_label}: {mode_freq:.2f} Hz')

            plt.title(f"FDD Analysis - Window: {FDD_DURATION_SEC}s - {timestamp}")
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("Magnitude")
            plt.legend()
            plt.grid(True, which='both', alpha=0.3)
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()

        q_mqtt.put({
            "type": "FDD",
            "timestamp": timestamp,
            "mode_1_hz": mode_1, "mode_2_hz": mode_2, "mode_3_hz": mode_3
        })
        logger.info(f"FDD completata: Mode1={mode_1:.2f}Hz, Mode2={mode_2:.2f}Hz, Mode3={mode_3:.2f}Hz")

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
