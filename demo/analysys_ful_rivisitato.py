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
FDD_INTERVAL_SEC = 600      # Ogni quanto eseguire la FDD (es. 3600s = ogni ora)

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
    FDD - Frequency Domain Decomposition.

    SCALETTA:

      Step 1 - Costruzione matrice spettrale S_xx(f)
               S_xx[i,j,f] = Cross-PSD tra sensore i e sensore j alla frequenza f.
               signal.csd fa Welch tra due segnali:
                 i==j -> auto-PSD del sensore i
                 i!=j -> cross-PSD tra sensore i e j
               Chiamato 9x9=81 volte fuori dal loop su f,
               cosi ogni chiamata restituisce tutto l'array di frequenze in una volta.

               Cosa fa signal.csd(x, y) internamente:
                 1. Divide entrambi i segnali x e y in blocchi da nperseg campioni
                 2. Applica una finestra (Hann di default) su ogni blocco
                 3. Calcola la FFT di ogni blocco -> X_k e Y_k
                 4. Moltiplica X_k * conj(Y_k) per ogni blocco
                 5. Media su tutti i blocchi -> risultato finale
               Quando x == y il passo 4 diventa X_k * conj(X_k) = |X_k|^2
               che e identico a signal.welch.

      Step 2 - SVD di S_xx ad ogni frequenza
               S_xx(f) = U * diag(sigma) * V^H
               sigma_1(f) = primo valore singolare.
               I picchi di sigma_1(f) sono le frequenze dominanti della struttura.

      Step 3 - Peak picking su sigma_1(f)
               Trova i picchi di sigma_1, salva le prime 3 frequenze dominanti.

      Step 4 - Plot di sigma_1(f) con picchi e frequenze dominanti evidenziati.

      Step 5 - Salvataggio DB e invio MQTT.
    """
    logger.info(f"Avvio calcolo FDD asincrono: blocco terminante al {timestamp}")
    session = db_session_maker()
    try:
        n_campioni, n_sensori = data_matrix.shape
        nperseg = 1024

        # Step 1: Costruzione matrice spettrale S_xx(f)
        # Asse delle frequenze: fs/nperseg = 200/1024 ~ 0.2 Hz per bin, da 0 a 100 Hz (Nyquist)
        f = np.fft.rfftfreq(nperseg, d=1/FS_ACCEL)
        n_freq = len(f)

        # S_xx[i, j, k] = Cross-PSD tra sensore i e sensore j alla frequenza f[k]
        # signal.csd chiamato una volta per coppia -> restituisce tutto l'array di frequenze
        S_xx = np.zeros((n_sensori, n_sensori, n_freq), dtype=complex)
        for i in range(n_sensori):
            for j in range(n_sensori):
                # i==j -> auto-spettro (diagonale di S_xx), identico a signal.welch
                # i!=j -> cross-spettro (fuori diagonale di S_xx)
                _, S_xx[i, j, :] = signal.csd(data_matrix[:, i], data_matrix[:, j],
                                               fs=FS_ACCEL, nperseg=nperseg)

        # Step 2: SVD di S_xx ad ogni frequenza
        # Per ogni frequenza k, S_xx[:,:,k] e una matrice 9x9 complessa.
        # La SVD restituisce i valori singolari in ordine decrescente:
        # sigma_1[k] e il valore piu grande -> rappresenta il modo dominante a f[k]
        sigma_1 = np.zeros(n_freq)
        for k in range(n_freq):
            _, sigma, _ = np.linalg.svd(S_xx[:, :, k])
            sigma_1[k] = sigma[0]

        # Step 3: Peak picking su sigma_1
        # I picchi di sigma_1(f) corrispondono alle frequenze dominanti della struttura.
        # height=0.3 -> considera solo picchi almeno al 30% del massimo
        # distance=10 -> almeno 10 bin di distanza tra picchi (~2 Hz a nperseg=1024, fs=200)
        picchi, _ = signal.find_peaks(sigma_1, height=np.max(sigma_1) * 0.3, distance=10)

        # Ordina i picchi per ampiezza decrescente e prendi le prime 3 frequenze dominanti
        if len(picchi) > 0:
            picchi_ord = picchi[np.argsort(sigma_1[picchi])[::-1]]
            fn1 = float(f[picchi_ord[0]]) if len(picchi_ord) > 0 else 0.0
            fn2 = float(f[picchi_ord[1]]) if len(picchi_ord) > 1 else 0.0
            fn3 = float(f[picchi_ord[2]]) if len(picchi_ord) > 2 else 0.0
        else:
            fn1, fn2, fn3 = 0.0, 0.0, 0.0

        # Output frequenze dominanti in console
        print("=============================================")
        print(f"  FDD - Frequenze Dominanti  [{timestamp}]")
        print("=============================================")
        print(f"  fn1 = {fn1:.3f} Hz" if fn1 > 0 else "  fn1 = non identificata")
        print(f"  fn2 = {fn2:.3f} Hz" if fn2 > 0 else "  fn2 = non identificata")
        print(f"  fn3 = {fn3:.3f} Hz" if fn3 > 0 else "  fn3 = non identificata")
        print("=============================================")

        # Step 4: Plot
        if ENABLE_PLOTS:
            fdd_plot_dir = os.path.join(plot_dir, "FDD_System")
            ensure_dir(fdd_plot_dir)
            plot_path = os.path.join(fdd_plot_dir, f"{timestamp}_fdd_{FDD_DURATION_SEC}s.png")

            plt.figure(figsize=(12, 6))
            # Curva sigma_1(f) completa
            plt.semilogy(f, sigma_1, color='blue', label='sigma_1(f)')
            # Pallini rossi sui picchi identificati
            if len(picchi) > 0:
                plt.semilogy(f[picchi], sigma_1[picchi], 'ro', markersize=6, label='Picchi')
            # Linee verticali sulle frequenze dominanti
            for fn, etichetta in [(fn1, f'fn1={fn1:.2f}Hz'), (fn2, f'fn2={fn2:.2f}Hz'), (fn3, f'fn3={fn3:.2f}Hz')]:
                if fn > 0:
                    plt.axvline(x=fn, linestyle='--', alpha=0.7, label=etichetta)

            plt.title(f"FDD - sigma_1(f) | Window: {FDD_DURATION_SEC}s | {timestamp}")
            plt.xlabel("Frequenza [Hz]")
            plt.ylabel("sigma_1(f) [scala log]")
            plt.legend()
            plt.grid(True, which='both', alpha=0.3)
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()

        # Step 5: Salvataggio DB e MQTT
        session.add(FDDData(timestamp=timestamp, mode_1_freq=fn1, mode_2_freq=fn2, mode_3_freq=fn3))
        session.commit()

        q_mqtt.put({"type": "FDD", "timestamp": timestamp,
                    "mode_1_hz": fn1, "mode_2_hz": fn2, "mode_3_hz": fn3})
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

                    # Verifichiamo se e ora di far partire la FDD e se abbiamo abbastanza dati nel buffer
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
