import os

import queue

import uuid

import time

import threading

import numpy as np

import pandas as pd

from scipy import signal

from collections import defaultdict, deque

from scipy.signal import find_peaks



import matplotlib

matplotlib.use('Agg')  # Previene l'apertura di finestre GUI

import matplotlib.pyplot as plt



from lib.utils import get_logger, ensure_dir

from lib.database import StatData, FFTData, PSDData, FDDData

from sensor_map import SENSOR_FREQUENCIES, SENSOR_TYPES



from pyoma2.setup import SingleSetup

from pyoma2.algorithms import FDD



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



FS_ACCEL = 200  # Hz (frequenza campionamento accelerometri)

FDD_SENSORS = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6']



# --- Pre-processing comune (FFT/PSD/FDD) -------------------------------------

# Setup: trave 3.5 m, 6 accelerometri triassiali, fs=200 Hz (Nyquist=100 Hz).

# Prima frequenza propria attesa ~21 Hz, frequenze successive > 30 Hz.

# Banda di interesse strutturale: 1-95 Hz (esclude DC/drift e bordo Nyquist).

STRUCT_BAND         = (1.0, 95.0)  # Banda di ricerca picchi per FFT/PSD/FDD [Hz]



# --- Parametri Welch comuni a PSD (per-canale) e FDD (multivariato) ----------

# Lo stesso nperseg in entrambi garantisce la stessa risoluzione frequenziale

# df = fs / nperseg, quindi PSD per-canale e CSD/SVD della FDD sono confrontabili

# bin per bin sull'asse delle frequenze.

# A fs=200 Hz: nperseg=2048 -> segmento ~10.24 s, df ~ 0.098 Hz.

#   PSD su 60 s con 50% overlap  -> ~10 medie

#   FDD su 600 s con 50% overlap -> ~115 medie (alta stabilita')

WELCH_NPERSEG       = 2048

WELCH_OVERLAP_RATIO = 0.5

# ==========================================



# Buffer per FDD: fdd_history mantiene abbastanza dati per coprire la durata richiesta

fdd_sync_buffer = defaultdict(dict)

fdd_history_x = deque(maxlen=FDD_DURATION_SEC * FS_ACCEL)

fdd_history_y = deque(maxlen=FDD_DURATION_SEC * FS_ACCEL)

fdd_history_z = deque(maxlen=FDD_DURATION_SEC * FS_ACCEL)



# Variabile di controllo per l'intervallo FDD (basata sui secondi di dati processati)

seconds_since_last_fdd = 0



# Parametri per Analisi Modale (FDD via pyoma2)

# nxseg uguale a WELCH_NPERSEG -> stessa df di PSD per confronto diretto.

FDD_NXSEG          = WELCH_NPERSEG

FDD_FREQ_MIN_PICK  = STRUCT_BAND[0]   # Esclude DC e drift bassissimi

FDD_FREQ_MAX_PICK  = STRUCT_BAND[1]   # Estende la ricerca fino a quasi-Nyquist

FDD_FREQ_MAX_PLOT  = 100.0            # Zoom asse x grafici [Hz]

FDD_NUM_MODES      = 6                # Numero di frequenze proprie da estrarre

FDD_PROMINENCE_DB  = 5.0              # Prominenza min dei picchi su sigma_1 in dB

FDD_PEAK_DIST_HZ   = 0.5              # Distanza minima tra picchi [Hz]

FDD_DF_MPE         = 0.2              # Banda +/- DF per fdd.mpe (Hz)





# ==========================================

# HELPER DI PRE-PROCESSING

# ==========================================

def _preprocess_signal(x):

    """Pre-processing classico: detrend lineare (rimuove media + drift).



    Accetta segnali 1D oppure matrici 2D (n_campioni, n_sensori).

    """

    x = np.asarray(x, dtype=float)

    return signal.detrend(x, axis=0, type="linear")





def _find_structural_peaks(freq, magnitude, band=STRUCT_BAND,

                           n_peaks=6,

                           prominence_ratio=0.02):

    """Peak picking nella banda strutturale.



    Ordina i picchi per prominenza decrescente, restituisce i primi n_peaks

    riordinati per frequenza crescente. Soglia di prominenza relativa al

    range del segnale (robusta a livelli di ampiezza diversi).

    """

    fmin, fmax = band

    mask = (freq >= fmin) & (freq <= fmax)

    f_band = freq[mask]

    m_band = magnitude[mask]

    if f_band.size == 0:

        return np.array([]), np.array([])

    span = float(m_band.max() - m_band.min())

    prom = max(span * prominence_ratio, 1e-20)

    df = f_band[1] - f_band[0] if f_band.size > 1 else 1.0

    dist = max(1, int(0.3 / df))  # almeno 0.3 Hz tra due picchi distinti

    peaks, props = find_peaks(m_band, prominence=prom, distance=dist)

    if peaks.size == 0:

        return np.array([]), np.array([])

    order = np.argsort(props["prominences"])[-min(n_peaks, peaks.size):]

    peaks_top = np.sort(peaks[order])

    return f_band[peaks_top], m_band[peaks_top]





# ==========================================

# FUNZIONE FDD ASINCRONA (pyoma2)

# ==========================================

def calculate_fdd_async(axes, data_matrix, timestamp, db_session_maker, q_mqtt, plot_dir):

    #FDD via pyoma2 sul blocco multivariato di accelerazioni per un asse.

    #data_matrix: shape (n_campioni, n_sensori) - convenzione pyoma2.

    logger.info(f"Avvio FDD (pyoma2) asse {axes}: blocco terminante al {timestamp}")

    session = db_session_maker()



    try:

        # 1. Pre-processing classico: detrend lineare (rimuove media + drift)

        data = _preprocess_signal(data_matrix)



        # 2. Setup pyoma2 + algoritmo FDD

        ss = SingleSetup(data, fs=FS_ACCEL)

        fdd = FDD(name=f"FDD_{axes}", nxseg=FDD_NXSEG, method_SD="per")

        ss.add_algorithms(fdd)

        ss.run_all()



        # 3. Peak picking automatico sul 1o valore singolare in dB

        freq   = fdd.result.freq

        sv1    = fdd.result.S_val[0, 0, :].real

        sv1_dB = 10 * np.log10(sv1 / sv1.max() + 1e-20)



        df_bin   = freq[1] - freq[0]

        dist_min = max(1, int(FDD_PEAK_DIST_HZ / df_bin))   # min 0.5 Hz tra picchi



        peaks, props = find_peaks(sv1_dB, prominence=FDD_PROMINENCE_DB, distance=dist_min)



        # Maschera banda utile

        mask  = (freq[peaks] > FDD_FREQ_MIN_PICK) & (freq[peaks] < FDD_FREQ_MAX_PICK)

        peaks = peaks[mask]

        proms = props["prominences"][mask]



        if len(peaks) == 0:

            logger.warning(f"FDD asse {axes}: nessun picco identificato nella banda "

                           f"[{FDD_FREQ_MIN_PICK}-{FDD_FREQ_MAX_PICK}] Hz")

            return



        # Top-N per prominenza, poi ordinati per frequenza

        top_order = np.argsort(proms)[-min(FDD_NUM_MODES, len(peaks)):]

        peaks_top = np.sort(peaks[top_order])

        sel_freq  = freq[peaks_top].tolist()



        # 4. Estrazione frequenze proprie con mpe

        fdd.mpe(sel_freq=sel_freq, DF=FDD_DF_MPE)

        freq_picchi = np.array(fdd.result.Fn)

        freq_str = ", ".join([f"{f:.4f} Hz" for f in freq_picchi])

        logger.info(f"FDD asse {axes} - Frequenze proprie identificate: [{freq_str}]")



        # 5. Plot CMIF con picchi sovrapposti

        if ENABLE_PLOTS:

            fdd_plot_dir = os.path.join(plot_dir, "FDD_System")

            ensure_dir(fdd_plot_dir)



            fig_cmif, ax_cmif = fdd.plot_CMIF(freqlim=(0, FDD_FREQ_MAX_PLOT))

            ax_cmif.plot(freq[peaks_top], sv1_dB[peaks_top], "o", mfc="none",

                         mec="red", ms=9, mew=2.1, label="Frequenze proprie", zorder=10)

            ax_cmif.legend(loc="upper right")

            ax_cmif.set_title(f"FDD CMIF - asse {axes} - {timestamp}")



            out_path = os.path.join(fdd_plot_dir, f"{axes}_{timestamp}_fdd_{FDD_DURATION_SEC}s_CMIF.png" )

            fig_cmif.savefig(out_path, dpi=120, bbox_inches="tight")

            plt.close(fig_cmif)



        # 5. (Opzionale) Salvataggio DB e MQTT - da abilitare quando lo schema FDDData è definito

        # record = FDDData(

        #     timestamp=timestamp,

        #     mode_1_freq=float(freq_picchi[0]) if len(freq_picchi) > 0 else None,

        #     mode_2_freq=float(freq_picchi[1]) if len(freq_picchi) > 1 else None,

        #     mode_3_freq=float(freq_picchi[2]) if len(freq_picchi) > 2 else None,

        # )

        # session.add(record)

        # session.commit()

        # q_mqtt.put({"type": "FDD", "axes": axes, "timestamp": timestamp,

        #             "frequencies": freq_picchi.tolist()})



    except Exception as e:

        logger.error(f"Errore durante FDD asse {axes}: {e}")

        session.rollback()

    finally:

        session.close()



# ==========================================

# WORKER PRINCIPALE

# ==========================================

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

            # 1. Rimuoviamo l'estensione e dividiamo per ogni "_"

            parts = filename.replace('.csv', '').split('_')



            # 2. Assegniamo le variabili (Unpacking)

            if 'A' in filename or 'MT' in filename or 'I' in filename:

                sensor_name, axis, timestamp = parts

                sensor_axes = f"{sensor_name}_{axis}"

            else:

                sensor_name, timestamp = parts

                sensor_axes = sensor_name

            fs = SENSOR_FREQUENCIES.get(sensor_name, 200)

            sensor_type = SENSOR_TYPES.get(sensor_name, 200)



            df = pd.read_csv(filepath, header=None)

            new_samples = df[0].values



            # --- 1. GESTIONE BUFFER STANDARD (Stat, FFT, PSD) ---

            if sensor_axes not in sensor_buffers:

                max_samples = ANALYSIS_INTERVAL_SEC * fs

                sensor_buffers[sensor_axes] = deque(maxlen=max_samples)



            sensor_buffers[sensor_axes].extend(new_samples)

            current_buffer = np.array(sensor_buffers[sensor_axes])



            if len(current_buffer) >= (ANALYSIS_INTERVAL_SEC * fs):

                # Esecuzione Analisi Statistiche

                if ENABLE_STATISTICS:

                    # Get stats

                    max_val = float(np.max(current_buffer))

                    min_val = float(np.min(current_buffer))

                    mean_val = float(np.mean(current_buffer))

                    median_val = float(np.median(current_buffer))



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

                    ts_start = ts_now - 60000  # Esempio: inizio lettura 1 secondo fa

                    time_window = 60000

                    frequency = fs

                    version = "1.0"

                    sensor_data = [

                        {"rowId": "max", "Value": [max_val]},

                        {"rowId": "min", "Value": [min_val]},

                        {"rowId": "mean", "Value": [mean_val]},

                        {"rowId": "median", "Value": [median_val]}]

                    q_mqtt.put({"MsgId": msg_id, "IDGateway": gateway_id, "SensorId": sensor_id,

                                "SensorType": sensor_type, "Timestamp": ts_now, "TimestampStart": ts_start,

                                "Timewindow": time_window, "Frequency": frequency, "Version": version,

                                "SensorData": sensor_data})



                is_dynamic = "temp" not in sensor_name and "inclinometer" not in sensor_name



                # Esecuzione FFT

                if ENABLE_FFT and is_dynamic:

                    # Pre-processing classico: detrend lineare

                    sig_proc = _preprocess_signal(current_buffer)



                    n = len(sig_proc)

                    win = signal.windows.hann(n)

                    cg = win.mean()  # coherent gain della finestra di Hann (~0.5)

                    fft_vals = np.fft.rfft(sig_proc * win)

                    fft_freq = np.fft.rfftfreq(n, d=1 / fs)

                    # Ampiezza a singolo lato in unita' fisiche (m/s^2)

                    amplitudes = np.abs(fft_vals) / (n * cg) * 2.0

                    amplitudes[0] /= 2.0           # bin DC non si raddoppia

                    if n % 2 == 0:

                        amplitudes[-1] /= 2.0      # bin Nyquist non si raddoppia



                    # Picco dominante nella banda strutturale

                    peaks_f, peaks_a = _find_structural_peaks(

                        fft_freq, amplitudes, band=STRUCT_BAND,

                    )

                    if peaks_f.size > 0:

                        idx_dom = int(np.argmax(peaks_a))

                        peak_freq = float(peaks_f[idx_dom])

                        peak_amp  = float(peaks_a[idx_dom])

                    else:

                        mask_band = (fft_freq >= STRUCT_BAND[0]) & (fft_freq <= STRUCT_BAND[1])

                        idx_band = np.where(mask_band)[0]

                        p_idx = idx_band[np.argmax(amplitudes[idx_band])]

                        peak_freq = float(fft_freq[p_idx])

                        peak_amp  = float(amplitudes[p_idx])



                    session.add(FFTData(

                        sensor_name=sensor_name, timestamp=timestamp,

                        peak_freq=peak_freq, peak_amplitude=peak_amp

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

                    sig_proc = _preprocess_signal(current_buffer)

                    nperseg = min(len(sig_proc), WELCH_NPERSEG)

                    noverlap = int(nperseg * WELCH_OVERLAP_RATIO)

                    f_psd, p_mag = signal.welch(

                        sig_proc, fs, window='hann',

                        nperseg=nperseg, noverlap=noverlap,

                        detrend=False, scaling='density',

                    )



                    # Picco dominante nella banda strutturale

                    peaks_f, peaks_p = _find_structural_peaks(

                        f_psd, p_mag, band=STRUCT_BAND,

                    )

                    if peaks_f.size > 0:

                        idx_dom = int(np.argmax(peaks_p))

                        peak_freq = float(peaks_f[idx_dom])

                        peak_psd  = float(peaks_p[idx_dom])

                    else:

                        mask_band = (f_psd >= STRUCT_BAND[0]) & (f_psd <= STRUCT_BAND[1])

                        idx_band = np.where(mask_band)[0]

                        p_idx = idx_band[np.argmax(p_mag[idx_band])]

                        peak_freq = float(f_psd[p_idx])

                        peak_psd  = float(p_mag[p_idx])



                    session.add(PSDData(

                        sensor_name=sensor_name, timestamp=timestamp,

                        peak_freq=peak_freq, peak_psd=peak_psd

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

                sensor_buffers[sensor_axes].clear()



            # --- 2. GESTIONE BUFFER FDD (Sincronizzato su N sensori) ---

            if ENABLE_FDD and "A" in sensor_axes and ("X" in sensor_axes or "Y" in sensor_axes or "Z" in sensor_axes):

                fdd_sync_buffer[timestamp][sensor_axes] = new_samples



                # Quando abbiamo il pacchetto di dati per tutti i sensori e per le 3 assi per questo specifico timestamp

                if len(fdd_sync_buffer[timestamp]) == len(FDD_SENSORS) * 3:

                    matrice_min = np.column_stack([fdd_sync_buffer[timestamp][s] for s in fdd_sync_buffer[timestamp] if 'X' in s])

                    fdd_history_x.extend(matrice_min)

                    matrice_min = np.column_stack([fdd_sync_buffer[timestamp][s] for s in fdd_sync_buffer[timestamp] if 'Y' in s])

                    fdd_history_y.extend(matrice_min)

                    matrice_min = np.column_stack([fdd_sync_buffer[timestamp][s] for s in fdd_sync_buffer[timestamp] if 'Z' in s])

                    fdd_history_z.extend(matrice_min)

                    del fdd_sync_buffer[timestamp]



                    # Incrementiamo il contatore dei dati processati (basandoci sulla lunghezza del CSV appena letto)

                    # Assumendo che ogni file CSV contenga 1 secondo di dati:

                    seconds_since_last_fdd += 60



                    # Verifichiamo se è ora di far partire la FDD e se abbiamo abbastanza dati nel buffer

                    if seconds_since_last_fdd >= FDD_INTERVAL_SEC:

                        if (len(fdd_history_x) >= (FDD_DURATION_SEC * FS_ACCEL)

                                and len(fdd_history_y) >= (FDD_DURATION_SEC * FS_ACCEL)

                                and len(fdd_history_z) >= (FDD_DURATION_SEC * FS_ACCEL)):



                            # Estraiamo esattamente la finestra temporale richiesta (Duration)

                            dati_fdd_x = np.array(fdd_history_x)

                            dati_fdd_y = np.array(fdd_history_y)

                            dati_fdd_z = np.array(fdd_history_z)



                            threading.Thread(

                                target=calculate_fdd_async,

                                args=('X', dati_fdd_x, timestamp, db_session_maker, q_mqtt, plot_dir),

                                daemon=True

                            ).start()



                            threading.Thread(

                                target=calculate_fdd_async,

                                args=('Y', dati_fdd_y, timestamp, db_session_maker, q_mqtt, plot_dir),

                                daemon=True

                            ).start()



                            threading.Thread(

                                target=calculate_fdd_async,

                                args=('Z', dati_fdd_z, timestamp, db_session_maker, q_mqtt, plot_dir),

                                daemon=True

                            ).start()



                            # Reset del timer per il prossimo intervallo

                            seconds_since_last_fdd = 0



        except Exception as e:

            logger.error(f"Errore nel file {filepath}: {e}")

            session.rollback()



    session.close()
