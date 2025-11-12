import wfdb
import os
import neurokit2 as nk
import numpy as np
import json

import warnings
warnings.filterwarnings("ignore")

baseline_win_ms = 40.0
st_evaluation_delay_ms = 60
horiz_thresh_mV = 0.02 


def compute_st_form(signal, j_idx, baseline, fs, st_eval_delay_ms=st_evaluation_delay_ms,
                    horiz_thresh_mV=horiz_thresh_mV, j_window_ms=10, eval_window_ms=10):
    """Estimate ST form robustly.

    Strategy:
    - Compute a small averaged J-point level (j_level) using a short window after the J-index.
    - Compute a small averaged ST level (st_level) around a point st_eval_delay_ms after J-point.
    - Use both the absolute delta (st_level - baseline) and the slope (st_level - j_level) / time_ms
      to classify into 'horizontal', 'upslope', or 'declination'.
    - This reduces sensitivity to single-sample noise and normalizes slope by time.

    Returns one of: 'horizontal', 'upslope', 'declination'
    """
    n = len(signal)
    # J-level averaging window
    j_win_samples = max(1, int(j_window_ms * fs / 1000.0))
    j_start = j_idx
    j_end = min(n - 1, j_idx + j_win_samples - 1)
    try:
        j_level = float(np.mean(signal[j_start:j_end + 1]))
    except Exception:
        j_level = float(signal[j_idx]) if 0 <= j_idx < n else 0.0

    # ST evaluation index and averaging window
    st_eval_idx = min(n - 1, j_idx + int(st_eval_delay_ms * fs / 1000.0))
    eval_win_samples = max(1, int(eval_window_ms * fs / 1000.0))
    half = eval_win_samples // 2
    eval_start = max(0, st_eval_idx - half)
    eval_end = min(n - 1, st_eval_idx + half)
    try:
        st_level = float(np.mean(signal[eval_start:eval_end + 1]))
    except Exception:
        st_level = float(signal[st_eval_idx])

    # Delta relative to baseline (mV)
    delta = st_level - baseline

    # Slope normalized by time (mV/ms)
    time_ms = (st_eval_idx - j_idx) / fs * 1000.0
    if time_ms == 0:
        time_ms = 1.0  # avoid division by zero; effectively treats as very short interval
    slope_mV_per_ms = (st_level - j_level) / time_ms

    # derive a slope threshold related to absolute threshold and evaluation delay
    slope_thresh = max(horiz_thresh_mV / max(st_eval_delay_ms, 1) * 1.5, 0.0003)

    # Classify: require both a meaningful delta and slope magnitude to call non-horizontal
    if abs(delta) <= horiz_thresh_mV:
        return "horizontal"
    if slope_mV_per_ms > slope_thresh:
        return "upslope"
    if slope_mV_per_ms < -slope_thresh:
        return "declination"

    # Fallback: if delta is significant but slope is marginal, use delta sign for direction
    return "upslope" if delta > 0 else "declination"

def process_ecg_signal(ecg_dir):
    for filename in os.listdir(ecg_dir):
        if filename.endswith('.hea'):
            record_id = os.path.splitext(filename)[0]
            record_path = os.path.join(ecg_dir, record_id)
            record = wfdb.rdrecord(record_path)
            yield record
            
def extract_ecg_features(record):
    fs = record.fs
    baseline_win_samples = int(baseline_win_ms * fs / 1000.0)
    
    lead_names = record.sig_name
    twelve_leads_ecg_signals = record.p_signal.T
    
    global_features = {}
    heart_rate_list = []
    max_rr_list = []
    min_rr_list = []
    mean_rr_list = []
    median_rr_list = []
    
    lead_features = {}
    for idx, lead_name in enumerate(lead_names):
        
        ecg_signal = twelve_leads_ecg_signals[idx]
        
        _, r_peaks = nk.ecg_peaks(ecg_signal, sampling_rate=fs)
        
        ecg_signal = nk.ecg_clean(ecg_signal, sampling_rate=fs, method="neurokit")
        
        _, waves_peak = nk.ecg_delineate(ecg_signal, r_peaks, sampling_rate=fs, method="prominence")

        p_onsets, p_offsets, p_peaks = waves_peak["ECG_P_Onsets"], waves_peak["ECG_P_Offsets"], waves_peak["ECG_P_Peaks"]
        q_peaks, s_peaks = waves_peak["ECG_Q_Peaks"], waves_peak["ECG_S_Peaks"]
        r_onsets, r_offsets = waves_peak["ECG_R_Onsets"], waves_peak["ECG_R_Offsets"]
        t_onsets, t_offsets, t_peaks = waves_peak["ECG_T_Onsets"], waves_peak["ECG_T_Offsets"], waves_peak["ECG_T_Peaks"]

        p_peaks = np.nan_to_num(p_peaks, nan=0).astype(int)
        p_onsets = np.nan_to_num(p_onsets, nan=0).astype(int)
        p_offsets = np.nan_to_num(p_offsets, nan=0).astype(int)
        p_durations_ms = np.array([x - y if x > y else 0 for x, y in zip(p_offsets, p_onsets)]) / fs * 1000
        p_durations_ms = np.round(p_durations_ms, 1)

        def compute_amplitude(signal, peaks, onsets, fs):
            amps = []
            for i, peak in enumerate(peaks):
                onset = onsets[i]
                start = max(0, onset - baseline_win_samples)
                baseline = np.mean(signal[start:onset])
                amplitude = signal[peak] - baseline
                amps.append(amplitude)
            return np.round(amps, 3)
        p_amplitudes = np.nan_to_num(compute_amplitude(ecg_signal, p_peaks, p_onsets, fs), 0)

        r_onsets = np.nan_to_num(r_onsets, nan=0).astype(int)
        pr_ms = np.array([x - y if x > y else 0 for x, y in zip(r_onsets, p_onsets)]) / fs * 1000

        qrs_dur_ms = []
        qrs_amp_list = []
        qt_ms = []
        qtc_ms = []
        st_dur_ms = []
        st_forms = []
        t_durations_ms = []
        t_amplitudes = []

        beats = min(len(r_onsets), len(r_offsets), len(t_onsets), len(t_offsets))
        for i in range(beats):
            r_on = r_onsets[i] # np.nan_to_num(r_onsets[i], nan=0).astype(int) is above
            r_off = int(r_offsets[i]) if not np.isnan(r_offsets[i]) else 0
            t_on = int(t_onsets[i]) if not np.isnan(t_onsets[i]) else 0
            t_peak = int(t_peaks[i]) if not np.isnan(t_peaks[i]) else 0
            t_off = int(t_offsets[i]) if not np.isnan(t_offsets[i]) else 0
            
            if r_off <= r_on or t_off <= t_on:
                qrs_amp_list.append(0.0)
                qrs_dur_ms.append(0.0)
                qt_ms.append(0.0)
                st_dur_ms.append(0.0)
                t_durations_ms.append(0.0)
                t_amplitudes.append(0.0)
                st_forms.append("horizontal")
                continue

            qrs_segment = ecg_signal[r_on : r_off + 1]
            qrs_amp = np.max(qrs_segment) - np.min(qrs_segment) # peak-to-peak amplitude inside the QRS interval
            qrs_amp_list.append(round(float(qrs_amp), 3))

            qrs_dur = (r_off - r_on) / fs * 1000.0
            qrs_dur_ms.append(round(float(qrs_dur), 1))

            qt_val = (t_off - r_on) / fs * 1000.0
            qt_ms.append(round(float(qt_val), 1))

            st_val = (t_on - r_off) / fs * 1000.0
            st_dur_ms.append(round(float(st_val), 1))

            t_durations_ms.append(round(float((t_off - t_on) / fs * 1000.0), 1))
            
            # T Amplitude (mV) = signal[T_peak] - baseline, where baseline = mean(signal[start_baseline : R_onset]) 
            # and start_baseline = R_onset - baseline_win_samples (default baseline window = 40 ms).
            bstart = max(0, r_on - baseline_win_samples)
            baseline = float(np.mean(ecg_signal[bstart : r_on])) if bstart < r_on else float(np.mean(ecg_signal[max(0, r_on - 1) : r_on + 1]))
            t_amplitudes.append(round(float(ecg_signal[t_peak]) - baseline, 3))

            # st form estimation (use robust helper)
            j_idx = int(r_off)  # assume j-point is at the end of the QRS-wave
            form = compute_st_form(ecg_signal, j_idx, baseline, fs)
            st_forms.append(form)


        # global rr intervals
        r_peaks = r_peaks["ECG_R_Peaks"]
        r_peaks = np.nan_to_num(r_peaks, nan=0).astype(int)
        rr_samples_next = np.diff(r_peaks)
        rr_samples_next[rr_samples_next < 0] = 0
        rr_ms = rr_samples_next * 1000 / fs
        rr_ms = np.round(rr_ms, 1)
        max_rr_list.append(np.max(rr_ms))
        min_rr_list.append(np.min(rr_ms))
        mean_rr_list.append(np.mean(rr_ms[rr_ms > 0]))
        median_rr_list.append(np.median(rr_ms[rr_ms > 0]))
        
        # QTc using _bazett algorithm
        rr_s = rr_samples_next / fs
        qt_s = np.array(qt_ms[:len(rr_s)]) / 1000
        qtc_s = np.nan_to_num(qt_s / np.sqrt(rr_s), 0.0)
        qtc_ms = np.round(qtc_s * 1000, 1)
        
        # heart rate
        hr_bpm = np.nan_to_num(60.0 / rr_s, nan=0.0)
        heart_rate_list.append(np.round(np.mean(hr_bpm[hr_bpm > 0]), 1))

        lead_features[f"{lead_name}"] = {
            "P Amplitude (mV)": str(p_amplitudes.tolist()),
            "P Duration (ms)": str(p_durations_ms.tolist()),
            "PR Interval (ms)": str(pr_ms.tolist()),
            "QRS Amplitude (mV)": str(qrs_amp_list),
            "QRS Duration (ms)": str(qrs_dur_ms),
            "QT Interval(ms)": str(qt_ms),
            "QTC Interval(ms)": str(qtc_ms.tolist()),
            "ST Duration (ms)": str(st_dur_ms),
            "ST Form": str(st_forms),
            "T Amplitude (mV)": str(t_amplitudes),
            "T Duration (ms)": str(t_durations_ms),
        }

    global_features["Heart Rate (bpm)"] = str(np.round(np.mean(heart_rate_list), 1))
    global_features["Max RR Interval (ms)"] = str(np.round(np.mean(max_rr_list), 1))
    global_features["Min RR Interval (ms)"] = str(np.round(np.mean(min_rr_list), 1))
    global_features["Mean RR Interval (ms)"] = str(np.round(np.mean(mean_rr_list), 1))
    global_features["Median RR Interval (ms)"] = str(np.round(np.mean(median_rr_list), 1))

    return {"global_features": global_features, "lead_features": lead_features}

if __name__ == '__main__':
    ecg_dir = './test/ecg_signal'
    output_dir = "./"
    for record in process_ecg_signal(ecg_dir):
        features = extract_ecg_features(record)
        # dump the features to a json file
        record_id = record.record_name
        print(f"record: {record_id}")
        output_path = os.path.join(output_dir, f"{record_id}_features.json")
        with open(output_path, 'w') as f:
            json.dump(features, f, indent=2, 
                      default=lambda x: 0 if (np.isnan(x)) else x)
    
    print("done!")