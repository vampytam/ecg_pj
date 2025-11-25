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


def detect_r_peaks_with_fallback(ecg_signal, fs):
    _, r_peaks_dict = nk.ecg_peaks(ecg_signal, sampling_rate=fs)
    r_peaks_arr = r_peaks_dict.get("ECG_R_Peaks", np.array([]))
    r_peaks_arr = np.nan_to_num(r_peaks_arr, nan=0).astype(int)
    
    valid_peaks = np.count_nonzero(r_peaks_arr > 0)
    if valid_peaks >= 2:
        return r_peaks_dict
    
    try:
        from neurokit2 import signal as nk_signal
        signal_inverted = -ecg_signal
        peaks_fallback, _ = nk_signal.signal_findpeaks(signal_inverted, height=0)
        if len(peaks_fallback) >= 2:
            r_peaks_dict["ECG_R_Peaks"] = np.array(peaks_fallback, dtype=float)
            return r_peaks_dict
    except Exception:
        pass
    
    try:
        from scipy import signal as sp_signal
        peaks, _ = sp_signal.find_peaks(ecg_signal, distance=int(0.4 * fs), prominence=np.max(ecg_signal) * 0.1)
        if len(peaks) >= 2:
            r_peaks_dict["ECG_R_Peaks"] = np.array(peaks, dtype=float)
            return r_peaks_dict
    except Exception:
        pass
    
    return r_peaks_dict


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
    j_win_samples = max(1, int(j_window_ms * fs / 1000.0))
    j_start = j_idx
    j_end = min(n - 1, j_idx + j_win_samples - 1)
    try:
        j_level = float(np.mean(signal[j_start:j_end + 1]))
    except Exception:
        j_level = float(signal[j_idx]) if 0 <= j_idx < n else 0.0

    st_eval_idx = min(n - 1, j_idx + int(st_eval_delay_ms * fs / 1000.0))
    eval_win_samples = max(1, int(eval_window_ms * fs / 1000.0))
    half = eval_win_samples // 2
    eval_start = max(0, st_eval_idx - half)
    eval_end = min(n - 1, st_eval_idx + half)
    try:
        st_level = float(np.mean(signal[eval_start:eval_end + 1]))
    except Exception:
        st_level = float(signal[st_eval_idx])

    delta = st_level - baseline

    time_ms = (st_eval_idx - j_idx) / fs * 1000.0
    if time_ms == 0:
        time_ms = 1.0
    slope_mV_per_ms = (st_level - j_level) / time_ms

    slope_thresh = max(horiz_thresh_mV / max(st_eval_delay_ms, 1) * 1.5, 0.0003)

    if abs(delta) <= horiz_thresh_mV:
        return "horizontal"
    if slope_mV_per_ms > slope_thresh:
        return "upslope"
    if slope_mV_per_ms < -slope_thresh:
        return "declination"

    return "upslope" if delta > 0 else "declination"
            
def extract_ecg_features(record):
    fs = record.fs
    baseline_win_samples = int(baseline_win_ms * fs / 1000.0)
    
    lead_names = record.sig_name
    twelve_leads_ecg_signals = record.p_signal.T
    
    global_features = {}
    # We'll keep per-lead RR/HR summaries and then choose a reference lead
    per_lead_rr_stats = {}
    
    lead_features = {}
    for idx, lead_name in enumerate(lead_names):
        
        ecg_signal = twelve_leads_ecg_signals[idx]
        
        ecg_signal = nk.ecg_clean(ecg_signal, sampling_rate=fs, method="neurokit")
        
        r_peaks = detect_r_peaks_with_fallback(ecg_signal, fs)

        # Ensure we have enough R-peaks before running delineation (neurokit2 expects >=2 peaks)

        r_peaks_arr = r_peaks.get("ECG_R_Peaks") if isinstance(r_peaks, dict) else np.array(r_peaks)
        r_peaks_arr = np.nan_to_num(r_peaks_arr, nan=0).astype(int) if r_peaks_arr is not None else np.array([], dtype=int)

        if np.count_nonzero(r_peaks_arr > 0) < 2:
            # Not enough R-peaks to perform delineation reliably; create empty/default wave dict
            waves_peak = {
                "ECG_P_Onsets": np.array([]),
                "ECG_P_Offsets": np.array([]),
                "ECG_P_Peaks": np.array([]),
                "ECG_Q_Peaks": np.array([]),
                "ECG_S_Peaks": np.array([]),
                "ECG_R_Onsets": np.array([]),
                "ECG_R_Offsets": np.array([]),
                "ECG_T_Onsets": np.array([]),
                "ECG_T_Offsets": np.array([]),
                "ECG_T_Peaks": np.array([]),
            }
        else:
            # Extract the R-peak indices and ensure they are integer indices
            _, waves_peak = nk.ecg_delineate(ecg_signal, r_peaks_arr, sampling_rate=fs, method="prominence")

        p_onsets, p_offsets, p_peaks = waves_peak["ECG_P_Onsets"], waves_peak["ECG_P_Offsets"], waves_peak["ECG_P_Peaks"]
        q_peaks, s_peaks = waves_peak["ECG_Q_Peaks"], waves_peak["ECG_S_Peaks"]
        r_onsets, r_offsets = waves_peak["ECG_R_Onsets"], waves_peak["ECG_R_Offsets"]
        t_onsets, t_offsets, t_peaks = waves_peak["ECG_T_Onsets"], waves_peak["ECG_T_Offsets"], waves_peak["ECG_T_Peaks"]
        p_peaks = np.nan_to_num(p_peaks, nan=0).astype(int)
        # Keep P-onsets as integers with 0 meaning missing (np.nan -> 0)
        p_onsets = np.nan_to_num(p_onsets, nan=0).astype(int)
        p_offsets = np.nan_to_num(p_offsets, nan=0).astype(int)
        p_durations_ms = np.array([x - y if x > y and y > 0 else 0 for x, y in zip(p_offsets, p_onsets)]) / fs * 1000
        p_durations_ms = np.round(p_durations_ms, 1)

        def compute_amplitude(signal, peaks, onsets, fs):
            # Be robust to mismatched lengths between peaks and onsets.
            # Iterate over the minimum length for paired computation, then pad results
            # so the returned array has the same length as `peaks`.
            amps = []
            peaks = np.array(peaks)
            onsets = np.array(onsets)
            n_pairs = min(len(peaks), len(onsets))

            for i in range(n_pairs):
                peak = int(peaks[i]) if not np.isnan(peaks[i]) else 0
                onset = int(onsets[i]) if not np.isnan(onsets[i]) else 0

                # Guard against missing peaks/onsets (encoded as 0)
                if peak <= 0 or onset <= 0:
                    amps.append(0.0)
                    continue

                start = max(0, onset - baseline_win_samples)
                # robust baseline for amplitude: if onset equals start, take a small neighboring window
                if onset > start:
                    baseline_local = float(np.mean(signal[start:onset]))
                else:
                    baseline_local = float(np.mean(signal[max(0, onset - 1):onset + 1]))

                amplitude = float(signal[peak]) - baseline_local
                amps.append(amplitude)

            # If there are more peaks than onsets, pad with zeros for missing onsets
            if len(peaks) > n_pairs:
                amps.extend([0.0] * (len(peaks) - n_pairs))

            return np.round(np.array(amps), 3)

        p_amplitudes = np.nan_to_num(compute_amplitude(ecg_signal, p_peaks, p_onsets, fs), nan=0.0)

        r_onsets = np.nan_to_num(r_onsets, nan=0).astype(int)
        
        pr_ms_list = []
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
            qrs_amp = np.max(qrs_segment) - np.min(qrs_segment)
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

            # PR interval (ms) - compute per beat and handle missing P-onsets
            # p_onsets uses 0 to indicate missing; in that case we set PR to 0.0
            p_on = int(p_onsets[i]) if i < len(p_onsets) else 0
            if p_on <= 0 or r_on <= 0 or r_on <= p_on:
                pr_val_ms = 0.0
            else:
                pr_val_ms = round(float((r_on - p_on) / fs * 1000.0), 1)
            pr_ms_list.append(pr_val_ms)

            # st form estimation (use robust helper)
            j_idx = int(r_off)  # assume j-point is at the end of the QRS-wave
            form = compute_st_form(ecg_signal, j_idx, baseline, fs)
            st_forms.append(form)


        # global rr intervals computed for this lead
        r_peaks_vals = r_peaks["ECG_R_Peaks"]
        r_peaks_vals = np.nan_to_num(r_peaks_vals, nan=0).astype(int)
        rr_samples_next = np.diff(r_peaks_vals) if len(r_peaks_vals) > 1 else np.array([], dtype=int)
        if rr_samples_next.size > 0:
            rr_samples_next[rr_samples_next <= 0] = 0
        rr_ms = rr_samples_next * 1000 / fs if rr_samples_next.size > 0 else np.array([], dtype=float)
        rr_ms = np.round(rr_ms, 1) if rr_ms.size > 0 else rr_ms

        # Filter to valid (>0) intervals for aggregation
        valid_rr_ms = rr_ms[rr_ms > 0] if rr_ms.size > 0 else np.array([], dtype=float)
        if len(valid_rr_ms) > 0:
            max_rr = float(np.max(valid_rr_ms))
            min_rr = float(np.min(valid_rr_ms))
            mean_rr = float(np.mean(valid_rr_ms))
            median_rr = float(np.median(valid_rr_ms))
        else:
            max_rr = min_rr = mean_rr = median_rr = 0.0

        # QTc using Bazett algorithm
        valid_rr_mask = rr_samples_next > 0 if rr_samples_next.size > 0 else np.array([], dtype=bool)
        rr_s = np.zeros_like(rr_samples_next, dtype=float) if rr_samples_next.size > 0 else np.array([], dtype=float)
        if rr_samples_next.size > 0 and valid_rr_mask.any():
            rr_s[valid_rr_mask] = rr_samples_next[valid_rr_mask] / fs

        qt_s = np.array(qt_ms[:len(rr_s)]) / 1000 if rr_s.size > 0 else np.array([], dtype=float)
        qtc_s = np.zeros_like(qt_s)
        if rr_s.size > 0 and len(qt_s) > 0:
            valid_mask = (rr_s > 0) & (np.isfinite(qt_s))
            qtc_s[valid_mask] = qt_s[valid_mask] / np.sqrt(rr_s[valid_mask])
        qtc_ms = np.round(np.nan_to_num(qtc_s, nan=0.0, posinf=0.0, neginf=0.0) * 1000, 1) if qtc_s.size > 0 else np.array([], dtype=float)

        # heart rate - only where rr_s > 0
        hr_bpm = np.zeros_like(rr_s) if rr_s.size > 0 else np.array([], dtype=float)
        if rr_s.size > 0 and (rr_s > 0).any():
            hr_mask = rr_s > 0
            hr_bpm[hr_mask] = 60.0 / rr_s[hr_mask]
        mean_hr = float(np.round(np.mean(hr_bpm[hr_bpm > 0]), 1)) if hr_bpm.size > 0 and np.any(hr_bpm > 0) else 0.0

        # Save per-lead summary so global features can be taken from a reference lead
        per_lead_rr_stats[lead_name] = {
            "Heart Rate (bpm)": mean_hr,
            "Max RR Interval (ms)": max_rr,
            "Min RR Interval (ms)": min_rr,
            "Mean RR Interval (ms)": mean_rr,
            "Median RR Interval (ms)": median_rr,
            "QTC Interval(ms)": qtc_ms.tolist() if hasattr(qtc_ms, 'tolist') else [],
        }

        lead_features[f"{lead_name}"] = {
            "P Amplitude (mV)": str(p_amplitudes.tolist()),
            "P Duration (ms)": str(p_durations_ms.tolist()),
            "PR Interval (ms)": str(pr_ms_list),
            "QRS Amplitude (mV)": str(qrs_amp_list),
            "QRS Duration (ms)": str(qrs_dur_ms),
            "QT Interval(ms)": str(qt_ms),
            "QTC Interval(ms)": str(qtc_ms.tolist()),
            "ST Duration (ms)": str(st_dur_ms),
            "ST Form": str(st_forms),
            "T Amplitude (mV)": str(t_amplitudes),
            "T Duration (ms)": str(t_durations_ms),
        }

    # Choose reference lead for global features: prefer lead 'II'
    ref_lead = None
    for name in lead_names:
        if name.strip().upper() == 'II':
            ref_lead = name
            break
    if ref_lead is None and len(lead_names) >= 2:
        # fallback to the second lead (commonly lead II in many recordings)
        ref_lead = lead_names[1]
    if ref_lead is None and len(lead_names) >= 1:
        ref_lead = lead_names[0]

    if ref_lead in per_lead_rr_stats:
        stats = per_lead_rr_stats[ref_lead]
        global_features["Heart Rate (bpm)"] = str(np.round(stats.get("Heart Rate (bpm)", 0.0), 1))
        global_features["Max RR Interval (ms)"] = str(np.round(stats.get("Max RR Interval (ms)", 0.0), 1))
        global_features["Min RR Interval (ms)"] = str(np.round(stats.get("Min RR Interval (ms)", 0.0), 1))
        global_features["Mean RR Interval (ms)"] = str(np.round(stats.get("Mean RR Interval (ms)", 0.0), 1))
        global_features["Median RR Interval (ms)"] = str(np.round(stats.get("Median RR Interval (ms)", 0.0), 1))
    else:
        # fallback: average across available per-lead summaries
        hr_vals = [v.get("Heart Rate (bpm)", 0.0) for v in per_lead_rr_stats.values()]
        max_vals = [v.get("Max RR Interval (ms)", 0.0) for v in per_lead_rr_stats.values()]
        min_vals = [v.get("Min RR Interval (ms)", 0.0) for v in per_lead_rr_stats.values()]
        mean_vals = [v.get("Mean RR Interval (ms)", 0.0) for v in per_lead_rr_stats.values()]
        med_vals = [v.get("Median RR Interval (ms)", 0.0) for v in per_lead_rr_stats.values()]
        global_features["Heart Rate (bpm)"] = str(np.round(np.mean(hr_vals) if hr_vals else 0.0, 1))
        global_features["Max RR Interval (ms)"] = str(np.round(np.mean(max_vals) if max_vals else 0.0, 1))
        global_features["Min RR Interval (ms)"] = str(np.round(np.mean(min_vals) if min_vals else 0.0, 1))
        global_features["Mean RR Interval (ms)"] = str(np.round(np.mean(mean_vals) if mean_vals else 0.0, 1))
        global_features["Median RR Interval (ms)"] = str(np.round(np.mean(med_vals) if med_vals else 0.0, 1))

    return {"global_features": global_features, "lead_features": lead_features}


def process_ecg_signal(ecg_dir):
    for filename in os.listdir(ecg_dir):
        if filename.endswith('.hea'):
            record_id = os.path.splitext(filename)[0]
            record_path = os.path.join(ecg_dir, record_id)
            record = wfdb.rdrecord(record_path)
            yield record
            
if __name__ == '__main__':
    import traceback
    ecg_dir = './test/ecg_signal'
    output_dir = "./output"
    err_dir = "./err"
    
    for subdir_name in os.listdir(ecg_dir):
        subdir = os.path.join(ecg_dir, subdir_name)
        if os.path.isdir(subdir):
            for filename in os.listdir(subdir):
                if filename.endswith('.hea'):
                    record_id = os.path.splitext(filename)[0]
                    record_path = os.path.join(subdir, record_id)
                    record = wfdb.rdrecord(record_path)
                    record_id = record.record_name
                    output_subdir = os.path.join(output_dir, subdir_name)
                    os.makedirs(output_subdir, exist_ok=True)
                    output_path = os.path.join(output_subdir, f"{record_id}.json")
                    
                    try:
                        features = extract_ecg_features(record)
                        print(f"extracted {record_id} ")
                        with open(output_path, 'w') as f:
                            json.dump(features, f, indent=2, 
                                      default=lambda x: 0 if (np.isnan(x)) else x)
                    except Exception as e:
                        os.makedirs(err_dir, exist_ok=True)
                        err_path = os.path.join(err_dir, f"{record_id}.txt")
                        content = (
                            f'Exception Type: {type(e).__name__}\n'
                            f'Exception Message: {e}\n'
                            '\nTraceback (most recent call last):\n'
                            f'{"".join(traceback.format_tb(e.__traceback__))}'
                        )
                        with open(err_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        print(f"err happened {record_id} ")
                    
    
    print("done!")