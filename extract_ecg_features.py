import wfdb
import os
import neurokit2 as nk
import numpy as np
import json

baseline_win_ms = 40.0
st_evaluation_delay_ms = 60
horiz_thresh_mV = 0.02 

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
    for idx, name in enumerate(lead_names):
        print(f"Lead {idx+1}: {name}")
        
        ecg_signal = twelve_leads_ecg_signals[idx]
        
        _, r_peaks = nk.ecg_peaks(ecg_signal, sampling_rate=fs)
        
        ecg_signal = nk.ecg_clean(ecg_signal, sampling_rate=fs, method="neurokit")
        
        _, waves_peak = nk.ecg_delineate(ecg_signal, r_peaks, sampling_rate=fs, method="prominence")

        p_onsets = waves_peak["ECG_P_Onsets"]
        p_offsets = waves_peak["ECG_P_Offsets"]
        p_peaks  = waves_peak["ECG_P_Peaks"]

        q_peaks = waves_peak["ECG_Q_Peaks"]

        r_onsets = waves_peak["ECG_R_Onsets"]
        r_offsets = waves_peak["ECG_R_Offsets"]

        s_peaks = waves_peak["ECG_S_Peaks"]

        t_onsets = waves_peak["ECG_T_Onsets"]
        t_offsets = waves_peak["ECG_T_Offsets"]
        t_peaks = waves_peak["ECG_T_Peaks"]

        p_durations_ms = np.array([x - y for x, y in zip(p_offsets, p_onsets)]) / fs * 1000
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
        p_amplitudes = compute_amplitude(ecg_signal, p_peaks, p_onsets, fs)


        pr_ms = np.array([x - y for x, y in zip(r_onsets, p_onsets)]) / fs * 1000

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
            r_on = int(r_onsets[i])
            r_off = int(r_offsets[i])
            t_on = int(t_onsets[i])
            t_peak = int(t_peaks[i])
            t_off = int(t_offsets[i])

            qrs_segment = ecg_signal[r_on : r_off + 1]
            qrs_amp = np.max(qrs_segment) - np.min(qrs_segment)
            qrs_amp_list.append(float(qrs_amp))
            qrs_dur = (r_off - r_on) / fs * 1000.0
            qrs_dur_ms.append(float(qrs_dur))

            qt_val = (t_off - r_on) / fs * 1000.0
            qt_ms.append(float(qt_val))

            st_val = (t_on - r_off) / fs * 1000.0
            st_dur_ms.append(float(st_val))

            t_durations_ms.append(round(float((t_off - t_on) / fs * 1000.0), 1))
            bstart = max(0, r_on - baseline_win_samples)
            baseline = float(np.mean(ecg_signal[bstart : r_on])) if bstart < r_on else float(np.mean(ecg_signal[max(0, r_on - 1) : r_on + 1]))
            t_amplitudes.append(round(float(ecg_signal[t_peak]) - baseline, 3))

            # st form estimatation
            j_idx = r_off # assume j-point is at the end of the QRS-wave
            st_eval_idx = min(j_idx + int(st_evaluation_delay_ms * fs / 1000.0), len(ecg_signal) - 1)
            st_level_at_60 = float(ecg_signal[st_eval_idx])
            delta = st_level_at_60 - baseline
            slope = st_level_at_60 - float(ecg_signal[j_idx])
            if abs(delta) <= horiz_thresh_mV:
                form = "horizontal"
            else:
                # slope positive -> upslope, slope negative -> declination (downsloping)
                form = "upslope" if slope > 0 else "declination"
            st_forms.append(form)


        rr_samples_next = np.diff(r_onsets)
        rr_intervals_ms = rr_samples_next * 1000 / fs
        rr_intervals_ms = np.round(rr_intervals_ms, 1)
        max_rr_list.append(np.max(rr_intervals_ms))
        min_rr_list.append(np.min(rr_intervals_ms))
        mean_rr_list.append(np.mean(rr_intervals_ms))
        median_rr_list.append(np.median(rr_intervals_ms))
        
        rr_s = rr_samples_next / fs
        rr_s_per_beat = np.empty(len(r_onsets))
        rr_s_per_beat[:-1] = rr_s
        rr_s_per_beat[-1] = rr_s[-1] if len(rr_s) > 0 else np.nan  # fallback
        qtc_ms = qt_ms / np.sqrt(rr_s_per_beat) # _bazett algorithm
        
        hr_bpm = 60.0 / rr_s_per_beat
        heart_rate_list.append(np.mean(hr_bpm))

        lead_features["Lead I"] = {
            "P Amplitude (mV)": p_amplitudes.tolist(),
            "P Duration (ms)": p_durations_ms.tolist(),
            "PR Interval (ms)": pr_ms.tolist(),
            "QRS Amplitude (mV)": qrs_amp_list,
            "QRS Duration (ms)": qrs_dur_ms,
            "QT Interval(ms)": qt_ms,
            "QTC Interval(ms)": qtc_ms.tolist(),
            "ST Duration (ms)": st_dur_ms,
            "ST Form": st_forms,
            "T Amplitude (mV)": t_amplitudes,
            "T Duration (ms)": t_durations_ms,
        }

    global_features["Heart Rate (bpm)"] = np.mean(heart_rate_list)
    global_features["Max RR Interval (ms)"] = np.mean(max_rr_list)
    global_features["Min RR Interval (ms)"] = np.mean(min_rr_list)
    global_features["Mean RR Interval (ms)"] = np.mean(mean_rr_list)
    global_features["Median RR Interval (ms)"] = np.mean(median_rr_list)

    return {"gloal_features": global_features, "lead_features": lead_features}

if __name__ == '__main__':
    ecg_dir = './test/ecg_signal'
    output_dir = "./"
    for record in process_ecg_signal(ecg_dir):
        features = extract_ecg_features(record)
        # dump the features to a json file
        record_id = record.record_name
        output_path = os.path.join(output_dir, f"{record_id}_features.json")
        with open(output_path, 'w') as f:
            json.dump(features, f, indent=2)
    
    print("done!")