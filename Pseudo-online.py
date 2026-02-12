"""
© 2025 Alexandra Mikhael. All Rights Reserved.
"""

import os
import argparse
import pickle
import numpy as np
import mne
from datetime import datetime
import matplotlib.pyplot as plt
from pyriemann.classification import MDM
from Utils.preprocessing import concatenate_streams  # reuse from generate script
from Generate_Reimannian_COURSEPROJECT import load_and_merge_gdf, segment_epochs
from pyriemann.utils.mean import invsqrtm
from pyriemann.utils.geodesic import geodesic_riemann
from pyriemann.estimation import Shrinkage
from sklearn.covariance import LedoitWolf
import config


def classify_trial_sliding(
    raw, start_sample, end_sample, sfreq, model,
    window_size_sec=0.5, step_size_sec=1/16
):
    """
    Classify one trial using sliding windows and adaptive recentering.

    Parameters:
        raw: MNE Raw object (preloaded)
        start_sample: trial start (in samples)
        end_sample: trial end (in samples)
        sfreq: Sampling frequency
        model: Trained Riemannian classifier
        window_size_sec: window size in seconds
        step_size_sec: step size in seconds

    Returns:
        List of probabilities over time (for correct class)
    """


    # Pre-allocate
    proba_trace = []
    Prev_T = None
    counter = 0

    # Extract trial data from raw
    data = raw.get_data(start=start_sample, stop=end_sample)  # shape (n_channels, n_samples)
    n_samples = data.shape[1]
    window_samples = int(window_size_sec * sfreq)
    step_samples = int(step_size_sec * sfreq)

    for i in range(0, n_samples - window_samples + 1, step_samples):
        segment = data[:, i:i+window_samples]

        # Convert segment to MNE RawArray
        info = raw.info.copy()
        raw_segment = mne.io.RawArray(segment, info)

        # Drop mastoids if needed
        for ch in ["M1", "M2"]:
            if ch in raw_segment.ch_names:
                raw_segment.drop_channels([ch])

        # Filtering (Notch + Bandpass)
        raw_segment.notch_filter(60, method="iir")
        raw_segment.filter(8, 12, method="iir")

        # Select motor channels if needed
        if config.SELECT_MOTOR_CHANNELS:
            raw_segment = select_motor_channels(raw_segment)

        # Compute covariance
        X = raw_segment.get_data()
        cov = (X @ X.T) / np.trace(X @ X.T)

        # Apply shrinkage
        if config.LEDOITWOLF:
            cov = LedoitWolf().fit(cov).covariance_
        else:
            cov = Shrinkage(shrinkage=config.SHRINKAGE_PARAM).fit_transform(cov[np.newaxis, :, :])[0]

        # Adaptive recentering
        if config.RECENTERING:
            if counter == 0:
                Prev_T = cov
            T_test = geodesic_riemann(Prev_T, cov, 1 / (counter + 1))
            Prev_T = T_test
            counter += 1
            T_inv = invsqrtm(T_test)
            cov = T_inv @ cov @ T_inv.T

        cov = np.expand_dims(cov, axis=0)  # shape (1, channels, channels)
        probs = model.predict_proba(cov)[0]

        # Determine MI or REST trial
        correct_label = 200 if raw.annotations.description[start_sample] == '770' else 100
        class_index = np.where(model.classes_ == correct_label)[0][0]
        proba_trace.append(probs[class_index])

    return proba_trace



def main(model_dir, working_dir):
    """
    Pseudo-online replay of MI/REST decoding using a trained Riemannian classifier.

    Parameters:
        model_dir (str): Directory containing the trained .pkl decoder file.
        working_dir (str): Directory containing the GDF files to analyze.
    """
    from pyriemann.utils.geodesic import geodesic_riemann
    from pyriemann.utils.mean import invsqrtm
    from pyriemann.estimation import Shrinkage
    from sklearn.covariance import LedoitWolf

    mne.set_log_level("WARNING")
    print("🚀 Starting pseudo-online analysis...")
    print(f"📁 Data Directory: {working_dir}")
    print(f"📦 Model Directory: {model_dir}")

    # 🔍 Locate model
    model_candidates = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]
    if len(model_candidates) != 1:
        raise RuntimeError(f"🚨 Expected exactly one .pkl model file in {model_dir}, found {len(model_candidates)}.")
    model_path = os.path.join(model_dir, model_candidates[0])
    print(f"📦 Using decoder model: {model_candidates[0]}")

    # Load model
    with open(model_path, 'rb') as f:
        decoder = pickle.load(f)
    print("✅ Loaded trained decoder.")

    # 🔄 Find all .gdf files in the working directory
    run_filenames = sorted([f for f in os.listdir(working_dir) if f.endswith(".gdf")])
    if not run_filenames:
        raise RuntimeError(f"🚨 No GDF files found in: {working_dir}")
    print(f"📑 Found {len(run_filenames)} GDF files to analyze.")

    for run_name in run_filenames:
        run_path = os.path.join(working_dir, run_name)
        print(f"\n🧠 Processing run: {run_name}")

        raw = mne.io.read_raw_gdf(run_path, preload=True)
        events, event_id = mne.events_from_annotations(raw)

        # Step 1: Preprocessing
        raw.pick_types(eeg=True)

        montage = mne.channels.make_standard_montage("standard_1020")

        # Drop mastoids
        for ch in ["M1", "M2"]:
            if ch in raw.ch_names:
                raw.drop_channels([ch])

        # Select target channels
        target_channels = [
            'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8',
            'FC5', 'FC1', 'FC2', 'FC6',
            'T7', 'C3', 'CZ', 'C4', 'T8',
            'CP5', 'CP1', 'CP2', 'CP6',
            'P7', 'P3', 'PZ', 'P4', 'P8',
            'O1', 'OZ', 'O2',
        ]
        raw.pick_channels([ch for ch in target_channels if ch in raw.ch_names])

        raw.notch_filter(60, method="iir")
        raw.filter(config.LOWCUT, config.HIGHCUT, method="iir")
        # Step 2: Segment each trial using 769/770 → 7691/7701
        sfreq = int(raw.info['sfreq'])
        marker_values = events[:, 2]
        marker_times = events[:, 0]

        EPOCHS_START_END = {4: 5, 8: 9}
        all_trials = []

        for start_code, end_code in EPOCHS_START_END.items():
            start_indices = np.where(marker_values == start_code)[0]
            end_indices = np.where(marker_values == end_code)[0]

            if len(start_indices) != len(end_indices):
                min_len = min(len(start_indices), len(end_indices))
                start_indices = start_indices[:min_len]
                end_indices = end_indices[:min_len]

            for s_idx, e_idx in zip(start_indices, end_indices):
                start_sample = marker_times[s_idx]
                end_sample = marker_times[e_idx]
                label = start_code  # 4 = REST, 8 = MI

                all_trials.append((start_sample, end_sample, label))

        print(f"🔎 Found {len(all_trials)} valid trials.")

        # Step 3: Slide through each trial with adaptive recentering
        trial_results = []

        for trial_num, (start_sample, end_sample, label) in enumerate(all_trials):
            baseline_start_sample = max(0, int(start_sample - sfreq))  # 1s before trial
            baseline_end_sample = int(start_sample)
            baseline_data = raw.get_data(start=baseline_start_sample, stop=baseline_end_sample)
            baseline_mean = baseline_data.mean(axis=1, keepdims=True)

            data = raw.get_data(start=int(start_sample), stop=int(end_sample))
            data = data - baseline_mean  # 🧼 baseline correction

            n_samples = data.shape[1]
            win_samples = int(config.CLASSIFY_WINDOW / 1000 * sfreq)
            step_samples = int(1 / 16 * sfreq)

            Prev_T = None
            counter = 0
            proba_trace = []

            for i in range(0, n_samples - win_samples + 1, step_samples):
                segment = data[:, i:i + win_samples]
                raw_seg = mne.io.RawArray(segment, raw.info.copy())

                eeg = raw_seg.get_data()
                cov = (eeg @ eeg.T) / np.trace(eeg @ eeg.T)

                if config.LEDOITWOLF:
                    cov = LedoitWolf().fit(cov).covariance_
                else:
                    cov = Shrinkage(shrinkage=config.SHRINKAGE_PARAM).fit_transform(cov[np.newaxis])[0]

                if config.RECENTERING:
                    if counter == 0:
                        Prev_T = cov
                    T = geodesic_riemann(Prev_T, cov, 1 / (counter + 1))
                    Prev_T = T
                    counter += 1
                    T_inv = invsqrtm(T)
                    cov = T_inv @ cov @ T_inv.T

                cov = np.expand_dims(cov, axis=0)
                probs = decoder.predict_proba(cov)[0]

                proba_trace.append(probs)

            trial_results.append({
                "label": label,
                "proba_trace": proba_trace
            })


        print(f"✅ Completed {len(trial_results)} trials with probability traces.")


if __name__ == "__main__":
    model_dir = "/home/arman-admin/Documents/NE Course project data/Subject_205_Session_002_FES_Offline_Visual"
    working_dir = "/home/arman-admin/Documents/NE Course project data/Subject_205_FES_Online/Subject_205_Session_001_FES_Online_Visual"


    main(model_dir, working_dir)
