import os
import numpy as np
import pickle
import mne
from datetime import datetime
import re
from mne import SourceEstimate
from sklearn.metrics import accuracy_score
from pyriemann.estimation import Shrinkage
from pyriemann.classification import MDM, FgMDM
from pyriemann.estimation import Covariances, XdawnCovariances
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, lfilter, lfilter_zi
import glob  # Required for multi-file loading
from scipy.stats import zscore
from pyriemann.utils.mean import mean_riemann
from scipy.linalg import sqrtm
import seaborn as sns
from sklearn.covariance import LedoitWolf
from pyriemann.preprocessing import Whitening




def load_and_merge_gdf(folder_path):
    """
    Load and concatenate multiple GDF files from a folder.

    Parameters:
        folder_path (str): Path to the folder containing .gdf files

    Returns:
        raw (mne.io.Raw): Concatenated MNE Raw object
        events (np.ndarray): Events array
        event_id (dict): Dictionary of event labels and their integer codes
    """
    # Get sorted list of .gdf files (ignore .log files)
    gdf_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".gdf")])
    if not gdf_files:
        raise FileNotFoundError(f"No .gdf files found in {folder_path}")

    print(f"Found {len(gdf_files)} GDF files:")
    for f in gdf_files:
        print(f" - {f}")

    raw_list = []

    for filename in gdf_files:
        full_path = os.path.join(folder_path, filename)
        raw = mne.io.read_raw_gdf(full_path, preload=True)
        raw_list.append(raw)

    # Concatenate the runs
    raw_combined = mne.concatenate_raws(raw_list)

    # Extract annotations and convert to events
    events, event_id = mne.events_from_annotations(raw_combined)

    print(f"🧠 Merged Raw has {len(events)} total events.")
    print(f"📌 Event ID mapping: {event_id}")

    return raw_combined, events, event_id



def segment_epochs(epochs, window_size=0.2, step_size=0.1):
    """
    Slice each epoch into smaller overlapping windows for training.

    Parameters:
        epochs (mne.Epochs): The full 5s epochs.
        window_size (float): Length of each training segment (e.g., 0.5s).
        step_size (float): Overlap between consecutive windows (e.g., 0.1s).
    
    Returns:
        np.ndarray: Segmented data (n_segments, n_channels, n_timepoints).
        np.ndarray: Corresponding labels for each segment.
    """
    window_size = window_size/1000
    sfreq = epochs.info["sfreq"]  # Sampling frequency
    step_samples = int(step_size * sfreq)  # Convert step size to samples
    window_samples = int(window_size * sfreq)  # Convert window size to samples

    segmented_data = []
    segmented_labels = []

    for i, epoch in enumerate(epochs.get_data()):  # Iterate over each full epoch
        label = epochs.events[i, -1]  # Get the class label

        for start in range(0, epoch.shape[1] - window_samples + 1, step_samples):
            end = start + window_samples
            segmented_data.append(epoch[:, start:end])
            segmented_labels.append(label)  # Each slice gets the same label

    return np.array(segmented_data), np.array(segmented_labels)



def main():
    """
    Main function to generate a Riemannian-based EEG decoder.
    """
    mne.set_log_level("WARNING")

    print("🔄 Loading GDF data...")

    # Set path to folder containing the .gdf files
    folder_path = "/Users/alexandra/Desktop/PhD/Neural Engineering/Project/Subject 205/Subject_205_FES_Online/Subject_205_Session_003_FES_Online_Visual"

    # Load and merge the GDF recordings
    raw, events, event_id = load_and_merge_gdf(folder_path)

    print(f"✅ Successfully loaded and merged GDF recordings.")
    print(f"📌 Event ID map: {event_id}")

    # ─────────────────────────────────────────────────────────
    # Simulate `eeg_stream` and `marker_stream` from raw/events
    # ─────────────────────────────────────────────────────────

    eeg_stream = {
        "time_series": raw.get_data().T,  # (n_samples, n_channels)
        "time_stamps": raw.times          # (n_samples,)
    }

    # Build marker stream from MNE events (sample, 0, event_id)
    marker_stream = {
        "time_series": [
            [int(ev[2]), raw.times[int(ev[0])]] for ev in events
        ]
    }

    # Now use your existing logic from here onward ⬇️
    eeg_timestamps = np.array(eeg_stream['time_stamps'])
    eeg_data = np.array(eeg_stream['time_series']).T  # (n_channels, n_samples)

    # Use channel names from the raw object
    channel_names = raw.info['ch_names']

    marker_values = np.array([int(m[0]) for m in marker_stream['time_series']])
    marker_timestamps = np.array([float(m[1]) for m in marker_stream['time_series']])
    # Load standard 10-20 montage
    # Load a 64-channel montage that supports topomap plotting
    montage = mne.channels.make_standard_montage("standard_1005")  # or try "biosemi64"
    raw.set_montage(montage, on_missing="warn", match_case=False)
    print("✅ Applied 64-channel montage (standard_1005)")


    # Case-sensitive renaming dictionary


    # First, pick EEG channels only (drop stim, EOG, etc.)
    raw.pick_types(eeg=True)
    print(f"✅ Picked EEG channels: {len(raw.ch_names)} remaining")

    # 🖨️ Print the final EEG channels
    print(f"\n🎯 Final EEG Channels Used ({len(raw.ch_names)} total):")
    print(raw.ch_names)



    # Remove Mastoid channels if they exist
    if "M1" in raw.ch_names and "M2" in raw.ch_names:
        raw.drop_channels(["M1", "M2"])
        print("Removed Mastoid Channels: M1, M2")
    else:
        print("No Mastoid Channels Found in Data")

    # Select your analysis-specific channel subset (64-cap based)
    target_channels = [
        'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8',
        'FC5', 'FC1', 'FC2', 'FC6',
        'T7', 'C3', 'CZ', 'C4', 'T8',
        'CP5', 'CP1', 'CP2', 'CP6',
        'P7', 'P3', 'PZ', 'P4', 'P8',
        'O1', 'OZ', 'O2',
        # Extend this list if needed
    ]
    raw.pick_channels(target_channels)

    # 🖨️ Print the final EEG channels
    print(f"\n🎯 Final EEG Channels Used ({len(raw.ch_names)} total):")
    print(raw.ch_names)


    # Apply Notch Filtering (Remove Powerline Noise)
    raw.notch_filter(60, method="iir")  

    # **Apply Bandpass Filtering with State Preservation**
    #b, a = butter_bandpass(config.LOWCUT, config.HIGHCUT, sfreq, order=4)
    #raw = apply_stateful_filter(raw, b, a)

    raw.filter(l_freq=8, h_freq=12, method="iir")  # Bandpass filter (8-16Hz)

    # Apply Common Average Reference (CAR)
    #raw.set_eeg_reference("average")

    # Print remaining channels to confirm
    # Define start and end markers for each condition
    # Define start and end markers for each condition
    # Define start markers and corresponding possible end markers
    EPOCHS_START_END = {
    5: [6, 7],   # REST
    9: [10, 11]  # MI
}

    # Updated epoch timing to include baseline window
    TRIAL_START = -1.0
    TRIAL_END = 5.0
    sfreq = 512 

    events = []
    event_id_map = {}
    trial_counts = {}

    for start_marker, possible_end_markers in EPOCHS_START_END.items():
        start_indices = np.where(marker_values == int(start_marker))[0]
        end_indices_all = np.where(np.isin(marker_values, possible_end_markers))[0]

        matched_starts = []
        matched_ends = []

        for start_idx in start_indices:
            start_time = marker_timestamps[start_idx]

            # Find the first end marker that occurs after this start marker
            valid_ends = end_indices_all[marker_timestamps[end_indices_all] > start_time]
            if len(valid_ends) > 0:
                end_idx = valid_ends[0]
                matched_starts.append(start_idx)
                matched_ends.append(end_idx)

        if len(matched_starts) == 0:
            print(f"⚠️ No valid trials found for start marker {start_marker}")
            continue

        trial_counts[start_marker] = len(matched_starts)

        for start_idx in matched_starts:
            start_sample = np.searchsorted(eeg_timestamps, marker_timestamps[start_idx])
            events.append([start_sample, 0, int(start_marker)])
            event_id_map[str(start_marker)] = int(start_marker)

    # Sort events chronologically
    events = np.array(events)
    if len(events) == 0:
        raise ValueError("🚨 No trials found — check your marker values or event matching logic.")
    events = events[np.argsort(events[:, 0])]

    # Trial count summary
    label_map = {5: "REST", 9: "MI"}
    print("\n🧪 Trial Count Summary:")
    for marker, count in trial_counts.items():
        label = label_map.get(marker, f"Code {marker}")
        print(f" - {label} trials: {count}")

    #  Add average EEG reference as a projector
    raw.set_eeg_reference(projection=True)

    # Load the data so projectors can be applied
    raw.load_data()

    # (Optional but recommended) Apply the projectors immediately
    raw.apply_proj()

    # Create epochs including pre-trial baseline
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id_map,
        tmin=TRIAL_START,
        tmax=TRIAL_END,
        baseline=None,  # Let TFR handle the baseline
        detrend=1,
        preload=True
    )



    # === ERD/ERS Time-Frequency Analysis ===
    print("⚡ Computing ERD/ERS topographic maps...")

    time_start = -1.0
    baseline_period = 1.0
    time_end = 5.0
    lowband, highband = 8, 30
    baseline = (time_start, time_start + baseline_period)
    window_size = 0.5
    time_windows = np.arange(0, 5, window_size)
    marker_labels = {"5": "REST", "9": "MI"}

    tfr_data = {}

    for marker in ["5", "9"]:
        if marker in event_id_map:
            freqs = np.linspace(lowband, highband, highband - lowband)
            tfr = epochs[str(event_id_map[marker])].compute_tfr(
                method="multitaper",
                freqs=freqs,
                tmin=time_start,
                tmax=time_end,
                n_cycles=2.5,
                use_fft=True,
                return_itc=False
            )
            tfr.apply_baseline(baseline=baseline, mode="logratio")
            tfr_data[marker] = tfr.average()

    all_erd_values = np.concatenate([tfr_avg.data.flatten() for tfr_avg in tfr_data.values()])
    vmin, vmax = np.percentile(all_erd_values, [2, 98])
    print(f"🎨 ERD/ERS Color Scale: vmin={vmin:.2f}, vmax={vmax:.2f}")

    figures = {}
    skip_factor = 2

    for marker, tfr_avg in tfr_data.items():
        selected_indices = range(0, len(time_windows), skip_factor)
        fig, axes = plt.subplots(1, len(selected_indices), figsize=(15, 4), constrained_layout=True)
        im = None

        for ax, i in zip(axes, selected_indices):
            t_start = time_windows[i]
            t_end = t_start + window_size

            img = tfr_avg.plot_topomap(
                tmin=t_start,
                tmax=t_end,
                axes=ax,
                cmap="viridis",
                show=False,
                vlim=(vmin, vmax),
                colorbar=False
            )

            if hasattr(ax, "collections") and len(ax.collections) > 0:
                im = ax.collections[0]

            ax.set_title(f"{t_start:.1f}-{t_end:.1f}s")

        if im is not None:
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            sm = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.05, pad=0.1)
            cbar.set_label("ERD/ERS (logratio)", fontsize=12)

        marker_label = marker_labels.get(marker, f"Marker {marker}")
        fig.suptitle(f"ERD/ERS Over Time - {marker_label}", fontsize=14)
        figures[marker] = fig

    plt.show()

    # === Source Localization Analysis ===
   
    # Compute noise covariance

    subjects_dir = "/Users/alexandra/mne_data/MNE-fsaverage-data"
    subject = "fsaverage"

    # Fetch the fsaverage subject if you haven't already
    mne.datasets.fetch_fsaverage(subjects_dir=subjects_dir)

    # Now fsaverage data will be in the specified subjects_dir
    # 4. Compute noise covariance
    noise_cov = mne.compute_covariance(epochs, tmin=-1.0, tmax=0.0, method='shrunk')

    # Create BEM model and source space
    subject = "fsaverage"  # Or your own subject with T1 MRI
    subjects_dir = os.path.dirname(mne.datasets.fetch_fsaverage(verbose=True))
    print("Subjects dir:", subjects_dir)
    print("Contents:", os.listdir(subjects_dir))
    # Define subject and subjects_dir
    subjects_dir = "/Users/alexandra/mne_data/MNE-fsaverage-data"
    subject = "fsaverage"


    mne.datasets.fetch_fsaverage(subjects_dir="/Users/alexandra/mne_data/MNE-fsaverage-data", verbose=True)

    # Set up the source space
    src = mne.setup_source_space(subject, spacing='oct6', subjects_dir=subjects_dir, add_dist=False)

    # Set up the BEM model (3-layer)
    bem_model = mne.make_bem_model(subject, subjects_dir=subjects_dir)
    bem = mne.make_bem_solution(bem_model)
    trans = 'fsaverage'  # Or your own trans file (must match montage/MRI)

    # Perform forward solution using the 3-layer BEM model
    fwd = mne.make_forward_solution(raw.info, trans=trans, src=src, bem=bem, eeg=True, meg=False)

    # Create inverse operator (weighted MNE)
    inverse_operator = mne.minimum_norm.make_inverse_operator(raw.info, fwd, noise_cov,loose=0.2, depth=0.8)

    # Assume epochs are already epoched and cleaned

    # Define frequency band of interest
    fmin, fmax = 1, 30  # Beta band

    # Compute power (source estimate) using inverse operator on epochs
    power_mi = mne.minimum_norm.apply_inverse_epochs(
    epochs['9'], inverse_operator, lambda2=1. / 9., method='MNE', pick_ori="normal"
)

    power_rest = mne.minimum_norm.apply_inverse_epochs(
    epochs['5'], inverse_operator, lambda2=1. / 9., method='MNE', pick_ori="normal"
)

    # Average power over time and trials
    mi_power = np.mean([np.mean(np.square(stc.data), axis=1) for stc in power_mi], axis=0)
    rest_power = np.mean([np.mean(np.square(stc.data), axis=1) for stc in power_rest], axis=0)

    # Compute log-ratio ERD/ERS (in dB)
    epsilon = 1e-10
    log_erd_ers = 10 * np.log10((mi_power + epsilon) / (rest_power + epsilon))  # dB scale

    # Create SourceEstimate to visualize
    stc_template = power_mi[0]  # Use any STC as a template
    erd_stc = mne.SourceEstimate(
    log_erd_ers[:, np.newaxis],  # Add time axis back
    vertices=stc_template.vertices,
    tmin=0,
    tstep=1,  # Single timepoint
    subject='fsaverage'
)

## === Interactive time plot ===
    brain = erd_stc.plot(
    subject='fsaverage',
    subjects_dir=subjects_dir,
    hemi='both',
    views='lateral',
    size=(1000, 700),
    background='white',
    foreground='black',
    time_unit='s',
    colormap='RdBu_r',
    clim=dict(kind='value', lims=[-5, 0, 5]),  # Adjust as needed
    time_viewer=True  # <-- This enables interactive time slider!
)

    brain.show_view('dorsal')

    brain = stc_template.plot(subject=subject, subjects_dir=subjects_dir, hemi='both', views='lat')
    brain.save_movie('/Users/alexandra/Desktop/PhD/Neural Engineering/Project/205posttms.mp4', time_dilation=10)  # Slower playback 


if __name__ == "__main__":
    main()
