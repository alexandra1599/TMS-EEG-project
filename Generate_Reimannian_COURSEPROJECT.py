"""
© 2025 Alexandra Mikhael. All Rights Reserved.
"""

import os
import numpy as np
import pickle
import mne
from datetime import datetime
import re
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from pyriemann.estimation import Shrinkage
from pyriemann.classification import MDM, FgMDM
from pyriemann.estimation import Covariances, XdawnCovariances
import config
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from Utils.stream_utils import load_xdf, get_channel_names_from_xdf
from scipy.signal import butter, lfilter, lfilter_zi
from Utils.preprocessing import butter_bandpass, concatenate_streams, select_motor_channels
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


def plot_posterior_probabilities(posterior_probs):
    """
    Plots the histogram of posterior probabilities for each class.

    Parameters:
        posterior_probs (dict): Dictionary containing posterior probabilities for each class.
    """
    plt.figure(figsize=(10, 6))
    bins = np.linspace(0, 1, 20)  # Set bins for histogram

    # Convert numerical labels to "Rest" and "MI"
    label_map = {4: "Rest", 7: "MI"}
    renamed_probs = {label_map.get(label, f"Label {label}"): probs for label, probs in posterior_probs.items()}

    for label, probs in renamed_probs.items():
        probs = np.array(probs).flatten()
        sns.histplot(probs, bins=bins, alpha=0.6, label=f"{label} Probability", kde=True)

    plt.xlabel("Predicted Probability")
    plt.ylabel("Frequency")
    plt.title("Posterior Probability Distribution Across Classes")
    plt.legend(title="True Class")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()




# Stateful Filtering Function
def apply_stateful_filter(raw, b, a):
    filter_states = {}  # Initialize state tracking dictionary
    for ch_idx in range(len(raw.ch_names)):
        if ch_idx not in filter_states:
            filter_states[ch_idx] = lfilter_zi(b, a) * raw._data[ch_idx][0]  # Initialize filter state
        raw._data[ch_idx], filter_states[ch_idx] = lfilter(b, a, raw._data[ch_idx], zi=filter_states[ch_idx])
    return raw

def center_cov_matrices_riemannian(cov_matrices):
    """
    Center a set of covariance matrices around the identity matrix using the Riemannian mean.

    Parameters:
        cov_matrices (np.ndarray): Array of shape (n_matrices, n_channels, n_channels)

    Returns:
        np.ndarray: Centered covariance matrices
    """
    # Compute the Riemannian mean of the covariance matrices
    mean_cov = mean_riemann(cov_matrices, maxiter = 5000)

    # Compute the inverse square root of the mean covariance
    eigvals, eigvecs = np.linalg.eigh(mean_cov)
    inv_sqrt_mean_cov = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

    # Apply whitening transformation to center around the identity
    centered_matrices = np.array([inv_sqrt_mean_cov @ C @ inv_sqrt_mean_cov for C in cov_matrices])

    return centered_matrices

def segment_epochs(epochs, window_size=config.CLASSIFY_WINDOW, step_size=0.1):
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

def train_riemannian_model(cov_matrices, labels, n_splits=8, shrinkage_param=config.SHRINKAGE_PARAM):
    """
    Train an MDM classifier with k-fold cross-validation using Riemannian geometry 
    and plot posterior probability histograms.

    Parameters:
        cov_matrices (np.ndarray): Covariance matrices of shape (n_samples, n_channels, n_channels).
        labels (np.ndarray): Corresponding labels for the segments.
        n_splits (int): Number of splits for cross-validation.
        shrinkage_param (float): Regularization strength for Shrinkage.

    Returns:
        mdm (MDM): Trained MDM model.
    """

    print("\n🚀 Starting K-Fold Cross Validation with Riemannian MDM...\n")


    
    # Compute the reference matrix (Riemannian mean)
    '''
    reference_matrix = mean_riemann(cov_matrices, maxiter=1000)

    # Center covariance matrices
    
    cov_matrices = np.array([np.linalg.inv(reference_matrix) @ cov @ np.linalg.inv(reference_matrix)
                              for cov in cov_matrices])
    
    '''

    #cov_matrices = center_cov_matrices_riemannian(cov_matrices)
    #cov_matrices = center_cov_matrices(cov_matrices, reference_matrix)
    # Apply Shrinkage-based regularization

    if config.LEDOITWOLF:
        # Compute covariance matrices with optimized shrinkage
        cov_matrices_shrinked = np.array([LedoitWolf().fit(cov).covariance_ for cov in cov_matrices])
        cov_matrices = cov_matrices_shrinked    
    else:
        shrinkage = Shrinkage(shrinkage=shrinkage_param)
        cov_matrices = shrinkage.fit_transform(cov_matrices)  

    
    # Apply Riemannian whitening
    if config.RECENTERING:
        whitener = Whitening(metric="riemann")  # Use Riemannian mean for whitening
        cov_matrices = whitener.fit_transform(cov_matrices)
    
    #print(mean_riemann(cov_matrices))
    
    
    # Initialize cross-validation
    kf = KFold(n_splits=n_splits, shuffle=False)
    mdm = MDM()
    accuracies = []
    posterior_probs = {label: [] for label in np.unique(labels)}  # Store probabilities per class

    for fold_idx, (train_index, test_index) in enumerate(kf.split(cov_matrices), start=1):
        X_train, X_test = cov_matrices[train_index], cov_matrices[test_index]
        Y_train, Y_test = labels[train_index], labels[test_index]

        # Train and evaluate model
        mdm.fit(X_train, Y_train)
        Y_pred = mdm.predict(X_test)
        Y_predProb = mdm.predict_proba(X_test)  # Get class probabilities

        accuracy = accuracy_score(Y_test, Y_pred)
        accuracies.append(accuracy)
        print(f"\n ✅ Fold {fold_idx} Accuracy: {accuracy:.4f}")

        # Store probabilities per class
        for idx, true_label in enumerate(Y_test):
            class_idx = np.where(mdm.classes_ == true_label)[0][0]
            posterior_probs[true_label].append(Y_predProb[idx, class_idx])

    # Convert probability lists to numpy arrays
    for label in posterior_probs:
        posterior_probs[label] = np.array(posterior_probs[label])

    # Plot probability histograms
    plot_posterior_probabilities(posterior_probs)

    # Print overall accuracy
    avg_accuracy = np.mean(accuracies)
    print(f"\n🚀 **Final Average Accuracy:** {avg_accuracy:.4f}")

    # Retrain the model on all data
    mdm.fit(cov_matrices, labels)

    return mdm

def main():
    """
    Main function to generate a Riemannian-based EEG decoder.
    """
    mne.set_log_level("WARNING")

    print("🔄 Loading GDF data...")

    # Set path to folder containing the .gdf files
    folder_path = "/home/arman-admin/Documents/NE Course project data/Subject_205_Session_002_FES_Offline_Visual"

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
    montage = mne.channels.make_standard_montage("standard_1020")

    # Case-sensitive renaming dictionary


    # First, pick EEG channels only (drop stim, EOG, etc.)
    raw.pick_types(eeg=True)
    print(f"✅ Picked EEG channels: {len(raw.ch_names)} remaining")

    # 🖨️ Print the final EEG channels
    print(f"\n🎯 Final EEG Channels Used ({len(raw.ch_names)} total):")
    print(raw.ch_names)


    # Apply the standard 10-20 montage with fallback for unrecognized channels
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, on_missing="ignore")
    print("✅ Applied 10-20 montage (ignored non-matching channels)")

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
    EPOCHS_START_END = {
        4: 5,  # REST: start → end
        7: 8   # MI: start → end
    }


    # Configurable trial timing (in seconds)
    BASELINE_START = -1.0
    BASELINE_END = 0
    TRIAL_START = 1.0
    TRIAL_END = 5.0
    sfreq = 512
    events = []
    event_id_map = {}
    trial_counts = {}
    for start_marker, end_marker in EPOCHS_START_END.items():
        start_indices = np.where(marker_values == int(start_marker))[0]
        end_indices = np.where(marker_values == int(end_marker))[0]
        if len(start_indices) != len(end_indices):
            print(f"⚠️  Unequal markers: {start_marker} (start) and {end_marker} (end). Adjusting to shortest.")
            min_length = min(len(start_indices), len(end_indices))
            start_indices = start_indices[:min_length]
            end_indices = end_indices[:min_length]

        trial_counts[start_marker] = len(start_indices)

        for start_idx, end_idx in zip(start_indices, end_indices):
            # Convert marker timestamps to sample indices
            start_sample = np.searchsorted(eeg_timestamps, marker_timestamps[start_idx])
            end_sample = np.searchsorted(eeg_timestamps, marker_timestamps[end_idx])

            # Baseline correction
            baseline_start_sample = max(0, start_sample + int(sfreq * BASELINE_START))
            baseline_end_sample = start_sample + int(sfreq * BASELINE_END)
            baseline_mean = raw._data[:, baseline_start_sample:baseline_end_sample].mean(axis=1, keepdims=True)
            raw._data -= baseline_mean  # Subtract baseline mean from entire recording

            # Add to event list
            events.append([start_sample, 0, int(start_marker)])
            event_id_map[str(start_marker)] = int(start_marker)
    # Sort and convert to MNE-compatible format
    events = np.array(events)

    if len(events) == 0:
        raise ValueError("🚨 No trials found — check your marker values or event matching logic.")

    events = events[np.argsort(events[:, 0])]

    # Print trial counts
    label_map = {4: "REST", 7: "MI"}  # Marker 4 (769) = REST, 7 (770) = MI
    print("\n🧪 Trial Count Summary:")
    for marker, count in trial_counts.items():
        label = label_map.get(marker, f"Code {marker}")
        print(f" - {label} trials: {count}")

    # Create epochs
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id_map,
        tmin=TRIAL_START,
        tmax=TRIAL_END,
        baseline=None,  # Already manually baseline corrected
        detrend=1,
        preload=True
    )

    # Define Rejection Criteria (Artifact Removal)
    # Compute max per epoch
    max_per_epoch = np.max(np.abs(epochs.get_data()), axis=(1, 2))  

    # Compute z-scores
    z_scores = zscore(max_per_epoch)

    # Define a rejection criterion (e.g., 3 standard deviations)
    reject_z_threshold = 3.0  
    bad_epochs = np.where(np.abs(z_scores) > reject_z_threshold)[0]  

    # Drop the bad epochs
    epochs.drop(bad_epochs)
    print(f"Dropped {len(bad_epochs)} bad epochs based on z-score method.")

   # Slice Epochs into Smaller Training Windows (e.g., 0.5s)
    print(f"Segmenting epochs into {config.CLASSIFY_WINDOW}ms training windows...")
    segments, labels = segment_epochs(epochs, window_size=config.CLASSIFY_WINDOW, step_size=1/16)

    print(f"🔹 Segmented Data Shape: {segments.shape}")  # Debugging output

    # Compute Covariance Matrices (for Riemannian Classifier)
    print("Computing Covariance Matrices...")

    cov_matrices = []
    info = epochs.info  # Use the same info as the original epochs
    
    # Compute Covariance Matrices (for Riemannian Classifier)
    print("Computing Covariance Matrices...")
    #cov_matrices = np.array([np.cov(segment) for segment in segments])
    cov_matrices = np.array([ (segment @ segment.T) / np.trace(segment @ segment.T) for segment in segments ])

    '''
    for segment in segments:
        # Convert segment into an MNE EpochsArray (shape needs to be (n_epochs, n_channels, n_samples))
        segment = mne.EpochsArray(segment[np.newaxis, :, :], info)  # Ensure correct shape

        # Compute covariance matrix using OAS regularization
        cov = mne.compute_covariance(segment, method="oas")

        # Extract covariance matrix and store
        cov_matrices.append(cov["data"])
    '''
    # Convert list to numpy array (shape: (n_epochs, n_channels, n_channels))
    cov_matrices = np.array(cov_matrices)
    #print(cov_matrices[0])
    print(f"Computed {len(cov_matrices)} covariance matrices with shape: {cov_matrices.shape}")
    #print(f" Sample cov matrix: {cov_matrices[0]}")

    # Train Riemannian MDM Model
    #print(cov_matrices)
    print("Training Riemannian Classifier...")
    Reimans_model = train_riemannian_model(cov_matrices, labels)

    # Extract subject + session from folder name
    folder_basename = os.path.basename(folder_path)
    subject_match = re.search(r"Subject_(\d+)", folder_basename)
    session_match = re.search(r"Session_(\d+)", folder_basename)

    subject_id = f"sub-{subject_match.group(1)}" if subject_match else "sub-UNKNOWN"
    session_id = f"ses-{session_match.group(1)}" if session_match else "ses-UNKNOWN"

    # Gather decoder metadata
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    shrink_val = round(config.SHRINKAGE_PARAM, 3)
    lw_flag = config.LEDOITWOLF

    # Construct filename
    model_filename = f"{subject_id}_{session_id}_decoder_lw_{lw_flag}_shrink_{shrink_val}_{timestamp}.pkl"
    subject_model_path = os.path.join(folder_path, model_filename)

    # Save the model
    with open(subject_model_path, 'wb') as f:
        pickle.dump(Reimans_model, f)

    print(f"✅ Trained model saved at: {subject_model_path}")
    #np.save(Training_mean_path, training_mean)
    #np.save(Training_std_path, training_std)
    #print(" Saved precomputed training mean and std for real-time use.")


if __name__ == "__main__":
    main()
