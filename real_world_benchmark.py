from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import mne
from mne.datasets.sleep_physionet import age

# Import CBraMod architecture from braindecode if available
try:
    from braindecode.models import CBraMod
    HAS_BRAINDECODE = True
except ImportError:
    HAS_BRAINDECODE = False


def fetch_and_preprocess_sleep_edf(subject_id: int = 0) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Downloads a real 8-hour sleep EEG recording from PhysioNet Sleep-EDF,
    resamples to 200 Hz, extracts 30-second epochs, and maps ground-truth sleep stages.
    """
    print(f"--> Fetching real-world Sleep-EDF data for Subject {subject_id} from PhysioNet...")
    [file_path] = age.fetch_data(subjects=[subject_id], recording=[1])
    
    # Load raw EDF recording and corresponding annotations (.edf / .txt)
    raw = mne.io.read_raw_edf(file_path[0], preload=True, verbose=False)
    annot = mne.read_annotations(file_path[1])
    raw.set_annotations(annot, emit_warning=False)

    # Standardize channel configuration and apply 0.5-35 Hz bandpass filter
    raw.filter(l_freq=0.5, h_freq=35.0, verbose=False)

    # Map AASM Sleep Stage annotations to integer target classes
    annotation_mapping = {
        "Sleep stage W": 0,
        "Sleep stage 1": 1,
        "Sleep stage 2": 2,
        "Sleep stage 3": 3,
        "Sleep stage 4": 3,  # Merge N4 into N3 per AASM guidelines
        "Sleep stage R": 4
    }

    # Extract 30-second continuous epochs matching AASM staging
    events, event_id = mne.events_from_annotations(
        raw, event_id=annotation_mapping, chunk_duration=30.0, verbose=False
    )

    tmax = 30.0 - (1.0 / raw.info["sfreq"])
    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_id,
        tmin=0.0,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose=False
    )

    # Resample epochs to CBraMod's target 200 Hz sampling rate
    print("--> Resampling real EEG epochs to 200 Hz...")
    epochs.resample(sfreq=200.0, verbose=False)

    data = epochs.get_data(units="uV")  # Shape: [Epochs, Channels, Time_Samples (6000)]
    labels = epochs.events[:, -1]       # Integer sleep stage labels (0..4)

    return data, labels, epochs.info["sfreq"]


class CBraModRealWorldBenchmark(nn.Module):
    """CBraMod backbone coupled with a 5-class sleep staging head."""
    def __init__(self, num_channels: int, num_classes: int = 5):
        super().__init__()
        if HAS_BRAINDECODE:
            self.backbone = CBraMod(
                n_outputs=200,
                n_chans=num_channels,
                sfreq=200.0,
                return_encoder_output=True
            )
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(num_channels * 30 * 200, 256),
                nn.ELU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
        else:
            # Fallback mock architecture if braindecode package is not installed
            self.backbone = None
            self.head = nn.Sequential(
                nn.Conv1d(num_channels, 64, kernel_size=25, stride=5),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(10),
                nn.Flatten(),
                nn.Linear(64 * 10, num_classes)
            )

    def forward(self, x):
        if HAS_BRAINDECODE:
            feats = self.backbone(x)
            return self.head(feats)
        else:
            return self.head(x)
