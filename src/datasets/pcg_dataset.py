#!/usr/bin/env python3
"""
Datasets for PCG-based reduced-LVEF detection.

PCGDataset
    - Loads WAVs listed in a split CSV.
    - Applies resampling, 20-800 Hz band-pass filtering, and fixed-length center crop/pad.
    - Builds MFCC or gammatone time-frequency representations.
    - Normalises with dataset-level mean/std if provided.
    - Outputs tensors shaped (3, image_size, image_size) plus binary label.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

import soundfile as sf
import torchaudio.functional as AF
import torchaudio.transforms as AT

# Optional dependency for gammatone representations.
try:
    from gammatone.gtgram import gtgram
    HAVE_GAMMATONE = True
except ImportError:
    HAVE_GAMMATONE = False


def ef_to_label(ef: float) -> int:
    """Map ejection fraction to binary label: 1 if EF <= 40 else 0."""
    return int(float(ef) <= 40.0)


class PCGDataset(Dataset):
    """
    On-the-fly dataset that reads WAV files and builds time-frequency images.

    CSV requirements:
        - Columns: path, label, patient_id, device, ef
        - Optional: position
    """

    def __init__(
        self,
        csv_path: str,
        representation: str = "mfcc",
        sample_rate: int = 2000,
        fixed_duration: float = 4.0,
        image_size: int = 224,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        device_filter: Optional[List[str]] = None,
        position_filter: Optional[List[str]] = None,
        clamp: bool = True,
    ):
        """
        Args:
            csv_path: Path to a split CSV (e.g. splits/metadata_train.csv).
            representation: "mfcc" or "gammatone".
            sample_rate: Target sampling rate in Hz.
            fixed_duration: Waveform duration (seconds) after cropping/padding.
            image_size: Output H=W size for the time-frequency image.
            mean, std: Optional z-score normalisation parameters.
            device_filter: Optional list of device names to keep.
            position_filter: Optional list of auscultation sites to keep.
            clamp: Whether to clamp the final image values to a fixed range.
        """
        super().__init__()

        if representation not in ("mfcc", "gammatone"):
            raise ValueError("representation must be 'mfcc' or 'gammatone'")

        self.csv_path = csv_path
        self.representation = representation
        self.sample_rate = sample_rate
        self.fixed_duration = fixed_duration
        self.n_samples = int(sample_rate * fixed_duration)
        self.image_size = image_size

        self.mean = mean
        self.std = std
        self.clamp = clamp

        # Load recording metadata.
        df = pd.read_csv(csv_path, dtype={"patient_id": str})

        required_cols = {"path", "label", "patient_id", "device", "ef"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"CSV {csv_path} is missing required columns: {missing}")

        if device_filter is not None:
            df = df[df["device"].isin(device_filter)]
        if position_filter is not None and "position" in df.columns:
            df = df[df["position"].isin(position_filter)]

        df = df.reset_index(drop=True)
        if len(df) == 0:
            raise ValueError("After filtering, no rows remain in the dataset.")

        self.df = df

        # Configure the requested time-frequency transform.
        if self.representation == "mfcc":
            # MFCC configuration.
            n_mfcc = 40
            n_fft = 128
            hop_length = int(0.01 * sample_rate)

            self.tf_transform = AT.MFCC(
                sample_rate=sample_rate,
                n_mfcc=n_mfcc,
                melkwargs={
                    "n_fft": n_fft,
                    "hop_length": hop_length,
                    "n_mels": 64,
                    "f_min": 20.0,
                    "f_max": 800.0,
                    "center": True,
                    "power": 2.0,
                },
            )
        else:
            # Gammatone configuration via the external `gammatone` package.
            if not HAVE_GAMMATONE:
                raise ImportError(
                    "Representation 'gammatone' requires the 'gammatone' package.\n"
                    "Install it with: pip install gammatone"
                )
            # Store the gammatone settings for use in _to_tf_representation.
            self.gt_window_time = 0.04
            self.gt_hop_time = 0.01
            self.gt_channels = 64
            self.gt_f_min = 20.0

    # Audio loading and preprocessing.
    def __len__(self) -> int:
        return len(self.df)

    def _load_waveform(self, path: str) -> Tuple[torch.Tensor, int]:
        """
        Load waveform from disk using soundfile.

        Returns:
            waveform: Tensor [channels, time]
            orig_sr: original sampling rate (int)
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"WAV file not found: {path}")

        data, sr = sf.read(path, dtype="float32")

        if data.ndim == 1:
            data = data[np.newaxis, :]
        else:
            data = data.T

        waveform = torch.from_numpy(data)
        return waveform, int(sr)

    def _resample_if_needed(self, waveform: torch.Tensor, orig_sr: int) -> torch.Tensor:
        """
        Resample waveform to self.sample_rate if needed.
        waveform shape: [channels, time]
        """
        if orig_sr == self.sample_rate:
            return waveform
        return AF.resample(waveform, orig_freq=orig_sr, new_freq=self.sample_rate)

    def _bandpass_20_800(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply a simple 20-800 Hz band-pass using high-pass + low-pass biquads.
        waveform shape: [channels, time]
        """
        waveform = AF.highpass_biquad(
            waveform, self.sample_rate, cutoff_freq=20.0, Q=0.707
        )
        waveform = AF.lowpass_biquad(
            waveform, self.sample_rate, cutoff_freq=800.0, Q=0.707
        )
        return waveform

    def _fix_length(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Center-crop or zero-pad waveform to self.n_samples.
        waveform shape: [channels, time]
        """
        num_samples = waveform.shape[-1]
        if num_samples == self.n_samples:
            return waveform
        elif num_samples > self.n_samples:
            start = (num_samples - self.n_samples) // 2
            end = start + self.n_samples
            return waveform[..., start:end]
        else:
            pad_total = self.n_samples - num_samples
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            return F.pad(waveform, (pad_left, pad_right))

    # Time-frequency conversion and image formatting.
    def _to_tf_representation(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Convert waveform to a time-frequency representation.

        waveform shape: [channels, time]
        Returns:
            spec: Tensor [freq, time]
        """
        # Use a single channel if the recording contains multiple channels.
        if waveform.shape[0] > 1:
            waveform = waveform[:1, :]

        if self.representation == "mfcc":
            spec = self.tf_transform(waveform)
            spec = spec[0]
        else:
            # Compute the gammatone representation with the external package.
            wave_np = waveform.squeeze(0).cpu().numpy()

            spec_np = gtgram(
                wave_np,
                fs=self.sample_rate,
                window_time=self.gt_window_time,
                hop_time=self.gt_hop_time,
                channels=self.gt_channels,
                f_min=self.gt_f_min,
            )
            spec = torch.from_numpy(spec_np).to(torch.float32)

        return spec

    def _spec_to_image(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Convert a [freq, time] representation to a (3, image_size, image_size) tensor.

        Steps:
        - Add channel and batch dimensions
        - Resize with bilinear interpolation
        - Repeat to 3 channels
        - Sanitize NaNs/Infs
        - Apply z-score normalisation using dataset-level stats
        - Clamp to a reasonable range to avoid exploding activations
        """
        spec = spec.unsqueeze(0).unsqueeze(0)
        spec = F.interpolate(
            spec,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        spec = spec.squeeze(0)
        img = spec.repeat(3, 1, 1)

        # Replace non-finite values before normalisation.
        img = torch.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

        mean = self.mean
        std = self.std
        if (mean is not None) and (std is not None):
            img = (img - mean) / (std + 1e-8)

        if self.clamp:
            # Clamp after normalisation to keep values in a moderate range.
            img = torch.clamp(img, min=-10.0, max=10.0)

        return img

    # Dataset protocol.
    def _row_to_example(
        self, row: pd.Series
    ) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        path = row["path"]
        patient_id = str(row["patient_id"])
        device = row["device"]
        ef = float(row["ef"])

        label = ef_to_label(ef)

        waveform, orig_sr = self._load_waveform(path)
        waveform = self._resample_if_needed(waveform, orig_sr)
        waveform = self._bandpass_20_800(waveform)
        waveform = self._fix_length(waveform)

        spec = self._to_tf_representation(waveform)
        img = self._spec_to_image(spec)

        meta = {
            "patient_id": patient_id,
            "device": device,
            "ef": ef,
            "path": path,
        }
        if "position" in row:
            meta["position"] = row["position"]
        if "filename" in row:
            meta["filename"] = row["filename"]

        return img, label, meta

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        row = self.df.iloc[idx]
        return self._row_to_example(row)
