#!/usr/bin/env python3
"""
Datasets for PCG-based low LVEF detection.

PCGDataset
    - Loads WAVs listed in a split CSV.
    - Applies resampling, 20-800 Hz band-pass, fixed-length crop/pad.
    - Builds MFCC or gammatone time-frequency images.
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

# Optional gammatone dependency
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
    On-the-fly dataset that reads WAV files and builds spectrogram "images".

    CSV requirements:
        - Columns: path, label, patient_id, device, ef
        - Optional: position
    """

    def __init__(
        self,
        csv_path: str,
        representation: str = "mfcc",  # "mfcc" or "gammatone"
        sample_rate: int = 2000,
        fixed_duration: float = 4.0,  # seconds
        image_size: int = 224,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        device_filter: Optional[List[str]] = None,  # e.g. ["iphone"]
        position_filter: Optional[List[str]] = None,  # e.g. ["M"]
        clamp: bool = True,
    ):
        """
        Args:
            csv_path: Path to a split CSV (e.g. splits/metadata_train.csv).
            representation: "mfcc" or "gammatone".
            sample_rate: Target sampling rate in Hz.
            fixed_duration: Waveform duration (seconds) after cropping/padding.
            image_size: Output H=W size for the spectrogram image.
            mean, std: Optional z-score normalisation parameters.
            device_filter: Optional list of device names to keep
                (values from metadata 'device' column).
            position_filter: Optional list of positions to keep
                (values "A", "P", "M", "T").
            clamp: Whether to clamp the final image values to a fixed range.
        """
        super().__init__()

        assert representation in ("mfcc", "gammatone"), (
            "representation must be 'mfcc' or 'gammatone'"
        )

        self.csv_path = csv_path
        self.representation = representation
        self.sample_rate = sample_rate
        self.fixed_duration = fixed_duration
        self.n_samples = int(sample_rate * fixed_duration)
        self.image_size = image_size

        self.mean = mean
        self.std = std
        self.clamp = clamp

        # ---------------------------------------------------------------------
        # Load metadata
        # ---------------------------------------------------------------------
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

        # ---------------------------------------------------------------------
        # Configure time-frequency transforms
        # ---------------------------------------------------------------------
        if self.representation == "mfcc":
            # MFCC configuration:
            n_mfcc = 40
            n_fft = 128  # -> n_freqs = 65 (OK for 64 mels)
            hop_length = int(0.01 * sample_rate)  # ~10 ms

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
            # Gammatonegram config using external `gammatone` package.
            if not HAVE_GAMMATONE:
                raise ImportError(
                    "Representation 'gammatone' requires the 'gammatone' package.\n"
                    "Install it with: pip install gammatone"
                )
            # Store gammatone parameters; computation is in _to_tf_representation.
            self.gt_window_time = 0.04  # 40 ms windows
            self.gt_hop_time = 0.01  # 10 ms hop
            self.gt_channels = 64  # number of gammatone filters
            self.gt_f_min = 20.0  # lowest frequency (Hz)

    # -------------------------------------------------------------------------
    # Audio loading and preprocessing
    # -------------------------------------------------------------------------
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

        data, sr = sf.read(path, dtype="float32")  # data: np.ndarray

        if data.ndim == 1:
            data = data[np.newaxis, :]       # [1, time]
        else:
            data = data.T                    # [channels, time]

        waveform = torch.from_numpy(data)    # [channels, time]
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

    # -------------------------------------------------------------------------
    # Time-frequency and image conversion
    # -------------------------------------------------------------------------
    def _to_tf_representation(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Convert waveform to a time-frequency representation.

        waveform shape: [channels, time]
        Returns:
            spec: Tensor [freq, time]
        """
        # Ensure single-channel
        if waveform.shape[0] > 1:
            waveform = waveform[:1, :]  # [1, time]

        if self.representation == "mfcc":
            # torchaudio MFCC expects [channel, time]
            spec = self.tf_transform(waveform)  # [1, n_mfcc, time]
            spec = spec[0]                      # [n_mfcc, time]
        else:
            # Gammatonegram via external gammatone.gtgram.gtgram
            wave_np = waveform.squeeze(0).cpu().numpy()

            spec_np = gtgram(
                wave_np,
                fs=self.sample_rate,
                window_time=self.gt_window_time,
                hop_time=self.gt_hop_time,
                channels=self.gt_channels,
                f_min=self.gt_f_min,
            )
            # gtgram returns [channels, time] (nonnegative magnitudes)
            spec = torch.from_numpy(spec_np).to(torch.float32)  # [channels, time]

        return spec

    def _spec_to_image(self, spec: torch.Tensor, device: str) -> torch.Tensor:
        """
        Convert [freq, time] spec to (3, image_size, image_size) tensor.

        Steps:
        - Add channel and batch dimensions
        - Resize with bilinear interpolation
        - Repeat to 3 channels
        - Sanitize NaNs/Infs
        - Apply z-score normalisation using dataset-level stats
        - Clamp to a reasonable range to avoid exploding activations
        """
        # [freq, time] -> [1, 1, freq, time]
        spec = spec.unsqueeze(0).unsqueeze(0)
        spec = F.interpolate(
            spec,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )  # [1, 1, H, W]
        spec = spec.squeeze(0)           # [1, H, W]
        img = spec.repeat(3, 1, 1)       # [3, H, W]

        # Make sure there are no NaNs or Infs in the image
        img = torch.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

        mean = self.mean
        std = self.std
        if (mean is not None) and (std is not None):
            img = (img - mean) / (std + 1e-8)

        if self.clamp:
            # Clamp after normalisation: keep values in a moderate range
            img = torch.clamp(img, min=-10.0, max=10.0)

        return img

    # -------------------------------------------------------------------------
    # Dataset protocol
    # -------------------------------------------------------------------------
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
        img = self._spec_to_image(spec, device=device)

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


class PCGBagDataset(PCGDataset):
    """
    Patient-level dataset for MIL-style pooling.

    Each item is a bag of recordings from the same patient:
    - bag: Tensor [N, 3, H, W]
    - label: binary patient label
    - meta: patient_id and bag summary
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        grouped = self.df.groupby("patient_id", sort=True)
        label_counts = grouped["label"].nunique()
        conflicts = label_counts[label_counts > 1]
        if not conflicts.empty:
            raise ValueError(
                "Conflicting labels within patient_id group; "
                "patient-level MIL requires consistent labels."
            )

        self.patient_ids = grouped.size().index.tolist()
        self.bag_indices = {
            pid: sorted(idx_list)
            for pid, idx_list in grouped.indices.items()
        }
        self.patient_df = grouped[["label", "ef"]].first().reset_index()

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        patient_id = self.patient_ids[idx]
        row_indices = self.bag_indices[patient_id]
        if not row_indices:
            raise ValueError(f"No recordings found for patient_id={patient_id}")

        imgs = []
        devices = []
        positions = []

        label = None
        for row_idx in row_indices:
            row = self.df.iloc[row_idx]
            img, row_label, meta = self._row_to_example(row)
            imgs.append(img)
            devices.append(meta.get("device", ""))
            if "position" in meta:
                positions.append(meta["position"])
            if label is None:
                label = row_label
            elif label != row_label:
                raise ValueError(
                    f"Label mismatch within bag for patient_id={patient_id}"
                )

        bag = torch.stack(imgs, dim=0)  # [N, 3, H, W]
        meta = {
            "patient_id": str(patient_id),
            "n_recordings": int(len(imgs)),
            "device_set": ",".join(sorted(set(devices))),
        }
        if positions:
            meta["position_set"] = ",".join(sorted(set(positions)))
        return bag, int(label), meta


def bag_collate_fn(batch):
    """
    Collate variable-length bags into padded tensors.

    Returns:
        bags: [B, Nmax, 3, H, W]
        labels: [B]
        mask: [B, Nmax] (True where valid)
        metas: list of dicts
    """
    bags, labels, metas = zip(*batch)
    max_len = max(bag.shape[0] for bag in bags)
    batch_size = len(bags)
    c, h, w = bags[0].shape[1:]

    padded = bags[0].new_zeros((batch_size, max_len, c, h, w))
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    for i, bag in enumerate(bags):
        n = bag.shape[0]
        padded[i, :n] = bag
        mask[i, :n] = True

    labels = torch.tensor(labels, dtype=torch.float32)
    return padded, labels, mask, list(metas)
