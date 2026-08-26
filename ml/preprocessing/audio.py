"""
PRISM Audio Preprocessing Pipeline
==================================
Handles format conversion, sample rate normalization (16kHz mono),
loudness normalization, silence trimming, Voice Activity Detection (VAD),
and audio quality scoring.
"""

import io
import math
import os
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
from scipy import signal


@dataclass
class AudioMetadata:
    duration_sec: float
    sample_rate: int
    channels: int
    snr_db: float
    quality_score: float
    is_valid_speech: bool
    num_samples: int


@dataclass
class SpeechSegment:
    segment_id: int
    start_sec: float
    end_sec: float
    audio: np.ndarray
    duration_sec: float


class AudioPreprocessor:
    """Production-grade audio preprocessor for Indian-language ASR."""

    def __init__(
        self,
        target_sample_rate: int = 16000,
        target_channels: int = 1,
        target_dbfs: float = -20.0,
        silence_threshold_db: float = -40.0,
        min_speech_duration_sec: float = 0.3,
        max_audio_duration_sec: float = 600.0,
        frame_duration_ms: int = 30,
        energy_threshold: float = 0.015,
    ):
        self.target_sample_rate = target_sample_rate
        self.target_channels = target_channels
        self.target_dbfs = target_dbfs
        self.silence_threshold_db = silence_threshold_db
        self.min_speech_duration_sec = min_speech_duration_sec
        self.max_audio_duration_sec = max_audio_duration_sec
        self.frame_duration_ms = frame_duration_ms
        self.energy_threshold = energy_threshold

    def load_audio(
        self,
        audio_input: Union[str, bytes, io.BytesIO, np.ndarray],
        original_sr: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """Loads audio from various input formats and returns (audio_array, sample_rate)."""
        if isinstance(audio_input, np.ndarray):
            sr = original_sr or self.target_sample_rate
            audio = audio_input.astype(np.float32)
            if audio.ndim > 1 and audio.shape[1] > 1:
                audio = np.mean(audio, axis=1)
            return audio, sr

        if isinstance(audio_input, str):
            if not os.path.exists(audio_input):
                raise FileNotFoundError(f"Audio file not found: {audio_input}")
            audio, sr = sf.read(audio_input, dtype="float32")
            if audio.ndim > 1 and audio.shape[1] > 1:
                audio = np.mean(audio, axis=1)
            return audio, sr

        if isinstance(audio_input, bytes):
            buffer = io.BytesIO(audio_input)
            audio, sr = sf.read(buffer, dtype="float32")
            if audio.ndim > 1 and audio.shape[1] > 1:
                audio = np.mean(audio, axis=1)
            return audio, sr

        if isinstance(audio_input, io.BytesIO):
            audio_input.seek(0)
            audio, sr = sf.read(audio_input, dtype="float32")
            if audio.ndim > 1 and audio.shape[1] > 1:
                audio = np.mean(audio, axis=1)
            return audio, sr

        raise ValueError(f"Unsupported audio input type: {type(audio_input)}")

    def resample(self, audio: np.ndarray, orig_sr: int) -> np.ndarray:
        """Resamples audio to target sample rate using high-quality polyphase filtering."""
        if orig_sr == self.target_sample_rate:
            return audio

        num_target_samples = int(len(audio) * float(self.target_sample_rate) / orig_sr)
        if num_target_samples == 0:
            return np.zeros(0, dtype=np.float32)

        resampled = signal.resample(audio, num_target_samples).astype(np.float32)
        return resampled

    def normalize_loudness(self, audio: np.ndarray) -> np.ndarray:
        """Normalizes audio loudness to target dBFS without aggressive clipping."""
        if len(audio) == 0:
            return audio

        rms = np.sqrt(np.mean(np.square(audio)) + 1e-12)
        current_dbfs = 20.0 * np.log10(rms + 1e-12)

        gain_db = self.target_dbfs - current_dbfs
        # Cap maximum gain to avoid amplifying background noise excessively
        gain_db = np.clip(gain_db, -20.0, 18.0)
        gain_linear = 10.0 ** (gain_db / 20.0)

        normalized = audio * gain_linear
        # Peak normalization safety limiter
        peak = np.max(np.abs(normalized)) + 1e-12
        if peak > 0.95:
            normalized = normalized * (0.95 / peak)

        return normalized.astype(np.float32)

    def trim_silence(self, audio: np.ndarray, threshold_db: Optional[float] = None) -> np.ndarray:
        """Removes leading and trailing silence based on energy threshold."""
        if len(audio) == 0:
            return audio

        thresh = threshold_db or self.silence_threshold_db
        thresh_linear = 10.0 ** (thresh / 20.0)

        frame_len = int(self.target_sample_rate * 0.02)  # 20ms frames
        if len(audio) < frame_len * 2:
            return audio

        num_frames = len(audio) // frame_len
        frame_energies = [
            np.sqrt(np.mean(np.square(audio[i * frame_len : (i + 1) * frame_len])))
            for i in range(num_frames)
        ]

        active_indices = [i for i, e in enumerate(frame_energies) if e > thresh_linear]
        if not active_indices:
            return audio

        start_idx = max(0, (active_indices[0] - 1) * frame_len)
        end_idx = min(len(audio), (active_indices[-1] + 2) * frame_len)

        return audio[start_idx:end_idx]

    def compute_snr_and_quality(self, audio: np.ndarray) -> Tuple[float, float]:
        """Estimates Signal-to-Noise Ratio (SNR) and speech quality score (0.0 - 1.0)."""
        if len(audio) == 0:
            return 0.0, 0.0

        frame_len = int(self.target_sample_rate * 0.03)  # 30ms frames
        if len(audio) < frame_len:
            return 10.0, 0.5

        num_frames = len(audio) // frame_len
        frame_powers = [
            np.mean(np.square(audio[i * frame_len : (i + 1) * frame_len]))
            for i in range(num_frames)
        ]

        if not frame_powers:
            return 0.0, 0.0

        sorted_powers = sorted(frame_powers)
        # Noise floor estimated from bottom 15% energy frames
        noise_cutoff = max(1, int(0.15 * len(sorted_powers)))
        noise_power = np.mean(sorted_powers[:noise_cutoff]) + 1e-12

        # Signal power estimated from top 40% energy frames
        signal_cutoff = max(1, int(0.60 * len(sorted_powers)))
        signal_power = np.mean(sorted_powers[signal_cutoff:]) + 1e-12

        snr = 10.0 * np.log10(signal_power / noise_power)
        snr = float(np.clip(snr, 0.0, 45.0))

        # Quality score scaled from 0.0 to 1.0 based on SNR, dynamic range, and clipping
        clipping_ratio = np.sum(np.abs(audio) >= 0.98) / len(audio)
        quality = (snr / 40.0) * (1.0 - min(clipping_ratio * 5.0, 0.5))
        quality = float(np.clip(quality, 0.1, 1.0))

        return snr, quality

    def detect_voice_activity(
        self, audio: np.ndarray, sr: int = 16000
    ) -> List[Tuple[float, float, bool]]:
        """
        Energy and Zero-Crossing based Voice Activity Detection (VAD).
        Returns a list of tuples: (start_time_sec, end_time_sec, is_speech).
        """
        frame_size = int(sr * (self.frame_duration_ms / 1000.0))
        num_frames = len(audio) // frame_size

        if num_frames == 0:
            return [(0.0, len(audio) / sr, True)]

        speech_flags = []
        for i in range(num_frames):
            frame = audio[i * frame_size : (i + 1) * frame_size]
            energy = np.sqrt(np.mean(np.square(frame)))
            zero_crossings = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * len(frame))

            # Speech heuristic: significant energy with natural zero-crossing characteristics
            is_speech = energy > self.energy_threshold and zero_crossings < 0.45
            speech_flags.append(is_speech)

        # Smooth flags using 3-frame median filter
        smoothed_flags = []
        for i in range(len(speech_flags)):
            window = speech_flags[max(0, i - 1) : min(len(speech_flags), i + 2)]
            smoothed_flags.append(sum(window) >= (len(window) / 2))

        # Segment into continuous intervals
        segments = []
        if not smoothed_flags:
            return [(0.0, len(audio) / sr, True)]

        current_state = smoothed_flags[0]
        start_frame = 0

        for i, state in enumerate(smoothed_flags):
            if state != current_state:
                segments.append(
                    (
                        start_frame * self.frame_duration_ms / 1000.0,
                        i * self.frame_duration_ms / 1000.0,
                        current_state,
                    )
                )
                start_frame = i
                current_state = state

        segments.append(
            (
                start_frame * self.frame_duration_ms / 1000.0,
                num_frames * self.frame_duration_ms / 1000.0,
                current_state,
            )
        )

        return segments

    def segment_speech(
        self, audio: np.ndarray, sr: int = 16000, max_segment_duration: float = 12.0
    ) -> List[SpeechSegment]:
        """Segments long audio into continuous speech chunks for robust ASR inference."""
        vad_intervals = self.detect_voice_activity(audio, sr)
        speech_intervals = [
            (start, end)
            for (start, end, is_speech) in vad_intervals
            if is_speech and (end - start) >= self.min_speech_duration_sec
        ]

        if not speech_intervals:
            # Fallback if no specific VAD boundary found
            duration = len(audio) / sr
            return [
                SpeechSegment(
                    segment_id=0,
                    start_sec=0.0,
                    end_sec=duration,
                    audio=audio,
                    duration_sec=duration,
                )
            ]

        # Merge adjacent speech intervals that have brief silence (< 0.5s)
        merged = []
        cur_start, cur_end = speech_intervals[0]

        for n_start, n_end in speech_intervals[1:]:
            if n_start - cur_end < 0.5 and (n_end - cur_start) <= max_segment_duration:
                cur_end = n_end
            else:
                merged.append((cur_start, cur_end))
                cur_start, cur_end = n_start, n_end
        merged.append((cur_start, cur_end))

        result = []
        for idx, (s, e) in enumerate(merged):
            s_idx = int(s * sr)
            e_idx = int(e * sr)
            seg_audio = audio[s_idx:e_idx]
            result.append(
                SpeechSegment(
                    segment_id=idx,
                    start_sec=round(s, 3),
                    end_sec=round(e, 3),
                    audio=seg_audio,
                    duration_sec=round(e - s, 3),
                )
            )

        return result

    def process(
        self,
        audio_input: Union[str, bytes, io.BytesIO, np.ndarray],
        original_sr: Optional[int] = None,
    ) -> Tuple[np.ndarray, AudioMetadata]:
        """
        Complete audio processing pipeline:
        Load -> Resample (16kHz) -> Mono -> Loudness Normalization -> Silence Trim -> Validation.
        """
        raw_audio, orig_sr = self.load_audio(audio_input, original_sr)
        resampled = self.resample(raw_audio, orig_sr)
        normalized = self.normalize_loudness(resampled)
        trimmed = self.trim_silence(normalized)

        if len(trimmed) == 0:
            trimmed = normalized

        duration = len(trimmed) / self.target_sample_rate
        snr, quality = self.compute_snr_and_quality(trimmed)

        metadata = AudioMetadata(
            duration_sec=round(duration, 3),
            sample_rate=self.target_sample_rate,
            channels=self.target_channels,
            snr_db=round(snr, 2),
            quality_score=round(quality, 3),
            is_valid_speech=duration >= self.min_speech_duration_sec,
            num_samples=len(trimmed),
        )

        return trimmed, metadata

    @staticmethod
    def save_temp_wav(audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Saves audio array to an isolated temporary WAV file for inference."""
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(temp_file.name, audio, sample_rate, format="WAV", subtype="PCM_16")
        temp_file.close()
        return temp_file.name
