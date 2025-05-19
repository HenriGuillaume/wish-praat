from typing import List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
import numpy as np
import scipy as sp
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch, torchaudio
import whisperx
import os

VOWELS = 'aɑeɛɛəiœøoɔuyɑ̃ɛ̃œ̃ɔ̃'
CONSONANTS = 'bdfgklmnɲŋpʁsʃtvzʒjwɥ'
LONG_CONSONANTS = 'fmnsʃvzʒ'

@dataclass
class AudioData:
    signal: np.ndarray
    fs: int
    n_fft: int = 32
    hop_length: int = 16
    # most models play well with 16K Hz audio
    stft: tuple = field(init=False)
    signal_energy_time: np.ndarray = field(init=False)
    signal_energy: np.ndarray = field(init=False)

    def __post_init__(self):
        # we work with normalized signals
        self.convert_to_mono()
        self.normalize()
        f, t, Zxx = sp.signal.stft(self.signal, fs=self.fs, window='hann',
                         nperseg=self.n_fft)
        self.stft = (f, t, Zxx)

    def convert_to_mono(self):
        if self.signal.ndim > 1:
            self.signal = np.mean(self.signal, axis=1)
    
    def normalize(self):
        self.signal /= (np.max(np.abs(self.signal)) + 1e-10)


def load_audio(filepath: str) -> AudioData:
    fs, signal = sp.io.wavfile.read(filepath)
    signal = signal.astype(np.float32)
    return AudioData(signal=signal, fs=fs)


def reformat_tensor(signal_tensor: torch.Tensor, fs: int, new_fs: int = 16000) -> torch.Tensor:
    '''
    Preprocessing necessary to make a tensor suitable for the application of most models,
    our standard here is mono, 16K Hz, float32 normalized tensors.
    '''
    # ensure float32 format
    signal_tensor = signal_tensor.clone().detach().to(torch.float32)
    # convert to mono
    if signal_tensor.ndim > 1:
        signal_tensor = signal_tensor.mean(dim=1)
    # ensure normalization
    signal_tensor = torch.nn.functional.normalize(signal_tensor, dim=0)
    signal_tensor = torch.clamp(signal_tensor, -1.0, 1.0)
    signal_tensor = signal_tensor.unsqueeze(0)
    # resample
    resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=new_fs)

    signal_tensor = resampler(signal_tensor)
    return signal_tensor.squeeze(0)

def sanitize_tensor(signal_tensor: torch.Tensor | np.ndarray, fs:int):
    if type(signal_tensor) != torch.Tensor:
        signal_tensor = torch.Tensor(signal_tensor)
    if fs != 16000 or signal_tensor.ndim != 1:
        return reformat_tensor(signal_tensor, fs)


def smooth_energy(energy: np.ndarray, conv_window_size: int = 10) -> np.ndarray:
    return np.convolve(energy, np.ones(conv_window_size)/conv_window_size, mode='same')


def rms(signal: np.ndarray, frame_size: int = 2048, hop_size: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    num_frames = 1 + (len(signal) - frame_size) // hop_size
    energy = np.sqrt(np.array([
        np.mean(signal[i * hop_size:i * hop_size + frame_size] ** 2)
        for i in range(num_frames)
    ]))
    energy_time =  np.array([
        (i * hop_size) / sample_rate
        for i in range(num_frames)
    ])
    return (energy_time, energy)


def get_local_maxima(energy: np.ndarray) -> np.ndarray:
    '''
    This function is built separately (not in init) because is it advised to smooth the
    signal before finding extrema in it's RMS curve.
    '''
    return sp.signal.argrelextrema(self.energy)[0]


def global_energy_VAD(signal: np.ndarray, fs: int = 16000):
    intervals = librosa.effects.split(signal)
    segments = [(start / fs, end / fs, 1) for start, end in intervals]
    return 
