import numpy as np
import scipy as sp
from typing import List, Tuple, Any, Callable
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch, torchaudio
import whisperx
import processing
import logging
from sklearn.svm import SVC
# Felix KREUK
from argparse import Namespace
from utils import (detect_peaks, max_min_norm, replicate_first_k_frames)
from next_frame_classifier import NextFrameClassifier

# Vowel and consonant sets for statistical extension
VOWELS = 'aɑeɛɛəiœøoɔuyɑ̃ɛ̃œ̃ɔ̃'
CONSONANTS = 'bdfgklmnɲŋpʁsʃtvzʒjwɥ'
LONG_CONSONANTS = 'fmnsʃvzʒ'
SHORT_CONSONANTS = 'bdgkpt'


# === Conversion ===

def whisperx_dict_to_segment_list(whisperx_dict: dict) -> List[Tuple[float, float, str]]:
    """Convert WhisperX output dictionary to list of (start, end, text) tuples"""
    segments = []
    for segment in whisperx_dict.get("segments", []):
        for word in segment.get("words", []):
            segments.append((
                word.get("start", 0.0),
                word.get("end", 0.0),
                word.get("word", "")
            ))
    return segments

# === Local segment-level functions ===

def local_segment_whisperx_transcript(
    signal: np.ndarray,
    fs: int,
    start: float,
    end:float,
    model: Any,
    model_a: Any,
    metadata: Any,
    device: str,
    language: str = 'fr'
) -> List[dict]:
    """Forced-alignment word-level transcription using WhisperX"""
    segment = signal[int(start * fs): int(end * fs)]
    try:
        result = model.transcribe(segment, language="fr")
        aligned = whisperx.align(
            result["segments"], model_a, metadata, segment, device,
            return_char_alignments=False
        )
        return whisperx_dict_to_segment_list(aligned)
    except Exception as e:
        logging.warning(e)
        return []


def local_segment_phonemizer_transcript(
    signal_tensor: torch.Tensor,
    fs: int,
    start: float,
    end: float,
    model: Wav2Vec2ForCTC,
    processor: Wav2Vec2Processor,
) -> str:
    """Phoneme-level transcription (no temporal alignment)"""
    segment_tensor = signal_tensor[int(start * fs): int(end * fs)]
    model.eval()
    try:
        input_values = processor(segment_tensor, sampling_rate=fs, return_tensors="pt").input_values
        with torch.no_grad():
            logits = model(input_values).logits
        pred_ids = torch.argmax(logits, dim=-1)
        phonemes = processor.batch_decode(pred_ids)
        return phonemes[0]
    except Exception as e:
        logging.warning(e)
        return ""


def local_ctc_phoneme_alignment(
    signal_tensor: torch.Tensor,
    fs: int,
    start: float,
    end: float,
    model: Wav2Vec2ForCTC,
    processor: Wav2Vec2Processor,
) -> List[Tuple[float, float, str]]:
    """CTC-based frame-level phoneme transcription and alignment"""
    segment_tensor = signal_tensor[int(start * fs): int(end * fs)]
    try:
        # Process input for the model
        input_values = processor(segment_tensor, sampling_rate=fs, return_tensors="pt").input_values

        # Predict logits and apply softmax
        with torch.no_grad():
            logits = model(input_values).logits
        probs = torch.nn.functional.softmax(logits[0], dim=-1)
        pred_ids = torch.argmax(probs, dim=-1).cpu().numpy()

        # Compute time per step (frame duration)
        time_per_step = segment_tensor.shape[-1] / fs / logits.shape[1]

        # Decode and align
        segments = []
        prev_id = processor.tokenizer.pad_token_id
        for i, idx in enumerate(pred_ids):
            if idx != prev_id:
                phon = processor.tokenizer.decode([idx]).strip()
                if phon:
                    start = i * time_per_step
                    end = start + time_per_step
                    segments.append((start, end, phon))
            prev_id = idx

        return segments
    except Exception as e:
        logging.warning(e)
        return []


# === Global wrappers ===

def global_whisperx_transcript(
    signal: np.ndarray,
    fs: int,
    segment_list: List[Tuple[float, float, Any]],
    drool: float = 0.01,
    model_name: str = 'large',
    model_language: str = 'fr',
    device: str = 'cuda'
) -> List[List[dict]]:
    """
    Transcribe each segment with WhisperX forced alignment.
    Returns list of aligned segment lists per input segment.
    """
    CT = ['float32', 'float16'][int(device == 'cuda')]
    model = whisperx.load_model(model_name, device=device, compute_type=CT)
    model_a, metadata = whisperx.load_align_model(language_code=model_language, device=device)
    results = []
    for start, end, _ in segment_list:
        # extract numpy audio and convert
        aligned = local_segment_whisperx_transcript(signal, fs, start, end, model, model_a, metadata, device)
        results.extend(aligned)
    return results


def global_phonemizer_transcript(
    signal: np.ndarray,
    fs: int,
    segment_list: List[Tuple[float, float, Any]],
    model_name: str = "bofenghuang/phonemizer-wav2vec2-ctc-french"
) -> List[str]:
    """Phoneme transcription for each segment (no alignment)"""
    processor = Wav2Vec2Processor.from_pretrained(
            model_name
    )
    model = Wav2Vec2ForCTC.from_pretrained(
            model_name
    )
    results = []
    tensor = torch.tensor(signal, dtype=torch.float32)
    tensor = processing.reformat_tensor(tensor, fs, 16000)
    for start, end, _ in segment_list:
        tensor = torch.tensor(tensor, dtype=torch.float32)
        tensor = processing.reformat_tensor(tensor, fs, 16000)
        phon = local_segment_phonemizer_transcript(tensor, 16000, start, end, model, processor)
        results.append((start, end, phon))
    return results


def global_ctc_phoneme_alignment(
    signal: np.ndarray,
    fs: int,
    segment_list: List[Tuple[float, float, Any]],
    model_name: str = "bofenghuang/phonemizer-wav2vec2-ctc-french"
) -> List[Tuple[float, float, str]]:
    """Frame-level phoneme alignment for each segment"""
    processor = Wav2Vec2Processor.from_pretrained(
            model_name
    )
    model = Wav2Vec2ForCTC.from_pretrained(
            model_name
    )
    results = []
    # whole audio is tensorized, passed as a reference, local functions deal with the
    # slicing
    tensor = torch.tensor(signal, dtype=torch.float32)
    tensor = processing.reformat_tensor(tensor, fs, 16000)
    for start, end, _ in segment_list:
        aligned = local_ctc_phoneme_alignment(tensor, 16000, start, end, model, processor)
        results.extend(aligned)
    return results



def global_VAD(
        wav_path: str
) -> List[Tuple[float, float, int]]:
    """Global voice-activity detection using Silero VAD"""
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
    (get_speech_timestamps, _, read_audio, _, _) = utils

    wav = read_audio(wav_path)
    speech_timestamps = get_speech_timestamps(
      wav,
      model,
      return_seconds=True,  # Return speech timestamps in seconds (default is samples)
    ) 
    return [(s['start'], s['end'], 'True') for s in speech_timestamps]

def snap_close_boundaries(
    segments: List[Tuple[float, float, Any]],
    extrema_times: np.ndarray
) -> List[Tuple[float, float, Any]]:
    """Adjust close boundaries to the nearest extrema time"""
    out = [list(s) for s in segments]
    for i in range(len(out)-1):
        e1 = out[i][1]
        s2 = out[i+1][0]
        if abs(e1 - s2) < 0.15:
            cands = extrema_times[(extrema_times>=e1)&(extrema_times<=s2)]
            snap = cands.min() if cands.size else (e1+s2)/2
            out[i][1] = snap
            out[i+1][0] = snap
    return [tuple(s) for s in out]


# === Felix Kreuk tools ===

def kreuk_phoneme_preds(
        signal: np.ndarray,
        fs: int,
        ckpt: str
        ) -> torch.tensor:
    '''Uses felix KREUK's https://arxiv.org/pdf/2007.13465 model
    to detect precise phoneme boundaries'''
    #load model
    ckpt = torch.load(ckpt, map_location=lambda storage, loc: storage, weights_only=False)
    hp = Namespace(**dict(ckpt["hparams"]))

    model = NextFrameClassifier(hp)
    weights = ckpt["state_dict"]
    weights = {k.replace("NFC.", ""): v for k,v in weights.items()}
    model.load_state_dict(weights)
    # prepare input
    tensor = torch.tensor(signal, dtype=torch.float32)
    tensor = processing.reformat_tensor(tensor, fs, 16000)
    tensor = tensor.unsqueeze(0)
    # forward pass
    preds = model(tensor)[1][0]
    preds = replicate_first_k_frames(preds, k=1, dim=1)  # padding
    preds = 1 - max_min_norm(preds)
    return preds

def kreuk_preds_to_seg(
        preds: torch.tensor,
        fs: int = 16000,
        prominence: float = 0.025,
        distance: int = 5,
        width: float = 0.1
        ) -> List[Tuple[float, float, str]]:
    '''infer segments from the probabilities returned byt the model'''
    peaks = detect_peaks(x=preds,
                         lengths=[preds.shape[1]],
                         prominence=prominence,
                         width=width,
                         distance=distance)
    peaks = peaks[0] * 160 / fs  # transform frame indices to seconds
    phoneme_segments = []
    for i in range(len(peaks) - 1):
        phoneme_segments.append([peaks[i], peaks[i + 1], '~'])
    return phoneme_segments


def global_kreuk_seg(
        signal: np.ndarray,
        fs: int,
        segment_list: List[Tuple[float, float, Any]],
        ckpt: str
        ) -> List[Tuple[float, float, str]]:
    '''Detect phoneme boundaries within each given segment'''
    kreuk_seg = []
    for start, end, _ in segment_list:
        si, ei = int(start*fs), int(end*fs)
        local_kreuk_preds = kreuk_phoneme_preds(signal[si:ei], fs, ckpt)
        local_kreuk_segs = kreuk_preds_to_seg(local_kreuk_preds)
        for s in local_kreuk_segs:
            s[0] += start
            s[1] += start
        kreuk_seg.extend(local_kreuk_segs)
    return kreuk_seg


def ctc_kreuk_heuristic_match(
        ctc_seg: List[Tuple[float, float, str]],
        kreug_seg: List[Tuple[float, float, str]]
        ) -> List[Tuple[float, float, str]]:
    '''Matches a ctc segmentation to boundaries infered by kreuk's model
    using simple heuritics'''
    new_phon_seg = [list(s) for s in ctc_seg]
    # build a dicitonnary of intersecting segments to make search faster
    intersection_dict = dict()
    for k_start, k_end, _ in kreug_seg:
        # find the ctc phonemes that intersect the kreuk segment
        candidates_idx = [
            i for i, (c_start, c_end, _) in enumerate(new_phon_seg)
            if k_start < c_start < k_end or
            k_start < c_end < k_end  # true intersection
        ]
        intersection_dict[(k_start, k_end)] = candidates_idx
    # apply heuristics
    # if a kreuk segment only has one candidate, fill it
    for k_start, k_end in intersection_dict.keys():
        candidates_idx = intersection_dict[(k_start, k_end)]
        if len(candidates_idx) == 1:
            idx = candidates_idx[0]
            new_phon_seg[idx][0] = k_start
            new_phon_seg[idx][1] = k_end
    # maximally extend first and last phoneme to kreuk edges
    for k_start, k_end in intersection_dict.keys():
        candidates_idx = intersection_dict[(k_start, k_end)]
        # keep in mind intersections have changed now
        # we find the segments that are fully inside
        for idx in candidates_idx:
            p_start, p_end, _ = new_phon_seg[idx]
            if p_start < k_start or p_end > k_end:
                candidates_idx.remove(idx)
        if len(candidates_idx) == 0:
            continue
        f_idx, l_idx = candidates_idx[0], candidates_idx[-1]
        new_phon_seg[f_idx][0] = k_start
        new_phon_seg[l_idx][1] = k_end
        # resolve conflicts between ctc segments sharing a kreuk seg
        # could make something fancier, might do it later
        if len(candidates_idx) >= 2:
            for i in range(len(candidates_idx) - 1):
                s1, e1, p1 = new_phon_seg[candidates_idx[i]]
                s2, e2, p2 = new_phon_seg[candidates_idx[i+1]]
                if p1 in SHORT_CONSONANTS:
                    s2 = e1
                elif p2 in SHORT_CONSONANTS:
                    e1 = s2
                else:
                    midpoint = (e1 + s2) / 2
                    e1 = midpoint
                    s2 = midpoint
                new_phon_seg[candidates_idx[i]] = [s1, e1, p1]
                new_phon_seg[candidates_idx[i+1]] = [s2, e2, p2]
    return [tuple(p) for p in new_phon_seg]
