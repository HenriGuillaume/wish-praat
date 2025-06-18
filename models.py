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

CTC_MODELS = {
    'fr':"bofenghuang/phonemizer-wav2vec2-ctc-french",
    'en':"speechbrain/asr-wav2vec2-commonvoice-en",
    'nl':"Clementapa/wav2vec2-base-960h-phoneme-reco-dutch",
    'nl2':"GroNLP/wav2vec2-dutch-large-ft-cgn"
}

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
    segment_tensor: torch.Tensor,
    fs: int,
    model: Wav2Vec2ForCTC,
    processor: Wav2Vec2Processor,
    ) -> List[Tuple[float, float, str]]:
    """CTC-based frame-level phoneme transcription and alignment"""
    try:
        input_values = processor(segment_tensor, sampling_rate=fs, return_tensors="pt").input_values

        with torch.no_grad():
            logits = model(input_values).logits
        probs = torch.nn.functional.softmax(logits[0], dim=-1)
        pred_ids = torch.argmax(probs, dim=-1).cpu().numpy()

        time_per_step = segment_tensor.shape[-1] / fs / logits.shape[1]

        segments = []
        prev_id = processor.tokenizer.pad_token_id
        for i, idx in enumerate(pred_ids):
            if idx != prev_id:
                phon = processor.tokenizer.decode([idx]).strip()
                if phon:
                    seg_start = i * time_per_step
                    seg_end = seg_start + time_per_step
                    segments.append((seg_start, seg_end, phon))
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
    model_tag: str = "fr"
) -> List[str]:
    """Phoneme transcription for each segment (no alignment)"""
    model_name = CTC_MODELS[model_tag]
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
    model_tag: str = "fr"
) -> List[Tuple[float, float, str]]:
    """Frame-level phoneme alignment for each segment"""
    model_name = CTC_MODELS[model_tag]
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)

    results = []
    for start, end, _ in segment_list:
        # Slice *before* resampling
        segment = signal[int(start * fs): int(end * fs)]
        tensor = torch.tensor(segment, dtype=torch.float32)
        tensor = processing.reformat_tensor(tensor, fs, 16000)
        duration = tensor.shape[-1] / 16000.0
        aligned = local_ctc_phoneme_alignment(tensor, 16000, model, processor)
        aligned = [(start + s, start + e, p) for s, e, p in aligned]
        results.extend(aligned)
    return results



def global_VAD(
        wav_path: str,
        threshold: float = 0.05
) -> List[Tuple[float, float, int]]:
    """Global voice-activity detection using Silero VAD"""
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
    (get_speech_timestamps, _, read_audio, _, _) = utils

    wav = read_audio(wav_path)
    speech_timestamps = get_speech_timestamps(
      wav,
      model,
      threshold,
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


def kreuk_ctc_refinement(
        signal: np.ndarray,
        fs: int,
        ctc_segment_list: List[Tuple[float, float, Any]],
        ckpt: str,
        VAD_segments: List[Tuple[float, float, Any]] = None,
        inter_phoneme_thresh: float = 0.2
        ) -> List[Tuple[float, float, str]]:
    '''Refine CTC segmentation by applying Kreuk's model between
    segment pairs. If in voiced region or under inter_phoneme_thresh,
    adjust shared boundary to Kreuk-predicted one, else keep midpoint.'''

    refined = [list(ctc_segment_list[0])]  # start with first segment

    for i in range(len(ctc_segment_list) - 1):
        s1, e1, ph1 = refined[-1]
        s2, e2, ph2 = ctc_segment_list[i + 1]

        insert_boundary = False
        refined_boundary = (e1 + s2) / 2  # default

        # Check for voiced overlap
        if VAD_segments is not None:
            for vad_start, vad_end, _ in VAD_segments:
                if e1 >= vad_start and s2 <= vad_end:
                    insert_boundary = True
                    break

        # Or check time distance
        if not insert_boundary and (s2 - e1) < inter_phoneme_thresh:
            insert_boundary = True
        
        if insert_boundary:
            start_idx = int(e1 * fs)
            end_idx = int(s2 * fs)
            if s2 - e1 < 0.03:  # too short to process
                refined_boundary = (e1 + s2) / 2
            else:
                segment = signal[start_idx:end_idx]
                preds = kreuk_phoneme_preds(segment, fs, ckpt)
                boundaries = kreuk_preds_to_seg(preds, fs)
                if boundaries:
                    refined_boundary = e1 + boundaries[0][0]  # first boundary                                                                                                            

            # Update current and next segment boundaries
            refined[-1][1] = refined_boundary
            refined.append([refined_boundary, e2, ph2])
        else:
            refined.append([s2, e2, ph2])

    return [tuple(seg) for seg in refined]


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
