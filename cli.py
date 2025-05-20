import argparse
import models
import processing
import conv
import sys, os
import torch
from typing import List, Tuple


def silence_long_phonemes(
        segments: List[Tuple[float, float, str]],
        length_thresh = 1.5
        ):
    '''Mark phonemes longer than a threshold as silent'''
    for i, (start, end, phon) in enumerate(segments):
        if abs(start - end) > length_thresh:
            segments[i] = (start, end, '~')
    return segments


def merge_repeated_phonemes(
        segments: List[Tuple[float, float, str]],
        ):
    '''Merge adjacent phonemes that are the same'''
    i = 0
    while i < len(segments) - 1:
        start1, end1, phon1 = segments[i]
        start2, end2, phon2 = segments[i + 1]
        if phon1 == phon2:
            segments[i] = (start1, end2, phon1)
            del segments[i + 1]
        else:
            i += 1
    return segments


def ctc_kreuk_pipeline(
        audio_pth: str,
        ckpt_pth: str
        ):
    audio_data = processing.load_audio(audio_pth)
    # use whole segment for CTC, no VAD is required
    speech_seg = [(0, len(audio_data.signal) / audio_data.fs, 1)]
    # inference
    ctc_seg = models.global_ctc_phoneme_alignment(audio_data.signal, audio_data.fs, speech_seg)
    kreuk_preds = models.kreuk_phoneme_preds(audio_data.signal, audio_data.fs, ckpt_pth)
    kreuk_seg = models.kreuk_preds_to_seg(kreuk_preds)
    # adjusting boundaries
    phon_seg = models.ctc_kreuk_heuristic_match(ctc_seg, kreuk_seg)
    # tweaks
    phon_seg = silence_long_phonemes(phon_seg)
    phon_seg = merge_repeated_phonemes(phon_seg)
    # save
    conv.seg_to_textgrid(phon_seg, './outputs/kreuk_ajusted.TextGrid')
    return phon_seg



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CTC-Kreuk phoneme alignment pipeline.")
    parser.add_argument("--audio", required=True, help="Path to input audio file.")
    parser.add_argument("--ckpt", required=True, help="Path to Kreuk model checkpoint.")
    args = parser.parse_args()

    ctc_kreuk_pipeline(args.audio, args.ckpt)
