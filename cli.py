import argparse
import models
import processing
import conv
import sys, os
import torch
from typing import List, Tuple

def merge_nasal_phonemes(
        segments: List[Tuple[float, float, str]]
        ) -> List[Tuple[float, float, str]]:
    """
    Merge adjacent segments like ('a', '~') into ('ã') using combining tilde.
    """
    nasal_merge_map = {
        'a': 'ã',
        'ɑ': 'ã',
        'e': 'ẽ',
        'o': 'õ',
        'ɔ': 'õ',
        'ɛ': 'œ̃',
        'œ': 'œ̃'
    }
    tilde = '\u0303'
    merged = []
    i = 0
    while i < len(segments):
        start, end, label = segments[i]
        label = label.strip()

        if i + 1 < len(segments):
            next_start, next_end, next_label = segments[i + 1]
            next_label = next_label.strip()

            if next_label == tilde and label in nasal_merge_map.keys():
                merged_label = nasal_merge_map[label]  # e.g., 'a' + '̃'
                merged.append([start, next_end, merged_label])
                i += 2
                continue

        merged.append((start, end, label))
        i += 1

    return merged


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
        ckpt_pth: str,
        out_pth: str,
        voice_detection = True
        ):
    audio_data = processing.load_audio(audio_pth)
    whole_speech_seg = [(0, len(audio_data.signal) / audio_data.fs, 1)]
    if voice_detection:
        active_seg = models.global_VAD(audio_pth)
    else:
        active_seg = whole_speech_seg
    # inference
    ctc_seg = models.global_ctc_phoneme_alignment(audio_data.signal, audio_data.fs, whole_speech_seg)
    # run phoneme boundary detection on active segments
    kreuk_seg = models.global_kreuk_seg(audio_data.signal, audio_data.fs, active_seg, ckpt_pth)
    # adjusting boundaries
    phon_seg = models.ctc_kreuk_heuristic_match(ctc_seg, kreuk_seg)
    # tweaks
    phon_seg = silence_long_phonemes(phon_seg)
    phon_seg = merge_repeated_phonemes(phon_seg)
    phon_seg = merge_nasal_phonemes(phon_seg)
    # save
    base_name = os.path.splitext(os.path.basename(audio_pth))[0]
    output_path = os.path.join(out_pth, f"{base_name}_seg.TextGrid")
    conv.seg_to_textgrid(phon_seg, output_path)
    return phon_seg



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CTC-Kreuk phoneme alignment pipeline.")
    parser.add_argument("--audio", required=True, help="Path to input audio file.")
    parser.add_argument("--ckpt", required=True, help="Path to Kreuk model checkpoint.")
    parser.add_argument("--out", required=True, help="Path to output TextGrid.")
    args = parser.parse_args()

    ctc_kreuk_pipeline(args.audio, args.ckpt, args.out)
