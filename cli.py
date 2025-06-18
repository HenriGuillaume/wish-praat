import argparse
import models
import processing
import conv
import sys, os
from typing import List, Tuple, Any

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


def extend_phonemes_to_vad_edges(
    phonemes: List[Tuple[float, float, str]],
    vad_segments: List[Tuple[float, float, Any]]
) -> List[Tuple[float, float, str]]:
    """
    Extend the first and last phonemes of each VAD segment to match the VAD boundaries,
    unless the phoneme already goes beyond the VAD edge.
    """
    extended = []
    i = 0  # phoneme index

    for vad_start, vad_end, _ in vad_segments:
        # collect phonemes within this VAD segment
        group = []
        while i < len(phonemes):
            p_start, p_end, label = phonemes[i]
            if p_end <= vad_start:
                i += 1
                continue
            if p_start >= vad_end:
                break
            group.append([p_start, p_end, label])
            i += 1

        if not group:
            continue  # no phonemes in this VAD segment

        # Extend first phoneme to vad_start if needed
        if group[0][0] > vad_start:
            group[0][0] = vad_start

        # Extend last phoneme to vad_end if needed
        if group[-1][1] < vad_end:
            group[-1][1] = vad_end

        extended.extend(group)

    return [tuple(p) for p in extended]


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


def limit_plosive_duration(
    segments: List[Tuple[float, float, str]],
    max_duration: float = 0.1,
    short_consonants: str = 'bdgkpt'
) -> List[Tuple[float, float, str]]:
    """
    Shortens overlong plosives to `max_duration` and distributes the excess
    duration equally to their immediate neighbors (if present).
    """
    segments = [list(seg) for seg in segments]  # make mutable
    n = len(segments)

    for i, (start, end, label) in enumerate(segments):
        duration = end - start
        if label in short_consonants and duration > max_duration:
            excess = duration - max_duration
            mid = (start + end) / 2
            new_start = mid - max_duration / 2
            new_end = mid + max_duration / 2

            # Adjust current plosive
            segments[i][0] = new_start
            segments[i][1] = new_end

            # Extend previous
            if i > 0:
                prev_start, prev_end, prev_label = segments[i - 1]
                if prev_end == start:  # safe to extend
                    segments[i - 1][1] = new_start

            # Extend next
            if i < n - 1:
                next_start, next_end, next_label = segments[i + 1]
                if next_start == end:  # safe to extend
                    segments[i + 1][0] = new_end

    return [tuple(seg) for seg in segments]


def ctc_kreuk_pipeline(
        audio_pth: str,
        ckpt_pth: str,
        out_pth: str,
        lang: str,
        voice_detection = True
        ):
    audio_data = processing.load_audio(audio_pth)
    whole_speech_seg = [(0, len(audio_data.signal) / audio_data.fs, 1)]
    if voice_detection:
        active_seg = models.global_VAD(audio_pth)
    else:
        active_seg = whole_speech_seg
    # inference
    #conv.seg_to_textgrid(active_seg, "../active.TextGrid")
    print('CTC...')
    ctc_seg = models.global_ctc_phoneme_alignment(audio_data.signal, audio_data.fs, 
                                                  active_seg, lang)
    #conv.seg_to_textgrid(ctc_seg, './ctc.TextGrid')
    # run phoneme boundary detection on active segments
    print('Refinement')
    phon_seg = models.kreuk_ctc_refinement(audio_data.signal, audio_data.fs,
                                           ctc_seg, ckpt_pth, active_seg)
    
    # tweaks
    phon_seg = silence_long_phonemes(phon_seg)
    phon_seg = merge_repeated_phonemes(phon_seg)
    phon_seg = merge_nasal_phonemes(phon_seg)
    phon_seg = extend_phonemes_to_vad_edges(phon_seg, active_seg)
    phon_seg = limit_plosive_duration(phon_seg)
    # save
    base_name = os.path.splitext(os.path.basename(audio_pth))[0]
    output_path = os.path.join(out_pth, f"{base_name}_seg.TextGrid")
    conv.seg_to_textgrid(phon_seg, output_path)
    return phon_seg



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CTC-Kreuk phoneme alignment pipeline.")
    parser.add_argument("--audio", required=True, help="Path to input audio file.")
    parser.add_argument("--lang", required=True, help="Path to input audio file.")
    parser.add_argument("--ckpt", required=True, help="Path to Kreuk model checkpoint.")
    parser.add_argument("--out", required=True, help="Path to output TextGrid.")
    args = parser.parse_args()

    ctc_kreuk_pipeline(args.audio, args.ckpt, args.out, args.lang)
