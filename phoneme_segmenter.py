import numpy as np
import scipy as sp
from scipy.signal import stft
import pandas as pd
from matplotlib import pyplot as plt
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import torchaudio
import whisperx # whisper alignment is terrible, see (https://github.com/m-bain/whisperX)
import csv
from textgrid import TextGrid, IntervalTier
from sklearn.cluster import KMeans, AgglomerativeClustering
import difflib


AUDIO_PTH = './mfa/audio/'
TRANSCRIPT_PTH = './mfa/transcriptions/'
OUT_PTH = './mfa/out/'
DEVICE='cpu'

def csv_to_textgrid(csv_path, tg_path):
    tg = TextGrid(minTime=0)
    tier = IntervalTier(name="Labels", minTime=0)

    with open(csv_path, 'r') as f:
        for line in f:
            start, end, label = line.strip().split(',')
            tier.add(float(start), float(end), label)

    tg.append(tier)
    tg.write(tg_path)


class Audio:
    def __init__(self, filename, word_seg_path=None, syll_seg_path=None):
        self.filename = filename
        self.audio_pth = AUDIO_PTH + filename

        self.fs, self.signal = sp.io.wavfile.read(self.audio_pth)
        self.signal = self.signal.astype(np.float32)
        self.signal /= np.max(np.abs(self.signal)) + 1e-10
        self.nperseg, self.hop_size = 256, 128
        self.f, self.t, self.Zxx = stft(self.signal, fs=self.fs, nperseg=self.nperseg)

        self.energy_time, self.energy, self.energy_grad = self.get_energy()
        self.energy_thresh = 0.1
        self.local_minima, self.local_minima_times = None, None
        self.silence_segments = self.get_silence()

        self.whole_text = None

        self.whspr_dct = None
        self.word_seg = None # words stored as words (i.e. 'bonjour')
        self.phon_word_seg = None # words stored as syll separated phonemes (bɔ̃.ʒuʁ)
        self.syll_seg = None
        self.phon_seg = None
        
        if syll_seg_path != None:
            self.syll_seg = self.csv_to_seg(syll_seg_path)

    # CONVERSION
    def seg_to_csv(self, seg, csv_path):
        with open(csv_path, mode="w", newline='') as file:
            writer = csv.writer(file)
            for interval in seg:
                writer.writerow(list(interval))

    def csv_to_seg(self, csv_path):
        with open(csv_path, newline='') as f:
            reader = csv.reader(f)
            seg = [
            (float(start), float(end), label) 
            for start, end, label in reader
            ]
            return seg

    def speech_to_whspr_dct(self):
        device = DEVICE
        CT = ['float32', 'float16'][int(DEVICE == 'cuda')]
        model = whisperx.load_model("tiny", device=device, compute_type=CT)
        audio = whisperx.load_audio(self.audio_pth)
        result = model.transcribe(audio, language="fr")
        model_a, metadata = whisperx.load_align_model(language_code='fr', device=device)
        return whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    def local_seg_transcript(self, segment_list, drool=0.01):
        '''
        Gives a local phonetic transcription of each segment in the segment list
        This doesn't change the segment list you pass, it returns the modified version
        '''
        new_segments = []

        # Load processor and model from Hugging Face
        processor = Wav2Vec2Processor.from_pretrained("bofenghuang/phonemizer-wav2vec2-ctc-french")
        model = Wav2Vec2ForCTC.from_pretrained("bofenghuang/phonemizer-wav2vec2-ctc-french")
        
        # Set model to evaluation mode
        model.eval()
        
        max_time = segment_list[-1][1]
        for s in segment_list:
            seg_start, seg_end = s[0], s[1]
            # drool a little on the sides
            large_seg_start = max(seg_start-drool, 0)
            large_seg_end = min(seg_end + drool, max_time)
            
            # Slice segment using original fs
            start_idx = int(large_seg_start * self.fs)
            end_idx = int(large_seg_end * self.fs)
            segment = self.signal[start_idx:end_idx]

            # Convert to torch tensor
            segment_tensor = torch.tensor(segment, dtype=torch.float32)
            
            # Convert to mono if stereo
            if segment_tensor.ndim > 1:
                segment_tensor = segment_tensor.mean(dim=0)
            
            # Add batch dimension
            segment_tensor = segment_tensor.unsqueeze(0)
            try:
                resampler = torchaudio.transforms.Resample(orig_freq=self.fs, new_freq=16000)
                segment_tensor = resampler(segment_tensor)
                
                # Remove batch dimension (processor expects 1D audio tensor)
                segment_tensor = segment_tensor.squeeze(0)
                
                # Pass to processor
                input_values = processor(segment_tensor, sampling_rate=16000, return_tensors="pt").input_values


                with torch.no_grad():
                    logits = model(input_values).logits

                # Decode phonemes using CTC decoding
                predicted_ids = torch.argmax(logits, dim=-1)
                phonemes = processor.batch_decode(predicted_ids)
                
                new_segments.append((seg_start, seg_end, phonemes[0]))
            except:
                print("Couldnt label")

        return new_segments

    def ctc_align_phonemes_per_segment(self, segment_list, drool=0.01):
        """
        Performs frame-level CTC alignment for phonemes within each syllable or word segment.
        Returns a list of (start_time, end_time, phoneme) using CTC timestamps.
        """
        processor = Wav2Vec2Processor.from_pretrained("bofenghuang/phonemizer-wav2vec2-ctc-french")
        model = Wav2Vec2ForCTC.from_pretrained("bofenghuang/phonemizer-wav2vec2-ctc-french")
        model.eval()

        phoneme_segments = []
        max_time = segment_list[-1][1]

        for s in segment_list:
            seg_start, seg_end = s[0], s[1]
            large_start = max(seg_start - drool, 0)
            large_end = min(seg_end + drool, max_time)

            start_idx = int(large_start * self.fs)
            end_idx = int(large_end * self.fs)
            segment = self.signal[start_idx:end_idx]

            # To tensor and resample
            segment_tensor = torch.tensor(segment, dtype=torch.float32)
            if segment_tensor.ndim > 1:
                segment_tensor = segment_tensor.mean(dim=0)
            segment_tensor = segment_tensor.unsqueeze(0)

            try:
                resampler = torchaudio.transforms.Resample(orig_freq=self.fs, new_freq=16000)
                segment_tensor = resampler(segment_tensor).squeeze(0)

                input_values = processor(segment_tensor, sampling_rate=16000, return_tensors="pt").input_values
                with torch.no_grad():
                    logits = model(input_values).logits  # (1, T, V)

                # Decode timestamps
                probs = torch.nn.functional.softmax(logits[0], dim=-1)
                pred_ids = torch.argmax(probs, dim=-1).cpu().numpy()
                time_per_step = segment_tensor.shape[-1] / 16000 / logits.shape[1]

                # Collapse repeated CTC tokens and remove blanks (ID 0 for CTC blank)
                previous = -1
                for i, idx in enumerate(pred_ids):
                    if idx != previous and idx != processor.tokenizer.pad_token_id:
                        phoneme = processor.tokenizer.decode([idx]).strip()
                        if phoneme:
                            t_start = large_start + i * time_per_step
                            t_end = t_start + time_per_step
                            phoneme_segments.append((t_start, t_end, phoneme))
                    previous = idx

            except Exception as e:
                print(f"Could not process segment {s}: {e}")

        self.phon_seg = phoneme_segments
        return phoneme_segments

    
    def merge_phonemes(self):
        '''
        If two segments are close and correspond to the same phoneme, merge them
        '''
        if self.whole_text is None:
            self.whole_text_phonemizer()
        # merge adjacent repeated phonemes
        repeated_indices = []
        resolved = []
        for i, s in enumerate(self.phon_seg):
            start1, end1, phon1 = s
            start2, end2, phon2 = self.phon_seg[i+1]
            if phon1 == phon2 and (abs(start2 - end1) < 0.1 or start2 < end1):
                # merge segments
                repeated_indices.append(i+1)
                end1 = end2
            if i not in repeated_indices:
                resolved.append((start1, end1, phon1))
        self.phon_seg = resolved
        return resolved

    def fix_transcription(self, word_phon, candidates):
        '''
        Only used by adjust_phonetic_word_seg, given a local word transcription from phonemizer,
        we only allow phoneme deletions and additions on the edges of the phoneme (liaisons).
        Modifications are fixed with the transcription candidates that are pulled from a lexicon. 
        '''
        # only keep close matches
        word_lst = word_phon.split(' ')
        longest = max(word_lst, key=len)
        longest_idx = word_lst.index(longest)
        if len(word_phon) != 0:
            candidates = difflib.get_close_matches(longest, candidates)
        if not candidates:
            return word_phon
        # repair changes in the word
        matcher = difflib.SequenceMatcher(None, candidates[0], longest)
        # change modifications
        repaired_word = list(longest)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            # If there's a modification ('replace'), revert it to the original version
            if tag == 'replace':
                # Revert the changed portion of the string back to the original
                repaired_word[j1:j2] = candidates[0][i1:i2]

        word_lst[longest_idx] = ''.join(repaired_word)
        return ''.join(word_lst)

    def adjust_phonetic_word_seg(self, lexicon_pth='ult.csv'):
        '''
        Given the transcription via phonemizer, and the whisperx transcription, we check that
        the phonetic transcription is coherent with what we find from a lexicon
        '''
        if self.word_seg is None or self.phon_word_seg is None:
            print('This feature requires both whisper transcription and local phonemizing')
            return
        adjusted = []
        df = pd.read_csv(lexicon_pth, sep=',', header=0)
        words = df.iloc[:, 0].values
        phons = df.iloc[:, 1].values
        for i, seg in enumerate(self.word_seg):
            start, end, word = seg
            idx = np.searchsorted(words, word, side='left')
            # same word can have several transcriptions
            results = []
            while idx < len(words) and words[idx] == word:
                results.append(phons[idx].replace('.', ''))
                idx += 1
            # now find the most likely utterance based on the pronounced word
            word_phon = self.phon_word_seg[i][2]
            new_word_phon = self.fix_transcription(word_phon, results)
            print(f'word:{word} / audio transcription: {word_phon} / repaired: {new_word_phon}')
            adjusted.append((start, end, new_word_phon))
        self.phon_word_seg = adjusted
        return adjusted


    def whole_text_phonemizer(self):
        processor = Wav2Vec2Processor.from_pretrained("bofenghuang/phonemizer-wav2vec2-ctc-french")
        model = Wav2Vec2ForCTC.from_pretrained("bofenghuang/phonemizer-wav2vec2-ctc-french")
        
        # Set model to evaluation mode
        model.eval()
        audio_tensor = torch.tensor(self.signal, dtype=torch.float32)
        if audio_tensor.ndim > 1:
            audio_tensor = audio_tensor.mean(dim=0)
            audio_tensor = audio_tensor.unsqueeze(0)

        try:
            resampler = torchaudio.transforms.Resample(orig_freq=self.fs, new_freq=16000)
            audio_tensor = resampler(segment_tensor).squeeze(0)
        except:
            print('Could not resample, this may affect transcription quality')

        input_values = processor(audio_tensor, sampling_rate=16000, return_tensors="pt").input_values


        with torch.no_grad():
            logits = model(input_values).logits

        # Decode phonemes using CTC decoding
        predicted_ids = torch.argmax(logits, dim=-1)
        phonemes = processor.batch_decode(predicted_ids)
        self.whole_text = phonemes
        return phonemes

    def lexicon_text_to_phonetic_transcript(self, lexicon_pth):
        '''
        Using the whisperX transcript, looks up words in a phonetic lexicon (csv format), returns 
        the word segments, this provides no phoneme-level alignment and may not match the actual 
        pronunciation of each word.
        '''
        pass


    def whspr_dct_to_word_seg(self):
        word_seg = []
        for segment in self.whspr_dct["segments"]:
            for word in segment["words"]:
                word_start = word['start']
                word_end = word['end']
                txt = word['word'].lower().strip("~-.,!?…'’ ")
                word_seg.append((word_start, word_end, txt))
        self.word_seg = word_seg
        return word_seg

    # PROCESSING

    def get_energy(self):
        spectrogram_db = 20 * np.log10(np.abs(self.Zxx) + 1e-10)
        energy = []
        for i in range(0, len(self.signal) - self.nperseg, self.hop_size):
            frame = self.signal[i:i + self.nperseg]
            rms = np.sqrt(np.mean(frame ** 2))
            energy.append(rms)
        energy = np.array(energy)
        energy_time = np.arange(len(energy)) * self.hop_size / self.fs
        energy_grad = np.gradient(energy, self.hop_size / self.fs)
        self.energy_time, self.energy, self.energy_grad = energy_time, energy, energy_grad
        return energy_time, energy, energy_grad

    def smooth_energy(self, conv_window_size=10):
        smoothed_energy = np.convolve(self.energy, 
                                      np.ones(conv_window_size)/conv_window_size, mode='same')
        self.energy = smoothed_energy
        return smoothed_energy

    def get_local_minima(self):
        #energy_grad_grad = np.gradient(self.energy_grad, self.hop_size / self.fs)
        #local_minima = np.where((np.abs(self.energy_grad < 0.01)) & (energy_grad_grad > 0))[0]
        local_minima = sp.signal.argrelmin(self.energy)[0]
        local_minima_times = self.energy_time[local_minima]
        self.local_minima = local_minima
        self.local_minima_times = local_minima_times
        return local_minima, local_minima_times


    def get_silence(self):
        '''
        To do, try to plot the energy distribution, should get a small peak corresponding to
        silences, deduce threshold accordingly
        '''
        self.silence_segments = []
        self.get_local_minima()
        
        min_spacing = int(0.1 * self.fs / self.hop_size)
        silence_segments = []
        if len(self.local_minima) == 0:
            return silence_segments
        local_minima = self.local_minima
        current_segment = [local_minima[0], local_minima[0]]
        for m in local_minima[1:]:
            if m - current_segment[-1] <= min_spacing:
                current_segment[1] = m
            else:
                silence_segments.append(current_segment.copy())
                current_segment = [m, m]
        silence_segments.append(current_segment)
        return [(self.energy_time[start], self.energy_time[end]) for start, end in silence_segments]

    def snap_close_boundaries_to_minimum(self, segment_list):
        """
        For word boundary pairs closer than threshold seconds,
        snap both to the nearest local energy minimum.
        Does not modify the segments in place
        """
        # Compute local minima
        minima_indices = self.local_minima
        minima_times = self.local_minima_times

        updated_segments = [list(seg) for seg in segment_list]

        for i in range(len(segment_list) - 1):
            start1, end1, label1 = segment_list[i]
            start2, end2, label2 = segment_list[i + 1]
            # find syllables that are close
            if abs(end1 - start2) < 0.15:
                # Snap to nearest minimum in between
                in_range = (minima_times >= end1) & (minima_times <= start2)
                if np.any(in_range):
                    indices_in_range = minima_indices[in_range]
                    min_idx = indices_in_range[np.argmin(self.energy[indices_in_range])]
                    snap_time = self.energy_time[min_idx]
                else:
                    mid = (end1 + start2) / 2
                    snap_time = mid
                updated_segments[i][1] = snap_time
                updated_segments[i+1][0] = snap_time

        return updated_segments


def main():
    print('In main')
    # load audio and CSV
    filename = "ddh2.wav"
    audio_obj = Audio(filename)
    audio_obj.syll_seg = audio_obj.csv_to_seg('/home/bigh/Documents/prog/stage/segmentation/mfa/transcriptions/sylber_ddh.csv')
    # clean the syllable boundaries and audio energy
    for i in range(3):
        audio_obj.smooth_energy()
    # get word level transcript
    audio_obj.whspr_dct = audio_obj.speech_to_whspr_dct()
    audio_obj.whspr_dct_to_word_seg()
    # adjust it
    audio_obj.word_seg = audio_obj.snap_close_boundaries_to_minimum(audio_obj.word_seg)
    # locally transcribe words with phonemizer
    audio_obj.phon_word_seg = audio_obj.local_seg_transcript(audio_obj.word_seg)
    #audio_obj.adjust_phonetic_word_seg()
    # align the phonemes
    audio_obj.ctc_align_phonemes_per_segment(audio_obj.word_seg)
    audio_obj.phon_seg = audio_obj.snap_close_boundaries_to_minimum(audio_obj.phon_seg)
    # save the aligned phonemes
    audio_obj.seg_to_csv(audio_obj.phon_seg, './ddh2phon.csv')
    csv_to_textgrid('./ddh2phon.csv', './ddh2phon.TextGrid')
    exit()

if __name__ == "__main__":
    main()
