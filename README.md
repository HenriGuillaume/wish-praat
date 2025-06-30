# PHONEME ALIGNER
This software uses ctc alignment, as well as phoneme boundary detection provided by https://github.com/felixkreuk/UnsupSeg
The aim was to have a full python/torch pipeline to segment audio, that is more often reliable than MFA.
It works by performing CTC phoneme transcription/alginment (https://huggingface.co/bofenghuang/phonemizer-wav2vec2-ctc-french), then refines the 
alignment using boundaires found by the first model, as well as simple heuristics.
If you wish to modify the code, do not hesitate to contact me, I enjoy it when things work.

# INSTALL
## CLONE REPO
```
git clone https://github.com/HenriGuillaume/phoneme_aligner.git
cd phoneme_aligner
```

## CREATE ENVIRONMENT
```
conda create -n yourenvname python=3.10
conda activate yourvenvname
pip install -r requirements.txt
```

# USE
```
python main.py --audio path/to/file.wav --ckpt path/to/ckpt.pth
```

# TO DO
- Implement VAD that works
- Replace boundary conflict heuristics with something more fancy
- Mapping CTC phonemes to phoneme boundaries could be done through a dynamic programming approach, I need to find a proper formulation, heuristics will do for now
