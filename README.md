# PHONEME ALIGNER
This software uses ctc alignment, as well as phoneme boundary detection provided by https://github.com/felixkreuk/UnsupSeg
The aim was to have a full python/torch pipeline to segment audio, that doesn't sh*t its pants on the regular.
It works by performing CTC phoneme transcription/alginment (https://huggingface.co/bofenghuang/phonemizer-wav2vec2-ctc-french), then refines the 
alignment using boundaires found by the first model, as well as simple heuristics.
If you wish to modify the code, do not hesitate to contact me, I enjoy it when things work.

# INSTALL
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
