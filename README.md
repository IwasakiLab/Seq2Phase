# Seq2Phase

LLPS-related protein predictor from amino acid sequence


## Description

![Image](image.jpg)
Seq2Phase is an LLPS (Liquid-Liquid Phase Separation) client and scaffold predictor that uses only the amino acid sequence of proteins for prediction. 
LLPS is a membraneless cellular compartmentalization process involving two types of proteins: "scaffolds" that form condensates and "clients" that localize in the condensates.

## Dependencies

Install [Bio Embeddings](https://github.com/sacdallago/bio_embeddings)
```bash
pip install bio-embeddings
```

Install [PyTorch](https://pytorch.org) suitable for your machine
```bash
pip3 install torch torchvision torchaudio
```

## Usage

Run `train_model.py` first, then `seq2phase.py`.
```bash
python train_model.py
python seq2phase.py -i input.fasta -o output.tsv
```

The output file contains the client and scaffold scores.
If both scores of a protein are high (>0.5), it is considered a scaffold protein.



### Contact

Kazuki Miyata (The University of Tokyo) miyatakazuki381@g.ecc.u-tokyo.ac.jp