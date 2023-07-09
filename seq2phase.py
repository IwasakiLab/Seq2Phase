#!/usr/bin/env python3
import argparse
import sys

def main(inputfile, outputfile, thread, cpu, model_dir):
    from bio_embeddings.embed.prottrans_t5_embedder import ProtTransT5XLU50Embedder
    from Bio import SeqIO
    import numpy as np
    import pandas as pd
    import torch
    from joblib import load
    import torch
        
    if thread != -1:
        torch.set_num_threads(thread)
    
    print("Loading model...")
    if cpu or not torch.cuda.is_available():
        embedder = ProtTransT5XLU50Embedder(device='cpu')
        print("on CPU")
    else:
        embedder = ProtTransT5XLU50Embedder(device='cuda')
        print("Model moved to CUDA")
        
    print("Embedding...")    
    sequences = []
    for record in SeqIO.parse(inputfile, "fasta"):
        sequences.append(record)
    embeddings = embedder.embed_many([str(s.seq) for s in sequences])
    embeddings=list(embeddings)
    embeddings=[x.mean(axis=0) for x in embeddings]
    ids=[str(s.id) for s in sequences]
    
    model_client = load(model_dir + 'trained_model_client.joblib')
    model_scaffold = load(model_dir + 'trained_model_scaffold.joblib')
    print("Predicting...")
    pred_client=model_client.predict_proba(embeddings)[:,1]
    pred_scaffold=model_scaffold.predict_proba(embeddings)[:,1]
    
    results=pd.DataFrame(index=ids, data={"ClientScore":pred_client, "ScaffoldScore":pred_scaffold})
    results.to_csv(outputfile, sep="\t")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-i', '--input', type=str, help='path to input fasta file')
    parser.add_argument('-o', '--output', type=str, default='output.tsv', help='path to output file (default: %(default)s)')
    parser.add_argument('-t', '--thread', type=int, default=-1, help='number of threads on CPU. -1 for all threads')
    parser.add_argument('--cpu', action='store_true', help='run on CPU')
    parser.add_argument('-d', '--model_dir', type=str, default='./', help='path to model file (default: %(default)s)')
    
    if len(sys.argv) == 1 or '-h' in sys.argv or '--help' in sys.argv:
        parser.print_help()
        sys.exit()
        
    args = parser.parse_args()
    
    if not args.input:
        parser.error("Input file is required.")

    main(args.input, args.output, args.thread, args.cpu, args.model_dir)