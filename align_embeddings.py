import numpy as np
import argparse
import sys
import os
from alignments import get_embedding_similarities
# ^ We'll assume you’ve put that def in a separate file or inline

def parse_args():
    parser = argparse.ArgumentParser(description="Compute alignment from two existing embeddings.")
    
    parser.add_argument('--emb1', type=str, required=True,
                        help='Path to .npy file for first set of embeddings (e.g., male).')
    parser.add_argument('--emb2', type=str, required=True,
                        help='Path to .npy file for second set of embeddings (e.g., female).')
    parser.add_argument('--sim_measure', type=str, default='euclidean',
                        help='Type of similarity measure to use. Options: euclidean or cosine.')
    parser.add_argument('--num_top', type=int, default=None,
                        help='If set, use KD-tree to get only top K similarities.')
    parser.add_argument('--output', type=str, default='alignment_matrix.npy',
                        help='Where to save the alignment/similarity matrix.')
    
    return parser.parse_args()

def main(args):
    # 1) Load embeddings from .npy
    emb1 = np.load(args.emb1)
    emb2 = np.load(args.emb2)
    print("Loaded emb1 shape:", emb1.shape)
    print("Loaded emb2 shape:", emb2.shape)

    # 2) Compute similarity or alignment matrix
    alignment_matrix = get_embedding_similarities(emb1, emb2,
                                                  sim_measure=args.sim_measure,
                                                  num_top=args.num_top)

    # 3) Save it
    np.save(args.output, alignment_matrix)
    print(f"Alignment/similarity matrix saved to: {args.output}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
