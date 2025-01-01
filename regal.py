#!/usr/bin/env python3
"""
regal.py

Example REGAL pipeline adapted for TWO separate graphs.

Usage:
  python regal.py \
      --input1 path/to/male_connectome_graph.txt \
      --input2 path/to/female_connectome_graph.txt \
      --output1 path/to/male_emb.npy \
      --output2 path/to/female_emb.npy \
      --untillayer 2 \
      --alpha 0.01 \
      --k 10 \
      --align

Steps:
  1) Reads two edgelist files (graph1, graph2).
  2) Learns xNetMF embeddings for each graph separately.
  3) Saves each embedding to a .npy file (e.g., male_emb.npy, female_emb.npy).
  4) Optionally, if --align is set, computes an alignment matrix between the two embeddings.
"""

import argparse
import numpy as np
import networkx as nx
import time
import sys
from scipy.sparse import csr_matrix

# These imports assume your local REGAL modules are in the same directory
import xnetmf
from config import *
from alignments import *

def parse_args():
    parser = argparse.ArgumentParser(description="Run REGAL on two separate graphs.")

    # Graph inputs
    parser.add_argument('--input1', required=True, help="Edgelist for Graph 1 (e.g., male)")
    parser.add_argument('--input2', required=True, help="Edgelist for Graph 2 (e.g., female)")

    # Output embedding paths
    parser.add_argument('--output1', required=True, help="Where to save Graph 1's embedding (.npy)")
    parser.add_argument('--output2', required=True, help="Where to save Graph 2's embedding (.npy)")

    # Node attributes (optional)
    parser.add_argument('--attributes', nargs='?', default=None,
                        help="Path to .npy file of node attributes, or int for synthetic attributes.")
    parser.add_argument('--attrvals', type=int, default=2,
                        help="Number of attribute values if synthetic attributes are generated.")

    # xNetMF/REGAL parameters
    parser.add_argument('--dimensions', type=int, default=128,
                        help="Number of embedding dimensions. Default=128.")
    parser.add_argument('--k', type=int, default=10,
                        help="Controls # of landmarks (default=10).")
    parser.add_argument('--untillayer', type=int, default=2,
                        help="Max layer for xNetMF (default=2). 0 => no limit.")
    parser.add_argument('--alpha', type=float, default=0.01,
                        help="Discount factor for further layers (default=0.01).")
    parser.add_argument('--gammastruc', type=float, default=1,
                        help="Weight on structural similarity (default=1).")
    parser.add_argument('--gammaattr', type=float, default=1,
                        help="Weight on attribute similarity (default=1).")
    parser.add_argument('--numtop', type=int, default=0,
                        help="KD-tree top similarities. 0 => compute all pairwise.")
    parser.add_argument('--buckets', type=float, default=2,
                        help="Base of log for degree binning (default=2).")

    # Alignment flag
    parser.add_argument('--align', action='store_true',
                        help="If set, compute an alignment matrix between the two embeddings.")

    return parser.parse_args()

def learn_representations_for_edgelist(edgelist_file, rep_method, attributes=None):
    """
    Reads an edgelist file, constructs a Graph object, and runs xNetMF to get the embedding.
    """
    print(f"\n=== Reading edgelist from: {edgelist_file} ===")
    # For directed/weighted graphs, adjust the read_edgelist call, e.g.:
    # nx_graph = nx.read_edgelist(edgelist_file, nodetype=int, data=[('weight', float)], create_using=nx.DiGraph())
    nx_graph = nx.read_edgelist(edgelist_file, nodetype=int, comments="%")
    print("  Graph has", nx_graph.number_of_nodes(), "nodes and", nx_graph.number_of_edges(), "edges")

    # Build adjacency
    node_list = sorted(nx_graph.nodes())  # ensures consistent ordering
    adj = nx.adjacency_matrix(nx_graph, nodelist=node_list)
    print("  Constructed adjacency matrix shape:", adj.shape)

    # Create a Graph object as used by xNetMF
    graph_obj = Graph(adj, node_attributes=attributes)

    # Run xNetMF
    print("  Running xNetMF get_representations...")
    embedding = xnetmf.get_representations(graph_obj, rep_method, verbose=True)
    print("  xNetMF finished. Embedding shape:", embedding.shape)
    return embedding

def main(args):
    # Handle attributes if provided
    if args.attributes is not None:
        # Could be a .npy file or an int specifying synthetic attribute dimension
        try:
            attributes = np.load(args.attributes)
            print("Loaded external attributes of shape:", attributes.shape)
        except ValueError:
            # e.g. if it's an int, you'd generate synthetic attributes
            # (not implemented in this minimal example)
            attributes = None
    else:
        attributes = None

    # Create the RepMethod config
    max_layer = args.untillayer if args.untillayer != 0 else None
    if args.buckets == 1:
        buckets = None
    else:
        buckets = args.buckets

    rep_method = RepMethod(
        max_layer=max_layer,
        alpha=args.alpha,
        k=args.k,
        num_buckets=buckets,
        normalize=True,      # typically True for REGAL
        gammastruc=args.gammastruc,
        gammaattr=args.gammaattr
    )
    print("\n=== REGAL/xNetMF Parameters ===")
    print("  max_layer =", max_layer)
    print("  alpha     =", args.alpha)
    print("  k         =", args.k)
    print("  buckets   =", buckets)
    print("  gammastruc=", args.gammastruc)
    print("  gammaattr =", args.gammaattr)

    # Learn embeddings for each graph
    print("\n=== Learning Embeddings for Graph 1 ===")
    start_time_1 = time.time()
    emb1 = learn_representations_for_edgelist(args.input1, rep_method, attributes)
    np.save(args.output1, emb1)
    print(f"  Saved Graph 1 embedding to {args.output1}")
    elapsed1 = time.time() - start_time_1
    print(f"  Time for Graph 1 embedding: {elapsed1:.2f} seconds")

    print("\n=== Learning Embeddings for Graph 2 ===")
    start_time_2 = time.time()
    emb2 = learn_representations_for_edgelist(args.input2, rep_method, attributes)
    np.save(args.output2, emb2)
    print(f"  Saved Graph 2 embedding to {args.output2}")
    elapsed2 = time.time() - start_time_2
    print(f"  Time for Graph 2 embedding: {elapsed2:.2f} seconds")

    # If requested, compute alignment matrix (similarities)
    if args.align:
        print("\n=== Computing Alignment between emb1 & emb2 ===")
        align_start = time.time()
        # By default, we compute the full pairwise similarities
        # If your method needs partial similarity (num_top), you can pass that
        alignment_matrix = get_embedding_similarities(emb1, emb2, sim_measure="euclidean", num_top=None)
        align_time = time.time() - align_start
        print("  Alignment matrix shape:", alignment_matrix.shape)
        print(f"  Computed alignment in {align_time:.2f} seconds")

        # If you have ground-truth node matches, you could call score_alignment_matrix here.
        # e.g.: score_alignment_matrix(alignment_matrix, topk=1, true_alignments=...)
    else:
        print("\nNo alignment requested (--align not set). Done!")

if __name__ == "__main__":
    args = parse_args()
    main(args)
