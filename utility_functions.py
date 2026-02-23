import re
import numpy as np
import pandas as pd
import scipy
import networkx as nx
import argparse
import math

from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csgraph
from scipy.spatial.distance import pdist, squareform
from sklearn import preprocessing
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import fcluster


def parse_arguments():
    parser = argparse.ArgumentParser(description="HapFIRE: Haplotype-based Frequency Inference and accession REconstruction from pool-seq data",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    input_group = parser.add_argument_group("Input options")
    input_group.add_argument('-vcf',type = str, action= 'store',dest='vcf',help='bgzipped and phased vcf file from the founders')
    input_group.add_argument('-bam',type = str, action= 'store',dest='bam',help = "pool-seq alignment file in the bam format")
    input_group.add_argument('-fa',type = str, action= 'store',dest='reference',help = "reference genome fasta file")
    
    output_group = parser.add_argument_group("Output option")
    output_group.add_argument('-output',type = str, action = 'store', dest = 'output',help = "the prefix of the output files")

    model1_group = parser.add_argument_group("SNP and accession frequency estimation")
    model1_group.add_argument('-r1',type = float, action = 'store', dest = 'corr',default = 0.1,help = "r2 for independent LD blocks")
    model1_group.add_argument('-stepsize',type = int, action = 'store', dest = 'window',default = 100,help = "sliding window size for calculating independent LD blocks")
    model1_group.add_argument('-maf',type = float, action = 'store', dest = 'maf',default = 0.05)
    
    model2_group = parser.add_argument_group("naive chromosome painting")
    model2_group.add_argument('-painting',type = bool, action= 'store',dest='painting',default = False,help = "performing genome-wide accession frequency estimation per window")
    model2_group.add_argument('-windowsize',type = int, action= 'store',dest='windowsize',default = 50000,help = "size of genomic window (bp) to calculate accession frequency per window")

    
    model3_group = parser.add_argument_group("haplotype frequency estimation")
    model3_group.add_argument('-haplotype',type = bool, action= 'store',dest='haplotype_frequency',default = False,help = "performing haplotype frequency estimation")
    model3_group.add_argument('-block',type = str, action= 'store',dest='block',default = "bigld",help="block partition method: bigld or previously defined blocks")
    model3_group.add_argument('-r2',type = float, action = 'store', dest = 'CLQcut',default = 0.5,help = "BigLD parameter: r2 for defining fine haplotype blocks")
    model3_group.add_argument('-hc',type = str, action = 'store', dest = 'clustering_algorithm',help = "haplotype clustering method: xmeans, KNN, modularity, local. Default: KNN",default = "KNN")
    
    args = parser.parse_args()

    return(args)

def keep_largest_overlaps(blocks):
	"""
	blocks: list of [start, end]
	Return only the largest block in each overlapping cluster.
	"""

	# convert to tuples
	ivals = [(int(s), int(e)) for s, e in blocks]
	n = len(ivals)
	if n <= 1:
		return blocks

	# build overlap graph
	adj = [[] for _ in range(n)]
	for i in range(n):
		for j in range(i + 1, n):
			a0, a1 = ivals[i]
			b0, b1 = ivals[j]
			if not (a1 < b0 or b1 < a0):  # overlap
				adj[i].append(j)
				adj[j].append(i)

	visited = [False] * n
	result = []

	for i in range(n):
		if visited[i]:
			continue

		# DFS to collect overlapping cluster
		stack = [i]
		visited[i] = True
		cluster = [i]

		while stack:
			u = stack.pop()
			for v in adj[u]:
				if not visited[v]:
					visited[v] = True
					stack.append(v)
					cluster.append(v)

		# choose largest block in this cluster
		best = max(
			cluster,
			key=lambda k: (ivals[k][1] - ivals[k][0], -ivals[k][0], ivals[k][1])
		)

		result.append(list(ivals[best]))

	sorted_results = sorted(result, key=lambda x: x[0])
	return(sorted_results)


