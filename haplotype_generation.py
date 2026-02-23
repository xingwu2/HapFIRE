import numpy as np
import pandas as pd
import multiprocessing as mp
import subprocess
import time
import re
import sys
import os
import cvxpy as cp
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csgraph
from scipy.spatial.distance import pdist, squareform
from sklearn import preprocessing
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import fcluster
import community as community_louvain
import networkx as nx
import math


import utility_functions as UF

def find_ld(i,snps,cutoff,window_size):  ###find the correlation of every SNP to a window
	n_inds,n_snps = snps.shape
	left = max(i - window_size, 0)  #define the boundary of the interval
	right = min(i + window_size,n_snps) #define the boundary of the interval

	left_snps_window = snps[:,left:(i+1)]
	right_snps_window = snps[:,i:(right+1)]
	left_cor = np.matmul(np.transpose(left_snps_window),snps[:,i])/n_inds
	left_cor_rev = np.flip(left_cor)
	right_cor = np.matmul(np.transpose(right_snps_window),snps[:,i])/n_inds
	
	left_list_ = np.where(left_cor_rev**2 > cutoff)[0]
	for j in range(len(left_list_)-1):
		if left_list_[j+1] - left_list_[j] > 25:
			left_list_ = left_list_[:j+1]
			break
	left_list_ = np.flip(left_list_) * -1

	right_list_ = np.where(right_cor**2 > cutoff)[0]
	for j in range(len(right_list_)-1):
		if right_list_[j+1] - right_list_[j] > 25:
			right_list_ = right_list_[:j+1]
			break
		
	SNPinLD_index = np.unique(np.concatenate((left_list_, right_list_))) + i
	return(SNPinLD_index)

def CompleteLDPartition(standardized_genotype_matrix,cutoff,window_size):
	
	#define variables
	n_inds,n_snps = standardized_genotype_matrix.shape
	snp_list = {}
	cummax_list = []
	max_list = []
	boundary = []

	alone_SNPs_index = []
	for i in range(n_snps):
		snp_list[i] = find_ld(i,snps=standardized_genotype_matrix,cutoff=0.1,window_size=50)
		if len(snp_list[i]) == 1:
			alone_SNPs_index.append(i)
	if len(alone_SNPs_index) > 0:
		print("QUALITY CHECK: identify %d snps that are not in LD (r2 < %f) with its 50 up/downstream neighbours. You may consider remove these SNPs" %(len(alone_SNPs_index),0.1))

	print("window_size is %d, and the correlation cutoff is %f" %(window_size, cutoff))
	for i in range(n_snps):
		snp_list[i] = find_ld(i,snps=standardized_genotype_matrix,cutoff=cutoff,window_size=window_size)

	for i in range(len(snp_list)):
		if len(snp_list[i]) > 0:
			max_list.append(np.max(snp_list[i]))
		else:
			max_list.append(i)
	
	cummax_list.append(max_list[0])
	for i in range(1,len(max_list)):
		if max_list[i] > cummax_list[i-1]:
			cummax_list.append(max_list[i])
		else:
			cummax_list.append(cummax_list[i-1])

	idx = np.where( cummax_list - np.array(range(n_snps)) == 0)
	idx = idx[0]

	boundary_ = np.concatenate(([-1],np.array(idx)))
	for i in range(len(boundary_)-1):
		left = boundary_[i]+1
		right = boundary_[i+1]
		if right - left > 0:
			boundary.append([left,right])
		else:
			boundary.append([left])

	j = 0

	while j < len(boundary):
		if len(boundary[j]) == 1:
			if j ==0:
				boundary[j+1][0] = boundary[j][0]
				del boundary[j]
			else:
				boundary[j-1][1] = boundary[j][0]
				del boundary[j]
		else:
			j += 1

	return(boundary,alone_SNPs_index)

def BigLD_partition(DIR,IndepLD_breakpoints_index,geno_matrix,variant_names,variant_positions,CLQcut,prefix,maf):
	fine_breakpoints_ch = []

	#generate geno and SNPinfo for BigLD
	for I in range(len(IndepLD_breakpoints_index)):
		left = IndepLD_breakpoints_index[I][0]
		right = IndepLD_breakpoints_index[I][1]
		tmp_names = variant_names[left:right+1]
		tmp_positions = variant_positions[left:right+1]
		tmp_matrix = pd.DataFrame(geno_matrix[:,left:right+1],columns= tmp_names)
		tmp_matrix.to_csv(prefix+"_"+str(I)+"_geno_matrix"+".btmp",sep="\t",header=True,index=False)
		INFO = open(prefix+"_"+str(I)+"_snpINFO"+".btmp","w")
		INFO.write("chrN\trsID\tbp\n")
		for j in range(len(tmp_positions)):
			INFO.write(str(1)+"\t"+str(tmp_names[j])+"\t"+str(tmp_positions[j])+"\n")
		INFO.close()

		BigLD_command = "Rscript "+DIR+"/BigLD.R -g "+prefix+"_"+str(I)+"_geno_matrix"+".btmp"+ " -s "+prefix+"_"+str(I)+"_snpINFO"+".btmp" + " -c " + str(CLQcut) + " -m " + str(maf) + " -o " + prefix+"_"+str(I)
		try:
			blocks = []
			subprocess.check_call(BigLD_command,shell=True)
			tmp_file = prefix+"_"+str(I)+"_res_btmp.txt"
			with open(tmp_file,"r") as INPUT:
				header = INPUT.readline()
				for line in INPUT:
					items = line.split("\t")
					blocks.append([variant_positions.index(int(items[5])),variant_positions.index(int(items[6]))])
			blocks[-1][1] = right

			blocks = UF.keep_largest_overlaps(blocks)

			fine_breakpoints_ch.extend(blocks)

		except subprocess.CalledProcessError as e:
			print("BigLD cannot further partition blocks in this region %i - %i" %(left,right))
			fine_breakpoints_ch.append(IndepLD_breakpoints_index[I])
		rm_command = "rm "+prefix+"_"+str(I)+"_*btmp*"
		subprocess.check_call(rm_command,shell = True)

	return(fine_breakpoints_ch)

def convert_independent_genomewide_breakpoints(common_breakpoints,common_index,r):
	gw_breakpoints = []
	for i in range(len(common_breakpoints)):  ## there is a small bug issue when the first or the last snp is the block itself
		if len(common_breakpoints[i]) == 1:
			if common_breakpoints[i][0] == 0:
				left = 0
				right = common_index[common_breakpoints[i+1][0]] -1 
			else:
				left = common_index[common_breakpoints[i][0] -1] + 1
				right = common_index[common_breakpoints[i][0]]
		else:
			common_left = common_breakpoints[i][0]
			common_right = common_breakpoints[i][1]
			#print(common_left,common_right)
			if common_left == 0:
				left = 0
				right = common_index[common_right]
			elif common_right == len(common_index) -1:
				left = common_index[common_left - 1] + 1
				right = r -1
			else:
				left = common_index[common_left - 1] + 1
				right = common_index[common_right]
		gw_breakpoints.append([left,right])

	j = 0
	while j < len(gw_breakpoints):
		if gw_breakpoints[j][1] - gw_breakpoints[j][0] < 2000:
			if j == 0:
				gw_breakpoints[j+1] = [gw_breakpoints[j][0],gw_breakpoints[j+1][1]]
				del gw_breakpoints[j]
			else:
				gw_breakpoints[j-1] = [gw_breakpoints[j-1][0],gw_breakpoints[j][1]]
				del gw_breakpoints[j]
		else:
			j += 1

	return(gw_breakpoints)


def convert_fine_genomewide_breakpoints(common_breakpoints,common_index,r,gw_independent_breakpoints):

	IndepLD_breakpoints = []

	for k in range(len(gw_independent_breakpoints)):
		IndepLD_breakpoints.append(gw_independent_breakpoints[k][0])
		IndepLD_breakpoints.append(gw_independent_breakpoints[k][1])

	print("IndepLD_breakpoints",IndepLD_breakpoints)
	gw_breakpoints = []
	for i in range(len(common_breakpoints)): #SHOULD NOT HAPPEN BUT IF HAPPENED THERE COULD BE A SMALL BUG
		common_left = common_breakpoints[i][0]
		common_right = common_breakpoints[i][1]
		if i == 0:
			left = 0
			right = common_index[common_right]

		elif i == len(common_breakpoints) -1:
			common_left_prev = common_breakpoints[i-1][1]
			left = common_index[common_left_prev] + 1
			right = r -1

		else:
			common_left_prev = common_breakpoints[i-1][1]
			left = common_index[common_left_prev] + 1
			right = common_index[common_right]
		gw_breakpoints.append([left,right])

	j = 0
	print("before",len(gw_breakpoints),gw_breakpoints)
	while j < len(gw_breakpoints):
		if gw_breakpoints[j][1] - gw_breakpoints[j][0] < 6:
			print("you motherfucker",gw_breakpoints[j])

			if gw_breakpoints[j][0] in IndepLD_breakpoints or gw_breakpoints[j][1] in IndepLD_breakpoints:
				print("WTF",gw_breakpoints[j])
				print("left,right,you son of bitch",IndepLD_breakpoints)
				
				if j == 0:
					gw_breakpoints[j+1] = [gw_breakpoints[j][0],gw_breakpoints[j+1][1]]
					del gw_breakpoints[j]
				else:
					j += 1

			else:
				gw_breakpoints[j-1] = [gw_breakpoints[j-1][0],gw_breakpoints[j][1]]
				del gw_breakpoints[j]
		else:
			j += 1
	print("after",len(gw_breakpoints),gw_breakpoints)
	return(gw_breakpoints)


def custom_fine_partition(block):
	genomewide_breakpoints = {}
	with open(block, "r") as f:
		for line in f:
			line = line.strip("\n")
			items = line.split("\t")
			ch = items[0]
			if ch not in genomewide_breakpoints:
				genomewide_breakpoints[ch] = [[int(items[1]),int(items[2])]]

			else:
				genomewide_breakpoints[ch].append([int(items[1]),int(items[2])])
	return(genomewide_breakpoints)

def snp_table_construction(ch,haplotypes,pos,reference,alternative):
	dedup_haplotypes = haplotypes.drop_duplicates(keep = 'first')
	r,c = dedup_haplotypes.shape
	header = [ch,"Ref"]
	header.extend(list(np.arange(r)))
	snp_table_list = [header]

	for i in range(c):
		snp_numeric = np.array(dedup_haplotypes.iloc[:,i])
		snp_i = [pos[i],reference[i]]
		for j in range(len(snp_numeric)):
			if snp_numeric[j] == 0:
				snp_i.append(reference[i])
			else:
				snp_i.append(alternative[i])
		snp_table_list.append(snp_i)
	snp_table = pd.DataFrame(snp_table_list)
	return(snp_table,dedup_haplotypes)

def find_nearest(np_array, value):
    array = np.asarray(np_array)
    idx = (np.abs(np_array - value)).argmin()
    return(idx)

def haplotype_frequency_estimation(ch,block_partition,hap_matrix_d1,hap_matrix_d2,reference_alleles,alternative_alleles,variant_positions,variant_names,bam,reference_file,prefix):
	snp_frequency = []
	haplotype_frequency = {}
	hap_matrix_d1_pd = pd.DataFrame(np.transpose(hap_matrix_d1),columns=variant_names)
	hap_matrix_d2_pd = pd.DataFrame(np.transpose(hap_matrix_d2),columns=variant_names)
	r,c = hap_matrix_d1.shape
	SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))	
	for i in range(len(block_partition)):
		left = block_partition[i][0]
		right = block_partition[i][1]
		hap1 = hap_matrix_d1_pd[variant_names[left:right+1]]
		hap2 = hap_matrix_d2_pd[variant_names[left:right+1]]
		haplotypes = pd.concat([hap1,hap2],ignore_index=True)
		snp_table,dedup_haplotypes = snp_table_construction(ch,haplotypes,variant_positions[left:right+1],reference_alleles[left:right+1],alternative_alleles[left:right+1])
		snp_table_name = prefix+ "_"+"snp_table_"+str(ch)+"_"+str(i)+".txt"
		snp_table.to_csv(snp_table_name,sep=",",header=False,index = False)
		read_left_bound = max(variant_positions[left]-100,1)
		read_right_bound = min(variant_positions[right]+100,variant_positions[-1])
		like_command = SCRIPT_DIR + "/src/harp like --bam "+ bam + " --region "+str(ch)+":"+str(read_left_bound)+ \
						"-"+str(read_right_bound)+" --refseq "+ reference_file + "  --snps  "+snp_table_name +"  --stem  " + prefix+"_"+str(ch)+"_"+str(i)
		subprocess.check_call(like_command,shell=True)
		freq_command = SCRIPT_DIR + "/src/harp freq --hlk " + prefix + "_" +str(ch)+"_"+str(i)+".hlk" + " --region "+str(ch)+":"+str(read_left_bound)+ \
						"-"+str(read_right_bound)
		try:
			subprocess.check_call(freq_command,shell=True)
			with open(prefix+ "_"+ str(ch)+"_"+str(i)+".freqs","r") as FREQ:
				line = FREQ.readline()
				line = line.strip("\n")
				items = line.split(" ")
				haplotype_frequency[i] = [float(items[i]) for i in range(3,len(items)-1)]
				snp_frequency.extend(np.matmul(haplotype_frequency[i],np.asarray(dedup_haplotypes)))
		except subprocess.CalledProcessError as e:
			print("There is no reads mapped to this region %i - %i" %(read_left_bound,read_right_bound))
			d_r,d_c = dedup_haplotypes.shape
			haplotype_frequency[i] = np.repeat(1.0/d_r,d_r)
			snp_frequency.extend(np.matmul(haplotype_frequency[i],np.asarray(dedup_haplotypes)))
		rm_command = "rm -r " + prefix + "_" + str(ch)+"_"+str(i)+"* " + snp_table_name
		subprocess.check_call(rm_command,shell=True)
	return(snp_frequency,haplotype_frequency)


def ecotype_frequency_estimation(gw_independent_breakpoints,independent_block_haplotype_frequency,hap_matrix_d1,hap_matrix_d2,variant_names,variant_positions):
	ecotype_frequency = np.zeros((len(gw_independent_breakpoints),hap_matrix_d1.shape[1]))
	haplotype_frequency = {}
	hap_matrix_d1_pd = pd.DataFrame(np.transpose(hap_matrix_d1),columns=variant_names)
	hap_matrix_d2_pd = pd.DataFrame(np.transpose(hap_matrix_d2),columns=variant_names)
	weights = []
	length = []
	for i in range(len(gw_independent_breakpoints)):
		left = gw_independent_breakpoints[i][0]
		right = gw_independent_breakpoints[i][1]
		hap1 = hap_matrix_d1_pd[variant_names[left:right+1]]
		hap2 = hap_matrix_d2_pd[variant_names[left:right+1]]
		haplotypes = pd.concat([hap1,hap2],ignore_index=True)
		dedup_haplotypes = np.asarray(haplotypes.drop_duplicates(keep = 'first'))
		length.append(dedup_haplotypes.shape[0])
		dictionary = {}
		for j in range(dedup_haplotypes.shape[0]):
			unique_haplotype_ = "".join(map(str,dedup_haplotypes[j,:]))
			dictionary[unique_haplotype_] = j

		independent_haplotype_DM_ = np.zeros((hap1.shape[0],dedup_haplotypes.shape[0]))


		for k in range(hap1.shape[0]):
			tmp_1 = "".join(map(str,hap1.values[k,:]))
			tmp_2 = "".join(map(str,hap2.values[k,:]))
			l_1 = dictionary[tmp_1]
			l_2 = dictionary[tmp_2]
			independent_haplotype_DM_[k,l_1] += 1
			independent_haplotype_DM_[k,l_2] += 1

		y = np.asarray(independent_block_haplotype_frequency[i]) * 2
		h = cp.Variable(hap1.shape[0])
		product = independent_haplotype_DM_.T @ h 
		diff  = product - y
		constraints = [0 <= h,sum(h) == 1]
		problem = cp.Problem(cp.Minimize(cp.norm(diff)),constraints)
		problem.solve(solver = 'SCS',verbose=False)
		ecotype_frequency[i,:] = h.value
		weights.append(variant_positions[right] - variant_positions[left])
	ecotype_frequency_avg = np.average(ecotype_frequency,axis=0,weights = weights)
	return(ecotype_frequency_avg)


def ecotype_frequency_estimation_selected(gw_independent_breakpoints,independent_block_haplotype_frequency,hap_matrix_d1,hap_matrix_d2,variant_names,variant_positions):
	ecotype_frequency = np.zeros((len(gw_independent_breakpoints),hap_matrix_d1.shape[1]))
	haplotype_frequency = {}
	hap_matrix_d1_pd = pd.DataFrame(np.transpose(hap_matrix_d1),columns=variant_names)
	hap_matrix_d2_pd = pd.DataFrame(np.transpose(hap_matrix_d2),columns=variant_names)
	weights = []
	length = []
	for i in range(len(gw_independent_breakpoints)):
		left = gw_independent_breakpoints[i][0]
		right = gw_independent_breakpoints[i][1]
		hap1 = hap_matrix_d1_pd[variant_names[left:right+1]]
		hap2 = hap_matrix_d2_pd[variant_names[left:right+1]]
		haplotypes = pd.concat([hap1,hap2],ignore_index=True)
		dedup_haplotypes = np.asarray(haplotypes.drop_duplicates(keep = 'first'))
		dictionary = {}
		for j in range(dedup_haplotypes.shape[0]):
			unique_haplotype_ = "".join(map(str,dedup_haplotypes[j,:]))
			dictionary[unique_haplotype_] = j

		independent_haplotype_DM_ = np.zeros((hap1.shape[0],dedup_haplotypes.shape[0]))


		for k in range(hap1.shape[0]):
			tmp_1 = "".join(map(str,hap1.values[k,:]))
			tmp_2 = "".join(map(str,hap2.values[k,:]))
			l_1 = dictionary[tmp_1]
			l_2 = dictionary[tmp_2]
			independent_haplotype_DM_[k,l_1] += 1
			independent_haplotype_DM_[k,l_2] += 1

		y = np.asarray(independent_block_haplotype_frequency[i]) * 2
		h = cp.Variable(hap1.shape[0])
		product = independent_haplotype_DM_.T @ h 
		diff  = product - y
		constraints = [0 <= h,sum(h) == 1]
		problem = cp.Problem(cp.Minimize(cp.norm(diff)),constraints)
		problem.solve(solver = 'SCS',verbose=False)
		ecotype_frequency[i,:] = h.value
		print(h.value)
		length.append(dedup_haplotypes.shape[0])
		# if dedup_haplotypes.shape[0] < hap1.shape[0] * 0.9:
		# 	weights.append(0)
		# else:
		# 	weights.append(variant_positions[right] - variant_positions[left])
	#ecotype_frequency_avg = np.average(ecotype_frequency,axis=0,weights = weights)
	return(ecotype_frequency,length)


def ecotype_frequency_estimation_lasso(gw_independent_breakpoints,independent_block_haplotype_frequency,hap_matrix_d1,hap_matrix_d2,variant_names,variant_positions,lambd):
	ecotype_frequency = np.zeros((len(gw_independent_breakpoints),hap_matrix_d1.shape[1]))
	haplotype_frequency = {}
	hap_matrix_d1_pd = pd.DataFrame(np.transpose(hap_matrix_d1),columns=variant_names)
	hap_matrix_d2_pd = pd.DataFrame(np.transpose(hap_matrix_d2),columns=variant_names)
	weights = []
	length = []
	for i in range(len(gw_independent_breakpoints)):
		left = gw_independent_breakpoints[i][0]
		right = gw_independent_breakpoints[i][1]
		hap1 = hap_matrix_d1_pd[variant_names[left:right+1]]
		hap2 = hap_matrix_d2_pd[variant_names[left:right+1]]
		haplotypes = pd.concat([hap1,hap2],ignore_index=True)
		dedup_haplotypes = np.asarray(haplotypes.drop_duplicates(keep = 'first'))
		dictionary = {}
		for j in range(dedup_haplotypes.shape[0]):
			unique_haplotype_ = "".join(map(str,dedup_haplotypes[j,:]))
			dictionary[unique_haplotype_] = j

		independent_haplotype_DM_ = np.zeros((hap1.shape[0],dedup_haplotypes.shape[0]))


		for k in range(hap1.shape[0]):
			tmp_1 = "".join(map(str,hap1.values[k,:]))
			tmp_2 = "".join(map(str,hap2.values[k,:]))
			l_1 = dictionary[tmp_1]
			l_2 = dictionary[tmp_2]
			independent_haplotype_DM_[k,l_1] += 1
			independent_haplotype_DM_[k,l_2] += 1

		y = np.asarray(independent_block_haplotype_frequency[i]) * 2
		h = cp.Variable(hap1.shape[0])
		product = independent_haplotype_DM_.T @ h 
		diff  = product - y
		constraints = [0 <= h,sum(h) == 1]
		problem = cp.Problem(cp.Minimize(cp.norm2(diff)**2 + lambd*cp.norm1(h)),constraints)
		problem.solve(solver = 'SCS',verbose=False)
		ecotype_frequency[i,:] = h.value
		weights.append(variant_positions[right] - variant_positions[left])
	ecotype_frequency_avg = np.average(ecotype_frequency,axis=0,weights = weights)
	return(ecotype_frequency_avg)


# def Spectral_clustering(np_array):
# 	r,c =np_array.shape
# 	k = max(int(r/20),5)

# 	dists = squareform(pdist(np_array))
# 	knn_distances = np.sort(dists, axis=0)[k]
# 	knn_distances = knn_distances[np.newaxis].T
# 	local_scale = knn_distances.dot(knn_distances.T)
# 	affinity_matrix = - dists * dists / local_scale
# 	affinity_matrix[np.where(np.isnan(affinity_matrix))] = 0.0
# 	affinity_matrix = np.exp(affinity_matrix)
# 	np.fill_diagonal(affinity_matrix, 0)

# 	L = csgraph.laplacian(affinity_matrix,normed = True)

# 	eig_val, eig_vec = np.linalg.eig(L)
# 	eig_val = np.real(eig_val)
# 	eig_vec = np.real(eig_vec)
	
# 	eig_vec = eig_vec[:,np.argsort(eig_val)]
# 	eig_val = eig_val[np.argsort(eig_val)]


# 	if sum(np.iscomplex(eig_val)) > 0:
# 		print("Spectral Clustering failed. Clusters are assigned by affinity_propagation.")
# 		print(np_array.shape)
# 		labels = affinity_propagation(np_array)
# 		if labels[0] == -1 or max(labels) == 0:
# 			labels = np.arange(np_array.shape[0])
# 			print("Affinity propagation failed")
		
# 	else:
# 		index_largest_gap = np.argsort(np.diff(eig_val))[::-1][0]
# 		#print(index_largest_gap)
# 		n_clusters = index_largest_gap + 2
# 		V = eig_vec[:,:n_clusters]
# 		Z = linkage(V, 'ward')
# 		labels = fcluster(Z, n_clusters, criterion='maxclust') - 1
# 	return(labels)

def KNN_Spectral(np_array,eps=1e-12, k_max_eigs=30):

	r,c =np_array.shape
	k = max(10, int(2*math.log(r)))
	k = min(k, r - 1)

	connectivity = kneighbors_graph(X=np_array, n_neighbors=k, mode='connectivity',metric="hamming",include_self=False)

	A = (1/2)*(connectivity + connectivity.T)
	
	L = csgraph.laplacian(A,normed = True)

	L = L.toarray()

	evals, evecs = np.linalg.eigh(L)

	# Use eigengap on the smallest part of the spectrum
	m = min(k_max_eigs, r - 1)
	small = evals[:m]
	gaps = np.diff(small)

	# Ignore the trivial first eigenvalue near 0; search gaps starting from index 1
	if len(gaps) < 2:
		n_clusters = 1
	else:
		j = 1 + np.argmax(gaps[1:])  # j is the split point
		n_clusters = j + 1

	# Spectral embedding: take first n_clusters eigenvectors
	V = evecs[:, :n_clusters]

	# Row-normalize (common in Ng-Jordan-Weiss style)
	V_norm = V / (np.linalg.norm(V, axis=1, keepdims=True) + eps)

	# Cluster embedding (Ward on Euclidean in embedding space is fine)
	Z = linkage(V_norm, method="ward")
	labels = fcluster(Z, t=n_clusters, criterion="maxclust") - 1

	#print(max(labels))

	return(labels)

def local_scale_modularity(np_array,eps=1e-12):
	
	r,c =np_array.shape
	k = max(10, int(2*math.log(r)))
	k_knn = min(k, r - 1)
	k_sigma = min(k, r - 2)


	D = squareform(pdist(np_array,metric="hamming"))

	# sigma_i = distance to k-th neighbor (excluding self at position 0)
	D_sorted = np.sort(D, axis=1)
	sigma = D_sorted[:, k_sigma + 1]  # +1 because D_sorted[:,0] is self-distance 0
	sigma = np.maximum(sigma, eps)

	# Local scaling: exp( -d^2 / (sigma_i * sigma_j) )
	S = np.outer(sigma, sigma)
	S = np.maximum(S, eps)

	W = np.exp(-(D * D) / S)
	np.fill_diagonal(W, 0.0)


	connectivity = kneighbors_graph(X=np_array, n_neighbors=k_knn, mode='connectivity',metric="hamming",include_self=False)
	connectivity = 0.5 * (connectivity + connectivity.T) ## make connectivity symmetric

	connectivity_matrix = connectivity.toarray()
	weighted_connectivity = np.multiply(W,connectivity_matrix)
	G = nx.from_numpy_array(weighted_connectivity)
	## this is a more stable function that the results can be reproduced due to the seed setting
	louvain_partition = nx.community.louvain_communities(G,seed=0)
	labels = np.empty(r, dtype=int)
	for cid, nodes in enumerate(louvain_partition):
		labels[list(nodes)] = cid

	return(labels)

def local_scale_Spectral(np_array,eps=1e-12,k_max_eigs=30):

	r,c =np_array.shape
	k = max(10, int(2*math.log(r)))
	k = min(k, r - 2)

	D = squareform(pdist(np_array,metric="hamming"))

	# sigma_i = distance to k-th neighbor (excluding self at position 0)
	D_sorted = np.sort(D, axis=1)
	sigma = D_sorted[:, k + 1]  # +1 because D_sorted[:,0] is self-distance 0
	sigma = np.maximum(sigma, eps)

	# Local scaling: exp( -d^2 / (sigma_i * sigma_j) )
	S = np.outer(sigma, sigma)
	S = np.maximum(S, eps)

	W = np.exp(-(D * D) / S)
	np.fill_diagonal(W, 0.0)
	L = csgraph.laplacian(W, normed=True)
	evals, evecs = np.linalg.eigh(L)

	# Use eigengap on the smallest part of the spectrum
	m = min(k_max_eigs, r - 1)
	small = evals[:m]
	gaps = np.diff(small)

	# Ignore the trivial first eigenvalue near 0; search gaps starting from index 1
	if len(gaps) < 2:
		n_clusters = 1
	else:
		j = 1 + np.argmax(gaps[1:])  # j is the split point
		n_clusters = j + 1

	# Spectral embedding: take first n_clusters eigenvectors
	V = evecs[:, :n_clusters]

	# Row-normalize (common in Ng-Jordan-Weiss style)
	V_norm = V / (np.linalg.norm(V, axis=1, keepdims=True) + eps)

	# Cluster embedding (Ward on Euclidean in embedding space is fine)
	Z = linkage(V_norm, method="ward")
	labels = fcluster(Z, t=n_clusters, criterion="maxclust") - 1

	#print(max(labels))


	return(labels)

def xmeans_clustering(array):
	# Prepare initial centers - amount of initial centers defines amount of clusters from which X-Means will
	# start analysis.
	amount_initial_centers = 2
	initial_centers = kmeans_plusplus_initializer(array, amount_initial_centers).initialize()
	# Create instance of X-Means algorithm. The algorithm will start analysis from 2 clusters, the maximum
	# number of clusters that can be allocated is 30.
	xmeans_instance = xmeans(array, initial_centers, 30)
	xmeans_instance.process()
	# Extract clustering results: clusters and their centers
	clusters_ = xmeans_instance.get_clusters()
	clusters = [0]*len(array)
	for i in range(len(clusters_)):
		for j in clusters_[i]:
			clusters[j] = i
	#print(clusters)
	return(clusters)

def haplotypes_clustering(np_array,algorithm):
	if algorithm == "KNN":
		clusters = KNN_Spectral(np_array)
	elif algorithm == "xmeans":
		clusters = xmeans_clustering(np_array)	
	elif algorithm == "local":
		clusters = local_scale_Spectral(np_array)
	elif algorithm == "modularity":
		clusters = local_scale_modularity(np_array)
	else:
		sys.exit("Unknown haplotype clustering algorithm")
	return(clusters)


def fine_haplotype_frequency_calculation(block_partitions,gw_independent_breakpoints,independent_block_haplotype_frequency,gw_breakpoints,hap_matrix_d1,hap_matrix_d2,variant_names,variant_positions,snp_frequency,ch,algorithm):	
	unique_haplotype_frequency = []
	haplotype_cluster_frequency = []
	vcf_haplotype_frequency = []
	unique_haplotype_names = []
	haplotype_cluster_names = []
	vcf_haplotype_cluster_frequency = []
	columns = []

	if gw_breakpoints[-1][1] != len(snp_frequency)-1:
		print(gw_breakpoints[-1][1],len(snp_frequency))
		sys.exit("The number of SNPs and genome wide partitions dont match!")
	else:
		hap_matrix_d1_pd = pd.DataFrame(np.transpose(hap_matrix_d1),columns=variant_names)
		hap_matrix_d2_pd = pd.DataFrame(np.transpose(hap_matrix_d2),columns=variant_names)

		unique_haplotype_DM = pd.DataFrame(index=range(hap_matrix_d1_pd.shape[0]),columns=columns)
		clustered_haplotype_DM = pd.DataFrame(index=range(hap_matrix_d1_pd.shape[0]),columns=columns)

		for key in block_partitions:
			ind_left = gw_independent_breakpoints[key][0]
			ind_right = gw_independent_breakpoints[key][1]
			ind_hap1 = hap_matrix_d1_pd[variant_names[ind_left:ind_right+1]]
			ind_hap2 = hap_matrix_d2_pd[variant_names[ind_left:ind_right+1]]
			ind_haplotypes = pd.concat([ind_hap1,ind_hap2],ignore_index=True)
			ind_dedup_haplotypes = np.asarray(ind_haplotypes.drop_duplicates(keep = 'first'))
			ind_haplotype_freq = independent_block_haplotype_frequency[key]

			for i in range(len(block_partitions[key])):
				left = block_partitions[key][i][0]
				right = block_partitions[key][i][1]
				hap1 = hap_matrix_d1_pd[variant_names[left:right+1]]
				hap2 = hap_matrix_d2_pd[variant_names[left:right+1]]
				haplotypes = pd.concat([hap1,hap2],ignore_index=True)
				dedup_haplotypes = np.asarray(haplotypes.drop_duplicates(keep = 'first'))

				relative_left = left - ind_left
				relative_right = right - ind_left
				

				block_snp_frequency = snp_frequency[left:right+1]
				r,c = dedup_haplotypes.shape

				# unique haplotype frequency calculations 
				unique_haplotype_DM_ = np.zeros((r,len(ind_haplotype_freq)))
			
				dictionary = {}
				for j in range(r):
					unique_haplotype_ = "".join(map(str,dedup_haplotypes[j,:]))
					dictionary[unique_haplotype_] = j

				# generate the design matrix
				for k in range(len(ind_haplotype_freq)):
					tmp = "".join(map(str,ind_dedup_haplotypes[k,relative_left:relative_right+1]))
					l = dictionary[tmp]
					unique_haplotype_DM_[l,k] += 1
				unique_haplotype_freq_ = np.matmul(unique_haplotype_DM_,ind_haplotype_freq)

				#calculate the VCF based haplotype frequency

				vcf_haplotype_DM_ = np.zeros((hap1.shape[0],r))

				for k in range(hap1.shape[0]):
					tmp_1 = "".join(map(str,hap1.values[k,:]))
					tmp_2 = "".join(map(str,hap2.values[k,:]))
					l_1 = dictionary[tmp_1]
					l_2 = dictionary[tmp_2]
					vcf_haplotype_DM_[k,l_1] += 1
					vcf_haplotype_DM_[k,l_2] += 1

				vcf_haplotype_freq_ = np.sum(vcf_haplotype_DM_,axis=0) / haplotypes.shape[0]
				unique_haplotype_names_ = [ch+"@"+str(variant_positions[left])+"-"+str(variant_positions[right])+'_'+str(l) for l in range(r)]
				#unique_haplotype_DM_ =pd.DataFrame(unique_haplotype_DM_,columns=haplotype_names_)
				#unique_haplotype_DM = pd.concat([unique_haplotype_DM,unique_haplotype_DM_],axis=1,ignore_index=True)


				# cluster unique haplotypes into haplotype clusters and then calculate cluster frequencies

				if dedup_haplotypes.shape[0] > 10:
					clusters = haplotypes_clustering(dedup_haplotypes,algorithm)
				else:
					clusters = np.arange(dedup_haplotypes.shape[0])
				haplotype_cluster_names_ = [ch+"@"+str(variant_positions[left])+"-"+str(variant_positions[right])+'_'+"haplotype_cluster"+str(l) for l in range(max(clusters)+1)]

				haplotype_cluster_frequency_ = []

				for i in range(max(clusters)+1):
					index = np.where(clusters == i)[0]
					haplotype_cluster_frequency_.append(sum(unique_haplotype_freq_[index]))

				#generate haplotype cluster design matrix
				dictionary = {}
				for j in range(r):
					unique_haplotype_ = "".join(map(str,dedup_haplotypes[j,:]))
					dictionary[unique_haplotype_] = clusters[j]

				haplotype_cluster_DM_ = np.zeros((hap1.shape[0],max(clusters)+1))

				for k in range(hap1.shape[0]):
					tmp_1 = "".join(map(str,hap1.values[k,:]))
					tmp_2 = "".join(map(str,hap2.values[k,:]))
					l_1 = dictionary[tmp_1]
					l_2 = dictionary[tmp_2]
					haplotype_cluster_DM_[k,l_1] += 1
					haplotype_cluster_DM_[k,l_2] += 1

				vcf_haplotype_cluster_freq_ = np.sum(haplotype_cluster_DM_,axis=0) / haplotypes.shape[0]
				haplotype_cluster_DM_ =pd.DataFrame(haplotype_cluster_DM_,columns=haplotype_cluster_names_)
				clustered_haplotype_DM = pd.concat([clustered_haplotype_DM,haplotype_cluster_DM_],axis=1,ignore_index=True)

				#print("predicted",haplotype_cluster_frequency_)
				#print("true",vcf_haplotype_cluster_freq_)
				
				unique_haplotype_frequency.extend(unique_haplotype_freq_)
				haplotype_cluster_frequency.extend(haplotype_cluster_frequency_)
				vcf_haplotype_frequency.extend(vcf_haplotype_freq_)
				unique_haplotype_names.extend(unique_haplotype_names_)
				haplotype_cluster_names.extend(haplotype_cluster_names_)
				vcf_haplotype_cluster_frequency.extend(vcf_haplotype_cluster_freq_)

	return(unique_haplotype_frequency,vcf_haplotype_frequency,unique_haplotype_names,haplotype_cluster_frequency,haplotype_cluster_names,vcf_haplotype_cluster_frequency)
			






















