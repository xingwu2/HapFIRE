import re
import numpy as np
import pandas as pd
import scipy
import sys

from cyvcf2 import VCF



def vcf2hapmatrix(vcf):
	hap_matrix_d1 = {} #haplotype 1 of individuals, chromosome number is the key for dict
	hap_matrix_d2 = {} #haplotype 2 of individuals, chromosome number is the key for dict
	variant_names = {}
	variant_positions = {}# Chromosome number is the key for dict
	ref = {} # Chromosome number is the key for dict
	alt = {} # Chromosome number is the key for dict
	chromosome = [] #Chromosome number
	
	with open(vcf,"r") as VCF:
		for line in VCF:
			if re.search("^##",line): ## skip the first annotation lines
				continue
			elif re.search("^#CHROM",line): ## acquire the sample name information
				line = line.strip("\n")
				ind_names = line.split("\t")[9:]
			else:
				line = line.strip("\n")
				items = line.split("\t")
				ch = items[0]
				if items[2] == "\.":
						sys.exit("Found at least one empty variant names, please name variant sites accordingly.")

				if ch not in chromosome:
					chromosome.append(ch)
					variant_names[ch] = [items[2]]
					variant_positions[ch] = [int(items[1])]
					ref[ch] = [items[3]]
					alt[ch] = [items[4]]
					hap_matrix_d1[ch] = []
					hap_matrix_d2[ch] = []
					genotype = items[9:]
					for i in range(len(genotype)):
						m = re.search('([0-9])\|([0-9])',genotype[i])
						hap_matrix_d1[ch].append(int(m.group(1)))
						hap_matrix_d2[ch].append(int(m.group(2)))
				else:
					variant_names[ch].append(items[2])
					variant_positions[ch].append(int(items[1]))
					ref[ch].append(items[3])
					alt[ch].append(items[4])
					genotype = items[9:]
					for i in range(len(genotype)):
						m = re.search('([0-9])\|([0-9])',genotype[i])
						hap_matrix_d1[ch].append(int(m.group(1)))
						hap_matrix_d2[ch].append(int(m.group(2)))

	for ch in chromosome:
		hap_matrix_d1[ch] = np.reshape(np.asarray(hap_matrix_d1[ch],dtype=int),(len(variant_names[ch]),len(ind_names)))
		hap_matrix_d2[ch] = np.reshape(np.asarray(hap_matrix_d2[ch],dtype=int),(len(variant_names[ch]),len(ind_names)))
		print(ref[ch][:20])
		print(alt[ch][:20])
		print(variant_positions[ch][:20])
		sys.exit()

	return(ind_names,hap_matrix_d1,hap_matrix_d2,variant_names,variant_positions,ref,alt,chromosome)

def vcf_processing(vcf):
	hap_matrix_d1 = {} #haplotype 1 of individuals, key as chromosome number
	hap_matrix_d2 = {} #haplotype 2 of individuals, key as chromosome number
	variant_names = {}
	variant_positions = {} #key as chromosome number
	chromosome = [] #key as chromosome number and value as number of SNPs per chromosome
	ref = {}
	alt = {}

	vcf_ = VCF(vcf)
	ind_names = vcf_.samples

	for v in vcf_:
		ch = v.CHROM
		if ch not in chromosome:
			chromosome.append(ch)
			variant_names[ch] = []
			hap_matrix_d1[ch] = []
			hap_matrix_d2[ch] = []
			variant_positions[ch] = []
			ref[ch] = []
			alt[ch] = []

		if v.ID == None:
			variant_names[ch].append(str(ch) + "_" + str(v.POS))
		else:
			variant_names[ch].append(v.ID)		
		variant_positions[ch].append(v.POS)
		hap_matrix_d1[ch].append(np.array(v.genotypes, dtype=np.int16)[:,0])
		hap_matrix_d2[ch].append(np.array(v.genotypes, dtype=np.int16)[:,1])
		ref[ch].append(v.REF)
		alt[ch].append(v.ALT[0])

		phased = np.sum(np.array(v.genotypes, dtype=np.int16)[:,2])

		if phased != len(np.array(v.genotypes, dtype=np.int16)[:,0]):
			sys.exit("FOUND unphased genotype, please make sure used completely phased vcf")
		if len(v.ALT) > 1:
			sys.exit("Found multi-allelic variants, please make sure used biallelic vcf")

	for ch in chromosome:
		hap_matrix_d1[ch] = np.vstack(hap_matrix_d1[ch])
		hap_matrix_d2[ch] = np.vstack(hap_matrix_d2[ch])

	return(ind_names,hap_matrix_d1,hap_matrix_d2,variant_names,variant_positions,ref,alt,chromosome)

