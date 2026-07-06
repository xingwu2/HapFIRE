<img width="9024" height="3718" alt="Untitled-9" src="https://github.com/user-attachments/assets/86fa1936-4523-471a-8813-e8b73136e7df" />

# HapFIRE

HapFIRE is a python-based pipeline to calculate allele and genotype frequencies from pool-seq data given founder SNP VCF file.

Please cite the following papers for HapFIRE

* Xing Wu et al. ,Rapid adaptation and extinction in synchronized outdoor evolution experiments of Arabidopsis.Science391,eadz0777(2026).DOI:10.1126/science.adz0777

* Kessner, Darren, Thomas L. Turner, and John Novembre. "Maximum likelihood estimation of frequencies of known haplotypes from pooled sequence data." Molecular biology and evolution 30.5 (2013): 1145-1158.


## Installation
HapFIRE relies on HARP to estimate the haplotype likelihood. To make everyone's life easier, we have included a pre-compiled version of harp (harp_linux_140925_103521). If you prefer to install another version, please replace the executable in the src folder. 

Install via mamba
```
mamba env create -f environment.yml
conda activate hapfire
```

## Required input
* Phased SNP vcf file as the reference panel
* sorted and index pool seq alignment bam file
* reference genome fasta file

## Options

    -v the phased vcf file 
    -b the sorted alignment bam file 
    -f reference fasta file 
    -s (calculate haplotype frequencies) (True/False ,default: False. ) This step will be very slow and computationally expensive if you perform genome-wide fine block partition 
    -p genome-wide block fine block partition algorithm (bigld) 
    -c the LD r2 cutoff for more defined blocks (default: 0.5) 
    -w window size for calculating independent blocks (default: 100) 
    -r the LD r2 cutoff for independent blocks (default: 0.1) 
    -o prefix of output files 

## Usage
The most common usage of hapfire is to estimate allele and genotype frequency from pool sequencing data

```
python3 hapFIRE.py -v example/example.recode.vcf -b example/example.bam -f example/example.fa -o test
```

If you want to estimate haplotype frequency, HapFIRE will first perform genome-wide block partition and then cluster unique haplotypes from each block, and then estimate haplotype cluster frequencies

```
python3 hapFIRE.py -v example/example.recode.vcf -b example/example.bam -f example/example.fa -s True -p bigld -o test
```

## Program output
* .snp_frequency.txt: The estimated frequencies of alternative alleles in the reference panel from the pool seq data.
* .ecotype_frequency.txt: The estimated frequencies of all genotypes in the reference panel from the pool seq data.
* .independent_genomewide_partition.txt: independent haplotype blocks (LD r2 0.1) identified by hapFIRE
* .fine_genomewide_partition.txt: more defined haplotype blocks identified by user specification (usually bigLD)
* .unique_haplotype_frequency.txt: The estimated frequencies of unique haplotypes with **-s** specified
* .clustered_haplotype_frequency.txt The estimated frequencies of clustered haplotypes with **-s** and user specified haplotype clustering algorithm
