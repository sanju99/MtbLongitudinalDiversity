#!/bin/bash
#SBATCH -c 8
#SBATCH -t 0-11:59
#SBATCH -p short
#SBATCH --mem=50G
#SBATCH -o /home/sak0914/Errors/zerrors_%j.out
#SBATCH -e /home/sak0914/Errors/zerrors_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=skulkarni@g.harvard.edu

conda activate /home/sak0914/anaconda3/envs/snakemake/envs/fastani

############## compute ANI between the two assemblies ##############
MAv_ASM_dir="/n/data1/hms/dbmi/farhat/rollingDB/TRUST/M_avium_ASMs"
out_dir="/home/sak0914/MtbLongitudinalDiversity/hybrid_assemblies/MAv"

# lengths: S0344-01: 5,575,292, S0346-01: 5,573,596. Differences = 1,696 bp
genome_1="$MAv_ASM_dir/S0344-01/FlyeAssembly_I3_PilonPolishing/pilon_IllPE_Polishing_I3_Asm_ChangeSNPsINDELsOnly/S0344-01.Flye.I3Asm.PilonPolished.fasta"
genome_2="$MAv_ASM_dir/S0346-01/FlyeAssembly_I3_PilonPolishing/pilon_IllPE_Polishing_I3_Asm_ChangeSNPsINDELsOnly/S0346-01.Flye.I3Asm.PilonPolished.fasta"

fastANI -q $genome_1 -r $genome_2 -o $out_dir/fastANI_S0344-01_S0346-01.out

############## Compute ANI to all publicly available complete <i>M avium</i> genomes (N = 86 on June 25, 2026 when downloaded) ##############
# get a file of the genome lengths
while read -r fasta; do
    length=$(awk '/^>/ {next} {sum += length($0)} END {print sum}' "$fasta")
    echo -e "${fasta}\t${length}" >> $out_dir/public_genome_lengths.txt
done < "$out_dir/MAv_NCBI_genomes.txt"

fastANI -q $genome_1 --rl $out_dir/MAv_NCBI_genomes.txt -o $(dirname "$genome_1")/fastANI_NCBI_genomes.out
fastANI -q $genome_2 --rl $out_dir/MAv_NCBI_genomes.txt -o $(dirname "$genome_2")/fastANI_NCBI_genomes.out

############## align the two assemblies to each other using minimap2 ##############
conda activate liftoff
minimap2 -ax asm5 $genome_1 $genome_2 > $out_dir/two_assembly_comparison/S0346-01_S0344-01.sam
minimap2 -ax asm5 $genome_2 $genome_1 > $out_dir/two_assembly_comparison/S0344-01_S0346-01.sam

conda activate bioinformatics
samtools view -b $out_dir/two_assembly_comparison/S0346-01_S0344-01.sam | samtools sort -o $out_dir/two_assembly_comparison/S0346-01_S0344-01.bam
samtools index $out_dir/two_assembly_comparison/S0346-01_S0344-01.bam

samtools view -b $out_dir/two_assembly_comparison/S0344-01_S0346-01.sam | samtools sort -o $out_dir/two_assembly_comparison/S0344-01_S0346-01.bam
samtools index $out_dir/two_assembly_comparison/S0344-01_S0346-01.bam

############## run genome-wide comparison using mummer ##############
conda activate circlator
mkdir -p $out_dir/mummer

nucmer -p $out_dir/mummer/alignment $genome_1 $genome_2

# highlight the SNP differences between them
mummerplot -p $out_dir/mummer/alignment --SNP $out_dir/mummer/alignment.delta -png

# dnadiff is part of mummer tools. Compute DNA differences between them
dnadiff $genome_1 $genome_2 -p $out_dir/two_assembly_comparison/dnadiff