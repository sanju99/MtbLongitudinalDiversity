#!/bin/bash
#SBATCH -c 1
#SBATCH -t 0-11:59
#SBATCH -p short
#SBATCH --mem=10G
#SBATCH -o /home/sak0914/Errors/zerrors_%j.out
#SBATCH -e /home/sak0914/Errors/zerrors_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=skulkarni@g.harvard.edu

set -o errexit # any error will cause the shell script to exit immediately. This is not native bash behavior

if ! [ $# -eq 1 ]; then
    echo "Please pass in 1 command line argument: a text file of the full names of BAM files to compress to CRAM"
    exit
fi

input_file=$1

# Read the TSV file line by line, skiping the header. IFS sets the field separator. Here, it is tab
while IFS=$'\t', read -r fName
do

    if [ -f $fName ] && [ ! -f "$fName.gz" ]; then
        gzip -c $fName > "$fName.gz"
        rm $fName
        echo "Finished gzipping $fName"
    else
        echo "Already finished $fName"
    fi

done < "$input_file"