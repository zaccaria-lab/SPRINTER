# Demo on running SPRINTER
: ex: set ft=markdown ;:<<'```shell' #

The following SPRINTER demo represents a guided example of running SPRINTER on the required input.
This demo uses [example data](https://doi.org/10.5281/zenodo.14060547) previous generated from the diploid and tetraploid ground truth datasets, generated in the related SPRINTER manuscript.
From this directory, simply run this file through BASH as a standard script to run the complete demo.
The demo can also be considered as a guided example to be executed line-by-line.

## Requirements and set up

The demo requires the SPRINTER algorithm to be installed, preferably through [conda]().
Specifically, please make sure that you can run the command `sprinter` in an environment activated before running this demo.
The demo includes the downloading of all the required files and will terminate in <20 minutes on machine with minimum requirements satisfied.

We also ask the demo to terminate in case of errors and to print a trace of the execution by the following commands
```shell
set -e
set -o xtrace
PS4='[\t]'
:<<'```shell' # Ignore this line
```

## Downloading of data

The demo auomatically downloads the required input files.
By default, it downloads and runs the tetraploid demo, and the diploid demo can be downloaded and run by changing the variable as indicated below.

```shell
export PLOIDY='tetraploid' # Change this to 'diploid' for running the diploid demo.
echo "Downloading ${PLOIDY} input file."
curl -L 'https://zenodo.org/records/14060548/files/gt_'${PLOIDY}'.tsv.gz?download=1' > sprinter.input.gt_${PLOIDY}.tsv.gz
export INPUT="sprinter.input.gt_${PLOIDY}.tsv.gz"
:<<'```shell' # Ignore this line
```

SPRINTER can be run just with the required input, however it is recommended to also provide the correct reference genome in order to compute the accurate GC content of all genomic regions.
As such, the demo download the corresponding reference genome hg19.

```shell
echo "Downloading human reference genome, please be patient as downloading time may vary."
curl -L https://hgdownload.cse.ucsc.edu/goldenpath/hg19/bigZips/analysisSet/hg19.p13.plusMT.no_alt_analysis_set.fa.gz | gzip -d > hg19.fa
export REF="hg19.fa"
:<<'```shell' # Ignore this line
```

## Generating SPRINTER input

SPRINTER is now run directly on the required input and providing the corresponding reference genome.
SPRINTER can be run with default parameters, or these can be changed according to different experimental settings.
For example, `cnreads` is the target number of reads to form bins to call CNAs in single cells and it controls the resolution of the inferred CNAs; for example, in this demo it set to 1000 reads that will provide smaller bins and thus resolution to infer smaller CNAs.
The results can be compared with other values like 2000.
Also, `minnumcells` defines the minimum number of cells to select clones, and together with `minpropcells` is used to define the size of clones that are selected (i.e., min(minnumcells, tot_num_of_cells * minpropcells)).
These values can be adjusted according to the experimental requirements and here `minnumcells` is set to 15.
Finally, please make sure to add the option `-j J` with J being the number of available cores, set to 8 as an example here but it should be increased to the maximum number of available processors for efficiency purposes.

```shell
sprinter ${INPUT} \
         --refgenome ${REF} \
         --minreads 100000 \
         --rtreads 200 \
         --cnreads 1000 \
         --minnumcells 15 \
         -j 8
echo "END"
exit $? # Ignore this line
```
