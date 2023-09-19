# SPRINTER <br/> <sub><u>S</u>ingle-cell <u>P</u>roliferation <u>R</u>ate <u>I</u>nference in <u>N</u>on-homogeneous <u>T</u>umours through <u>E</u>volutionary <u>R</u>outes</sub> #

SPRINTER is an algorithm that uses single-cell whole-genome DNA sequencing data to enable the accurate identification of actively replicating cells in both the S and G2 phases of the cell cycle and their assignment to distinct tumour clones, thus providing a proxy to estimate clone-specific proliferation rates.

This repository includes detailed instructions for installation and requirements, demos, a list of current issues, and contacts.
A fully reproducible capsule for testing SPRINTER is available in CodeOcean at:

[SPRINTER's CodeOcean capsule](https://codeocean.com/capsule/9392115)

## Contents ##

1. [Quick start](#quick)
2. [Setup](#setup)
3. [Usage](#usage)
    - [Input data](#requireddata)
    - [System requirements](#requirements)
    - [Demos](#demos)
4. [Contacts](#contacts)

<a name="quick"></a>
## Quick start

During review, the installation and execution of SPRINTER can be tested and the previously tested automatic runs reviewed using the reproducible capsule available in CodeOcean at:

[SPRINTER's CodeOcean capsule](https://codeocean.com/capsule/9392115)

<a name="setup"></a>
## Setup

SPRINTER is written in Python3 and will be packaged and distributed through [bioconda](https://bioconda.github.io/). During review, SPRINTER installation has been automatically tested using the available [SPRINTER CodeOcean capsule](https://codeocean.com/capsule/9392115) and can be installed directly from source following the instructions below.

1. [Manual installation](#manual): installs SPRINTER from source with conda.
2. [Basic requirements](#reqs): list of requirements.

<a name="manual"></a>
### Manual installation

SPRINTER can be installed manually in three steps by creating a custom environment with `conda` (that can be installed locally on any machine using either the compact [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or the complete [Anaconda](https://www.anaconda.com/)).

First, SPRINTER requires `bioconda` [requires](https://bioconda.github.io/), which can be used by setting the following channels in this exact order:
```shell
conda config --add channels defaults
conda config --add channels bioconda
conda config --add channels conda-forge
conda config --set channel_priority strict
```

Next, a custom `bioconda` environment with the required packages can be created using the following command
```
conda create -n sprinter -c bioconda python=3.9 numpy pandas scipy statsmodels hmmlearn matplotlib-base pybedtools scikit-learn seaborn
```

Lastly, `sprinter` can be used within the created environment (`conda activate sprinter`) by cloning this repository and using the following command
```shell
python bin/sprinter.py
```
as demonstrated in the [SPRINTER CodeOcean capsule](https://codeocean.com/capsule/9392115).

<a name="reqs"></a>
### Basic requirements

SPRINTER depends on the following standard python packages, which must be available in the python environment where the user runs SPRINTER.

| Package | Tested version |
|---------|----------------|
| hmmlearn | 0.3.0 |
| matplotlib-base | 3.7.2 |
| numpy | 1.25.2 |
| pandas | 2.1.0 |
| pybedtools | 0.9.1 |
| scikit-learn | 1.3.0 |
| scipy | 1.11.2 |
| seaborn | 0.12.2 |
| statsmodels | 0.14.0 |

<a name="usage"></a>
## Usage

1. [Required data](#requireddata)
2. [System requirements](#requirements)
3. [Demos](#demos)

<a name="requireddata"></a>
### Required input

SPRINTER requires a single input, which is a TSV dataframe file (which can be `gz` compressed) containing single-cell read counts per 50kb genomic regions across autosomes (the same as those specified in the [RT file](data/ext/rtscores.csv.gz) included in this repository).
This file can be automatically created using the `chisel_rdr` command of [CHISEL](https://github.com/raphael-group/chisel) starting from a standard barcoded single-cell BAM file.

In detail, the input TSV dataframe file has to contain the following columns:

| **Name** | **Description** |
|---------|----------------|
| CHROMOSOME | the name of a chromosome |
| START | the start coordinate of a genomic bin |
| END | the end coordinate of the genomic bin |
| CELL | the name of a cell |
| NORMAL | the number of sequencing reads from the matched-normal sample for the bin |
| COUNT | the number of sequencing reads from the cell CELL in the bin |
| RDR | the estimated RDR |

<a name="requirements"></a>
### System requirements

SPRINTER is highly parallelised in order to make the extensive computations performed for each cell efficient, often splitting independent computations to parallel processes. We recommend executing SPRINTER on multi-processing computing machines. The minimum system requirements that we have tested for running the demos are:
- CPU with at least 2 virtual cores
- 12GB of RAM

However, input data with higher number of cells will require machines with more memory (>50GB) and more processors (>12) to make the execution efficient.

<a name="demos"></a>
### Demos

Two demos applying SPRINTER on 1000 diploid and tetraploid cells from the generated ground truth are available in the [SPRINTER's CodeOcean capsule](https://codeocean.com/capsule/9392115).

<a name="contacts"></a>
## Contacts

SPRINTER's repository is actively maintained by Olivia Lucas, PhD student at the UCL Cancer Institute, and [Simone Zaccaria](https://simozacca.github.io/), group leader of the [Computational Cancer Genomics research group](https://www.ucl.ac.uk/cancer/zaccaria-lab) at the UCL Cancer Institute.
