# Output file `cn_all_corrected.tsv.gz`

A compressed TSV dataframe containing the inferred copy numbers across all genomic regions for every analysed cell, with the following columns.

| **Columns** | **Description** |
|-------------|-----------------|
| CELL | Analysed cell |
| CHR | Chromosome |
| START | Start position of the genomic region |
| END | End position of the genomic region |
| GENOME | Linear genomic coordinate |
| GC | GC content |
| consistent_rt | Reference replication timing classification |
| RDR | Read depth ratio used for replication timing analyses |
| RDR_CN | Read depth ratio used for copy-number analyses |
| CN_TOT | SPRINTER inferred copy number for the analysed cell in the corresponding genomic region |
| CN_CLONE | SPRINTER inferred copy number for the assigned, corresponding clone in the corresponding genomic region |
| BIN_REPINF | Genomic bin for replication timing analyses |
| BIN_CNSINF | Genomic bin for copy-number analyses |
| BIN_GLOBAL | Genomic bin for copy-number analyses across all cells |
| MERGED_CN_STATE | Copy-number state |
| RAW_CN_TOT | Uncorrected inferred copy number for the analysed cell in the corresponding genomic region |
| CLONE | Assigned, corresponding clone |
