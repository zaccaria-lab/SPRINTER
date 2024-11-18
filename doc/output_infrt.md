# Output file `rtinferred_clones.tsv.gz`

A compressed TSV dataframe containing the inferred altered replication timing across all genomic regions for every identified clone, with the following columns.

| **Columns** | **Description** |
|-------------|-----------------|
| CHR | Chromosome |
| START | Start position of the genomic region |
| END | End position of the genomic region |
| CLONE | SPRINTER inferred clone |
| GENOME | Linear genomic coordinate |
| consistent_rt | Reference replication timing classification |
| CNRT_PROFILE | Clone-specific replication-timing profile (RTP) |
| RT_INFERRED | Inferred replication timing classification |
| ALTERED_RT | Whether SPRINTER inferred altered replication timing |
