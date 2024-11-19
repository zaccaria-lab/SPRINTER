# Select cells to analyse

SPRINTER selects the cells to analyse only based on the total number of sequencing reads.
This is a SPRINTER argument that can be provided with the following argument

```shell
--minreads MINREADS
```

where `MINREADS` is the minimum number of total sequencing reads that a cell has to have in order to be selected for the SPRINTER analyses.
This paramter can be tuned and provided by the user, for example with `--minreads 100000`.
