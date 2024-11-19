# Controlling maximum cell ploidy

SPRINTER automatically infers single-cell copy numbers based on a maximum value for the identified modal copy number.
The modal copy number is the most common copy number identified across the whole genome of each cell.
By default, the maximum value for the modal copy number is `4` which reflects what has been identified in previous cancer studies.
However, certain tumours can have particularly high values of ploidy (for example, due to the accumulation of multiple whole-genome doublings that are retained in the cancer genomes).
Therefore, the user can increase the maximum value `MAXPLOIDY` of the modal copy number when there is any evidence or indication that the analysed cells might have higher tumour ploidy.
This can be achieved using the following SPRINTER flag

```shell
--maxploidy MAXPLOIDY
```

It is recommended to try re-running SPRINTER in parallel with increased maximum ploidy if the cancer cells are inferred with a modal copy number close to the maximum and if the cells appear to have a particularly instable genome in terms of copy-number alterations.
The raw ploidies inferred for each cell can be explored using the output figure `raw_G1G2_clones.png`.
When many cells with very high modal copy numbers are identified, SPRINTER can be re-run with `--maxploidy 5` or `--maxploidy 6` and the results can be compared based on the levels of noisy in the inferred copy numbers.
