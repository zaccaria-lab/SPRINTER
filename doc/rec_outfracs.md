# Vary the fraction of outlying cells

SPRINTER is given an expected fraction of outlying cells that will be excluded from the assignment to clones; this fraction is equally applied to both S and G1/2 phase cells in order to preserve their accurate fractions.
The fraction is required because a certain fraction of outlying cells is expected in most single-cell experiments.
This fraction can be specified and tuned with the following SPRINTER flag:

```shell
--propsphase PROPSPHASE
```

where `PROPSPHASE` is the fraction of cells to preserve; that is, `1 - PROPSPHASE` corresponds to the fraction of outlying cells.
For example, `--propsphase 0.8` will preserve 80% of cells and exclude from the clone assignment as outliers the 20%.
According to the analysed data, this fraction can be increased or decreased.
So, it is reccomended to re-run SPRINTER with a either higher or lower thresholds in the case that the inferred copy numbers appear less or more noisy in the final SPRINTER results for outlying cells (represented with the brown colour in the final SPRINTER figure).
