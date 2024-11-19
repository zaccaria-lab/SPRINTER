# Tune GC-bias correction

SPRINTER corrects GC-content bias in each cell independently by considering sperately early and late replicating genomic regions.
The GC-bias correction inferred in the two groups are aggregated using a less or more strict statistic.
By default, the less strict correction is used and has been demonstrated to accurately work in most datasets.
In case of particularly noisy datasets with high levels of GC-content bias, the statistic could however identify slightly more S phase cells than expected.
To correct for this, SPRINTER automatically uses the more strict statistic when high GC-content bias is identified.
However, the user can force the use of this more strict statistic with the following argument:
```shell
--strictgc
```
The extent of the identified GC-content bias can be explored in different genomic regions by analysing the output figures `gccorr_*`.
Forcing a strict GC-bias correction can be useful in datasets with particularly high GC-content bias to avoid affecting the identification of S phase cells.
