# Vary bin sizes

SPRINTER calculates a cell-specific bin size for each cell based on a target number of reads expected per bin.
Also, two distinct sets of bins are defined for either the replication or copy-number analyses.
The default values of the target number of reads are based on previous studies, showing that those number of reads are sufficient for accurate analyses.
The calculated average of bin sizes are displayed in the SPRINTER output log as follows:

```shell
[XXXX-XXX-XX XX:XX:XX]> Average width of RT bins: 383502
[XXXX-XXX-XX XX:XX:XX]> Average width of CN bins: 4800544
```

However, the values can be easily modified and tuned by each user using two SPRINTER flags:
1. `--rtreads RTREADS` where `RTREADS` is the target number of reads used for replication-timing analyses; for example `--rtreads 100`.
2. `--cnreads CNREADS` where `CNREADS` is the target number of reads used for copy-number analyses; for example `--cnreads 1000`.

While `RTREADS` should fit the purposes for most analyses, `CNREADS` can be varied in order to achieve a copy-number bin size appropriate for the resolution of the copy-number events that are expected or that should be investigated.
Also, multiple runs of SPRINTER can be compared using different values of `CNREADS` to identify the best tradeoff for the analysed data.
