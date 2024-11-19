# Selecting copy-number clones

SPRINTER identifies clones based on the inferred copy numbers per individual cell.
Two parameters influence the selection of clones in SPRINTER: the minimum number of cells per clone, and the maximum cell-to-cell distance between cells expected in the same clone.

## Minimum number of cells

First, the minimum number of cells per clones determines which groups of cells with similar copy numbers should be used to define selected clones.
This number is output in the SPRINTER output log with the following message:
```shell
[XXXX-XXX-XX XX:XX:XX]> Chosen mininimum number of cells per clone: 20
```
The minimum number of cells is chosen by SPRINTER using two user defined parameters:
1. `--minnumcells MINNUMCELLS` where `MINNUMCELLS` is the minimum expected number of cells to define clones;
2. `--minpropcells MINPROPCELLS` where `MINPROPCELLS` is the minimum proportion of cells to define clones.

The actual minimum number of chosen cells to define clones is thus defined as the `min(MINNUMCELLS, TOT_CELLS * MINPROPCELLS)` where `TOT_CELLS` is the total number of analysed cells.
The user can tune which clones should be selected by modying these parameters, for example with `--minnumcells 20 --minpropcells 1.0` only clones with at least 20 cells will be chosen (`MINPROPCELLS`=1.0 disables the use of the fraction of total cells in the choice).

## Maximum cell-to-cell distance

Second, the maximum cell-to-cell distance between cells expected in the same clone is calculated.
This represents the expected fraction of copy-number errors.
SPRINTER automatically infers this value using a mixture distribution, represented in the output plot `clone_est_threshold.png`.
This value is then used to identify copy-number groups of cells among G0/1/2 phase cells, among which clones are selected with the parameters reported above.
The result of this grouping is represented in the output figure `raw_G1G2_clones.png` and the final selected clones are depicted in another output figure `final_G1G2_clones.png`.
This method is able to identify accurate clones in most settings; however, the method might perform less accurately in contexts with particularly high copy-number error rates.
If the inspection of the figures above suggests that the method to identify groups has not worked as expected, or in case of other tuning required, the user can provide the value `CLONETHRESHOLD` of the maximum cell-to-cell distance using the following debug paramter, by appending this to the end of SPRINTER command:
```shell
dev --clonethreshold CLONETHRESHOLD
```
Moreover, pre-computed clones for all cells, that can be calculated with other methods can be provided to SPRINTER with the following development argument:
```shell
dev --fixclones FIXCLONES
```
where `FIXCLONES` is the path to a TSV dataframe containing fields `CELL` and corresponding `CLONE`.
