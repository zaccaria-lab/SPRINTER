import os
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.append(os.path.join(os.path.realpath(os.path.dirname(__file__)), '../libs/'))
from utils import *

import argparse
from multiprocessing import cpu_count
from io import StringIO

from statsmodels.stats.multitest import multipletests

import gccorrect as gc
import cellinfer as ci
import preprocess as pp
import callcn as cc
import inferclones as ic
import assignrepcells as ar
import inferG2 as ig



def parse_args(args=None, inputdata=None):
    description = "SPRINTER algorithm"
    parser = argparse.ArgumentParser(description=description)
    if inputdata is None:
        parser.add_argument("DATA", type=str, help='Input data as a CSV/TSV dataframe with fields: CELL, CHR, START, END, GENOME, RDR, consistent_rt, and GC')
    parser.add_argument("-e", "--segtest", required=False, default='CHR', type=str, help='Definition of segment to use for doing the test within [CHR, JOINT_SEG, MERGED_CN_STATE, CHR_MERGED_CN_STATE, WHOLE_GENOME] (default: CHR)')
    parser.add_argument("-s", "--summarystat", required=False, default='special', type=str, help='Method for measuring overlap between distributions (default: special)')
    parser.add_argument("-m", "--meanmethod", required=False, default='weighted_min', type=str, help='Mean method across segs to use in output of test stat: weighted_min, normal, or weighted_sum (default: weighted_min)')
    parser.add_argument("-g", "--gccorr", required=False, type=str, default='QUANTILE', help='GC correction mode to be used (default: QUANTILE, available methods are QUANTILE and MODAL)')
    parser.add_argument("-q", "--quantile", required=False, type=float, default=0.05, help='Left-side quantile for special summary stat for test (default: 0.05)')
    parser.add_argument("-a", "--alpha", required=False, type=float, default=0.05, help='Level of significance (default: 0.05)')
    parser.add_argument("-j", "--jobs", required=False, default=cpu_count(), type=int, help='Number of parallel jobs (default: all available)')
    parser.add_argument("-o", "--output", required=False, default=None, type=str, help='Name of output (default: stdout)')
    parser.add_argument("-O", "--outprofiles", required=False, default=None, type=str, help='Name of output for rtprofiles (default: not saved)')
    parser.add_argument("--repliseq", required=False, default='repliseq', type=str, help='Repliseq samples to use (default: repliseq, or it can be allnormal or all)')
    parser.add_argument("--subsample", required=False, type=int, default=None, help="Random subsample of cells (default: use all cells)")
    parser.add_argument("--minreads", required=False, type=int, default=100000, help="Minimum number of reads for cells (default: 100000)")
    parser.add_argument("--rtreads", required=False, type=int, default=100, help="Target RT number of reads (default: 100)")
    parser.add_argument("--cnreads", required=False, type=int, default=2000, help="Target CN number of reads (default: 2000)")
    parser.add_argument("--minrtreads", required=False, type=int, default=5, help="Min number of reads (default: 5)")
    parser.add_argument("--minfracbins", required=False, type=float, default=0.5, help="Min fraction of raw bins (default: 0.5)")
    parser.add_argument("--combinert", required=False, type=int, default=None, help="Force size of RT bins (default: None)")
    parser.add_argument("--combinecn", required=False, type=int, default=None, help="Force size of CN bins (default: None)")
    parser.add_argument("--maxgap", required=False, type=int, default=3, help="Maximum gap to allow in RT binning (default: 3)")
    parser.add_argument("--maxploidy", required=False, type=int, default=4, help="Maximum ploidy allowed in CN calling (default: 4)")
    parser.add_argument("--seed", required=False, type=int, default=None, help="Random seed for replication (default: None)")
    parser.add_argument("--nortbinning", required=False, default=False, action='store_true', help='Do not apply RT binning (default: False)')
    parser.add_argument("--nogcaggregate", required=False, default=False, action='store_true', help='Do not compute GC biases by aggregating and correcting (default: False)')
    parser.add_argument("--nocorrgcintercept", required=False, default=False, action='store_true', help='Do not correct GC intercept per replication time (default: False)')
    parser.add_argument("--commaseparator", required=False, default=False, action='store_true', help='Use comma separators (default: tab separators)')
    parser.add_argument("--clonethreshold", required=False, type=float, default=None, help="Fix threshold to find clones (default: None)")
    parser.add_argument("--minnocells", required=False, type=int, default=20, help="Min number of cells to define clones (default: 20)")
    parser.add_argument("--rescuethreshold", required=False, type=float, default=None, help="Fix threshold to rescue noisy cells (default: None)")
    parser.add_argument("--pvalcorr", required=False, default='fdr_by', type=str, help='Pval correction method (default: fdr_by)')
    parser.add_argument("--propsphase", required=False, type=float, default=0.7, help="Proportion of S phase and nonreplicating cells to assign to clones (default: 0.7)")
    parser.add_argument("--strictgc", required=False, default=False, action='store_true', help='Use stricter GC correction for more conservative S phase identification (default: False)')

    args = parser.parse_args(args)

    if inputdata is None and not os.path.isfile(args.DATA):
        raise ValueError('Data file does not exist')
    if args.seed and args.seed < 1:
        raise ValueError("The random seed  must be positive!")
    if args.segtest not in ['CHR', 'JOINT_SEG', 'MERGED_CN_STATE', 'CHR_MERGED_CN_STATE', 'WHOLE_GENOME']:
        raise ValueError("The segment definition must be one within [CHR, MERGED_CN_STATE, CHR_MERGED_CN_STATE, WHOLE_GENOME]!")
    if args.gccorr not in ['QUANTILE', 'MODAL', 'TEST']:
        raise ValueError("The gcorr method must be one within [QUANTILE and MODAL]!")
    if args.repliseq not in ['repliseq', 'allnormal', 'all']:
        raise ValueError("Unknown repliseq value provided: {}, it must be either repliseq, allnormal, or all!".format(args.repliseq))
    if args.maxgap < 1:
        raise ValueError("Maxgaps value must be at least 1!")
    if args.maxploidy < 1:
        raise ValueError("Maxploidy value must be at least 1!")

    return {
        "data" : 'provided' if inputdata is not None else os.path.abspath(args.DATA),
        "segtest" : args.segtest,
        "jobs" : args.jobs,
        "summarystat" : args.summarystat,
        "meanmethod" : args.meanmethod,
        "commasep" : args.commaseparator,
        "output" : args.output,
        "outprofiles" : args.outprofiles,
        "subsample" : args.subsample,
        "seed" : args.seed,
        "gccorr" : args.gccorr,
        "quantile" : args.quantile,
        "alpha" : args.alpha,
        "rt_reads" : args.rtreads,
        "combine_rt" : args.combinert,
        "min_rt_reads" : args.minrtreads,
        "min_frac_bins" : args.minfracbins,
        "cn_reads" : args.cnreads,
        "combine_cn" : args.combinecn,
        "repliseq" : args.repliseq,
        "minreads" : args.minreads,
        "nortbinning" : args.nortbinning,
        "nogcaggregate" : args.nogcaggregate,
        "nocorrgcintercept" : args.nocorrgcintercept,
        "maxgap" : args.maxgap,
        "maxploidy" : args.maxploidy,
        "clonethreshold" : args.clonethreshold,
        "minnocells" : args.minnocells,
        "rescuethreshold" : args.rescuethreshold,
        "pvalcorr" : args.pvalcorr,
        "propsphase" : args.propsphase,
        "strictgc" : args.strictgc
    }


def main(args=None, inputdata=None):
    log('Parsing and checking arguments', level='STEP')
    args = parse_args(args, inputdata)
    np.random.seed(args['seed'])
    log('\n'.join(['Arguments:'] + ['\t{} : {}'.format(a, args[a]) for a in args]), level='INFO')

    log('Processing input data and defining bins', level='STEP')
    data, rtprofiles, excluded = pp.process_input(args, inputdata)
    log('> Selected cells: {}'.format(data['CELL'].nunique()), level='INFO')
    log('> Cells discarded because not enough reads: {}'.format(excluded['CELL'].nunique()), level='INFO')
    log('> Average width of RT profiles: {:.1f}'.format((rtprofiles['END'] - rtprofiles['START']).mean()), level='INFO')
    get_size = (lambda D : (D['END'] - D['START']).mean())
    log('> Average width of RT bins: {:.1f}'.format(get_size(data.groupby(['CELL', 'BIN_REPINF']).agg({'START' : 'min', 'END' : 'max'}))), level='INFO')
    log('> Average width of CN bins: {:.1f}'.format(get_size(data.groupby(['CELL', 'BIN_CNSINF']).agg({'START' : 'min', 'END' : 'max'}))), level='INFO')
    data.to_csv('bins.tsv.gz', sep='\t', index=False)

    gcbiases = None
    if not args["nogcaggregate"]:
        log('Estimating GC bias across all cells', level='STEP')
        gcbiases, _ = gc.estimate_gcbiases(rtprofiles, gccorr=args['gccorr'], jobs=args['jobs'])

    log('Running inference across all cells', level='STEP')
    pvals, rtprofiles = ci.run(_data=rtprofiles,
                               _seg=args['segtest'],
                               _testmethod=args['summarystat'],
                               _meanmethod=args['meanmethod'],
                               jobs=args['jobs'],
                               gcbiases=gcbiases,
                               gccorr=args['gccorr'],
                               strictgc=args['strictgc'],
                               nocorrgcintercept=args['nocorrgcintercept'],
                               _quantile=args['quantile'])
    pvals = pvals.sort_values('CELL')
    rtprofiles = rtprofiles.sort_values(['CELL', 'CHR', 'START', 'END'])
    rtprofiles.to_csv('raw_rtprofiles.tsv.gz', sep='\t', index=False)

    log('Applying multiple hypothesis correction', level='STEP')
    pvals = corrections(pvals, args['summarystat'], args['alpha'], method=args['pvalcorr'])
    log('> S-phase fraction: {:.1%}'.format(pvals['IS-S-PHASE'].sum() / len(pvals)), level='INFO')
    pvals.to_csv('pvals.tsv.gz', sep='\t')

    log('Correcting S-phase cells for copy-number calling', level='STEP')
    rtprofiles = cc.correct_sphase(rtprofiles, pvals)
    rtprofiles.to_csv('raw_rtprofiles.tsv.gz', sep='\t', index=False)

    log('Calling copy numbers for G1/G2 cells', level='STEP')
    cn_g1g2 = cc.call_cns_g1g2(data, rtprofiles, pvals, max_ploidy=args['maxploidy'], jobs=args['jobs'])
    cn_g1g2.to_csv('cn_g1g2.tsv.gz', sep='\t', index=False)

    log('Infering G1/G2 clones', level='STEP')
    clones_g1g2, normal_clones, ploidy_ref = ic.infer_clones(cn_g1g2, pvals, clone_threshold=args['clonethreshold'], min_no_cells=args['minnocells'])
    log('> Base tumour ploidy identified for most cancer cells: {}'.format(ploidy_ref), level='INFO')
    clones_g1g2.to_csv('clones_g1g2.tsv.gz', sep='\t', index=False)

    log('Assigning S phase cells to clones', level='STEP')
    cn_all, clones_all = ar.assign_s_clones(data, rtprofiles, pvals, cn_g1g2, clones_g1g2, normal_clones, rescue_threshold=args['rescuethreshold'], prop_sphase=args['propsphase'], jobs=args['jobs'])
    cn_all.to_csv('cn_all.tsv.gz', sep='\t', index=False)
    clones_all.to_csv('clones_all.tsv.gz', sep='\t', index=False)

    log('Infering G2 cells', level='STEP')
    annotations = ig.infer_G2(data, pvals, clones_all, cn_all, normal_clones, jobs=args['jobs'])

    log('Writing output', level='STEP')
    annotations = pd.concat((annotations, excluded), axis=0, ignore_index=True, sort=False)
    res = None
    if args['output'] is not None:
        annotations.to_csv(args['output'], sep=',' if args['commasep'] else '\t', index=False)
    else:
        res = annotations

    log('KTHXBYE', level='STEP')
    return res


def corrections(pvals, summarystat, alpha, method='fdr_by'):
    pvals['REP_INF_EMP_BH'] = multipletests(pvals['PVAL_EMP'].values, alpha=alpha, method=method)[0]
    pvals['REP_INF_FIT_BH'] = multipletests(pvals['PVAL_FIT'].values, alpha=alpha, method=method)[0]
    pvals['REP_INF_COMB_BH'] = multipletests(pvals['PVAL_COMB'].values, alpha=alpha, method=method)[0]
    pvals['IS-S-PHASE'] = (pvals['REP_INF_COMB_BH'] & pvals['IS_MAGENTA_HIGHER']) if summarystat not in ['mean', 'esmean', 'median', 'esmedian', 'special'] else pvals['REP_INF_COMB_BH']
    return pvals


if __name__ == '__main__':
    pvals = main()
    if pvals is not None:
        output = StringIO()
        pvals.to_csv(output, sep='\t', index=False)
        output.seek(0)
        print(output.read())
