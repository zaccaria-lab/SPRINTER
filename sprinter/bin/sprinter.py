import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from ..libs.utils import *

import argparse
from multiprocessing import cpu_count
from io import StringIO

from statsmodels.stats.multitest import multipletests

from ..libs import gccorrect as gc
from ..libs import cellinfer as ci
from ..libs import preprocess as pp
from ..libs import rtcorrect as rc
from ..libs import callcn as cc
from ..libs import inferclones as ic
from ..libs import assignrepcells as ar
from ..libs import inferG2 as ig
from ..libs import cncorrect
from ..libs import rtestimate
from ..libs import logo



def parse_args(args=None, inputdata=None):
    parser = argparse.ArgumentParser(description="The SPRINTER algorithm (Single-cell Proliferation Rate Inference in Non-homogeneous Tumors through Evolutionary Routes)\n\n{}".format(logo.SPRINTER_LOGO))

    # Main arguments
    if inputdata is None:
        parser.add_argument("DATA", type=str, help='Input data as a TSV dataframe of 50kb bins across all autosomes with fields: CHR, START, END, CELL, NORM_COUNT, COUNT, RAW_RDR')
    parser.add_argument("--refgenome", required=False, default=None, type=str, help='Path to indexed FASTA reference genome (default: using pre-computed GC counts from hg19; providing the reference genome is always reccommended)')    
    parser.add_argument("--minreads", required=False, type=int, default=100000, help="Minimum number of reads for cells (default: 100000)")
    parser.add_argument("--rtreads", required=False, type=int, default=200, help="Target number of reads used for replication-timing analyses, determining related bin size per cell (default: 200)")
    parser.add_argument("--cnreads", required=False, type=int, default=1000, help="Target number of reads used for copy-number analyses, determining related bin size per cell (default: 1000)")
    parser.add_argument("--minnumcells", required=False, type=int, default=20, help="Minimum number of cells to define clones (default: 20)")
    parser.add_argument("--minpropcells", required=False, type=float, default=0.1, help="Minimum proportion of cells to define clones (default: 0.1, the minimum between this and the minimum number is chosen)")
    parser.add_argument("--propsphase", required=False, type=float, default=0.7, help="Proportion of non-outliying S phase and nonreplicating cells to retain and assign to clones (default: 0.7)")
    parser.add_argument("--strictgc", required=False, default=False, action='store_true', help='Use stricter GC correction for more conservative S phase identification (default: False)')
    parser.add_argument("--maxploidy", required=False, type=int, default=4, help="Maximum mode copy number allowed in CN calling (default: 4, increase it when expecting particularly high ploidy for analysed tumour clones)")
    parser.add_argument("-q", "--quantile", required=False, type=float, default=0.05, help='Left-side quantile for SPRINTER summary statistic for identifying S-phase cells (default: 0.05)')
    parser.add_argument("-a", "--alpha", required=False, type=float, default=0.05, help='Level of significance (default: 0.05)')
    parser.add_argument("--repliseq", required=False, default='repliseq', type=str, help='Repliseq profiles to use (default: repliseq, or it can be allnormal or all)')
    parser.add_argument("--visual", required=False, default=False, action='store_true', help='Do not bootstrap RDRs within non-overlapping windows used for replication analysis, ideal for visualization purposes (default: False)')
    parser.add_argument("--visualcn", required=False, default=False, action='store_true', help='Do not bootstrap RDRs within non-overlapping windows used for CNA calling (default: False)')
    parser.add_argument("-j", "--jobs", required=False, default=cpu_count(), type=int, help='Number of parallel jobs (default: all available)')
    parser.add_argument("--seed", required=False, type=int, default=None, help="Random seed for replication (default: None)")
    parser.add_argument("-o", "--output", required=False, default='sprinter.output.tsv.gz', type=str, help='Name of output (default: sprinter.output.tsv.gz)')

    # Development arguments
    subparsers = parser.add_subparsers(dest='subcommand', help='Help for development arguments')
    devparser = subparsers.add_parser('dev', description='These are optional arguments mostly used for dev')
    # parser.add_argument("--fixploidy", required=False, default=None, type=str, help='Fixed ploidy as a CSV/TSV dataframe containing fields: CELL, PLOIDY (default: not used, ploidies are inferred by SPRINTER)')
    # parser.add_argument("--fixcns", required=False, default=None, type=str, help='Fixed baseline copy numbers as a CSV/TSV dataframe containing fields: CELL, CN_TOT, or with |-separated allele-specific CNAs in CN_STATE or HAP_CN (default: not used, copy numbers are inferred by SPRINTER)')
    devparser.add_argument("--rtscores", required=False, default=None, type=str, help='Path to replication score file (default: using pre-computed included file resources/rtscores.csv.gz)')    
    devparser.add_argument("--gapsfile", required=False, default=None, type=str, help='Path to mapping gaps file (default: using pre-computed included file resources/gaps_hg19.tsv)')    
    devparser.add_argument("--gccont", required=False, default=None, type=str, help='Path to pre-computed GC content file only used if reference genome is not provided (default: pre-computed GC contenct from hg19)')    
    devparser.add_argument("--fixclones", required=False, default=None, type=str, help='Fixed clones as a TSV dataframe containing fields: CELL, CLONE (default: not used, clones are inferred by SPRINTER)')
    devparser.add_argument("--pvalcorr", required=False, default='hs', type=str, help='Multiple-hypothesis correction method (default: hs)')
    devparser.add_argument("--fixploidyref", required=False, type=float, default=None, help="Fixed the reference ploidy for tumour cells in the sample (default: None, not used, it gets inferred by SPRINTER)")
    devparser.add_argument("-e", "--segtest", required=False, default='CHR', type=str, help='Definition of segment to use for doing the test within [CHR, JOINT_SEG, MERGED_CN_STATE, CHR_MERGED_CN_STATE, WHOLE_GENOME] (default: CHR)')
    devparser.add_argument("-s", "--summarystat", required=False, default='special', type=str, help='Method for measuring overlap between distributions (default: special)')
    devparser.add_argument("-m", "--meanmethod", required=False, default='weighted_min', type=str, help='Mean method across segs to use in output of test stat: weighted_min, normal, or weighted_sum (default: weighted_min)')
    devparser.add_argument("-g", "--gccorr", required=False, type=str, default='QUANTILE', help='GC correction mode to be used (default: QUANTILE, available methods are QUANTILE and MODAL)')
    devparser.add_argument("--rtdata", required=False, default=None, type=str, help='Dataframe file with RT data (default: None)')
    devparser.add_argument("--subsample", required=False, type=int, default=None, help="Random subsample of cells (default: use all cells)")
    devparser.add_argument("--minrtreads", required=False, type=int, default=5, help="Minimum number of reads (default: 5)")
    devparser.add_argument("--minfracbins", required=False, type=float, default=0.5, help="Minimum fraction of raw RT bins (default: 0.5)")
    devparser.add_argument("--combinert", required=False, type=int, default=None, help="Force size of RT bins (default: None)")
    devparser.add_argument("--combinecn", required=False, type=int, default=None, help="Force size of CN bins (default: None)")
    devparser.add_argument("--maxgap", required=False, type=int, default=3, help="Maximum gap to allow in RT binning (default: 3)")
    devparser.add_argument("--nortbinning", required=False, default=False, action='store_true', help='Do not apply RT binning (default: False)')
    devparser.add_argument("--nogcaggregate", required=False, default=False, action='store_true', help='Do not compute GC biases by aggregating and correcting (default: False)')
    devparser.add_argument("--nocorrgcintercept", required=False, default=False, action='store_true', help='Do not correct GC intercept per replication time (default: False)')
    devparser.add_argument("--commaseparator", required=False, default=False, action='store_true', help='Use comma separators (default: tab separators)')
    devparser.add_argument("--clonethreshold", required=False, type=float, default=None, help="Fix threshold to find clones (default: None)")
    devparser.add_argument("--rescuethreshold", required=False, type=float, default=None, help="Fix threshold to rescue noisy cells (default: None)")
    devparser.add_argument("--fastsphase", required=False, default=True, action='store_false', help='Faster mode to identify S phase cells (default: True)')
    devparser.add_argument("--fastcns", required=False, default=True, action='store_false', help='Faster mode to infer CNAs (default: True)')
    devparser.add_argument("--devmode", required=False, default=False, action='store_true', help='Output and plot everything for development mode (default: False)')

    args = parser.parse_args(args)
    devargs = args if args.subcommand == 'dev' else devparser.parse_args([])

    if devargs.rtscores is None:
        devargs.rtscores = get_resource('rtscores.csv.gz')
        if not os.path.isfile(devargs.rtscores):
            raise ImportError('rtscores resource not found; something went wrong or changed in the installation!')
    if devargs.gapsfile is None:
        devargs.gapsfile = get_resource('gaps_hg19.tsv')
        if not os.path.isfile(devargs.gapsfile):
            raise ImportError('gapsfile resource not found; something went wrong or changed in the installation!')
    if devargs.gccont is None:
        devargs.gccont = get_resource('gccont.csv.gz')
        if not os.path.isfile(devargs.gccont):
            raise ImportError('gccont resource not found; something went wrong or changed in the installation!')
        
    if inputdata is None and not os.path.isfile(args.DATA):
        raise ValueError('Data file does not exist')
    if not os.path.isfile(devargs.rtscores):
        raise ValueError('The replication-scores file does not exist: \n{}'.format(devargs.rtscores))
    if not os.path.isfile(devargs.gapsfile):
        raise ValueError('The gaps file does not exist: \n{}'.format(devargs.gapsfile))
    if args.refgenome is not None and not os.path.isfile(args.refgenome):
        raise ValueError('Reference genome file does not exist')
    if devargs.gccont is not None and args.refgenome is None and not os.path.isfile(devargs.gccont):
        print(devargs.gccont)
        raise ValueError('GC count file does not exist')
    assert (args.refgenome is not None and os.path.isfile(args.refgenome)) or (devargs.gccont is not None and os.path.isfile(devargs.gccont))
    if devargs.fixclones is not None and not os.path.isfile(devargs.fixclones):
        raise ValueError('Fixed clone file does not exist')
    # if args.fixcns is not None and not os.path.isfile(args.fixcns):
    #     raise ValueError('Fixed clone file does not exist')
    # if args.fixploidy is not None and not os.path.isfile(args.fixploidy):
    #     raise ValueError('Fixed clone file does not exist')
    if devargs.rtdata is not None and not os.path.isfile(devargs.rtdata):
        raise ValueError('RTdata file does not exist')
    if args.seed and args.seed < 1:
        raise ValueError("The random seed  must be positive!")
    if devargs.segtest not in ['CHR', 'JOINT_SEG', 'MERGED_CN_STATE', 'CHR_MERGED_CN_STATE', 'WHOLE_GENOME']:
        raise ValueError("The segment definition must be one within [CHR, MERGED_CN_STATE, CHR_MERGED_CN_STATE, WHOLE_GENOME]!")
    if devargs.gccorr not in ['QUANTILE', 'MODAL', 'TEST']:
        raise ValueError("The gcorr method must be one within [QUANTILE and MODAL]!")
    if args.repliseq not in ['repliseq', 'allnormal', 'all']:
        raise ValueError("Unknown repliseq value provided: {}, it must be either repliseq, allnormal, or all!".format(mainargs.repliseq))
    if devargs.maxgap < 1:
        raise ValueError("Maxgaps value must be at least 1!")
    if args.maxploidy < 1:
        raise ValueError("Maxploidy value must be at least 1!")

    return {
        "data" : 'provided' if inputdata is not None else os.path.abspath(args.DATA),
        "refgenome" : args.refgenome,
        "minreads" : args.minreads,
        "rt_reads" : args.rtreads,
        "cn_reads" : args.cnreads,
        "minnocells" : args.minnumcells,
        "minpropcells" : args.minpropcells,
        "propsphase" : args.propsphase,
        "strictgc" : args.strictgc,
        "maxploidy" : args.maxploidy,
        "quantile" : args.quantile,
        "alpha" : args.alpha,
        "repliseq" : args.repliseq,
        "visual" : args.visual,
        "visualcn" : args.visualcn,
        "jobs" : args.jobs,
        "seed" : args.seed,
        "output" : args.output,
        "rtscores" : devargs.rtscores,
        "gapsfile" : devargs.gapsfile,
        "gccont" : devargs.gccont,
        "fixclones" : devargs.fixclones,
        "pvalcorr" : devargs.pvalcorr,
        "fixcns" : None, #args.fixcns,
        "fixploidy" : None, #args.fixploidy,
        "fixploidyref" : devargs.fixploidyref,
        "segtest" : devargs.segtest,
        "summarystat" : devargs.summarystat,
        "meanmethod" : devargs.meanmethod,
        "commasep" : devargs.commaseparator,
        "subsample" : devargs.subsample,
        "gccorr" : devargs.gccorr,
        "combine_rt" : devargs.combinert,
        "min_rt_reads" : devargs.minrtreads,
        "min_frac_bins" : devargs.minfracbins,
        "combine_cn" : devargs.combinecn,
        "rtdata" : devargs.rtdata,
        "nortbinning" : devargs.nortbinning,
        "nogcaggregate" : devargs.nogcaggregate,
        "nocorrgcintercept" : devargs.nocorrgcintercept,
        "maxgap" : devargs.maxgap,
        "clonethreshold" : devargs.clonethreshold,
        "rescuethreshold" : devargs.rescuethreshold,
        "fastsphase" : devargs.fastsphase,
        "fastcns" : devargs.fastcns,
        "devmode" : devargs.devmode
    }


def main(args=None, inputdata=None):
    log('Parsing and checking arguments', level='STEP')
    args = parse_args(args, inputdata)
    np.random.seed(args['seed'])
    log('\n'.join(['Arguments:'] + ['\t{} : {}'.format(a, args[a]) for a in args]), level='INFO')

    log('Processing input data and defining bins', level='STEP')
    data, total_counts, cn_size, gl_size, excluded = pp.process_input(args, inputdata)
    log('> Selected cells: {}'.format(data['CELL'].nunique()), level='INFO')
    log('> Cells discarded because not enough reads: {}'.format(excluded['CELL'].nunique()), level='INFO')
    get_size = (lambda D : (D['END'] - D['START']).mean())
    log('> Average width of RT bins: {:.1f}'.format(get_size(data.groupby(['CELL', 'BIN_REPINF']).agg({'START' : 'min', 'END' : 'max'}))), level='INFO')
    log('> Average width of CN bins: {:.1f}'.format(get_size(data.groupby(['CELL', 'BIN_CNSINF']).agg({'START' : 'min', 'END' : 'max'}))), level='INFO')
    if args['devmode']:
        data.to_csv('rawdata.tsv.gz', sep='\t', index=False)
        total_counts.to_frame('TOTAL_COUNTS').to_csv('total_counts.tsv.gz', sep='\t')
    data = data[['CELL', 'CHR', 'START', 'END', 'GENOME', 'BIN_REPINF', 'BIN_CNSINF', 'BIN_GLOBAL', 'FOR_REP', 'consistent_rt', 'GC', 'RAW_RDR']]

    gcbiases = None
    if not args["nogcaggregate"]:
        log('Estimating GC bias across all cells', level='STEP')
        gcbiases, _ = gc.estimate_gcbiases(data, gccorr=args['gccorr'], fastsphase=args['fastsphase'], jobs=args['jobs'])

        if max(np.median([gcbiases['EARLY'][cell]['SLOPE'] for cell in data['CELL'].unique() if not gcbiases['EARLY'][cell]['IS_OUT']]),
               np.median([gcbiases['LATE'][cell]['SLOPE'] for cell in data['CELL'].unique() if not gcbiases['LATE'][cell]['IS_OUT']]),
               np.median([gcbiases['ALL'][cell]['SLOPE'] for cell in data['CELL'].unique() if not gcbiases['ALL'][cell]['IS_OUT']])) > 2.:
            log('Strict GC correction has been activated due to the high GC sequencing bias', level='WARN')
            args['strictgc'] = True

    log('Running inference across all cells', level='STEP')
    pvals, rtprofiles = ci.run(_data=data,
                               _seg=args['segtest'],
                               _testmethod=args['summarystat'],
                               _meanmethod=args['meanmethod'],
                               jobs=args['jobs'],
                               gcbiases=gcbiases,
                               gccorr=args['gccorr'],
                               strictgc=args['strictgc'],
                               nocorrgcintercept=args['nocorrgcintercept'],
                               _quantile=args['quantile'],
                               fastsphase=args['fastsphase'])
    pvals = pvals.sort_values('CELL')
    rtprofiles = rtprofiles.sort_values(['CELL', 'CHR', 'START', 'END'])
    if args['devmode']:
        rtprofiles.to_csv('rtprofiles.tsv.gz', sep='\t', index=False)
    data = data.merge(rtprofiles[['CELL', 'CHR', 'START', 'END', 'RT_CN_STATE', 'MERGED_CN_STATE', 'MERGED_RDR_MEDIAN', 'RDR']],
                      on=['CELL', 'CHR', 'START', 'END'],
                      how='outer')\
               .sort_values(['CELL', 'CHR', 'START', 'END']).reset_index(drop=True)
    del rtprofiles

    log('Applying multiple hypothesis correction', level='STEP')
    pvals = corrections(pvals, args['summarystat'], args['alpha'], method=args['pvalcorr'])
    log('> S-phase fraction: {:.1%}'.format(pvals['IS-S-PHASE'].sum() / len(pvals)), level='INFO')
    if args['devmode']:
        pvals.to_csv('pvals.tsv.gz', sep='\t')

    log('Correcting S-phase cells for copy-number calling', level='STEP')
    data = rc.correct_sphase(data, pvals, cn_size=cn_size, gl_size=gl_size, fastcns=args['fastcns'], visualcn=args['visualcn'], jobs=args['jobs'])
    if args['devmode']:
        data.to_csv('data.tsv.gz', sep='\t', index=False)

    log('Calling copy numbers for G1/G2 cells', level='STEP')
    cn_g1g2 = cc.call_cns_g1g2(data, pvals, max_ploidy=args['maxploidy'], fixed_ploidy=args['fixploidy'], fixed_cns=args['fixcns'], fastcns=args['fastcns'], jobs=args['jobs'])
    if args['devmode']:
        cn_g1g2.to_csv('cn_g1g2.tsv.gz', sep='\t', index=False)

    log('Infering G0/1/2-phase clones', level='STEP')
    min_no_cells = min(args['minnocells'], int(ceil(args['minpropcells'] * pvals['CELL'].nunique())))
    log('> Chosen mininimum number of cells per clone: {}'.format(min_no_cells), level='INFO')
    clones_g1g2, normal_clones, ploidy_ref, fixedclones = ic.infer_clones(cn_g1g2, pvals, clone_threshold=args['clonethreshold'], min_no_cells=min_no_cells, fixed_clones=args['fixclones'], fixploidyref=args['fixploidyref'], fastcns=args['fastcns'])
    log('> Base tumour ploidy identified for most cancer cells: {}'.format(ploidy_ref), level='INFO')
    if args['devmode']:
        clones_g1g2.to_csv('clones_g1g2.tsv.gz', sep='\t', index=False)
    pd.Series(normal_clones, name='NORMAL_CLONE').to_csv('normal_clones.tsv.gz', sep='\t', index=False)

    log('Assigning S phase cells to clones', level='STEP')
    cn_all, clones_all, assignments, cn_clones = ar.assign_s_clones(data, pvals, cn_g1g2, clones_g1g2, normal_clones, rescue_threshold=args['rescuethreshold'], prop_sphase=args['propsphase'], fixedclones=fixedclones, fastcns=args['fastcns'], jobs=args['jobs'])
    clones_all.to_csv('clones_all.tsv.gz', sep='\t', index=False)
    assignments.to_csv('assignments.tsv.gz', sep='\t', index=False)
    cn_clones.to_csv('cn_clones.tsv.gz', sep='\t', index=False)
    if args['devmode']:
        cn_all.to_csv('cn_all.tsv.gz', sep='\t', index=False)

    log('Correcting focal and RT-specific errors in S-phase cells', level='STEP')
    cn_all, frac_cncorrect = cncorrect.correct_sphase_cns(data=data, pvals=pvals, cn_all=cn_all, cn_clones=cn_clones, clones_all=clones_all, thres_sphase_cns=5e6)
    log('> Fraction of overall genome corrected in S-phase cells: {:.3f}'.format(frac_cncorrect), level='INFO')
    cn_all.to_csv('cn_all_corrected.tsv.gz', sep='\t', index=False)

    log('Infering G2 cells', level='STEP')
    annotations = ig.infer_G2(total_counts, pvals, clones_all, cn_all, normal_clones, jobs=args['jobs'])

    log('Estimating clone-specific RT profile', level='STEP')
    cnrt, rtdf, cnrt_clone, rtdf_clone = rtestimate.estimate_rt(cn_all, annotations, normal_clones, jobs=args['jobs'])
    if cnrt is not None:
        log('ART frac: {:.3f}'.format(rtdf['ALTERED_RT'].sum() / len(rtdf['ALTERED_RT'])), level='INFO')
        rtdf.to_csv('rtinferred_sample.tsv.gz', sep='\t', index=False)
        rtdf_clone.to_csv('rtinferred_clones.tsv.gz', sep='\t', index=False)
        if args['devmode']:
            cnrt.to_csv('cnrt.tsv.gz', sep='\t', index=False)        
            cnrt_clone.to_csv('cnrt_clone.tsv.gz', sep='\t', index=False)
    else:
        log('No aneuploid clones, or no S phase cells in these clones, have been found so RT profiles will not be estimated', level='WARN')

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
