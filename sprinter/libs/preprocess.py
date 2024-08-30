from utils import *
from estrdrs import calc_rt_rdrs

from pybedtools import BedTool
import time





def process_input(args, inputdata):
    if inputdata is None:
        data = pd.read_csv(args['data'], sep=',' if args['commasep'] else '\t', nrows=10)
        if any(col not in data.columns for col in ['CELL', 'CHR', 'START']):
            data = pd.read_csv(args['data'], sep=',' if args['commasep'] else '\t', names=['CHR', 'START', 'END', 'CELL', 'NORM_COUNT', 'COUNT', 'RAW_RDR'])
        else:
            data = pd.read_csv(args['data'], sep=',' if args['commasep'] else '\t')
    else:
        data = inputdata
        assert any(col in data.columns for col in ['CHR', 'START', 'END', 'CELL', 'NORM_COUNT', 'COUNT', 'RAW_RDR'])
    data['CHR'] = data['chr' if 'chr' in data.columns else 'CHR'].astype(str).str.replace('chr', '').astype(int)
    data, excluded = select_cells(data, args)
    data, cn_size, gl_size = prepare_input(counts=data, rt_reads=args['rt_reads'], combine_rt=args['combine_rt'], min_rt_reads=args['min_rt_reads'], min_frac_bins=args['min_frac_bins'],
                                           cn_reads=args['cn_reads'], combine_cn=args['combine_cn'], repliseq_touse=args['repliseq'], rtdata=args['rtdata'],
                                           rtscores=args['rtscores'], gapsfile=args['gapsfile'], refgenome=args['refgenome'], gccont=args['gccont'],
                                           nortbinning=args['nortbinning'], maxgap=args['maxgap'], j=args['jobs'])
    total_counts = data.groupby('CELL')['COUNT'].sum()
    return data, total_counts, cn_size, gl_size, excluded


def select_cells(data, args):
    total = data.groupby('CELL')['COUNT'].sum()
    allcells = set(data['CELL'].unique())
    excluded = pd.DataFrame({'CELL' : total[total < args['minreads']].index})
    data = data[data['CELL'].isin(total[total >= args['minreads']].index)].sort_values(['CELL', 'CHR', 'START', 'END']).reset_index(drop=True)
    assert allcells == set(data['CELL'].unique()).union(set(excluded['CELL'].unique())), (allcells, set(data['CELL'].unique()).union(set(excluded['CELL'].unique())))
    return data, excluded


def prepare_input(counts, rt_reads, combine_rt, min_rt_reads, min_frac_bins, cn_reads, combine_cn, repliseq_touse, rtdata,
                  rtscores, gapsfile, refgenome, gccont, nortbinning=False, maxgap=5e3, maxrdr=3.0, E=.5, L=-.5, j=1):
    counts = fill_count_gaps(counts)
    counts = compute_gc_exclude_gaps(counts, gapsfile, refgenome, gccont)
    repliseq = prepare_repliseq(repliseq_touse, rtscores)
    if rtdata is None:
        counts = combine_repliseq(counts, repliseq, E=E, L=L)
    else:
        counts = prep_combine_rtdata(counts, rtdata, orig_repliseq=repliseq, E=E, L=L)
    counts, rt_size, cn_size, gl_size = define_bins(counts, rt_reads, cn_reads, combine_rt, combine_cn, nortbinning=nortbinning, maxgap=maxgap, E=E, L=L)
    counts = calc_rt_rdrs(counts, rt_size, maxgap=20, min_frac_bins=.1, min_bins=4, j=j)
    assert counts.groupby('CELL')['START'].count().nunique() == 1
    counts = select_for_rep(counts, min_rt_reads, min_frac_bins)
    return counts, cn_size, gl_size


def fill_count_gaps(_data):
    mappable = _data.groupby('CELL')['NORM_COUNT'].transform(lambda x : scipy.stats.norm.ppf(.01, *scipy.stats.norm.fit(x.values))).clip(lower=1).round().astype(int)
    data = _data[_data['NORM_COUNT'] >= mappable]
    counts = data[['CHR', 'START', 'END', 'CELL', 'COUNT']]
    counts = counts.pivot_table(index='CELL', columns=['CHR', 'START', 'END'], values='COUNT', fill_value=0).astype(int).unstack().rename('COUNT').reset_index()
    counts = counts.merge(data[['CHR', 'START', 'END', 'CELL', 'NORM_COUNT']], on=['CHR', 'START', 'END', 'CELL'], how='outer')
    counts['NORM_COUNT'] = counts['NORM_COUNT'].where(~pd.isnull(counts['NORM_COUNT']), counts.groupby(['CHR', 'START', 'END'])['NORM_COUNT'].transform('median')).astype(int)
    assert (counts['NORM_COUNT'] > 0).all()
    return counts


def compute_gc_exclude_gaps(count, gapsfile, refgenome, gccont, buff_exclude=int(2e5)):
    bins = count[['CHR', 'START', 'END']].drop_duplicates()
    bins['chr'] = 'chr' + bins['CHR'].astype(str)

    if refgenome is not None:
        bed = BedTool.from_dataframe(bins[['chr', 'START', 'END']]).nucleotide_content(fi=refgenome)
        bed = bed.to_dataframe(disable_auto_names=True).rename(columns={'#1_usercol' : 'chr', '2_usercol' : 'START', '3_usercol' : 'END',
                                                                        '4_pct_at' : 'AT', '5_pct_gc' : 'GC'})
        bed['CHR'] = bed['chr'].str.replace('chr', '').astype(int)
    else:
        bed = pd.read_csv(gccont, index_col=0)

    gaps = pd.read_csv(gapsfile, sep='\t')
    gaps['START_POS'] = (gaps['START_POS'] - buff_exclude).clip(lower=0)
    gaps['END_POS'] = gaps['END_POS'] + buff_exclude
    bed = bed.merge(gaps, on='CHR', how='inner')
    bed['OVERLAP'] = np.minimum(bed['END'], bed['END_POS']) - np.maximum(bed['START'], bed['START_POS'])
    bed['TOREMOVE'] = (bed['OVERLAP'] >= 0) |\
                      ((bed['TYPE'] == 'telomere_start') & (bed['START'] <= bed['END_POS'])) |\
                      ((bed['TYPE'] == 'telomere_end') & (bed['END'] >= bed['START_POS']))
    bed = bed.dropna().groupby(['CHR', 'START', 'END'])[['GC', 'TOREMOVE']].agg({'GC' : 'first', 'TOREMOVE' : 'any'}).reset_index()
    bed = bed[~bed['TOREMOVE']][['CHR', 'START', 'END', 'GC']]\
             .drop_duplicates()\
             .dropna()\
             .groupby(['CHR', 'START', 'END'])['GC'].first()\
             .reset_index()

    return count.merge(bed[['CHR', 'START', 'END', 'GC']].dropna(), on=['CHR', 'START', 'END'], how='inner')


def prepare_repliseq(repliseq_touse, repliseq_file):
    repliseq_samples = choose_repliseq(repliseq_touse)
    repliseq = pd.read_csv(repliseq_file, sep=',')
    repliseq['CHR'] = repliseq['chr'].str.replace('chr', '').astype(int)
    repliseq['START'] = repliseq['start'].astype(int)
    repliseq['END'] = repliseq['stop'].astype(int)
    repliseq = repliseq[['CHR', 'START', 'END'] + repliseq_samples].sort_values(['CHR', 'START', 'END']).reset_index(drop=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        repliseq['RD_RT'] = np.nan_to_num(np.nanmean(repliseq[repliseq_samples].values.T, axis=0), nan=0)
    return repliseq


def choose_repliseq(repliseq_touse):
    if repliseq_touse == 'repliseq':
        return ['HBEC3', 'T2P', 'TT1', 'HMEC', 'MCF10A']
    elif repliseq_touse == 'allnormal':
        return ['HBEC3', 'T2P', 'TT1', 'HMEC', 'MCF10A', 'BG02', 'BJ', 'HUVEC', 'IMR90', 'keratinocyte']
    elif repliseq_touse == 'all':
        return ['A549', 'H1650', 'H1792', 'H2009', 'H2170','H520', 'HBEC3', 'SW900', 'T2P', 'TT1',
                'HMEC', 'MCF10A', 'MDA453', 'SK-BR3', 'MCF-7', 'T47D',
                'A549encode', 'BG02', 'BJ', 'Caki2','HUVEC', 'G401', 'H460', 'HeLa-S3', 'HepG2', 'IMR90', 'keratinocyte', 'LNCAP', 'SK-N-MC', 'SK-N-SH']
    else:
        raise ValueError('Unknown repliseq samples to be used!')


def combine_repliseq(counts, repliseq, E=.5, L=-.5):
    counts = counts.merge(repliseq[['CHR', 'START', 'END', 'RD_RT']],
                          on=['CHR', 'START', 'END'], how='inner').sort_values(['CELL', 'CHR', 'START', 'END']).reset_index(drop=True)
    counts['GENOME'] = counts['START'] + (counts['CHR'].map(counts.groupby('CHR')['END'].max().cumsum() - counts.groupby('CHR')['END'].max()))
    ranks = counts[['CHR', 'START', 'END']].drop_duplicates().sort_values(['CHR', 'START', 'END']).reset_index(drop=True).reset_index().rename(columns={'index' : 'RANK'})
    counts = counts.merge(ranks, on=['CHR', 'START', 'END'], how='inner').sort_values(['CELL', 'CHR', 'START', 'END']).reset_index(drop=True)
    counts['consistent_rt'] = np.where(counts['RD_RT'] >= E, 'early',
                              np.where(counts['RD_RT'] <= L, 'late', 
                                                             'not_consistent'))
    return counts


def prep_combine_rtdata(counts, rtdata, orig_repliseq, E, L):
    repliseq = pd.read_csv(rtdata, sep='\t')
    assert all(col in repliseq.columns for col in ['CHR', 'START', 'END', 'RT'])
    repliseq = repliseq[['CHR', 'START', 'END', 'RT']]
    repliseq['RD_RT'] = np.where(repliseq['RT']=='early', E,
                        np.where(repliseq['RT']=='late',  L,
                                                          (E+L)/2.))
    repliseq = orig_repliseq[['CHR', 'START', 'END', 'RD_RT']].rename(columns={'RD_RT' : 'REF_RD_RT'})\
                                                              .merge(repliseq, on=['CHR', 'START', 'END'], how='left')
    repliseq['REF_RT'] = np.where(repliseq['REF_RD_RT'] >= E, 'early',
                         np.where(repliseq['REF_RD_RT'] <= L, 'late',
                                                              'not_consistent'))
    repliseq['consistent_rt'] = repliseq['RT'].where(~pd.isnull(repliseq['RT']), repliseq['REF_RT'])
    assert not pd.isnull(repliseq['consistent_rt']).any()
    repliseq['RD_RT'] = repliseq['RD_RT'].where(~pd.isnull(repliseq['RD_RT']), repliseq['REF_RD_RT'])
    assert not pd.isnull(repliseq['RD_RT']).any()
    counts = counts.merge(repliseq[['CHR', 'START', 'END', 'RD_RT', 'consistent_rt']],
                          on=['CHR', 'START', 'END'], how='inner').sort_values(['CELL', 'CHR', 'START', 'END']).reset_index(drop=True)
    counts['GENOME'] = counts['START'] + (counts['CHR'].map(counts.groupby('CHR')['END'].max().cumsum() - counts.groupby('CHR')['END'].max()))
    ranks = counts[['CHR', 'START', 'END']].drop_duplicates().sort_values(['CHR', 'START', 'END']).reset_index(drop=True).reset_index().rename(columns={'index' : 'RANK'})
    counts = counts.merge(ranks, on=['CHR', 'START', 'END'], how='inner').sort_values(['CELL', 'CHR', 'START', 'END']).reset_index(drop=True)
    return counts


def define_bins(counts, rt_reads, cn_reads, combine_rt, combine_cn, nortbinning=False, maxgap=1, maxsize_rt=1000000, maxsize_cn=10000000, E=.5, L=-.5):
    counts = counts.sort_values(['CELL', 'CHR', 'START', 'END']).reset_index(drop=True)
    if nortbinning:
        counts, _, rt_size = make_cell_specific_bins(counts, rt_reads, 'BIN_REPINF', combine_rt, maxgap=maxgap, maxsize=maxsize_rt)
    else:
        counts, rt_size = make_cell_specific_bins_rt(counts, rt_reads, 'BIN_REPINF', combine_rt, maxgap=maxgap, maxsize=maxsize_rt, E=E, L=L)
    counts, combine, cn_size = make_cell_specific_bins(counts, cn_reads, 'BIN_CNSINF', combine_cn, maxgap=maxgap, maxsize=maxsize_cn)
    bins = counts[['CHR', 'START', 'END', 'RANK']].drop_duplicates()
    bins['FORCE_BK'] = ((bins['CHR'] != bins['CHR'].shift(1)) | \
                        ((bins['RANK'] - bins['RANK'].shift(1)) > maxgap)).astype(int).cumsum()
    bins['BIN_GLOBAL'] = bins.groupby('FORCE_BK')['START'].transform(lambda chrdf : np.mod(np.ones_like(chrdf).cumsum() - 1, combine) == 0).astype(int).cumsum()
    return counts.merge(bins[['CHR', 'START', 'END', 'BIN_GLOBAL']], on=['CHR', 'START', 'END'], how='inner')\
                 .sort_values(['CELL', 'CHR', 'START', 'END']).reset_index(drop=True), \
           rt_size, \
           cn_size, \
           combine


def make_cell_specific_bins(data, target, label, _combine=None, maxgap=1, maxsize=None):
    combine = (target / data.groupby('CELL')['COUNT'].mean()).round().clip(lower=1).astype(int) if _combine is None else _combine
    combine = (combine.clip(upper=int(np.ceil(maxsize / (data['END'] - data['START']).median().round()))) if not isinstance(combine, int) else min(maxsize, combine))\
              if maxsize is not None else combine
    data['COMBINE'] = data['CELL'].map(combine) if isinstance(combine, pd.Series) else combine
    data['FORCE_BK'] = ((data['CELL'] != data['CELL'].shift(1)) | \
                        (data['CHR'] != data['CHR'].shift(1)) | \
                        ((data['RANK'] - data['RANK'].shift(1)) > maxgap)).astype(int).cumsum()
    data[label] = (np.mod(data.groupby('FORCE_BK')['START'].transform('rank', method='first').astype(int) - 1,
                          data['COMBINE'].values) == 0).astype(int).cumsum()
    data[label] = ((data['CELL'] != data['CELL'].shift(1)) | \
                   (data['CHR'] != data['CHR'].shift(1)) | \
                   (data['FORCE_BK'] != data['FORCE_BK'].shift(1)) | \
                   (data[label] != data[label].shift(1))).astype(int)
    data[label] = data.groupby('CELL')[label].transform('cumsum')
    assert not pd.isnull(data[label]).any()
    assert (data.groupby(['CELL', label])['START'].transform('count') <= data['COMBINE']).all()
    return data.drop(columns=['COMBINE', 'FORCE_BK']), max(1, int(round(data['COMBINE'].median()))), combine


def make_cell_specific_bins_rt(data, target, label, _combine=None, maxgap=-1, maxsize=None, E=.5, L=-.5):
    combine = (target / data.groupby('CELL')['COUNT'].mean()).round().clip(lower=1).astype(int) if _combine is None else _combine
    combine = (combine.clip(upper=int(np.ceil(maxsize / (data['END'] - data['START']).median().round()))) if not isinstance(combine, int) else min(maxsize, combine)) \
              if maxsize is not None else combine
    early, _, _ = make_cell_specific_bins(data[data['RD_RT'] >= E].reset_index(drop=True), target, label, _combine=combine, maxgap=maxgap)
    early[label] = 'early_' + early[label].astype(str)
    late, _, _ = make_cell_specific_bins(data[data['RD_RT'] <= L].reset_index(drop=True), target, label, _combine=combine, maxgap=maxgap)
    late[label] = 'late_' + late[label].astype(str)
    other, _, _ = make_cell_specific_bins(data[(L < data['RD_RT']) & (data['RD_RT'] < E)].reset_index(drop=True), target, label, _combine=combine, maxgap=maxgap)
    other[label] = 'other_' + other[label].astype(str)
    res = pd.concat((early, late, other), axis=0, ignore_index=True).sort_values(['CELL', 'CHR', 'START', 'END']).reset_index(drop=True)
    res[label] = res[label].map(dict(zip(res[label].unique(), np.arange(0, res[label].nunique())))).astype(int)
    assert res[['CELL', 'CHR', 'START', 'END']].equals(data[['CELL', 'CHR', 'START', 'END']])
    assert not pd.isnull(res[label]).any()
    return res, combine


def select_for_rep(counts, min_rt_reads, min_frac_bins):
    counts['NUM_RAW_BINS'] = counts.groupby(['CELL', 'CHR', 'BIN_REPINF'])['START'].transform('count')
    target = counts.groupby('CELL')['NUM_RAW_BINS'].transform('max')
    counts['FOR_REP'] = (counts.groupby(['CELL', 'CHR', 'BIN_REPINF'])['COUNT'].transform('sum') >= min_rt_reads) &\
                        (counts['NUM_RAW_BINS'] >= (target * min_frac_bins))
    return counts

