
from utils import *
from rtprofile import rt_normalise



def estimate_rt(cn_all, annotations, normal_clones, use_chr=True, fastsphase=True, frac_cnrt=.3, max_rdr=3., jobs=1):
    pvals = annotations[(annotations['IS-S-PHASE'] == True) & 
                        (~annotations['PREDICTED_CLONE'].isin(normal_clones)) & 
                        (~pd.isnull(annotations['PREDICTED_CLONE']))].reset_index(drop=True)
    
    selected = pvals[pvals['IS-S-PHASE']==True].sort_values('MEAN_DIFF', ascending=False)\
                                               .head(round(frac_cnrt * pvals['IS-S-PHASE'].sum()))['CELL'].unique()
    selected_cn_all = cn_all[cn_all['CELL'].isin(selected)]
    if len(selected_cn_all) == 0:
        return None, None, None, None
    else:
        cnrt = compute_cnrtprofile(selected_cn_all, use_chr=use_chr, fastsphase=fastsphase, max_rdr=max_rdr, j=jobs)

        rtdf = cnrt.groupby(['CHR', 'START', 'END'])\
                .agg({'GENOME' : 'first', 'consistent_rt' : 'first', 'CNRT_PROFILE' : 'median'})\
                .reset_index()
        rtdf = infer_rt(rtdf).sort_values(['CHR', 'START', 'END']).reset_index(drop=True)

        selected_clone = pvals[pvals['IS-S-PHASE']==True].sort_values('MEAN_DIFF', ascending=False)\
                                                .groupby('PREDICTED_CLONE')\
                                                .apply(lambda x : x.head(round(frac_cnrt * len(x))))\
                                                ['CELL'].unique()
        selected_cn_clone = cn_all[cn_all['CELL'].isin(selected_clone)]
        cnrt_clone = compute_cnrtprofile(selected_cn_clone, use_chr=use_chr, fastsphase=fastsphase, max_rdr=max_rdr, j=jobs)
        
        rtdf_clone = cnrt_clone.groupby(['CHR', 'START', 'END', 'CLONE'])\
                        .agg({'GENOME' : 'first', 'consistent_rt' : 'first', 'CNRT_PROFILE' : 'median'})\
                        .reset_index()
        rtdf_clone = rtdf_clone.groupby('CLONE')\
                            .apply(lambda x : infer_rt(x))\
                            .sort_values(['CHR', 'START', 'END'])\
                            .reset_index(drop=True)
        
        return cnrt, rtdf, cnrt_clone, rtdf_clone


def compute_cnrtprofile(cn_all, use_chr=True, fastsphase=True, max_rdr=3., j=1):
    if not fastsphase:
        data = cn_all[['CLONE', 'CELL', 'CHR', 'START', 'END', 'GENOME', 'consistent_rt', 'RDR', 'CN_TOT', 'CN_CLONE']].reset_index(drop=True)
    else:
        data = cn_all.groupby(['CELL', 'CHR', 'BIN_REPINF'])[['CLONE', 'GENOME', 'consistent_rt', 'RDR', 'CN_TOT', 'CN_CLONE']].first().reset_index()
    with Manager() as manager:
        shared = manager.list()
        with Pool(processes=j,
                  initializer=init_cell_cnrtprofile, 
                  initargs=(manager.dict({cell : celldf for cell, celldf in data.groupby('CELL')}),
                            use_chr,
                            max_rdr,
                            shared)) \
        as pool:
            bar = ProgressBar(total=data['CELL'].nunique(), length=30, verbose=False)
            progress = (lambda e : bar.progress(advance=True, msg="Cell {}".format(e)))
            bar.progress(advance=False, msg="Started")
            _ = [cell for cell in pool.imap_unordered(run_cell_cnrtprofile, data['CELL'].unique()) if progress(cell)]
        shared = pd.concat(shared)
        if fastsphase:
            shared = cn_all[['CLONE', 'CELL', 'CHR', 'START', 'END', 'BIN_REPINF']].merge(shared, on=['CLONE', 'CELL', 'CHR', 'BIN_REPINF'], how='outer')
        shared = shared.sort_values(['CELL', 'CHR', 'START', 'END']).reset_index(drop=True)
        assert shared[['CELL', 'CHR', 'START', 'END']].equals(cn_all[['CELL', 'CHR', 'START', 'END']].reset_index(drop=True))
        return shared


def init_cell_cnrtprofile(_data, _use_chr, _max_rdr, _shared):
    global DATA, USE_CHR, SHARED, MAX_RDR
    DATA = _data
    USE_CHR = _use_chr
    MAX_RDR = _max_rdr
    SHARED = _shared


def run_cell_cnrtprofile(cell):
    celldf = DATA[cell].sort_values(['CHR', 'GENOME']).reset_index(drop=True)
    if USE_CHR:
        celldf['RTSEG'] = celldf['CHR'].astype(str) + '_' + celldf['CN_TOT'].astype(str)
        celldf['RTSEG'] = celldf['RTSEG'].map(dict(zip(celldf['RTSEG'].unique(), np.arange(celldf['RTSEG'].nunique()))))
    else:
        celldf['RTSEG'] = celldf['CN_TOT'].values
    celldf['CNRT_PROFILE'], celldf['CNRT_MEDIAN'] = rt_normalise(celldf, seg='RTSEG', timing='consistent_rt', val='RDR', boots=100)
    celldf['CNRT_PROFILE'] = celldf['CNRT_PROFILE'].clip(0., MAX_RDR)
    SHARED.append(celldf)
    return cell


def infer_rt(rtdf, min_prob=.9, ci=0.9):
    model = scipy.stats.norm
    touse = rtdf[((rtdf['consistent_rt']=='early') & (rtdf['CNRT_PROFILE'] > 1)) |\
                 ((rtdf['consistent_rt']=='late') & (rtdf['CNRT_PROFILE'] < 1))]
    early = touse[touse['consistent_rt']=='early']['CNRT_PROFILE']
    early_params = np.mean([model.fit(early.sample(len(early), replace=True).values) for _ in range(100)], axis=0)
    late = touse[touse['consistent_rt']=='late']['CNRT_PROFILE']
    late_params = np.mean([model.fit(late.sample(len(late), replace=True).values) for _ in range(100)], axis=0)

    prob_early = model.pdf(rtdf['CNRT_PROFILE'].values, *early_params) * (len(early) / (len(early) + len(late)))
    prob_late = model.pdf(rtdf['CNRT_PROFILE'].values, *late_params) * (len(late) / (len(early) + len(late)))
    norm = prob_early + prob_late
    prob_early = prob_early / norm
    prob_late = prob_late / norm

    ci_early = model.interval(ci, *early_params)
    ci_late = model.interval(ci, *late_params)
    rtdf['RT_INFERRED'] = np.where((~pd.isnull(rtdf['CNRT_PROFILE'])) & (ci_early[0] <= rtdf['CNRT_PROFILE']) & (prob_early >= min_prob), 'early',
                          np.where((~pd.isnull(rtdf['CNRT_PROFILE'])) & (rtdf['CNRT_PROFILE'] <= ci_late[1]) & (prob_late >= min_prob),   'late',
                                                                                                                                          'unknown'))
    
    rtdf['ALTERED_RT'] = (rtdf['RT_INFERRED'] != 'unknown') & (rtdf['consistent_rt'] != 'not_consistent') & (rtdf['consistent_rt'] != rtdf['RT_INFERRED'])
    return rtdf

