from .utils import *
from .rtsegment import globalhmm
from .gccorrect import quantile_gccorrect_rdr, modal_gccorrect_rdr



def profile_gccorr(_data, _timing='consistent_rt', _minrdr=0.0, _maxrdr=3.0, toplot=False, 
                  gccorr='QUANTILE', strictgc=False, nocorrgcintercept=False, gcbiases=None, fastsphase=True):
    data = None
    if not fastsphase:
            data = _data
    else:
            data = _data.groupby(['CELL', 'CHR', 'BIN_REPINF'])\
                        .first().reset_index()\
                        .sort_values(['CHR', 'START', 'END'])\
                        .reset_index(drop=True)
    data = data[(~pd.isnull(data['RAW_RDR'])) & data['FOR_REP']].reset_index(drop=True)
    
    if gccorr == 'QUANTILE':
        gccorr_feat = quantile_gccorrect_rdr(data, gcbiases, nocorrgcintercept, strictgc, _timing, _maxrdr, toplot)
    elif gccorr == 'MODAL':
        gccorr_feat = modal_gccorrect_rdr(data, gcbiases, nocorrgcintercept, strictgc, _timing, _maxrdr, toplot)
    else:
        raise ValueError('GCCORR method does not exist! {}'.format(gccorr))
    data['GC_CORR'] = data['GC'] * gccorr_feat['GC_SLOPE'] + gccorr_feat['GC_INTERCEPT']
    data['GC_CORR'] = data['GC_CORR'].where(data['GC_CORR'] > 0, 1.)

    if fastsphase:
         data = _data.merge(data[['CELL', 'CHR', 'BIN_REPINF', 'GC_CORR']], on=['CELL', 'CHR', 'BIN_REPINF'], how='outer')\
                     .sort_values(['CHR', 'START', 'END'])\
                     .reset_index(drop=True)
 
    data['RDR'] = norm_mean((data['RAW_RDR'] / data['GC_CORR']).clip(lower=_minrdr, upper=_maxrdr))
    data['RDR'] = data['RDR'].clip(lower=_minrdr, upper=_maxrdr)
    assert not (data['RDR'] < 0.).any()
    assert data['RDR'].mean() > 0.
    return data, gccorr_feat


def profile_local(cell, data, seg, _timing='consistent_rt', _gmm=True, _maxcn=10, _reps=10, toplot=False):
    data = data[data[_timing].isin(['early', 'late'])].sort_values(['CHR', 'START', 'END']).dropna().reset_index(drop=True)
    data_early = segment_rt_specific(data[data[_timing]=='early'].reset_index(drop=True), _gmm, _reps, _maxcn, cell)
    data_late = segment_rt_specific(data[data[_timing]=='late'].reset_index(drop=True), _gmm, _reps, _maxcn, cell)
    data = pd.concat((data_early, data_late)).sort_values(['CHR', 'START', 'END']).reset_index(drop=True)
    data['IS_SEG_START'] = data['IS_SEG_START'] | (data['CHR'].shift(1) != data['CHR'])
    data['SEG'] = data['IS_SEG_START'].cumsum()
    data['RT_CN_STATE'] = data[_timing] + '_' + data['CN_STATE'].astype(str)
    data = combine_segments(data, _timing, seg)
    data['RT_PROFILE'], data['MERGED_RDR_MEDIAN'] = rt_normalise(data, seg='MERGED_CN_STATE', timing=_timing)
    data['CHR_MERGED_CN_STATE'] = data['CHR'].astype(str) + ',' + data['MERGED_CN_STATE'].astype(str)
    if toplot: 
        plot_profile(cell, data, data_early, data_late, _timing)
    return data


def segment_rt_specific(_data, _gmm, _reps, _maxcn, cell):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _data['CN_STATE'] = globalhmm(_data['RDR'].values, chrs=_data['CHR'].values, gmm=_gmm, repetitions=_reps, max_components=_maxcn, max_segs=300, tprior=None)[2]
        _data['IS_SEG_START'] = (_data['CN_STATE'].shift(1) != _data['CN_STATE']) & (~_data['CN_STATE'].shift(1).isna())
        return _data


def combine_segments(data, timing, seg, _minsize=3):
    assert np.all(data.groupby('SEG')['RT_CN_STATE'].nunique() <= 2)
    minsize = _minsize if len(data) > 500 else 0

    # groupby segment and select RT_CN_STATES (e.g. early3, late2) that have > minsize of bins
    segs = data.groupby('SEG').agg({'CHR' : 'first',
                                    'START' : 'min', 
                                    'END' : 'max', 
                                    'RT_CN_STATE' : (lambda v : tuple(sorted(st for st, num in v.value_counts().items() if num > minsize)))})\
                              .reset_index()\
                              .sort_values(['CHR', 'START', 'END'])\
                              .reset_index(drop=True)
    # find the early and late CN states from HMM
    segs['EARLY_CN_STATE'] = segs['RT_CN_STATE'].apply(lambda v : v[0] if any('early' in e for e in v) else np.nan)
    segs['LATE_CN_STATE'] = segs['RT_CN_STATE'].apply(lambda v : v[-1] if any('late' in e for e in v) else np.nan)    
    segs = segs[~(segs['EARLY_CN_STATE'].isna() & segs['LATE_CN_STATE'].isna()) ].reset_index(drop=True)

    segs['PREV_EARLY'] = segs.groupby('CHR', group_keys=False)['EARLY_CN_STATE'].apply(lambda chrom : get_prev_notna(chrom))
    segs['NEXT_EARLY'] = segs.groupby('CHR', group_keys=False)['EARLY_CN_STATE'].apply(lambda chrom : get_next_notna(chrom))
    segs['PREV_LATE'] = segs.groupby('CHR', group_keys=False)['LATE_CN_STATE'].apply(lambda chrom : get_prev_notna(chrom))
    segs['NEXT_LATE'] = segs.groupby('CHR', group_keys=False)['LATE_CN_STATE'].apply(lambda chrom : get_next_notna(chrom))

    early_next = np.where(segs['LATE_CN_STATE'] == segs['NEXT_LATE'], segs['NEXT_EARLY'], np.nan)
    early_prev = np.where(segs['LATE_CN_STATE'] == segs['PREV_LATE'], segs['PREV_EARLY'], np.nan)
    early_option = np.where(~pd.isnull(early_next), early_next, early_prev)
    early_option = np.where(pd.isnull(early_option) & (segs['NEXT_EARLY'] == segs['PREV_EARLY']), segs['NEXT_EARLY'], early_option)
    segs['EARLY_CN_STATE'] = segs['EARLY_CN_STATE'].where((~segs['EARLY_CN_STATE'].isna()) | ((segs['LATE_CN_STATE'] != segs['NEXT_LATE']) & 
                                                                                              (segs['LATE_CN_STATE'] != segs['PREV_LATE']) &
                                                                                              (segs['NEXT_EARLY'] != segs['PREV_EARLY'])),
                                                          early_option)
    late_next = np.where(segs['EARLY_CN_STATE'] == segs['NEXT_EARLY'], segs['NEXT_LATE'], np.nan)
    late_prev = np.where(segs['EARLY_CN_STATE'] == segs['PREV_EARLY'], segs['PREV_LATE'], np.nan)
    late_option = np.where(~pd.isnull(late_next), late_next, late_prev)
    late_option = np.where(pd.isnull(late_option) & (segs['NEXT_LATE'] == segs['PREV_LATE']), segs['NEXT_LATE'], late_option)
    segs['LATE_CN_STATE'] = segs['LATE_CN_STATE'].where((~segs['LATE_CN_STATE'].isna()) | ((segs['EARLY_CN_STATE'] != segs['NEXT_EARLY']) &
                                                                                           (segs['EARLY_CN_STATE'] != segs['PREV_EARLY']) &
                                                                                           (segs['NEXT_LATE'] != segs['PREV_LATE'])),
                                                        late_option)

    segs = segs[~(segs['EARLY_CN_STATE'].isna() | segs['LATE_CN_STATE'].isna()) ].reset_index(drop=True)
    
    # get columns of whether a bin is the end of an early/late segment (i.e. redraw breakpoints in early and late independently, but excluding above excl segs)
    segs['END_EARLY'] = segs['EARLY_CN_STATE'] != segs['EARLY_CN_STATE'].shift(-1, fill_value=(segs['EARLY_CN_STATE'].values[-1] if len(segs['EARLY_CN_STATE']) > 0 else np.nan))
    segs['END_LATE'] = segs['LATE_CN_STATE'] != segs['LATE_CN_STATE'].shift(-1, fill_value=(segs['LATE_CN_STATE'].values[-1] if len(segs['LATE_CN_STATE']) > 0 else np.nan))
    # check if start or end of chr 
    segs['IS_HEADCHR'] = segs['CHR'] != segs['CHR'].shift(1)
    segs['IS_TAILCHR'] = segs['CHR'] != segs['CHR'].shift(-1)
    # draw all possible breakpoints (if early, late or tail chr)
    segs['END_MERGED'] = segs['END_EARLY'] | segs['END_LATE'] | segs['IS_TAILCHR']

    segs['INNER_SEG'] = segs['END_MERGED'].shift(1, fill_value=False).astype(int).cumsum()
    binmap = segs[['SEG', 'INNER_SEG']].drop_duplicates()
    segs = segs.groupby('INNER_SEG').agg({'CHR' : 'first',
                                          'START' : 'min',
                                          'END' : 'max',
                                          'EARLY_CN_STATE' : 'first', 
                                          'LATE_CN_STATE' : 'first',
                                          'IS_HEADCHR' : 'any',
                                          'IS_TAILCHR' : 'any'})\
                                    .sort_values(['CHR', 'START', 'END'])\
                                    .reset_index()

    segs['WITHIN_REP_EARLY'] = ((segs['EARLY_CN_STATE'] == segs['EARLY_CN_STATE'].shift(-1)) & 
                                (segs['EARLY_CN_STATE'] == segs['EARLY_CN_STATE'].shift(1))) & \
                               ((~segs['IS_HEADCHR']) & (~segs['IS_TAILCHR']))
    segs['WITHIN_REP_LATE'] =  ((segs['LATE_CN_STATE']  == segs['LATE_CN_STATE'].shift(-1)) & 
                                (segs['LATE_CN_STATE']  == segs['LATE_CN_STATE'].shift(1))) & \
                               ((~segs['IS_HEADCHR']) & (~segs['IS_TAILCHR']))
    segs['WITHIN_REP'] = segs['WITHIN_REP_EARLY'] | segs['WITHIN_REP_LATE']

    is_safe_rep = segs['WITHIN_REP'] & \
                  ~(segs['WITHIN_REP_EARLY'] & (segs['WITHIN_REP_LATE'].shift(-1, fill_value=False) | segs['WITHIN_REP_LATE'].shift(1, fill_value=False))) & \
                  ~(segs['WITHIN_REP_LATE'] & (segs['WITHIN_REP_EARLY'].shift(-1, fill_value=False) | segs['WITHIN_REP_EARLY'].shift(1, fill_value=False)))
    is_next_safe_rep = segs['WITHIN_REP'].shift(-1, fill_value=False) & \
                       ~(segs['WITHIN_REP_EARLY'].shift(-1, fill_value=False) & segs['WITHIN_REP_LATE'].shift(1, fill_value=False)) & \
                       ~(segs['WITHIN_REP_LATE'].shift(-1, fill_value=False) & segs['WITHIN_REP_EARLY'].shift(1, fill_value=False))
    segs['END_MERGED'] = segs['IS_TAILCHR'] | (~is_safe_rep & ~is_next_safe_rep)

    segs['END_MERGED'] = segs['END_MERGED'] | (segs['CHR'] != segs['CHR'].shift(-1))
    segs['SEG_MERGED'] = segs['END_MERGED'].shift(1, fill_value=False).astype(int).cumsum()
    merged_cn_states = segs.groupby('SEG_MERGED')['EARLY_CN_STATE'].apply(lambda x : ','.join(np.sort(x.dropna().unique()))) + ',' + \
                       segs.groupby('SEG_MERGED')['LATE_CN_STATE'].apply(lambda  x : ','.join(np.sort(x.dropna().unique())))
    data = data.merge(segs[['INNER_SEG', 'SEG_MERGED']].drop_duplicates().merge(binmap))
    data['MERGED_CN_STATE'] = data['SEG_MERGED'].map(merged_cn_states)
    data['IS_JOINT_START'] = (data['SEG_MERGED'] != data['SEG_MERGED'].shift(1)) | (data['CHR'] != data['CHR'].shift(1))
    data['JOINT_SEG'] = data['IS_JOINT_START'].cumsum()
    data['IS_MERGED_START'] = data['MERGED_CN_STATE'] != data['MERGED_CN_STATE'].shift(1)

    sel = data.groupby('MERGED_CN_STATE')['RT_CN_STATE'].apply(lambda x : x.str.contains('early').any() and x.str.contains('late').any())
    data = data[data['MERGED_CN_STATE'].isin(sel[sel].index)]
    if seg != 'WHOLE_GENOME':
        sel = data.groupby(seg)[timing].apply(lambda x : (x == 'early').sum() > 0 and (x == 'late').sum() > 0)
        data = data[data[seg].isin(sel[sel].index)]
    return data


def rt_normalise(celldf, seg, timing, val='RDR', boots=1000):
    estmedian = (lambda early, late, size : np.mean([np.mean(np.concatenate((np.random.choice(early, size=size, replace=True) if len(early) > 0 else [], 
                                                                             np.random.choice(late, size=size, replace=True) if len(late) > 0 else [])))
                                                     for _ in range(boots)])
                                            if len(early) > 0 or len(late) > 0 else 1.0)
    meds = celldf.groupby(seg)[[val, timing]].apply(lambda r : estmedian(r[val][(r[timing] == 'early') & (~pd.isnull(r[val]))],
                                                                         r[val][(r[timing] == 'late') & (~pd.isnull(r[val]))],
                                                                         r[timing][~pd.isnull(r[val])].value_counts().min()))
    mergemedian = celldf[seg].map(meds)
    return celldf[val] / mergemedian, mergemedian


def plot_profile(cell, data, data_early, data_late, timing):
    # plot cell (with green and magenta)
    plt.figure(figsize=(20, 4))
    sns.scatterplot(data=data.sort_values(['CHR', 'START', 'END']).reset_index(), x='GENOME', y='RAW_RDR', hue='consistent_rt', hue_order=['early', 'late'], palette=['magenta', 'green'], s=3, legend=True)
    plt.ylim(0,3)
    xticks = data.groupby('CHR')['GENOME'].first()
    plt.plot((xticks.values, xticks.values), (np.full_like(xticks, 0.0), np.full_like(xticks, 3.0)), '--b', linewidth=1.0)
    plt.xticks(xticks.values, xticks.index, rotation=30)
    plt.ylabel('RAW_RDR')
    plt.xlabel('Chromosome')

    # plot cell (with green and magenta)
    plt.figure(figsize=(20, 4))
    sns.scatterplot(data=data.sort_values(['CHR', 'START', 'END']).reset_index(), x='GENOME', y='GC_CORR', hue='consistent_rt', hue_order=['early', 'late'], palette=['magenta', 'green'], s=3, legend=True)
    plt.ylim(0,3)
    xticks = data.groupby('CHR')['GENOME'].first()
    plt.plot((xticks.values, xticks.values), (np.full_like(xticks, 0.0), np.full_like(xticks, 3.0)), '--b', linewidth=1.0)
    plt.xticks(xticks.values, xticks.index, rotation=30)
    plt.ylabel('GC_CORR')
    plt.xlabel('Chromosome')

    # plot cell (in blue)
    plt.figure(figsize=(20, 4))
    sns.scatterplot(data=data.sort_values(['CHR', 'START', 'END']).reset_index(), x='GENOME', y='RDR', s=3)
    plt.ylim(0,3)
    xticks = data.groupby('CHR')['GENOME'].first()
    plt.plot((xticks.values, xticks.values), (np.full_like(xticks, 0.0), np.full_like(xticks, 3.0)), '--b', linewidth=1.0)
    plt.xticks(xticks.values, xticks.index, rotation=30)
    plt.ylabel('RDR')
    plt.xlabel('Chromosome')
    
    # plot cell (with green and magenta)
    plt.figure(figsize=(20, 4))
    sns.scatterplot(data=data.sort_values(['CHR', 'START', 'END']).reset_index(), x='GENOME', y='RDR', hue='consistent_rt', hue_order=['early', 'late'], palette=['magenta', 'green'], s=3, legend=True)
    plt.ylim(0,3)
    xticks = data.groupby('CHR')['GENOME'].first()
    plt.plot((xticks.values, xticks.values), (np.full_like(xticks, 0.0), np.full_like(xticks, 3.0)), '--b', linewidth=1.0)
    plt.xticks(xticks.values, xticks.index, rotation=30)
    plt.ylabel('RDR')
    plt.xlabel('Chromosome')

    # plot early and late HMM segmentation 
    plt.figure(figsize=(20, 4))
    sns.scatterplot(data=data_early, x='GENOME', y='RDR', hue='CN_STATE', s=3, palette='Set2', legend=False)
    plt.plot((data.groupby('CHR')['GENOME'].min(), data.groupby('CHR')['GENOME'].min()), (0, 2.5), '--k', lw=0.5)
    xticks = data.groupby('CHR')['GENOME'].first()
    plt.xticks(xticks.values, xticks.index, rotation=30)
    plt.ylim(0, 3)
    plt.ylabel('RDR')
    plt.xlabel('Chromosome')

    plt.figure(figsize=(20, 4))
    sns.scatterplot(data=data_late, x='GENOME', y='RDR', hue='CN_STATE', s=3, palette='Set2', legend=False)
    plt.plot((data.groupby('CHR')['GENOME'].min(), data.groupby('CHR')['GENOME'].min()), (0, 2.5), '--k', lw=0.5)
    xticks = data.groupby('CHR')['GENOME'].first()
    plt.xticks(xticks.values, xticks.index, rotation=30)
    plt.ylim(0, 3)
    plt.ylabel('RDR') 
    plt.xlabel('Chromosome')

    # plot all segments after merging
    plt.figure(figsize=(20, 4))
    sns.scatterplot(data=data, x='GENOME', y='RDR', hue='MERGED_CN_STATE', s=3, palette='Set2', legend=False)
    plt.plot((data.groupby('CHR')['GENOME'].min(), data.groupby('CHR')['GENOME'].min()), (0, 2.5), '--k', lw=0.5)
    plt.plot((data[data['IS_MERGED_START']]['GENOME'], data[data['IS_MERGED_START']]['GENOME']), (0, 2.5), '--', lw=0.3, c='blue')
    xticks = data.groupby('CHR')['GENOME'].first()
    plt.xticks(xticks.values, xticks.index, rotation=30)
    plt.ylim(0, 3)
    plt.ylabel('RDR')
    plt.xlabel('Chromosome')
    plt.title('MERGED_CN_STATE')

    # plot all segments after merging
    plt.figure(figsize=(20, 4))
    sns.scatterplot(data=data, x='GENOME', y='RDR', hue='JOINT_SEG', s=3, palette='Set2', legend=False)
    plt.plot((data.groupby('CHR')['GENOME'].min(), data.groupby('CHR')['GENOME'].min()), (0, 2.5), '--k', lw=0.5)
    plt.plot((data[data['IS_MERGED_START']]['GENOME'], data[data['IS_MERGED_START']]['GENOME']), (0, 2.5), '--', lw=0.3, c='blue')
    xticks = data.groupby('CHR')['GENOME'].first()
    plt.xticks(xticks.values, xticks.index, rotation=30)
    plt.ylim(0, 3)
    plt.ylabel('RDR')
    plt.xlabel('Chromosome')
    plt.title('JOINT_SEG')

    # coloured and ordered by CN state 
    df = data.sort_values('MERGED_CN_STATE').reset_index(drop=True).reset_index()
    plt.figure(figsize=(20, 4))
    sns.scatterplot(data=df, x='index', y='RDR', hue='MERGED_CN_STATE', s=3, palette='Set2', legend=False)
    plt.ylim(0, 3)

    # plot normalised 'replication profile' coloured by magenta and green 
    plt.figure(figsize=(20, 4))
    sns.scatterplot(data=data, x='GENOME', y='RT_PROFILE', hue=timing, hue_order=['late', 'early'], palette=['green', 'magenta'], 
                    s=10, edgecolor=None, linewidth=0, alpha=0.6, legend=True)
    plt.plot((data.groupby('CHR')['GENOME'].min(), data.groupby('CHR')['GENOME'].min()), (0, 2.5), '--k', lw=0.5)
    xticks = data.groupby('CHR')['GENOME'].first()
    plt.xticks(xticks.values, xticks.index, rotation=30)
    plt.ylim(0, 3)
    plt.ylabel('Normalised RDR')
    plt.xlabel('Chromosome')
