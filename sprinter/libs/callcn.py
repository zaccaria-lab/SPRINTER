from utils import *

from scipy.cluster import hierarchy
from hmmlearn import hmm
from sklearn.cluster import DBSCAN
from statsmodels.tools.eval_measures import bic
from statsmodels.tools.eval_measures import aic
from sklearn.neighbors import LocalOutlierFactor
import statsmodels.formula.api as smf



def correct_sphase(rtprofile, annotations):
    data = rtprofile.merge(annotations[['CELL', 'IS-S-PHASE']], on='CELL', how='inner')
    nonrep = data[data['IS-S-PHASE'] == False].reset_index(drop=True)
    nonrep['RDR_RTCORR'] = nonrep.groupby('CELL')['RDR'].transform(lambda array : array / array.mean())
    rep = data[data['IS-S-PHASE'] == True].reset_index(drop=True)
    rep['RT_CORR'] = rep.groupby(['CELL', 'RT_CN_STATE'])['RDR'].transform('median')
    rep['RDR_RTCORR'] = np.where(rep['RT_CORR'] > 0, (rep['RDR'] / rep['RT_CORR']) * rep['MERGED_RDR_MEDIAN'], 0.)
    rep['RDR_RTCORR'] = rep.groupby('CELL')['RDR_RTCORR'].transform(lambda array : (array / array.mean()) if array.mean() > 0 else 0.)
    assert (not np.isinf(rep['MERGED_RDR_MEDIAN'].values).any()) and (not pd.isnull(rep['MERGED_RDR_MEDIAN'].values).any()), rep['MERGED_RDR_MEDIAN']
    assert (not np.isinf(rep['RT_CORR'].values).any()) and (not pd.isnull(rep['RT_CORR'].values).any()), rep['RT_CORR']
    assert (not np.isinf(rep['RDR_RTCORR'].values).any()) and (not pd.isnull(rep['RDR_RTCORR'].values).any()), rep['RDR_RTCORR']
    return pd.concat([rep, nonrep])


def call_cns_g1g2(rawdata, rtprofile, annotations, max_ploidy=4, frac_covered=0.2, jobs=1):
    g1g2, data = process_cns_data(annotations, rawdata, rtprofile, frac_covered)
    with Manager() as manager:
        shared = manager.list()
        with Pool(processes=jobs, initializer=init_call_cns, initargs=(manager.dict({cell : celldf for cell, celldf in data.groupby('CELL')}),
                                                                       max_ploidy,
                                                                       shared)) as pool:
            bar = ProgressBar(total=data['CELL'].nunique(), length=30, verbose=False)
            progress = (lambda e : bar.progress(advance=True, msg="Cell {}".format(e)))
            bar.progress(advance=False, msg="Started")
            _ = [cell for cell in pool.imap_unordered(call_cns, (cell for cell in data['CELL'].unique())) if progress(cell)]
        data = pd.concat(shared)
    data = unify_cns(data, rawdata, g1g2)
    return data


def process_cns_data(annotations, rawdata, rtprofile, frac_covered, max_rdr=3.):
    g1g2 = annotations[annotations['IS-S-PHASE']==False]['CELL'].unique()
    data = rawdata[rawdata['CELL'].isin(g1g2)].merge(rtprofile[['CELL', 'CHR', 'BIN_REPINF', 'MERGED_CN_STATE', 'RDR_RTCORR']].assign(COV_RAW_BINS=1),
                                                     on=['CELL', 'CHR', 'BIN_REPINF'], how='inner')
    data['RDR_RTCORR'] = data['RDR_RTCORR'].clip(0, max_rdr)
    median_counts = rtprofile.groupby('CELL')['COUNT'].median()
    data['COUNT_RTCORR'] = data['RDR_RTCORR'] * data['CELL'].map(median_counts)
    data = data.groupby(['CELL', 'CHR', 'BIN_CNSINF']).agg({'START' : 'min',
                                                            'END' : 'max',
                                                            'GENOME' : 'min',
                                                            'NORM_COUNT' : 'sum',
                                                            'COUNT' : 'sum',
                                                            'GC' : 'mean',
                                                            'RDR_RTCORR' : 'median',
                                                            'COUNT_RTCORR' : 'sum',
                                                            'MERGED_CN_STATE' : (lambda v : v.value_counts().index[0] if (~pd.isnull(v)).any() else np.nan),
                                                            'COV_RAW_BINS' : 'sum'})\
                                                      .reset_index()\
                                                      .sort_values(['CELL', 'CHR', 'START', 'END'])\
                                                      .reset_index(drop=True)
    assert (not data['COUNT'].isna().any()) & (not data['NORM_COUNT'].isna().any()) & (not data['GC'].isna().any())
    tot_raw_bins = rawdata[rawdata['CELL'].isin(g1g2)][['CELL', 'CHR', 'BIN_CNSINF', 'START']]\
                   .groupby(['CELL', 'CHR', 'BIN_CNSINF'])['START'].count().rename('TOT_RAW_BINS').reset_index()
    data = data.merge(tot_raw_bins, on=['CELL', 'CHR', 'BIN_CNSINF'], how='inner')
    data['FRAC_COVERED'] = data['COV_RAW_BINS'] / data['TOT_RAW_BINS']
    assert (data['FRAC_COVERED'] <= 1.).all()
    data = data[data['FRAC_COVERED'] >= frac_covered].sort_values(['CELL', 'CHR', 'START', 'END']).reset_index(drop=True)
    return g1g2, data


def init_call_cns(_data, _max_ploidy, _shared):
    global DATA, MAXPLOIDY, SHARED
    DATA = _data
    MAXPLOIDY = _max_ploidy
    SHARED = _shared


def call_cns(cell):
    celldf = DATA[cell].sort_values(['CHR', 'START', 'END']).reset_index(drop=True)
    assert (not celldf['RDR_RTCORR'].isna().any()) & (not celldf['RDR_RTCORR'].apply(np.isinf).any())
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stderr(devnull):
            celldf['CN_TOT'] = infer_cns_free(celldf, repetitions=5, max_ploidy=MAXPLOIDY, ispoisson=False)[2]
    SHARED.append(celldf[['CELL', 'CHR', 'BIN_CNSINF', 'MERGED_CN_STATE', 'RDR_RTCORR', 'CN_TOT']])
    return cell


def infer_cns_free(data, repetitions, max_ploidy, min_aneu=0.1, ispoisson=False):
    ref = data[data['MERGED_CN_STATE'] == data.groupby('MERGED_CN_STATE')['START'].count().idxmax()]['RDR_RTCORR'].to_numpy()
    get_var = (lambda X : X.clip(*scipy.stats.norm.interval(0.99, *scipy.stats.norm.fit(X))).var(ddof=1))
    var = data.groupby('MERGED_CN_STATE')['RDR_RTCORR'].apply(lambda x : get_var(x) * len(x)).sum() / len(data)
    trans = data['MERGED_CN_STATE'].nunique() / len(data)
    if not ispoisson:
        rdrs = data['RDR_RTCORR'].to_numpy()
        brdr = np.median(ref)
    else:
        scale = data['COUNT'].median()
        rdrs = (data['RDR_RTCORR'] * scale).round().astype(int).to_numpy()
        brdr = np.round((np.median(ref) * scale)) # cn = rdr * gamma
    argmax = (lambda L : max(L, key=(lambda v : v[0])))
    compute_n = (lambda cn : argmax(infer_cns_hmm(rdrs, gamma=cn/brdr, var=var, trans=trans, ispoisson=ispoisson) for _ in range(repetitions)))
    candidate = min((compute_n(n) for n in range(2, max_ploidy + 1)), key=(lambda v : v[1]))
    if ((candidate[2] != pd.Series(candidate[2]).value_counts().index[0]).sum() / len(candidate[2])) >= min_aneu:
        return candidate
    else:
        return compute_n(2)


def infer_cns_hmm(data, gamma, var=None, trans=None, ispoisson=True, seed=None):
    n_components = 1 + int(np.round(data.max() * gamma))
    if not ispoisson:
        remodel = hmm.GaussianHMM(n_components=n_components, covariance_type="diag", algorithm='viterbi', 
                                  init_params='' if var is not None else 'c',
                                  params='s' if trans is not None and var is not None else \
                                        ('sc' if trans is not None and var is None else \
                                        ('st' if trans is None and var is not None else 'sct')),
                                  n_iter=10, random_state=seed)
        remodel.means_ = (np.arange(0, n_components) / gamma).reshape(-1, 1)
        if var is not None:
            remodel.covars_ = np.full(n_components, var).reshape(-1, 1)
    else:
        remodel = hmm.PoissonHMM(n_components=n_components, algorithm='viterbi', 
                                init_params='', params='st', n_iter=10, random_state=seed)
        remodel.lambdas_ = (np.arange(0, n_components) / gamma).reshape(-1, 1)
    if trans is None:
        remodel.transmat_ = np.array([np.random.dirichlet(alphas) for alphas in (np.identity(n_components) * 200) + 1])
    else:
        remodel.transmat_ = np.identity(n_components) * (1. - (trans * n_components)) + trans
        assert all(np.isclose(row.sum(), 1.) for row in np.identity(n_components) * (1. - (trans * n_components)) + trans)
    remodel.startprob_ = np.ones(n_components) / n_components
    fitted = remodel.fit(data.reshape(-1, 1))
    if any(~np.isclose(row.sum(), 1.) for row in fitted.transmat_):
        fitted.transmat_ = np.array([row if np.isclose(row.sum(), 1.) else np.ones_like(row) / len(row) for row in fitted.transmat_])
    if ~np.isclose(fitted.startprob_.sum(), 1.):
        fitted.startprob_ = np.ones_like(fitted.startprob_) / len(fitted.startprob_)
    
    _, hstates = fitted.decode(data.reshape(-1, 1))
    loglik = fitted.score(data.reshape(-1, 1))
    used_comp = pd.Series(hstates).nunique()
    if trans is not None and var is not None:
        score = bic(loglik, data.size, 3 + n_components)
    elif trans is None and var is None:
        score = bic(loglik, data.size, 1 + n_components + used_comp + (used_comp**2))
    elif trans is None:
        score = bic(loglik, data.size, 2 + n_components + (used_comp**2))
    else:
        score = bic(loglik, data.size, 2 + n_components + used_comp)

    T = pd.DataFrame({'RDR_RTCORR' : data, 'CN_TOT' : hstates}).reset_index()
    plt.figure(figsize=(14, 4))
    sns.scatterplot(data=T, x='index', y='RDR_RTCORR', hue='CN_TOT', s=7, palette='Set1')
    for mean in remodel.means_:
        plt.axhline(mean[0], ls='--', c='k', lw=1)
    plt.title('ll {} - bic {} - used comp {} - var {}'.format(loglik, score, used_comp, var))
    
    return (loglik, score, hstates, used_comp)


def unify_cns(data, rawdata, g1g2):
    data = data.merge(rawdata[rawdata['CELL'].isin(g1g2)][['CELL', 'CHR', 'START', 'END', 'GENOME', 'BIN_CNSINF', 'COUNT', 'BIN_GLOBAL']], 
                      on=['CELL', 'CHR', 'BIN_CNSINF'], how='outer')\
               .sort_values(['CELL', 'CHR', 'START'])\
               .reset_index(drop=True)
    data = data.groupby(['CELL', 'CHR', 'BIN_GLOBAL']).agg({'START' : 'min',
                                                            'END' : 'max',
                                                            'GENOME' : 'min',
                                                            'COUNT' : 'sum',
                                                            'MERGED_CN_STATE' : (lambda v : v.value_counts(dropna=False).index[0]),
                                                            'RDR_RTCORR' : 'median',
                                                            'CN_TOT' : (lambda v : v.value_counts(dropna=False).index[0])})\
                                                      .reset_index()\
                                                      .sort_values(['CELL', 'CHR', 'START', 'END'])\
                                                      .reset_index(drop=True)
    data['CN_TOT'] = data.groupby(['CELL', 'CHR'])['CN_TOT'].transform(lambda x : x.where(~pd.isnull(x), prev_else_next(x)))
    cell_mode = data.groupby('CELL')['CN_TOT'].transform(lambda x : x.value_counts().index[0])
    data['CN_TOT'] = data['CN_TOT'].where(~pd.isnull(data['CN_TOT']), cell_mode).round().astype(int)
    bins = data[['CHR', 'START', 'END']].drop_duplicates().sort_values(['CHR', 'START', 'END']).reset_index(drop=True)
    assert all(bins.equals(df.drop_duplicates().sort_values(['CHR', 'START', 'END']).reset_index(drop=True)) for _, df in data.groupby('CELL')[['CHR', 'START', 'END']])
    return data

