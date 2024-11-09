from .utils import *
from .estrdrs import prev_else_next_rdrs

import statsmodels.formula.api as smf
from sklearn.neighbors import LocalOutlierFactor
from hmmlearn import hmm
from statsmodels.tools.eval_measures import bic
from numba import jit



def correct_sphase(rtprofile, annotations, cn_size, gl_size, fastcns=True, visualcn=False, jobs=1, minrdr=0., maxrdr=3.):
    data = rtprofile.merge(annotations[['CELL', 'IS-S-PHASE']], on='CELL', how='inner')

    nonrep = data[data['IS-S-PHASE'] == False].reset_index(drop=True)
    assert (nonrep.groupby('CELL')['RDR'].mean() > 0.).all()
    nonrep['RDR_RTCORR'] = nonrep['RDR'].where(pd.isnull(nonrep['RDR']), nonrep['RDR'] / nonrep.groupby('CELL')['RDR'].transform('mean'))

    rep = data[data['IS-S-PHASE'] == True].reset_index(drop=True)
    rep['RT_CORR'] = rep.groupby(['CELL', 'RT_CN_STATE'])['RDR'].transform('median')
    sel = (~pd.isnull(rep['RT_CN_STATE'])) & (~pd.isnull(rep['MERGED_RDR_MEDIAN'])) & (~pd.isnull(rep['RT_CORR'])) & (rep['RT_CORR'] > 0) & (~pd.isnull(rep['RDR']))
    rep['RDR_RTCORR'] = np.where(sel, (rep['RDR'] / rep['RT_CORR']) * rep['MERGED_RDR_MEDIAN'], np.nan)
    assert (rep.groupby('CELL')['RDR_RTCORR'].mean() > 0.).all()
    rep['RDR_RTCORR'] = rep['RDR_RTCORR'].where(pd.isnull(rep['RDR_RTCORR']), rep['RDR_RTCORR'] / rep.groupby('CELL')['RDR_RTCORR'].transform('mean'))

    data = pd.concat([rep, nonrep])
    data['RDR_RTCORR'] = data['RDR_RTCORR'].clip(lower=minrdr, upper=maxrdr)
    data = calc_cn_rdrs(data, cn_size=cn_size, gl_size=gl_size, issphase=annotations.groupby('CELL')['IS-S-PHASE'].first(), fastcns=fastcns, visualcn=visualcn, j=jobs)

    issphase = data['CELL'].map(annotations[['CELL', 'IS-S-PHASE']].groupby('CELL')['IS-S-PHASE'].any())
    assert (~np.isinf(data['RDR_CN'].values) | issphase).all() and (~pd.isnull(data['RDR_CN'].values) | issphase).all(), data['RDR_CN']
    return data.sort_values(['CELL', 'CHR', 'START', 'END']).reset_index(drop=True)


def calc_cn_rdrs(counts, cn_size, gl_size, issphase, fastcns, maxgap=20, min_frac_bins=.1, min_bins=4, visualcn=False, j=1):
    with Manager() as manager:
        shared = manager.list()
        with Pool(processes=j, 
                  initializer=init_calc_cn_rdrs, 
                  initargs=(manager.dict({cell : celldf for cell, celldf in counts.groupby('CELL')}),
                            counts[['CHR', 'START', 'END']].drop_duplicates().sort_values(['CHR', 'START', 'END']).reset_index(drop=True),
                            shared,
                            manager.dict(cn_size.to_dict()),
                            gl_size,
                            maxgap,
                            min_frac_bins,
                            min_bins,
                            issphase,
                            fastcns,
                            visualcn)) \
        as pool:
            bar = ProgressBar(total=counts['CELL'].nunique(), length=30, verbose=False)
            progress = (lambda e : bar.progress(advance=True, msg="Cell {}".format(e)))
            bar.progress(advance=False, msg="Started")
            _ = [cell for cell in pool.imap_unordered(run_calc_cn_rdrs, counts['CELL'].unique()) if progress(cell)]
        return pd.concat(shared)
    

def init_calc_cn_rdrs(_counts, _allbins, _shared, _cn_size, _gl_size, _maxgap, _min_frac_bins, _min_bins, _issphase, _fastcns, _visualcn):
    global COUNTS, ALLBINS, SHARED, CN_SIZE, GL_SIZE, MAXGAP, MIN_FRAC_BINS, MIN_BINS, ISSPHASE, FASTCNS, VISUALCN
    COUNTS = _counts
    ALLBINS = _allbins
    SHARED = _shared
    CN_SIZE = _cn_size
    GL_SIZE = _gl_size
    MAXGAP = _maxgap
    MIN_FRAC_BINS = _min_frac_bins
    MIN_BINS = _min_bins
    ISSPHASE = _issphase
    FASTCNS = _fastcns
    VISUALCN = _visualcn


def run_calc_cn_rdrs(cell, minrdr=0., maxrdr=3.):
    counts = COUNTS[cell].sort_values(['CELL', 'CHR', 'START', 'END']).reset_index(drop=True)
    assert counts['RDR_RTCORR'].mean() > 0.
    counts['RAW_RDR_RTCORR'] = counts['RDR_RTCORR'].values

    if not ISSPHASE[cell]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            counts['CNS_GCCORR'] = smf.quantreg('RDR_RTCORR ~ GC + I(GC ** 2.0)', data=counts).fit(q=.5).predict(counts[['GC']])
        counts['RDR_RTCORR'] = (counts['RAW_RDR_RTCORR'] / np.where(counts['CNS_GCCORR'] > 0, counts['CNS_GCCORR'], 1.0)).clip(lower=minrdr, upper=maxrdr)
        assert counts['RDR_RTCORR'].mean() > 0.
        counts['RDR_RTCORR'] = counts['RDR_RTCORR'] / counts['RDR_RTCORR'].mean()

    else:
        excluded = counts[pd.isnull(counts['MERGED_CN_STATE']) | pd.isnull(counts['RDR_RTCORR'])].drop(columns=['RDR_RTCORR', 'RT_CORR']).reset_index(drop=True)
        counts = counts[~(pd.isnull(counts['MERGED_CN_STATE']) | pd.isnull(counts['RDR_RTCORR']))].reset_index(drop=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            counts['CNS_GCCORR'] = smf.quantreg('RDR_RTCORR ~ GC + I(GC ** 2.0)', data=counts).fit(q=.5).predict(counts[['GC']])
        counts['RDR_RTCORR'] = (counts['RAW_RDR_RTCORR'] / np.where((counts['CNS_GCCORR'] > 0) & (~pd.isnull(counts['CNS_GCCORR'])), counts['CNS_GCCORR'], 1.0))\
                               .clip(lower=minrdr, upper=maxrdr)
        counts['RDR_RTCORR'] = counts['RDR_RTCORR'].where((counts['CNS_GCCORR'] > 0) &\
                                                          (~pd.isnull(counts['CNS_GCCORR'])), np.nan)
        furexc = counts[  pd.isnull(counts['RDR_RTCORR'])].drop(columns=['RDR_RTCORR', 'RT_CORR']).reset_index(drop=True)
        counts = counts[~(pd.isnull(counts['RDR_RTCORR']))].reset_index(drop=True)

        assert counts['RDR_RTCORR'].mean() > 0.
        counts['RDR_RTCORR'] = counts['RDR_RTCORR'] / counts['RDR_RTCORR'].mean()

        counts = pd.concat((counts, excluded, furexc), axis=0)\
                   .sort_values(['CHR', 'START', 'END'])\
                   .reset_index(drop=True)

    assert len(counts) == len(COUNTS[cell])

    counts['RDR_CN'] = counts[::-1].groupby('CHR', sort=False)['RDR_RTCORR']\
                                   .rolling(CN_SIZE[cell], min_periods=1, center=False)\
                                   .median()\
                                   .droplevel(0)[::-1].sort_index()

    if not ISSPHASE[cell]:
        counts['RDR_CN'] = counts.groupby('CHR')['RDR_CN'].transform(prev_else_next_rdrs,
                                                                     engine='numba',
                                                                     engine_kwargs={'nopython': True, 'nogil': True, 'cache' : True, 'fastmath' : False})
        counts['RDR_CN'] = np.where(pd.isnull(counts['RDR_CN']) & (counts['RAW_RDR'] == 0.), 0., counts['RDR_CN'])
        counts['RDR_CN'] = counts['RDR_CN'].where(~pd.isnull(counts['RDR_CN']), counts['RAW_RDR'])

    if not VISUALCN:
        counts['RDR_CN'] = counts.groupby('BIN_CNSINF')['RDR_CN'].transform(boot_rdrs_cn,
                                                                            engine='numba',
                                                                            engine_kwargs={'nopython': True, 'nogil': True, 'cache' : True, 'fastmath' : False})
  
    counts['RDR_CN'] = counts['RDR_CN'].clip(lower=minrdr, upper=maxrdr)
    assert counts['RDR_CN'].mean() > 0.
    counts['RDR_CN'] = counts['RDR_CN'] / counts['RDR_CN'].mean()
    assert (not pd.isnull(counts['RDR_CN']).any()) or ISSPHASE[cell], 'Found a NaN in RDR_CN estimation for non-S phase cell: {}'.format(cell)
    assert ALLBINS.equals(counts[['CHR', 'START', 'END']])
    SHARED.append(counts)
    return cell


def correct_rt_within(counts, fastcns, minrdr=0., maxrdr=3.):
    assert not pd.isnull(counts['CNS_RTPROFILE']).any()
    assert counts['consistent_rt'].isin(['early', 'late']).all()
    early = counts[counts['consistent_rt']=='early'].reset_index(drop=True)
    late = counts[counts['consistent_rt']=='late'].reset_index(drop=True)
    if fastcns:
        early = early.groupby(['CELL', 'CHR', 'BIN_REPINF'])\
                     .first().reset_index()\
                     .sort_values(['CHR', 'START', 'END'])\
                     .reset_index(drop=True)
        late = late.groupby(['CELL', 'CHR', 'BIN_REPINF'])\
                   .first().reset_index()\
                   .sort_values(['CHR', 'START', 'END'])\
                   .reset_index(drop=True)

    early['CNS_RTCNSTATE'] = identify_rt_within(early)
    late['CNS_RTCNSTATE'] = identify_rt_within(late)

    if fastcns:
        early = counts[counts['consistent_rt']=='early'].merge(early[['CELL', 'CHR', 'BIN_REPINF', 'CNS_RTCNSTATE']],
                                                               on=['CELL', 'CHR', 'BIN_REPINF'],
                                                               how='outer')\
                                                        .sort_values(['CHR', 'START', 'END'])\
                                                        .reset_index(drop=True)
        late = counts[counts['consistent_rt']=='late'].merge(late[['CELL', 'CHR', 'BIN_REPINF', 'CNS_RTCNSTATE']],
                                                             on=['CELL', 'CHR', 'BIN_REPINF'],
                                                             how='outer')\
                                                      .sort_values(['CHR', 'START', 'END'])\
                                                      .reset_index(drop=True)

    counts = pd.concat((early, late), axis=0)
    counts['CNS_RTCORR'] = counts.groupby('CNS_RTCNSTATE')['RDR_RTCORR'].transform('median')
    counts['RDR_RTCORR'] = (counts['RDR_RTCORR'] / counts['CNS_RTCORR']) * counts['MERGED_RDR_MEDIAN']
    counts['RDR_RTCORR'] = counts['RDR_RTCORR'].where((counts['CNS_RTCORR'] > 0.) &\
                                                      (~pd.isnull(counts['CNS_RTCORR'])) &\
                                                      (counts['MERGED_RDR_MEDIAN'] > 0.) &\
                                                      (~pd.isnull(counts['MERGED_RDR_MEDIAN'])), np.nan).clip(lower=minrdr, upper=maxrdr)
    assert counts['RDR_RTCORR'].mean() > 0 and (not np.isnan(counts['RDR_RTCORR'].mean()))
    counts['RDR_RTCORR'] = counts['RDR_RTCORR'] / counts['RDR_RTCORR'].mean()

    return counts


def identify_rt_within(celldf, val='CNS_RTPROFILE', sensitivity=0.99, alpha=.01, gamma=4, n_reps=10):
    X = celldf[val].values.reshape(-1, 1)
    X = X[LocalOutlierFactor(n_jobs=1).fit_predict(X) == 1][:, 0]
    X = X[(np.quantile(X, alpha) < X) & (X < np.quantile(X, 1. - alpha))].reshape(-1, 1)

    bic1 = min(bic(llf=hmm.GaussianHMM(n_components=1, covariance_type="tied", algorithm='viterbi', implementation='scaling',
                                    init_params='smct', params='smct', n_iter=10, random_state=None).fit(X).score(X), nobs=len(X), df_modelwc=gamma*2)
            for _ in range(n_reps))

    bic2 = min(bic(llf=hmm.GaussianHMM(n_components=2, covariance_type="tied", algorithm='viterbi', implementation='scaling',
                                    init_params='smct', params='smct', n_iter=10, random_state=None).fit(X).score(X), nobs=len(X), df_modelwc=gamma*12)
            for _ in range(n_reps))

    if bic1 < bic2:
        return celldf['RT_CN_STATE']
    else:
        bestscore = None
        bestmodel = None
        for _ in range(n_reps):
            model2 = hmm.GaussianHMM(n_components=2, covariance_type="tied", algorithm='viterbi', implementation='scaling',
                                    init_params='smt', params='smt', n_iter=10, random_state=None)
            model2.covars_ = np.array([[(X.std(ddof=1) / 2) ** 2, ], ])
            model2 = model2.fit(X)
            score = model2.score(X)
            if bestscore is None or score > bestscore:
                bestscore = score
                bestmodel = model2

        X = celldf[val].to_numpy().reshape(-1, 1)
        pred = pd.Series(bestmodel.predict(X), index=celldf.index)
        ref = pred.value_counts().idxmax()
        out = pred.value_counts().idxmin()
        assert out == 1 - ref
        params = list(zip(bestmodel.means_[:, 0], np.sqrt(bestmodel.covars_[:, 0][:, 0])))
        comp1, comp2 = scipy.stats.norm(*params[ref]), scipy.stats.norm(*params[out])
        llk1, llk2 = comp1.pdf(X)[:, 0], comp2.pdf(X)[:, 0]
        norm = llk1 + llk2
        pdf1, pdf2 = llk1 / norm, llk2 / norm
        pred = pd.Series(np.argmax((pdf1, pdf2), axis=0), index=celldf.index)
        return celldf['RT_CN_STATE'] + '_' + np.where((pred == 1) & (pdf2 > sensitivity), 1, 0).astype(str)


@jit(nopython=True, fastmath=False, cache=True)
def boot_rdrs_cn(values, index, num_repeats=10):
    isna = np.isnan(values)
    final_res = np.full(values.shape[0], np.nan, dtype=np.float64)
    if np.all(isna):
        return final_res
    else:
        X = values[~isna].copy()
        res = np.empty(X.shape[0], dtype=np.float64)
        repeats = np.empty(num_repeats, dtype=np.float64)
        sample = np.empty(X.shape[0], dtype=np.float64)
        for pos in range(X.shape[0]):
            for rep in range(repeats.shape[0]):
                for b in range(X.shape[0]):
                    sample[b] = X[np.random.randint(0, X.shape[0])]
                repeats[rep] = np.mean(sample)
            res[pos] = np.mean(repeats)
        res[0] = X[0]
        if res.shape[0] > 1:
            mean = res[1:].mean()
            if mean > 0 and (not np.isnan(mean)):
                res[1:] = res[1:] * (res[0] / mean)
        final_res[~isna] = res
        final_res[isna] = res.mean()
        return final_res


