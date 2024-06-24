from utils import *

from scipy.cluster import hierarchy
from hmmlearn import hmm
from sklearn.cluster import DBSCAN
from statsmodels.tools.eval_measures import bic
from statsmodels.tools.eval_measures import aic
from sklearn.neighbors import LocalOutlierFactor
import statsmodels.formula.api as smf




def call_cns_g1g2(rawdata, annotations, max_ploidy=4, fixed_ploidy=None, fixed_cns=None, fastcns=True, frac_covered=0.2, jobs=1):
    fixed_ploidy = process_fixed_ploidy(fixed_ploidy) if fixed_ploidy is not None else None
    g1g2 = annotations[annotations['IS-S-PHASE']==False]['CELL'].unique()
    cn_g1g2 = rawdata[rawdata['CELL'].isin(g1g2)]
    if fixed_cns is not None:
        fixed_ploidy = fixed_cns.groupby('CELL')['CN_TOT'].mean().to_dict()
    with Manager() as manager:
        shared = manager.list()
        with Pool(processes=jobs, 
                  initializer=init_call_cns, 
                  initargs=(manager.dict({cell : celldf for cell, celldf in cn_g1g2.groupby('CELL')}),
                            max_ploidy,
                            shared,
                            manager.dict(fixed_ploidy) if fixed_ploidy is not None else None,
                            manager.dict({cell : celldf for cell, celldf in fixed_cns.groupby('CELL')}) if fixed_cns is not None else None,
                            fastcns)) \
        as pool:
            bar = ProgressBar(total=cn_g1g2['CELL'].nunique(), length=30, verbose=False)
            progress = (lambda e : bar.progress(advance=True, msg="Cell {}".format(e)))
            bar.progress(advance=False, msg="Started")
            _ = [cell for cell in pool.imap_unordered(call_cns, (cell for cell in cn_g1g2['CELL'].unique())) if progress(cell)]
        cn_g1g2 = pd.concat(shared)
    return cn_g1g2


def init_call_cns(_data, _max_ploidy, _shared, _fixed_ploidy, _fixed_cns, _fastcns):
    global DATA, MAXPLOIDY, SHARED, FIXED_PLOIDY, FIXED_CNS, FASTCNS
    DATA = _data
    MAXPLOIDY = _max_ploidy
    SHARED = _shared
    FIXED_PLOIDY = _fixed_ploidy
    FIXED_CNS = _fixed_cns
    FASTCNS = _fastcns


def call_cns(cell):
    if not FASTCNS:
        celldf = DATA[cell].sort_values(['CHR', 'START', 'END']).reset_index(drop=True)
    else:
        celldf = DATA[cell].groupby(['CELL', 'CHR', 'BIN_CNSINF'])\
                           .first().reset_index()\
                           .sort_values(['CHR', 'START', 'END'])\
                           .reset_index(drop=True)
        
    ret_cell = ''
    assert (not celldf['RDR_CN'].isna().any()) & (not celldf['RDR_CN'].apply(np.isinf).any())
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stderr(devnull): #, contextlib.redirect_stdout(devnull):
            if FIXED_PLOIDY is None or cell not in FIXED_PLOIDY:
                celldf['CN_TOT'] = infer_cns_free(celldf, repetitions=5, max_ploidy=MAXPLOIDY, ispoisson=False)[2]            
            else:
                celldf['CN_TOT'] = infer_cns_fixed(celldf, ploidy=FIXED_PLOIDY[cell], ispoisson=False)
                ret_cell = '_fixep'
    if FIXED_CNS is not None and cell in FIXED_CNS:
        overlapped_fixed = overlap_fixed_cns(celldf, FIXED_CNS[cell])
        celldf['CN_TOT'] = overlapped_fixed.where(~pd.isnull(overlapped_fixed), celldf['CN_TOT'])
        ret_cell += '_fixedcns'

    if not FASTCNS:
        SHARED.append(celldf[['CELL', 'CHR', 'START', 'END', 'GENOME', 'BIN_CNSINF', 'BIN_GLOBAL', 'MERGED_CN_STATE', 'RDR_CN', 'CN_TOT']])
    else:
        SHARED.append(DATA[cell][['CELL', 'CHR', 'START', 'END', 'GENOME', 'BIN_CNSINF', 'BIN_GLOBAL', 'MERGED_CN_STATE', 'RDR_CN']]\
                                .merge(celldf[['CELL', 'CHR', 'BIN_CNSINF', 'CN_TOT']],
                                       on=['CELL', 'CHR', 'BIN_CNSINF'],
                                       how='outer')\
                                .sort_values(['CHR', 'START', 'END'])\
                                .reset_index(drop=True))
    return cell + ret_cell


def infer_cns_free(data, repetitions, max_ploidy, min_aneu=0.1, ispoisson=False, min_ref_rdr=0.4):
    size_state = data.groupby('MERGED_CN_STATE')['START'].count()
    sel_states = data.groupby('MERGED_CN_STATE')['RDR_CN'].median()
    sel_states = sel_states[sel_states > min_ref_rdr]
    ref_state = size_state[sel_states.index].idxmax() if len(sel_states) > 0 else size_state.idxmax()
    ref = data[data['MERGED_CN_STATE'] == ref_state]['RDR_CN'].to_numpy()    
    get_var = (lambda X : X.clip(*scipy.stats.norm.interval(0.99, *scipy.stats.norm.fit(X))).var(ddof=1))
    var = np.median(scipy.stats.trimboth(data.groupby('MERGED_CN_STATE')['RDR_CN'].transform(get_var).dropna(), 0.05))
    segs = data[['MERGED_CN_STATE']].reset_index(drop=True)
    segs['SEG'] = ((segs['MERGED_CN_STATE'] != segs['MERGED_CN_STATE'].shift(1)) & 
                   (~(pd.isnull(segs['MERGED_CN_STATE']) & pd.isnull(segs['MERGED_CN_STATE'].shift(1))))).cumsum()
    segs = segs[segs['SEG'].isin((segs.groupby('SEG')['MERGED_CN_STATE'].count() > 2).index)]
    trans = segs['SEG'].nunique() / segs.shape[0]
    if not ispoisson:
        rdrs = data['RDR_CN'].to_numpy()
        brdr = np.median(ref)
    else:
        scale = data['COUNT'].median()
        rdrs = (data['RDR_CN'] * scale).round().astype(int).to_numpy()
        brdr = np.round((np.median(ref) * scale))
    argmax = (lambda L : max(L, key=(lambda v : v[0])))
    compute_n = (lambda cn : argmax(infer_cns_hmm(rdrs, gamma=cn/brdr, var=var, trans=trans, ispoisson=ispoisson) for _ in range(repetitions)))
    candidate = min((compute_n(n) for n in range(2, max_ploidy + 1)), key=(lambda v : v[1]))
    if ((candidate[2] != pd.Series(candidate[2]).value_counts().index[0]).sum() / len(candidate[2])) >= min_aneu:
        return candidate
    else:
        return compute_n(2)
    

def infer_cns_fixed(data, ploidy, ispoisson=False):
    assert not pd.isnull(data['RDR_CN']).any()
    get_var = (lambda X : X.clip(*scipy.stats.norm.interval(0.99, *scipy.stats.norm.fit(X))).var(ddof=1))
    var = np.median(scipy.stats.trimboth(data.groupby('MERGED_CN_STATE')['RDR_CN'].transform(get_var).dropna(), 0.05))
    segs = data[['MERGED_CN_STATE']].reset_index(drop=True)
    segs['SEG'] = ((segs['MERGED_CN_STATE'] != segs['MERGED_CN_STATE'].shift(1)) & 
                   (~(pd.isnull(segs['MERGED_CN_STATE']) & pd.isnull(segs['MERGED_CN_STATE'].shift(1))))).cumsum()
    segs = segs[segs['SEG'].isin((segs.groupby('SEG')['MERGED_CN_STATE'].count() > 2).index)]
    trans = segs['SEG'].nunique() / segs.shape[0]
    norm = (lambda v : v / v.mean())
    if not ispoisson:
        rdrs = norm(data['RDR_CN'].to_numpy())
    else:
        scale = data['COUNT'].median()
        rdrs = norm((data['RDR_CN'] * scale).round().astype(int).to_numpy())
    return infer_cns_hmm(rdrs, gamma=ploidy, var=var, trans=trans, ispoisson=ispoisson)[2]


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
        remodel.transmat_ = (np.identity(n_components) * (1. - trans - (trans / (n_components - 1)))) + (trans / (n_components - 1))
        assert np.all(remodel.transmat_ > 0.)
        assert all(np.isclose(row.sum(), 1.) for row in remodel.transmat_)
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

    return (loglik, score, hstates, used_comp)


def overlap_fixed_cns(ref, fixed):
    assert set(fixed.columns) == {'CHR', 'S', 'E', 'CN_TOT'}

    def overlap_cns(R, F):
        comb = R.merge(F, on='CHR', how='left')
        comb['OVERLAP'] = np.minimum(comb['END'], comb['E']) - np.maximum(comb['START'], comb['S'])
        return comb.groupby(['CHR', 'START', 'END'], sort=False).apply(lambda X : X.sort_values('OVERLAP', ascending=False).iloc[0])

    result = ref.groupby('CHR', sort=False).apply(lambda X : overlap_cns(X, fixed[fixed['CHR']==X.name]))
    assert ref[['CHR', 'START', 'END']].reset_index(drop=True).equals(result[['CHR', 'START', 'END']].reset_index(drop=True))
    return result['CN_TOT']


def process_fixed_ploidy(fdata):
    fixed_ploidy = pd.read_csv(fdata, sep='\t')
    return

