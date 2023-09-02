from utils import *

import contextlib
from multiprocessing.sharedctypes import Value

from hmmlearn import hmm
from sklearn.mixture import GaussianMixture
from statsmodels.tools.eval_measures import bic



def globalhmm(_data, chrs, repetitions=20, max_components=10, gmm=True):
    data = _data.reshape(-1, 1)
    assert np.all(chrs[:-1] <= chrs[1:]), 'Inputs must be sorted by chromosomes when provided'
    argmax = (lambda L : max(L, key=(lambda v : v[0])))
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stderr(devnull):
            compute_n = (lambda n : argmax(seghmm_iter(data, chrs=chrs, n_components=n, gmm=gmm) for _ in range(repetitions)))
            return min((compute_n(n) for n in range(1, max_components + 1)), key=(lambda v : v[1]))


def seghmm_iter(data, chrs, n_components, seed=None, gmm=True):
    remodel = hmm.GaussianHMM(n_components=n_components, covariance_type="diag", algorithm='viterbi', 
                              init_params = '' if gmm else 'mc', params = 'smct', n_iter=10, random_state=seed)
    remodel.transmat_ = np.array([np.random.dirichlet(alphas) for alphas in (np.identity(n_components) * 200) + 1])
    remodel.startprob_ = np.ones(n_components) / n_components
    if gmm:
        gmm = GaussianMixture(n_components=n_components, covariance_type='diag', n_init=10, random_state=seed).fit(data)
        remodel.means_ = gmm.means_
        remodel.covars_ = np.tile(gmm.covariances_.T[0][0], (n_components, 1))
    fitted = remodel.fit(data)
    if any(~np.isclose(row.sum(), 1.) for row in fitted.transmat_):
        fitted.transmat_ = np.array([row if np.isclose(row.sum(), 1.) else np.ones_like(row) / len(row) for row in fitted.transmat_])
    if ~np.isclose(fitted.startprob_.sum(), 1.):
        fitted.startprob_ = np.ones_like(fitted.startprob_) / len(fitted.startprob_)
    _, hstates = zip(*[fitted.decode(gdata) for gdata in splitchr(data, chrs)])
    loglik = fitted.score(data.reshape(-1, 1))
    hstates = np.concatenate(hstates)
    used_comp = pd.Series(hstates).nunique()
    return (loglik, bic(llf=loglik, nobs=data.size, df_modelwc=n_components+(2*used_comp)+(used_comp**2)), hstates, n_components)


splitchr = (lambda array, splitby : np.split(array, np.unique(splitby, return_index=True)[1][1:]))


def make_transmat(diag, K):
    offdiag = (1 - diag) / (K - 1)
    transmat_ = np.diag([diag - offdiag] * K) 
    transmat_ += offdiag
    return transmat_


def seghmm(_data, chrs=None, repetitions=20, max_components=10, gmm=True, glob=True):
    data = _data.reshape(-1, 1)
    assert chrs is None or np.all(chrs[:-1] <= chrs[1:]), 'Inputs must be sorted by chromosomes when provided'
    argmax = (lambda L : max(L, key=(lambda v : v[0])))
    if chrs is None:
        compute_n = (lambda n : argmax(hmm_iter(data, n_components=n, gmm=gmm) for _ in range(repetitions)))
        return min((compute_n(n) for n in range(1, max_components + 1)), key=(lambda v : v[1]))
    elif glob:
        compute_n = (lambda n : argmax(seghmm_iter(data, chrs=chrs, n_components=n, gmm=gmm) for _ in range(repetitions)))
        return min((compute_n(n) for n in range(1, max_components + 1)), key=(lambda v : v[1]))
    else:
        compute_n = (lambda n, cdata : argmax(hmm_iter(cdata, n_components=n, gmm=gmm) for _ in range(repetitions)))
        ll, bic, hstates, ns = zip(*[min((compute_n(n, cdata) for n in range(1, max_components + 1)), key=(lambda v : v[1])) for cdata in splitchr(data, chrs)])
        return ll, bic, np.concatenate(hstates), ns


def hmm_iter(data, n_components, seed=None, gmm=True):
    remodel = hmm.GaussianHMM(n_components=n_components, covariance_type="diag", algorithm='viterbi', 
                              init_params = '' if gmm else 'mc', params = 'smct', n_iter=10, random_state=seed)
    with contextlib.redirect_stderr(None):
        try:
            remodel.transmat_ = np.array([np.random.dirichlet(alphas) for alphas in (np.identity(n_components) * 200) + 1])
            remodel.startprob_ = np.ones(n_components, dtype=float) / n_components
            if gmm:
                gmm = GaussianMixture(n_components=n_components, covariance_type='diag', random_state=seed).fit(data)
                remodel.means_ = gmm.means_
                remodel.covars_ = np.tile(gmm.covariances_.T[0][0], (n_components, 1))
            loglik, hstates = remodel.fit(data).decode(data)
        except ValueError:
            return (-10000000, 1000000, [], n_components)
    return (loglik, bic(llf=loglik, nobs=data.size, df_modelwc=n_components), hstates, n_components)

