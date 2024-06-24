from utils import *

from callcn import infer_cns_fixed
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor



def assign_s_clones(data, annotations, cn_g1g2, clones_g1g2, normal_clones, 
                    rescue_threshold=None, prop_sphase=0.7, fixedclones=None, max_norm_error=0.02, max_error=0.5, combine_cns=10, frac_covered=0.2, fastcns=True, jobs=1):
    clones, clone_prevalence, max_clone_error, normal_error, normal_clones, selected_clones = process_clones(cn_g1g2, clones_g1g2, normal_clones, max_error, data, fastcns)
    allbins = data[['CHR', 'START', 'END']].drop_duplicates().sort_values(['CHR', 'START', 'END']).reset_index(drop=True)
    assert np.all(np.in1d(clones_g1g2['CELL'].unique(), annotations['CELL'].unique()))
    assert np.all(np.in1d(cn_g1g2['CELL'].unique(), data['CELL'].unique()))
    assert np.all(np.in1d(annotations['CELL'].unique(), data['CELL'].unique())) and np.all(np.in1d(data['CELL'].unique(), annotations['CELL'].unique()))
    cells_to_assign = np.setdiff1d(annotations['CELL'].unique(), clones_g1g2['CELL'].unique())
    assign_data = data[data['CELL'].isin(cells_to_assign)][['CELL', 'CHR', 'START', 'END', 'GENOME', 'BIN_CNSINF', 'BIN_GLOBAL', 'MERGED_CN_STATE', 'RDR_CN']]
    assert np.all(np.in1d(annotations[annotations['IS-S-PHASE']==True]['CELL'], cells_to_assign))
    cn_assignedcells, assignments = (None, None)
    with Manager() as manager:
        shared_cns = manager.list()
        shared_clones = manager.list()
        shared_probs = manager.list()
        with Pool(processes=jobs, 
                  initializer=init_assign_cells, 
                  initargs=(manager.dict({cell : celldf for cell, celldf in assign_data.groupby('CELL')}),
                            manager.dict({clone : clodf for clone, clodf in clones.groupby('CLONE')}),
                            manager.dict(clone_prevalence.to_dict()),
                            manager.dict(fixedclones) if fixedclones is not None else None,
                            shared_cns,
                            shared_clones,
                            shared_probs,
                            combine_cns,
                            frac_covered,
                            manager.dict(max_clone_error.to_dict()),
                            normal_clones,
                            max(normal_error, max_norm_error),
                            fastcns,
                            allbins)) \
        as pool:
            bar = ProgressBar(total=len(cells_to_assign), length=30, verbose=False)
            progress = (lambda e : bar.progress(advance=True, msg="Cell {}".format(e)))
            bar.progress(advance=False, msg="Started")
            _ = [cell for cell in pool.imap_unordered(assign_cells, cells_to_assign) if progress(cell)]
        cn_assignedcells = pd.concat(shared_cns)
        assignments = pd.DataFrame([r for r in shared_clones])
        pd.concat(shared_probs).to_csv('probs_assignments.tsv.gz', sep='\t', index=False)
    assert set(cn_assignedcells['CELL'].unique()) == set(cells_to_assign)
    assert 'CLONE' not in cn_assignedcells.columns
    cn_all = pd.concat([cn_g1g2[cn_g1g2['CELL'].isin(clones_g1g2['CELL'].unique())], cn_assignedcells], axis=0, ignore_index=True)
    clones_all = rescue_assignments(assignments, normal_clones, annotations, prop_sphase, selected_clones, clones_g1g2, normal_error, max_norm_error)
    clones_all['IS_REASSIGNED'] = clones_all['CELL'].isin(cells_to_assign)
    assert set(data['CELL'].unique()) == set(cn_all['CELL'].unique()) and set(data['CELL'].unique()) == set(clones_all['CELL'].unique())
    return cn_all, clones_all, assignments, clones


def process_clones(cn_g1g2, clones_g1g2, normal_clones, max_error, data, fastcns):
    selected_clones = clones_g1g2[['CELL', 'CLONE']].drop_duplicates()
    clone_prevalence = selected_clones.groupby('CLONE')['CELL'].nunique() / len(selected_clones)
    selected_clones_cn = selected_clones.merge(cn_g1g2, on='CELL', how='left')
    selected_clones_cn['CN_CLONE'] = selected_clones_cn.groupby(['CLONE', 'CHR', 'START', 'END'])['CN_TOT']\
                                                       .transform(lambda values, index : np.argmax(np.bincount(values.astype(np.int32))),
                                                                                         engine='numba',
                                                                                         engine_kwargs={'nopython': True, 'nogil': True, 'cache' : True, 'fastmath' : False})
    selected_clones_cn['CL_CN_ER'] = (selected_clones_cn['CN_CLONE'] != selected_clones_cn['CN_TOT']).astype(int)
    selected_clones['CLONE_ERROR'] = selected_clones['CELL'].map(selected_clones_cn.groupby('CELL')['CL_CN_ER'].apply(lambda err : err.sum() / len(err)))
    max_clone_error = selected_clones.groupby('CLONE')['CLONE_ERROR'].apply(lambda clone_error : np.quantile(clone_error, 0.99, method='closest_observation')).clip(0., max_error)
    max_clone_error = pd.Series(index=max_clone_error.index, data=max_clone_error.max())
    clones = selected_clones_cn[['CLONE', 'CHR', 'START', 'END', 'CN_CLONE']]\
                               .drop_duplicates()\
                               .sort_values(['CLONE', 'CHR', 'START', 'END'])\
                               .reset_index(drop=True)
    normal_error = selected_clones[selected_clones['CLONE'].isin(normal_clones)]['CLONE_ERROR'].max() if len(normal_clones) > 0 else 0.
    if len(normal_clones) == 0:
        norclone = clones['CLONE'].max() + 1
        normal_clones = np.array([norclone])
        assert norclone not in clones['CLONE'].unique() and norclone not in clones_g1g2['CLONE'].unique()
        addnor = clones[['CHR', 'START', 'END']].drop_duplicates().reset_index(drop=True)
        addnor['CLONE'] = norclone
        addnor['CN_CLONE'] = 2
        clones = pd.concat((clones, addnor), axis=0, ignore_index=True).sort_values(['CLONE', 'CHR', 'START', 'END']).reset_index(drop=True)
    clone_bins = clones[['CHR', 'START', 'END']].drop_duplicates().sort_values(['CHR', 'START', 'END']).reset_index(drop=True)
    assert clone_bins.equals(cn_g1g2[['CHR', 'START', 'END']].drop_duplicates().sort_values(['CHR', 'START', 'END']).reset_index(drop=True))
    assert clone_bins.equals(data[['CHR', 'START', 'END']].drop_duplicates().sort_values(['CHR', 'START', 'END']).reset_index(drop=True))
    return clones, clone_prevalence, max_clone_error, normal_error, normal_clones, selected_clones


def init_assign_cells(_data, _clones, _prevalence, _fixedclones, _shared, _shared_clones, _shared_probs, _combine, _frac_covered, _maxdiff, _normal_clones, _normal_error, _fastcns, _allbins):
    global DATA, CLONES, PREVALENCE, FIXEDCLONES, SHARED, SHAREDCLONES, SHAREDPROBS, COMBINE, FRACCOVERED, MAXDIFF, NORMALCLONES, NORMALERROR, FASTCNS, ALLBINS
    DATA = _data
    CLONES = _clones
    PREVALENCE = _prevalence
    FIXEDCLONES = _fixedclones
    SHARED = _shared
    SHAREDCLONES = _shared_clones
    SHAREDPROBS = _shared_probs
    COMBINE= _combine
    FRACCOVERED = _frac_covered
    MAXDIFF = _maxdiff
    NORMALCLONES = _normal_clones
    NORMALERROR = _normal_error
    FASTCNS = _fastcns
    ALLBINS = _allbins


def assign_cells(cell):
    cnsinf = None
    if not FASTCNS:
        celldf = DATA[cell].sort_values(['CHR', 'START', 'END']).reset_index(drop=True)
    else:
        celldf = DATA[cell].groupby(['CELL', 'CHR', 'BIN_CNSINF'])\
                           .first().reset_index()\
                           .sort_values(['CHR', 'START', 'END'])\
                           .reset_index(drop=True)
        cnsinf = DATA[cell][['CHR', 'START', 'END', 'BIN_CNSINF']].sort_values(['CHR', 'START', 'END']).drop_duplicates().reset_index(drop=True)

    celldf = celldf[~pd.isnull(celldf['RDR_CN'])].reset_index(drop=True)
    cnsinf = cnsinf.merge(celldf[['CHR', 'BIN_CNSINF']].drop_duplicates(), on=['CHR', 'BIN_CNSINF'], how='inner')\
                   .sort_values(['CHR', 'START', 'END'])\
                   .reset_index(drop=True)

    mgcn_size = celldf['MERGED_CN_STATE'].value_counts() / celldf.shape[0]
    mgcn_size = mgcn_size[mgcn_size > 0.02].index
    lb_std = celldf[celldf['MERGED_CN_STATE'].isin(mgcn_size)].groupby('MERGED_CN_STATE')['RDR_CN']\
                                                              .apply(lambda X : scipy.stats.norm.fit(X, floc=X.mean())[1]).min()\
             if len(mgcn_size) > 0 else\
             celldf.groupby('MERGED_CN_STATE')['RDR_CN'].apply(lambda X : scipy.stats.norm.fit(X, floc=X.mean())[1]).min()

    assert not celldf['RDR_CN'].isna().any(), celldf
    assert celldf['RDR_CN'].mean() > 0., celldf

    def clone_difference(clone):
        with open(os.devnull, 'w') as devnull, contextlib.redirect_stderr(devnull): #, contextlib.redirect_stdout(devnull):
            resdf = celldf.copy()
            clodf = CLONES[clone].reset_index(drop=True)
            resdf['CN_TOT'] = infer_cns_fixed(resdf, ploidy=np.mean([clodf.sample(len(clodf), replace=True)['CN_CLONE'].mean() for _ in range(50)]), ispoisson=False)

            if FASTCNS:
                clodf = clodf.merge(cnsinf, on=['CHR', 'START', 'END'], how='inner').sort_values(['CHR', 'START', 'END'])
                clodf['CN_CLONE'] = clodf.groupby(['CHR', 'BIN_CNSINF'])['CN_CLONE']\
                                         .transform(lambda values, index : np.argmax(np.bincount(values.astype(np.int32))),
                                                                             engine='numba',
                                                                             engine_kwargs={'nopython': True, 'nogil': True, 'cache' : True, 'fastmath' : False})
                clodf = clodf.groupby(['CHR', 'BIN_CNSINF'])\
                             .first().reset_index()\
                             .sort_values(['CHR', 'START', 'END'])\
                             .drop(columns=['BIN_CNSINF'])\
                             .reset_index(drop=True)
                assert clodf[['CHR', 'START', 'END']].reset_index(drop=True).equals(resdf[['CHR', 'START', 'END']].reset_index(drop=True))

            resdf = resdf.merge(clodf.assign(CELL=cell), on=['CELL', 'CHR', 'START', 'END'], how='inner')\
                         .sort_values(['CHR', 'START', 'END'])\
                         .reset_index(drop=True)
            loglk = est_clone_loglk(resdf, lb_std, clone)
            cell_diff = (resdf['CN_TOT'].astype(int) != resdf['CN_CLONE'].astype(int)).astype(int).sum() / len(resdf)

            if not FASTCNS:
                resdf = DATA[cell].merge(resdf[['CELL', 'CHR', 'START', 'END', 'CN_TOT']], on=['CELL', 'CHR', 'START', 'END'], how='outer')
            else:
                resdf = DATA[cell].merge(resdf[['CELL', 'CHR', 'BIN_CNSINF', 'CN_TOT']], on=['CELL', 'CHR', 'BIN_CNSINF'], how='outer')

            resdf = resdf.merge(CLONES[clone].assign(CELL=cell), on=['CELL', 'CHR', 'START', 'END'], how='outer')\
                         .sort_values(['CHR', 'START', 'END']).reset_index(drop=True)
            resdf['CN_TOT'] = resdf['CN_TOT'].where(~pd.isnull(resdf['CN_TOT']), resdf['CN_CLONE'])

            assert resdf[['CHR', 'START', 'END']].equals(CLONES[clone][['CHR', 'START', 'END']].sort_values(['CHR', 'START', 'END']).reset_index(drop=True))
            assert (~pd.isnull(resdf['CN_TOT']).any()) and (~pd.isnull(resdf['CN_CLONE']).any())
            return resdf[['CELL', 'CHR', 'START', 'END', 'GENOME', 'BIN_CNSINF', 'BIN_GLOBAL', 'MERGED_CN_STATE', 'RDR_CN', 'CN_TOT']], \
                   cell_diff, \
                   loglk + (np.log(PREVALENCE[clone]) if clone in PREVALENCE else min(PREVALENCE.values()))

    marginals = [clone_difference(clone) + (clone, MAXDIFF[clone] if clone not in NORMALCLONES else NORMALERROR) for clone in CLONES.keys()]
    if FIXEDCLONES is not None and cell in FIXEDCLONES:
        clone = FIXEDCLONES[cell]
        candidates = [clone_difference(clone) + (clone, MAXDIFF[clone] if clone not in NORMALCLONES else NORMALERROR)]
    else:
        candidates = list(filter(lambda x : x[1] <= x[-1], marginals))
    chosen = max(candidates, key=(lambda x : x[2])) if len(candidates) > 0 else max(marginals, key=(lambda x : x[2]))
    assert chosen[3] in CLONES.keys()
    assert chosen[1] <= chosen[4] or len(candidates) == 0 or (FIXEDCLONES is not None and cell in FIXEDCLONES)
    assert chosen[0][['CHR', 'START', 'END']].equals(DATA[cell][['CHR', 'START', 'END']].sort_values(['CHR', 'START', 'END']).reset_index(drop=True))
    assert ALLBINS.equals(chosen[0][['CHR', 'START', 'END']])

    SHAREDPROBS.append(pd.DataFrame([{'CELL' : cell,
                                      'CLONE' : cand[3],
                                      'MAX_CLONE_DIST' : cand[4],
                                      'DIST' : cand[1],
                                      'POST' : cand[2],
                                      'IS_MAPPED' : len(candidates) > 0}
                                      for cand in (candidates if len(candidates) > 0 else marginals)]))
    SHARED.append(chosen[0])    
    SHAREDCLONES.append({'CELL' : cell, 'CLONE' : chosen[3], 'MAX_CLONE_DIST' : chosen[4], 'DIST' : chosen[1], 'POST' : chosen[2], 'IS_MAPPED' : len(candidates) > 0})
    return cell


def est_clone_loglk(_data, lb_std, clone, ispoisson=False):
    assert not pd.isnull(_data['RDR_CN']).any()
    assert not pd.isnull(_data['CN_CLONE']).any()
    data = _data[['RDR_CN', 'CN_CLONE'] if not ispoisson else ['RDR_CN', 'CN_CLONE', 'COUNT']].reset_index(drop=True)
    norm = (lambda v : v / v.mean())
    if not ispoisson:
        rdrs = norm(data['RDR_CN'].to_numpy())
    else:
        scale = data['COUNT'].median()
        rdrs = (data['RDR_CN'] * scale).round().astype(int).to_numpy()
    ploidy = data['CN_CLONE'].mean()
    llkdf = pd.DataFrame({'RDR' : rdrs, 'CN' : data['CN_CLONE'].to_numpy()})
    get_llk = (lambda X, mean : scipy.stats.norm.logpdf(X, loc=mean, scale=max(scipy.stats.norm.fit(X, floc=mean)[1], lb_std)).sum())
    return llkdf.groupby('CN')['RDR'].apply(lambda x : get_llk(x, x.name / ploidy)).sum()


def rescue_assignments(assignments, normal_clones, annotations, prop_sphase, selected_clones, clones_g1g2, normal_error, max_norm_error):
    normal_cells = np.union1d(clones_g1g2[clones_g1g2['CLONE'].isin(normal_clones)]['CELL'].unique(),
                              assignments[assignments['CLONE'].isin(normal_clones)]['CELL'].unique())
    s_cancer_cells = annotations[(annotations['IS-S-PHASE']==True) & (~annotations['CELL'].isin(normal_cells))]['CELL'].unique()
    g_cancer_cells = annotations[(annotations['IS-S-PHASE']==False) & (~annotations['CELL'].isin(normal_cells))]['CELL'].unique()
    assert set(np.union1d(np.union1d(s_cancer_cells, g_cancer_cells), normal_cells)) == set(annotations['CELL'].unique())
    dist_g_cells = selected_clones[selected_clones['CELL'].isin(g_cancer_cells)][['CELL', 'CLONE', 'CLONE_ERROR']].rename(columns={'CLONE_ERROR' : 'DIST'})
    dist_g_cells = pd.concat((dist_g_cells, assignments[assignments['CELL'].isin(g_cancer_cells)][['CELL', 'CLONE', 'DIST']]), axis=0, ignore_index=True)
    dist_g_cells = dist_g_cells.sort_values('DIST', ascending=True).reset_index(drop=True)
    assert set(dist_g_cells['CELL'].unique()) == set(g_cancer_cells)
    dist_s_cells = assignments[assignments['CELL'].isin(s_cancer_cells)][['CELL', 'CLONE', 'DIST']].sort_values('DIST', ascending=True).reset_index(drop=True)
    assert set(dist_s_cells['CELL'].unique()) == set(s_cancer_cells)
    assign_cancer_cells = pd.concat((dist_g_cells[['CELL', 'CLONE']].head(int(round(len(dist_g_cells) * prop_sphase))),
                                     dist_s_cells[['CELL', 'CLONE']].head(int(round(len(dist_s_cells) * prop_sphase)))), axis=0, ignore_index=True)
    assign_cancer_cells = pd.concat((assign_cancer_cells,
                                     pd.DataFrame({'CELL' : np.setdiff1d(np.union1d(g_cancer_cells, s_cancer_cells), assign_cancer_cells['CELL'].unique()), 'CLONE' : np.nan})), 
                                     axis=0, ignore_index=True)
    assignments_normals = assignments[assignments['CLONE'].isin(normal_clones)].reset_index(drop=True)
    assign_normal_cells = pd.concat((clones_g1g2[clones_g1g2['CELL'].isin(normal_cells)][['CELL', 'CLONE']],
                                     assignments_normals[assignments_normals['DIST'] <= max(normal_error, max_norm_error)][['CELL', 'CLONE']]), axis=0, ignore_index=True)
    assign_normal_cells = pd.concat((assign_normal_cells,
                                     pd.DataFrame({'CELL' : np.setdiff1d(normal_cells, assign_normal_cells['CELL'].unique()), 'CLONE' : np.nan})), 
                                     axis=0, ignore_index=True)
    all_assignments = pd.concat((assign_cancer_cells, assign_normal_cells), axis=0, ignore_index=True)
    assert set(all_assignments['CELL'].unique()) == set(annotations['CELL'].unique())
    return all_assignments


def infer_rescue_threshold(distances, dist_quantile=.05, max_components=5):
    get_thres = (lambda gmm : min(list(zip(gmm.means_[:,0], np.sqrt(gmm.covariances_.T[0][0]))), key=(lambda p : p[0])))
    gmm_thres = (lambda X : get_thres(min((GaussianMixture(n_components=n, n_init=20).fit(X) for n in range(1, max_components)), key=(lambda gmm : gmm.bic(X)))))
    get_inter = (lambda params : scipy.stats.norm.interval(dist_quantile, loc=params[0], scale=params[1])[1])
    est_thres = get_inter(gmm_thres(distances.reshape(-1, 1)))
    plt.figure()
    sns.histplot(distances, bins=100, stat='density')
    plt.axvline(x=est_thres, color='red')
    plt.title(est_thres)
    plt.savefig('threshold_distance_rescue.png', dpi=300, bbox_inches='tight')
    return est_thres
