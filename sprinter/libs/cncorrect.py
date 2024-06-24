
from utils import *



def correct_sphase_cns(data, pvals, cn_all, cn_clones, clones_all, thres_sphase_cns=5e6):
    assert not pd.isnull(data['consistent_rt']).any()
    assert not pd.isnull(cn_all['CN_TOT']).any()
    assert not cn_clones.isnull().any().any()

    cn_all['CLONE'] = cn_all['CELL'].map(clones_all.set_index('CELL')['CLONE'])
    cn_all = cn_all.merge(cn_clones[['CLONE', 'CHR', 'START', 'END', 'CN_CLONE']], on=['CLONE', 'CHR', 'START', 'END'], how='left')\
                   .sort_values(['CELL', 'CHR', 'START', 'END'])\
                   .reset_index(drop=True)
    assert data[['CELL', 'CHR', 'START', 'END']].equals(cn_all[['CELL', 'CHR', 'START', 'END']])

    cn_all['consistent_rt'] = data['consistent_rt']
    cn_all['BIN_REPINF'] = data['BIN_REPINF']
    cn_all['GC'] = data['GC']
    cn_all['RDR'] = data['RDR']
    cn_all['RAW_CN_TOT'] = cn_all['CN_TOT'].copy().astype(int)
    cn_all['CN_TOT'] = cn_all['CN_TOT'].astype(int)
    cn_all['WIDTH'] = cn_all['END'] - cn_all['START']

    cn_all['IS-S-PHASE'] = cn_all['CELL'].map(pvals.set_index('CELL')['IS-S-PHASE'])
    cn_all['SEG_CLONE'] = (((cn_all['CLONE'] != cn_all['CLONE'].shift(1)) |\
                            (cn_all['CELL'] != cn_all['CELL'].shift(1)) |\
                            (cn_all['CHR'] != cn_all['CHR'].shift(1)) |\
                            (cn_all['CN_CLONE'] != cn_all['CN_CLONE'].shift(1))) &\
                           (~pd.isnull(cn_all['CN_CLONE']))).astype(int).cumsum().where(~pd.isnull(cn_all['CN_CLONE']), np.nan)

    cn_all['SEG_WIDTH'] = cn_all.groupby('SEG_CLONE')['WIDTH'].transform('sum')
    cn_all['IS_MIXED_RT'] = cn_all.groupby('SEG_CLONE', dropna=False)['consistent_rt'].transform(lambda x : 'early' in x.values and 'late' in x.values)

    cn_all['CN_TOT'] = cn_all['CN_CLONE'].where(cn_all['IS-S-PHASE'] &\
                                                (~pd.isnull(cn_all['CN_CLONE'])) &\
                                                ((cn_all['SEG_WIDTH'] <= thres_sphase_cns) | (~cn_all['IS_MIXED_RT'])),
                       cn_all['CN_TOT'])

    assert not pd.isnull(cn_all['CN_TOT']).any()
    return cn_all[['CELL', 'CHR', 'START', 'END', 'GENOME', 'GC', 'consistent_rt', 'BIN_REPINF', 'BIN_CNSINF', 'BIN_GLOBAL',
                   'MERGED_CN_STATE', 'RDR', 'RDR_CN', 'RAW_CN_TOT', 'CN_TOT', 'CLONE', 'CN_CLONE']].reset_index(drop=True),\
           (cn_all['RAW_CN_TOT'] != cn_all['CN_TOT']).sum() / len(cn_all)

