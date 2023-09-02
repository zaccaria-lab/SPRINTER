from utils import *

import rtprofile as rtp
import repcall as rep



default_values = {
        'gccorr' : 'QUANTILE',
        'strictgc' : False,
        'nocorrgcintercept' : False,
        'timing' : 'consistent_rt',
        'gmm'  : False,
        'maxcn' : 10,
        'reps' : 10,
        'permutations' : 10000,
        'permethod' : 'permutation',
        'rdr_profile' : 'RT_PROFILE',
        'maxrdr' : 3.0,
        'maxrtrdr' : 3.0,
        'shared_rtprofiles' : dict(),
        'toplot' : False,
        'quantile' : .05
}


def run(_data,
        _seg,
        _testmethod,
        _meanmethod,
        jobs,
        gcbiases,
        gccorr=default_values['gccorr'],
        strictgc=default_values['strictgc'],
        nocorrgcintercept=default_values['nocorrgcintercept'],
        _timing=default_values['timing'],
        _gmm=default_values['gmm'],
        _maxcn=default_values['maxcn'],
        _reps=default_values['reps'],
        _permutations=default_values['permutations'],
        _permethod=default_values['permethod'],
        _rdr_profile=default_values['rdr_profile'],
        _maxrdr=default_values['maxrdr'],
        _maxrtrdr=default_values['maxrtrdr'],
        _quantile=default_values['quantile']):

     results, rtprofiles = (None, None)
     with Manager() as manager:
        shared_rtprofiles = manager.dict()
        with Pool(processes=min(jobs, _data['CELL'].nunique()),
                  initializer=init_run,
                  initargs=(manager.dict({cell : celldf for cell, celldf in _data.groupby('CELL')}), 
                            _seg,
                            _testmethod,
                            _meanmethod,
                            _timing,
                            _gmm,
                            _maxcn,
                            _reps,
                            _permutations,
                            _permethod,
                            _rdr_profile,
                            _maxrdr,
                            _maxrtrdr,
                            _quantile,
                            shared_rtprofiles,
                            gccorr,
                            strictgc,
                            nocorrgcintercept,
                            manager.dict(gcbiases) if gcbiases is not None else None)) as pool:

                jobs = tuple(zip(_data['CELL'].unique(), np.random.randint(1e6, size=_data['CELL'].nunique())))
                bar = ProgressBar(total=len(jobs), length=30, verbose=False)
                progress = (lambda e : bar.progress(advance=True, msg="Cell {}".format(e)))
                bar.progress(advance=False, msg="Started")
                results = pd.DataFrame([res for res in pool.imap_unordered(infer, jobs) if progress(res['CELL'])])
                rtprofiles = pd.concat((shared_rtprofiles[cell] for cell in shared_rtprofiles))
     return results, rtprofiles


def init_run(_data, 
             _seg,
             _testmethod,
             _meanmethod,
             _timing,
             _gmm,
             _maxcn,
             _reps,
             _permutations,
             _permethod,
             _rdr_profile,
             _maxrdr,
             _maxrtrdr,
             _quantile,
             _shared_rtprofiles,
             _gccorr,
             _strictgc,
             _nocorrgcintercept,
             _gcbiases):
     global data, timing, gmm, maxcn, reps, maxrdr, data, seg, permutations,\
            permethod, testmethod, meanmethod, timing, rdr_profile, maxrtrdr, quantile,\
            shared_rtprofiles, gccorr, strictgc, nocorrgcintercept, gcbiases
     data = _data
     seg = _seg
     testmethod = _testmethod
     meanmethod = _meanmethod
     timing = _timing
     gmm = _gmm
     maxcn = _maxcn
     reps = _reps
     permutations = _permutations
     permethod = _permethod
     rdr_profile = _rdr_profile
     maxrdr = _maxrdr
     maxrtrdr = _maxrtrdr
     quantile = _quantile
     shared_rtprofiles = _shared_rtprofiles
     gccorr = _gccorr
     strictgc = _strictgc
     nocorrgcintercept = _nocorrgcintercept
     gcbiases = _gcbiases


def infer(job):
     cell, seed = job
     np.random.seed(seed)
     return infer_local(cell, _data=data, _seg=seg, _testmethod=testmethod, _meanmethod=meanmethod, _timing=timing,
                        _gmm=gmm, _maxcn=maxcn, _reps=reps, _permutations=permutations, _permethod=permethod,
                        _rdr_profile=rdr_profile, _maxrdr=maxrdr, _maxrtrdr=maxrtrdr, _quantile=quantile,
                        shared_rtprofiles=shared_rtprofiles, gccorr=gccorr, strictgc=strictgc, nocorrgcintercept=nocorrgcintercept, gcbiases=gcbiases)


def infer_local(cell,
                _data,
                _seg,
                _testmethod,
                _meanmethod,
                gcbiases,
                gccorr=default_values['gccorr'],
                strictgc=default_values['strictgc'],
                nocorrgcintercept=default_values['nocorrgcintercept'],
                _timing=default_values['timing'],
                _gmm=default_values['gmm'],
                _maxcn=default_values['maxcn'],
                _reps=default_values['reps'],
                _permutations=default_values['permutations'],
                _permethod=default_values['permethod'],
                _rdr_profile=default_values['rdr_profile'],
                _maxrdr=default_values['maxrdr'],
                _maxrtrdr=default_values['maxrtrdr'],
                _quantile=default_values['quantile'],
                shared_rtprofiles=default_values['shared_rtprofiles'],
                toplot=default_values['toplot']):

        columns = ['CELL', 'CHR', 'START', 'END', 'GENOME', 'BIN_REPINF', 'COUNT', 'NORM_COUNT', _timing, 'GC', 'RAW_RDR']
        data = _data[cell][columns].drop_duplicates().sort_values(['CHR', 'START', 'END']).reset_index(drop=True)

        data, gccorr_feat = rtp.profile_local(cell,
                                              _data=data,
                                              seg=_seg,
                                              shared_rtprofiles=shared_rtprofiles,
                                              _timing=_timing,
                                              _gmm=_gmm, 
                                              _maxcn=_maxcn,
                                              _reps=_reps,
                                              _maxrdr=_maxrdr,
                                              toplot=toplot,
                                              gccorr=gccorr,
                                              strictgc=strictgc,
                                              nocorrgcintercept=nocorrgcintercept,
                                              gcbiases=gcbiases)
     
        results = rep.infersphase_local(cell,
                                        data,
                                        seg=_seg,
                                        permutations=_permutations,
                                        permethod=_permethod,
                                        testmethod=_testmethod,
                                        meanmethod=_meanmethod, 
                                        timing=_timing,
                                        rdr_profile=_rdr_profile,
                                        maxrtrdr=_maxrtrdr,
                                        quantile=_quantile,
                                        toplot=toplot)

        results.update(gccorr_feat)
        return results
      