#!/usr/bin/env python

'''given a folder containing antfarm trials folders (like _2012cross)
launches not-yet-completed, not-currently-running summarize_segment_opencv.py

'''
import os,sys,re,time
from subprocess import Popen,PIPE
from glob import glob
import vidtools
import run_safe

MIN_VID_DUR = 8*60*60 #only analyze videos longer than 8hrs
QUEUE = 'long_serial'
MAX_RETRY = 2
DEFAULT_START=60
#DEFAULT_START=60*60*6 #6 hours in, to get to BW digs pronto 20130328
STOP_ON_MZOPT = False #if True, will not submit new jobs if any mzopt jobs are submitted or running
RERUN_COEFFS = [1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0]
MAX_ITERTIME = 480

DEFAULT_PARAMS = {'nframes':300, \
                  'nparts':60, \
                  'nstep':4, \
                  'param_queue':'normal_serial', \
                  'queue':'normal_serial', \
                  'seglen':1800, \
                  'ground_improvement':0.03, \
                  'ground_suppress':10, \
                  'outline_engine':'shapely', \
                  }


def get_current_run_count():
    return int(Popen('bjobs -w | grep %s | grep merge6mbit_720_ | wc -l' % QUEUE,shell=True,stdout=PIPE).stdout.read().strip())

skip_fn = lambda vid: os.path.splitext(vid)[0]+'-SKIP'
vid_from_cfg = lambda cfg: cfg.split('-config')[0]+'.mp4'
from viz_vidtools import cfg_fn as cfg_from_vid

def submit_one(cfg, default_start, \
               nframes,nparts,nstep,param_queue,seglen, \
               ground_improvement,ground_suppress,\
               outline_engine, \
               thresh_coeff=None, \
               run_local=False,**kwargs):

    vid = vid_from_cfg(cfg)
    donebase,vs = donebase_from_param(vid, \
                        nframes,nparts,nstep,seglen, \
                        ground_improvement,ground_suppress,\
                        outline_engine, \
                        thresh_coeff=thresh_coeff,return_vs=True)
    attempts_dir = donebase+'-ATTEMPTS'

    if thresh_coeff:
        cmd = 'summarize_segment_opencv.py -l %s -s %s -nf %s -np %s -ns %s -q %s -tc %s -gi %s -gs %s -oe %s -ac %s -vs %s --max_itertime %s %s' % (seglen,default_start,nframes,nparts,nstep,param_queue,thresh_coeff,ground_improvement,ground_suppress,outline_engine,cfg,vs, MAX_ITERTIME, vid)
    else:
        cmd = 'summarize_segment_opencv.py -l %s -s %s -nf %s -np %s -ns %s -q %s -gi %s -gs %s -oe %s -ac %s -vs %s --max_itertime %s %s' % (seglen,default_start,nframes,nparts,nstep,param_queue,ground_improvement,ground_suppress,outline_engine,cfg,vs,MAX_ITERTIME, vid)
    
    logfile = donebase+'.lsflog'
    ss = run_safe.safe_script(cmd,donebase,force_write=True)
    subcmd = 'bsub -q %s -o %s %s' % (QUEUE,logfile,ss)
    #print >> sys.stderr, '\n\t',subcmd
    if run_local:
        ret = os.system(ss)
    else:
        ret = os.system(subcmd)
    if ret == 0:
        if not os.path.exists(attempts_dir): os.makedirs(attempts_dir)
        at_ret = os.system('touch %s' % os.path.join(attempts_dir,'attempt'+time.strftime('%Y%m%d-%H%M%S')))
        if ret != 0:
            print >> sys.stderr, 'WRITING ATTEMPT FLAG TO %s FAILED' % os.path.join(attempts_dir,'attempt'+time.strftime('%Y%m%d-%H%M%S'))
    else:
        errstr = 'submission of job failed:\n%s' % subcmd
        raise OSError, errstr

    
def submit_runs(vidroot,default_start, \
                nframes,nparts,nstep,param_queue,seglen, \
                ground_improvement,ground_suppress, \
                outline_engine, \
                num_jobs_running_max,num_jobs_new_max,skip_fn=skip_fn):
    num_current = get_current_run_count()
    num_new = num_jobs_running_max - num_current
    if num_new < 1:
        print >> sys.stderr, 'number of currently running jobs (%s) meets or exceeds max concurrent (%s)' % (num_current, num_jobs_running_max)
        return None
    else:
        launched = 0
        for cfg in sorted(glob(os.path.join(vidroot,'*/*-config.dict'))):
            if launched == num_new or launched >= num_jobs_new_max: break
            currjobs = Popen('bjobs -w',shell=True,stdout=PIPE).stdout.read()
            print >> sys.stderr, cfg,'\t',
            vid = cfg.split('-config')[0]+'.mp4'
            if not os.path.exists(vid):
                print >> sys.stderr, 'video removed; skipping'
                continue
            if os.path.exists(skip_fn(vid)):
                print >> sys.stderr, 'skip flag found; skipping'
                continue
            if vidtools.vid_duration(vid) < MIN_VID_DUR: #only analyze videos longer than 8hrs
                print >> sys.stderr, 'too short; skip'
                continue
            donebase = '%s-l%snp%snf%sns%sgi%sgs%soe%s' % (vid[:-4],seglen,nparts,nframes,nstep,ground_improvement,ground_suppress,outline_engine)
            vs = 'np%snf%sns%sgi%sgs%soe%s' % (nparts,nframes,nstep,ground_improvement,ground_suppress,outline_engine)
            attempts_dir = donebase+'-ATTEMPTS'
            if os.path.exists(donebase+'.done'):
                print >> sys.stderr, 'done'
            elif donebase in currjobs:
                print >> sys.stderr, 'running'
            elif os.path.exists(attempts_dir) and len(glob(os.path.join(attempts_dir,'attempt*'))) >= MAX_RETRY:
                nrrc = next_rerun_condition(cfg,RERUN_COEFFS, \
                                            nframes,nparts,nstep,seglen, \
                                            ground_improvement,ground_suppress, \
                                            outline_engine,return_state=True)
                if nrrc is None:
                    print >> sys.stderr, 'too many attempts (%s) for all conditions (%s); see %s' % (len(glob(os.path.join(attempts_dir,'attempt*'))),RERUN_COEFFS,attempts_dir)
                else:
                    thresh_coeff,state = nrrc
                    print >> sys.stderr, 'rerun %s %s' % (thresh_coeff,state)
            else:
                cmd = 'summarize_segment_opencv.py -l %s -s %s -nf %s -np %s -ns %s -q %s -gi %s -gs %s -oe %s -ac %s -vs %s %s' % (seglen,default_start,nframes,nparts,nstep,param_queue,ground_improvement,ground_suppress,outline_engine,cfg,vs, vid)
                logfile = donebase+'.lsflog'
                ss = run_safe.safe_script(cmd,donebase,force_write=True)
                subcmd = 'bsub -q %s -o %s %s' % (QUEUE,logfile,ss)
                #print >> sys.stderr, '\n\t',subcmd
                ret = os.system(subcmd)
                launched += 1
                if ret == 0:
                    if not os.path.exists(attempts_dir): os.makedirs(attempts_dir)
                    at_ret = os.system('touch %s' % os.path.join(attempts_dir,'attempt'+time.strftime('%Y%m%d-%H%M%S')))
                    if ret != 0:
                        print >> sys.stderr, 'WRITING ATTEMPT FLAG TO %s FAILED' % os.path.join(attempts_dir,'attempt'+time.strftime('%Y%m%d-%H%M%S'))
                else:
                    errstr = 'submission of job failed:\n%s' % subcmd
                    raise OSError, errstr

def submit_reruns(vidroot,default_start,rerun_coeffs, \
                nframes,nparts,nstep,param_queue,seglen, \
                ground_improvement,ground_suppress, \
                outline_engine, \
                num_jobs_running_max,num_jobs_new_max,skip_fn=skip_fn):

    num_current = get_current_run_count()
    num_new = num_jobs_running_max - num_current

    if num_new < 1:
        print >> sys.stderr, 'number of currently running jobs (%s) meets or exceeds max concurrent (%s)' % (num_current, num_jobs_running_max)
        return None
    
    failed = get_failed_analyses(vidroot,rerun_coeffs, \
                                 nframes,nparts,nstep,seglen, \
                                 ground_improvement,ground_suppress, \
                                 outline_engine, \
                                 skip_fn=skip_fn)

    launched = 0
    for cfg in failed:
        if launched == num_new or launched >= num_jobs_new_max: break
        thresh_coeff = next_rerun_condition(cfg,rerun_coeffs, \
                                            nframes,nparts,nstep,seglen, \
                                            ground_improvement,ground_suppress, \
                                            outline_engine)
        if thresh_coeff is not None:
            print >> sys.stderr, '%s at %s\t' % (cfg,thresh_coeff),
        else:
            print >> sys.stderr, '%s has exhausted rerun conditions; no longer queueing' % (cfg)
            continue
        
        submit_one(cfg, default_start, \
                   nframes,nparts,nstep,param_queue,seglen, \
                   ground_improvement,ground_suppress,\
                   outline_engine, \
                   thresh_coeff=thresh_coeff)
        launched += 1

def get_currjobs():
    return Popen('bjobs -w',shell=True,stdout=PIPE).stdout.read()
        
def get_state(donebase,currjobs=None):
    if currjobs is None:
        currjobs = get_currjobs()
    
    attempts_dir = donebase+'-ATTEMPTS'
    if not os.path.exists(donebase+'.sh'):
        return 'absent'
    elif os.path.exists(donebase+'.done'):
        #print >> sys.stderr, 'done'
        return 'done'
    elif donebase in currjobs:
        #print >> sys.stderr, 'running'
        return 'running'
    elif os.path.exists(attempts_dir) and len(glob(os.path.join(attempts_dir,'attempt*'))) >= MAX_RETRY:
        #print >> sys.stderr, 'too many attempts (%s); see %s' % (len(glob(os.path.join(attempts_dir,'attempt*'))),attempts_dir)
        return 'failed'
    else:
        return 'ready'

def get_successful_analysis_dir(vid,rerun_coeffs, \
                        nframes,nparts,nstep,seglen, \
                        ground_improvement,ground_suppress,\
                        outline_engine, currjobs=None, \
                        **kwargs):
    donebase = donebase_from_param(vid,\
                        nframes,nparts,nstep,seglen, \
                        ground_improvement,ground_suppress,\
                        outline_engine, thresh_coeff=None)
    if get_state(donebase,currjobs) == 'done':
        return analysis_dir_from_vid(vid, \
                        nframes,nparts,nstep,seglen, \
                        ground_improvement,ground_suppress,\
                        outline_engine, \
                        thresh_coeff=None)
    else:
        for thresh_coeff in rerun_coeffs:
            donebase = donebase_from_param(vid,\
                        nframes,nparts,nstep,seglen, \
                        ground_improvement,ground_suppress,\
                        outline_engine, thresh_coeff=thresh_coeff)
            if get_state(donebase,currjobs) == 'done':
                return analysis_dir_from_vid(vid, \
                        nframes,nparts,nstep,seglen, \
                        ground_improvement,ground_suppress,\
                        outline_engine, \
                        thresh_coeff=thresh_coeff)

def analysis_dir_from_vid(vid, \
                        nframes,nparts,nstep,seglen, \
                        ground_improvement,ground_suppress,\
                        outline_engine, \
                        thresh_coeff=None,**kwargs):
    donebase = donebase_from_param(vid,\
                        nframes,nparts,nstep,seglen, \
                        ground_improvement,ground_suppress,\
                        outline_engine, \
                        thresh_coeff=None,return_vs=False,**kwargs)
    b,p = donebase.rsplit('-l%s' % seglen,1)
    if thresh_coeff is None:
        tcs = ''
    else:
        tcs = '_coeff-%0.3f' % thresh_coeff
    return glob('%s/analysis/*_seg%s%s*%s' % (b,seglen,tcs,p))[0]

def donebase_from_param(vid, \
                        nframes,nparts,nstep,seglen, \
                        ground_improvement,ground_suppress,\
                        outline_engine, \
                        thresh_coeff=None,return_vs=False,**kwargs):
    if thresh_coeff is not None:
        donebase = '%s-l%snp%snf%sns%stc%sgi%sgs%soe%s' % (vid[:-4],seglen,nparts,nframes,nstep,thresh_coeff,ground_improvement,ground_suppress,outline_engine)
    else:
        donebase = '%s-l%snp%snf%sns%sgi%sgs%soe%s' % (vid[:-4],seglen,nparts,nframes,nstep,ground_improvement,ground_suppress,outline_engine)

    if return_vs:
        vs = 'np%snf%sns%sgi%sgs%soe%s' % (nparts,nframes,nstep,ground_improvement,ground_suppress,outline_engine)
        return donebase,vs
    else:
        return donebase

def donebase_from_analysis_dir(analysis_dir):
    try:
        base,seglen,coeff,parstr = re.search('(^.+?)/analysis.+?_seg(\d+)_coeff-([\d\.]+)-.+?-(np.+$)',analysis_dir).groups()
        coeff = coeff.rstrip('0')
        if coeff[-1] == '.':
            coeff += '0'
        parstr = parstr.replace('gi','tc%sgi' % coeff)
    except:
        base,seglen, parstr = re.search('(^.+?)/analysis.+?_seg(\d+)_.+?-(np.+$)',analysis_dir).groups()
    donebase = '%s-l%s%s' % (base,seglen,parstr.rstrip('/'))
    return donebase

def analysis_dir_from_donebase(donebase):
    base,seglen,parstr = re.search('(^.+?)-l(\d+)(np.+$)',donebase).groups()
    if 'tc' in parstr:
        coeff = re.search('tc([\d\.]+)gi',parstr).groups()[0]
        parstr = re.sub('tc[\d\.]+gi','gi',parstr)
        globstr = '%s/analysis/*_seg%s_coeff-%0.3f-*%s' % (base,seglen,float(coeff),parstr)
    else:
        globstr = '%s/analysis/*_seg%s_auto*-%s' % (base,seglen,parstr)
    cand_dir = glob(globstr)
    if len(cand_dir) == 1:
        return cand_dir[0]
    else:
        errstr = '%s candidates found for %s: %s' % (len(cand_dir),globstr,cand_dir)
        raise ValueError, errstr

def next_rerun_condition(cfg,rerun_coeffs, \
                         nframes,nparts,nstep,seglen, \
                         ground_improvement,ground_suppress, \
                         outline_engine,return_state=False,**kwargs):
    """DOES NOT GUARANTEE THAT THE RETURNED CONDITION ISN'T RUNNING --
    CHECK BEFORE PROCEEDING!"""
    
    vid = vid_from_cfg(cfg)
    for thresh_coeff in rerun_coeffs:
        donebase = donebase_from_param(vid,nframes,nparts,nstep,seglen,ground_improvement,ground_suppress,outline_engine,thresh_coeff)
        state = get_state(donebase)
        if not state == 'failed':
            if return_state:
                return thresh_coeff,state
            else:
                return thresh_coeff
    #if we haven't found one, that's it
    return None
        

def get_failed_analyses(vidroot,rerun_coeffs, \
                        nframes,nparts,nstep,seglen, \
                        ground_improvement,ground_suppress, \
                        outline_engine, \
                        skip_fn=skip_fn,**kwargs):

    failed = []
    currjobs = Popen('bjobs -w',shell=True,stdout=PIPE).stdout.read()
    for cfg in sorted(glob(os.path.join(vidroot,'*/*-config.dict'))):
        #print >> sys.stderr, cfg,'\t',
        vid = cfg.split('-config')[0]+'.mp4'
        if not os.path.exists(vid):
            #print >> sys.stderr, 'video removed; skipping'
            continue
        if os.path.exists(skip_fn(vid)):
            #print >> sys.stderr, 'skip flag found; skipping'
            continue
        if vidtools.vid_duration(vid) < MIN_VID_DUR: #only analyze videos longer than 8hrs
            #print >> sys.stderr, 'too short; skip'
            continue
        donebase = donebase_from_param(vid, \
                                       nframes,nparts,nstep,seglen, \
                                       ground_improvement,ground_suppress,\
                                       outline_engine)
        if get_state(donebase) == 'failed':
            tc_good = None
            for thresh_coeff in rerun_coeffs:
                donebase = donebase_from_param(vid, \
                                               nframes,nparts,nstep,seglen, \
                                               ground_improvement,ground_suppress,\
                                               outline_engine, \
                                               thresh_coeff)
                if get_state(donebase) in ['running','done']:
                    tc_good = thresh_coeff
                    break
            if tc_good is None:
                failed.append(cfg)
    return failed


if __name__ == "__main__":
    import argparse
    
    ds =  ' [%(default)s]'
    #command parser
    parser = argparse.ArgumentParser(description='submits summarize_segment_opencv.py runs for antfarm video with configs (see viz_vidtools.get_antfarm_config)')

    parser.add_argument('-nf','--nframes',default=DEFAULT_PARAMS['nframes'],type=int,help='number of frames to analyze in each segment during parameter fitting (only if -mt not set)'+ds)
    parser.add_argument('-np','--nparts',default=DEFAULT_PARAMS['nparts'],type=int,help='number of segments to analyze during parameter fitting (only if -mt not set)'+ds)
    parser.add_argument('-ns','--nstep',default=DEFAULT_PARAMS['nstep'],type=int,help='number of segments to space parameter fit parts by during parameter fitting (only if -mt not set)'+ds)
    parser.add_argument('-q','--queue',default=DEFAULT_PARAMS['queue'],type=str,help='LSF queue for parameter fitting (only if -mt not set)'+ds)
    
    parser.add_argument('-l','--seglen',default=DEFAULT_PARAMS['seglen'],type=int,help='number of frames to average for object tracking background subtraction; see vidtools.init_frames()'+ds)

    parser.add_argument('-gi','--ground_improvement',default=DEFAULT_PARAMS['ground_improvement'],type=float,help='percent improvement a new ground must make for acceptance (only relevant if antfarm_config used)'+ds)
    parser.add_argument('-gs','--ground_suppress',default=DEFAULT_PARAMS['ground_suppress'],type=int,help='number of pixels below ground to extend "above ground" mask (burrows only scored below this mask; only relevant if antfarm_config used)'+ds)

    parser.add_argument('-oe','--outline_engine',default=DEFAULT_PARAMS['outline_engine'],type=str,choices=['homebrew','shapely'],help='switch between homebrew and shapely chain_outlines calls (shapely should be faster and more accurate, but requires working shapely python library and libgeos_c)'+ds)

    parser.add_argument('-nr','--num_jobs_running_max',default=32,type=int,help='submit jobs only up to this many concurrently running'+ds)
    parser.add_argument('-nn','--num_jobs_new_max',default=16,type=int,help='submit only this many new jobs'+ds)
    
    parser.add_argument('vidroot',help='directory to process')
    
    opts = parser.parse_args()
    if not os.path.exists(opts.vidroot):
        raise ValueError, 'invalid vidroot supplied'

    if STOP_ON_MZOPT:
        currjobs = Popen('bjobs -w',shell=True,stdout=PIPE).stdout.read()
        if 'mzopt' in currjobs:
            print >> sys.stderr, 'mzopt jobs running; will not proceed unless STOP_ON_MZOPT is disabled (see source)'
            raise ValueError
        

    submit_runs(opts.vidroot,DEFAULT_START, \
                opts.nframes,opts.nparts,opts.nstep,opts.queue,opts.seglen, \
                opts.ground_improvement,opts.ground_suppress, \
                opts.outline_engine, \
                opts.num_jobs_running_max,opts.num_jobs_new_max)

    if get_current_run_count() < opts.num_jobs_running_max:
        print >> sys.stderr, 'first-pass finished; launch reruns'

        submit_reruns(opts.vidroot,DEFAULT_START,RERUN_COEFFS, \
                      opts.nframes,opts.nparts,opts.nstep,opts.queue,opts.seglen, \
                      opts.ground_improvement,opts.ground_suppress, \
                      opts.outline_engine, \
                      opts.num_jobs_running_max,opts.num_jobs_new_max)

