#!/usr/bin/env python

from video_analysis import submit_summarize_runs,viz_vidtools
import LSF,run_safe
import time,os,sys
from glob import glob
os.chdir('/n/hoekstrafs2/burrowing/antfarms/data/_2012cross/')
logfile = '../rbt-logs/log'
currjobs = submit_summarize_runs.get_currjobs()

analysis_dirs = filter(None,[submit_summarize_runs.get_successful_analysis_dir(vid,submit_summarize_runs.RERUN_COEFFS,currjobs=currjobs,**submit_summarize_runs.DEFAULT_PARAMS) for vid in sorted(glob('*/merge6mbit_720_*.mp4')) if os.path.exists(viz_vidtools.cfg_fn(vid)) and 'end' in open(viz_vidtools.cfg_fn(vid)).read()])
trd = {}
for analysis_dir in analysis_dirs:
    #print >> sys.stderr, analysis_dir
    rbtdone = os.path.join(analysis_dir,'rainbowtron')
    cmd = 'run_rainbowtron.py %s' % analysis_dir
    run_safe.add_cmd(trd,rbtdone,cmd,force_write=True)

    
    
LSF.lsf_run_until_done(trd,logfile,'normal_serial','','rainbow',100,3)
                    
