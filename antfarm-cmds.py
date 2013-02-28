# GET CONFIGS
from video_analysis import viz_vidtools
from glob import glob

n_configs = 20

vids = [v for v in glob('*/merge6mbit_720_*.mp4') if not '59.94fps' in v and not '-0_' in v and not os.path.exists(viz_vidtools.cfg_fn(v))]

for vid in vids[:n_configs]:
    if os.path.exists(viz_vidtools.cfg_fn(vid)):
        print >> sys.stderr, 'configuration for %s present' % vid
    else:
        viz_vidtools.get_antfarm_config(vid)     



# SUBMIT RUNS
import os,sys,re
from subprocess import Popen, PIPE
from glob import glob

seglen = 1800
q = 'unrestricted_serial'

for cfg in sorted(glob('*/*-config.dict')):
    currjobs = Popen('bjobs -w',shell=True,stdout=PIPE).stdout.read()
    print >> sys.stderr, cfg,'\t',
    vid = cfg.split('-config')[0]+'.mp4'
    if vidtools.vid_duration(vid) < 8*60*60: #only analyze videos longer than 8hrs
        print >> sys.stderr, 'too short; skip'
        continue
    donebase = '%s-l%snp60nf300ns4' % (vid[:-4],seglen)
    if os.path.exists(donebase+'.done'):
        print >> sys.stderr, 'done'
    elif donebase in currjobs:
        print >> sys.stderr, 'running'
    else:
        cmd = 'summarize_segment_opencv.py -l %s -s 60 -nf 300 -np 60 -ns 4 -gi 0.03 -ac %s -vs np60nf300ns4 %s' % (seglen,cfg,vid)
        logfile = donebase+'.lsflog'
        ss = run_safe.safe_script(cmd,donebase,force_write=True)
        subcmd = 'bsub -q %s -o %s %s' % (q,logfile,ss)
        ret = os.system(subcmd)

