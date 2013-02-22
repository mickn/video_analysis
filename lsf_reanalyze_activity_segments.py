#!/usr/bin/env python
#
# lsf_reanalyze_activity_segments.py vid source_adir pad new_seglen
#
# identifies segments containing activity in <source_adir>
# pads by <pad>+1 intervals before and <pad> intervals following, extracts these frames from <vid>
# and launches reanalyses via LSF as:
# reanalyze_activity_segments.py <source_adir> <resulting_new_tdir> <new_seglen>

import os,sys, re, Util, LSF, time

from glob import glob
from video_analysis import vidtools

vid, source_adir, pad, new_seglen = sys.argv[1:]

print >> sys.stderr, 'reanalyze activity in %s from analysis in %s' % (vid,source_adir)

new_seglen=float(new_seglen)
match = re.search('^(.+)/([\d\w-]+)/([\d\.]+)fps.+?([\d\.]+)sec',source_adir)
if match:
    source_root = match.groups()[0]
    source_analysis_window = match.groups()[1]
    offset = 0
    if '-' in source_analysis_window:
        start = source_analysis_window.split('-')[0]
        if start != 'start':
            offset = int(start)
    fps = int(match.groups()[2])
    seglen = float(match.groups()[3])
    segment_step = int(seglen * fps)
    if segment_step == 0: segment_step = 1
    print >> sys.stderr, 'segment length was %s frames (%s seconds, %s fps) at %s sec offset\nNew length %s frames (%s sec)' % (segment_step,seglen,fps,offset,new_seglen*fps,new_seglen)
    unit = 'sec'
else:
    print >> sys.stderr,'fps and segment length could not be found in image path, please ensure these appear in source_adir (%s given)\n' % source_adir
    sys.exit(1)

print >> sys.stderr, 'determine activity intervals:'
print >> sys.stderr, '\tload'
newactouts = [(f.split('/')[-1],eval(open(f).read())) for f in sorted(glob(source_adir+'/*.newactout'))]
print >> sys.stderr, '\tcalc'
act_intervals = [[int(i) for i in k.split('.')[0].split('-')] for k,v in newactouts if len(v)>0]
frames_in = set([])
fpad = segment_step*int(pad)
null = [[frames_in.add(f) for f in range(i-fpad,j+fpad)] for i,j in act_intervals]
frames = sorted(list(frames_in))
wins = [(int(s/fps)+offset,int(e/fps)+offset) for s,e in Util.get_consecutive_value_boundaries(frames)]
print >> sys.stderr, 'The following windows will be re-analysed:\n\t%s\n' % wins

cropsdict = os.path.dirname(vid)+'/cropsdict.dict'
if os.path.exists(cropsdict):
    mouse = os.path.basename(source_root)
    cropsdict = eval(open(cropsdict).read())
    crops = cropsdict[mouse]
    print >>sys.stderr, 'crop frames: '+str(crops)
else:
    crops = None

exp_paths = ['%s/%05d-%05d/%sfps/' % (source_root,s,e,fps) for s,e in wins]
exist_paths = [p for p in exp_paths if os.path.exists(p)]
nonex_wins = [(s,e) for s,e in wins if not os.path.exists('%s/%05d-%05d/%sfps/' % (source_root,s,e,fps))]
if all([os.path.exists(p) for p in exp_paths]):
    print >> sys.stderr, ('\n'.join(exp_paths))+'\nall exist'
    tdirs = exp_paths
else:
    print >> sys.stderr, '%s exist, running %s' % (exist_paths,nonex_wins)
    new_tdirs = vidtools.parallel_v2p(vid, fps,tdir=source_root,queue='short_serial',num_jobs=20,crops=crops,extract_windows=nonex_wins)
    tdirs = exist_paths + new_tdirs

cmds = ['reanalyze_activity_segments.py %s %s %s' % (source_adir, d, new_seglen) for d in tdirs]

logfile = source_root+'/reanalyze-log'

final_summary = source_root+'/%0.1fsec_summary.pdf' % new_seglen
if os.path.exists(final_summary):
    print >>sys.stderr, 'merged summary %s exists; will not reanalyze' % final_summary
    do = False
else:
    do = True
passes = 0
while do and passes < 3:
    jids,ndict = LSF.lsf_jobs_submit(cmds,logfile,'normal_serial',jobname_base='reanalyze')
    LSF.lsf_wait_for_jobs(jids,logfile,ndict)
    time.sleep(10)
    unfinished = LSF.lsf_no_success_from_log(logfile)
    if unfinished:
        print >> sys.stderr, 'not finished: %s' % unfinished
        cmds = unfinished
    else:
        do = False
    passes += 1

if not os.path.exists(final_summary):
    print >> sys.stderr, 'write summary to '+final_summary
    os.system('gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite -sOutputFile=%s %s/*-*/30fps/analysis/1.0sec*/summary.pdf' % (final_summary,source_root))

        

