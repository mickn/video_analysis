#!/usr/bin/env python
import Util,sys,os,shutil
from glob import glob

vids = sys.argv[1:]

baseroot = os.path.split(vids[0])[0]
outroot = os.path.join(baseroot,'frames')
try:
    os.makedirs(outroot)
except OSError:
    pass

cmds = []
for i,v in enumerate(vids):
    out = '%03d_' % (i)
    out = os.path.join(outroot,out)
    cmds.append('ffmpeg -i %s %s%%07d.png' % (v, out))

outfile = os.path.join(outroot,'split-log')
jobids = Util.lsf_jobs_submit(cmds,outfile,'normal_serial')
Util.lsf_wait_for_jobs(jobids,outfile)

lastsrc = 0
lastframe = 0
inc = 0
for f in sorted(glob(outroot+'/*.png')):
    dn,fn = os.path.split(f)
    framenums = [int(i) for i in fn.rsplit('.',1)[0].split('_')]
    if framenums[0] != lastsrc:
        inc += lastframe
        lastsrc = framenums[0]
    frame = framenums[1]+inc
    shutil.move(f,os.path.join(dn,'%07d.png' % frame))
    lastframe = frame
    

cmd = 'ffmpeg -i %s/%%07d.png -r 30  %s/merge.mov' % (outroot,baseroot)
outfile = os.path.join(outroot,'merge-log')
jobids = Util.lsf_jobs_submit([cmd],outfile,'normal_serial')
Util.lsf_wait_for_jobs(jobids,outfile)
