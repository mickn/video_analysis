#!/usr/bin/env python
import os,sys,LSF
from PIL import Image
from glob import glob

if len(sys.argv[1:]) > 1 or sys.argv[1].endswith('*.png'):
    for f in sys.argv[1:]:
        im = Image.open(f)
        try:
            h=im.histogram()
        except:
            print 'invalid image: ' + f
            os.unlink(f)
else:
    outfile = sys.argv[1]+'/../drop-broken-pngs-log'
    print >> sys.stderr, 'master running, target dir %s output in %s' % (sys.argv[1],outfile)
    cmds = []
    images = glob(sys.argv[1]+'/*.png')
    for i in range(0,len(images),1000):
        cmds.append(sys.argv[0]+' '+(' '.join(images[i:i+1000])))
    jids,ndict = LSF.lsf_jobs_submit(cmds,outfile,jobname_base='pngdrop',num_batches=400)
    LSF.lsf_wait_for_jobs(jids,restart_outfile=outfile,namedict=ndict,restart_z=12)
