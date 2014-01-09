#!/usr/bin/env python

import os
from video_analysis import vidtools

cmds = []
for d in os.listdir('.'):
    if os.path.isdir(d):
        try:
            t = vidtools.vid_duration(d+'/merge6mbit_720.mp4')
            print d,t
        except:
            print d,'FAIL'
            cmds.append('cat %s/0*.MTS > %s/merge.MTS; ffmpeg -i %s/merge.MTS -r 29.97 -b 6000k -an -y -s 1280x720 %s/merge6mbit_720.mp4' % (d,d,d,d))

cmd = ';'.join(cmds)
shname = os.path.join(os.getcwd(),'run_mts_ffmpeg.sh')
open(shname,'w').write('#!/usr/bin/env sh\n'+cmd+'\n')
os.system('chmod +x %s' % shname)
os.system('bsub -q hoekstra -o cat-trans-log "%s"' % shname)
#print cmd
                                                                            
