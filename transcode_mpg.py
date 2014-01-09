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
            cmds.append('cat %s/M2U*.MPG > %s/merge.MPG; ffmpeg -i %s/merge.MPG -r 29.97 -b 6000k -an -y -s 480x320 %s/merge6mbit_320.mp4' % (d,d,d,d))

cmd = ';'.join(cmds)
shname = os.path.join(os.getcwd(),'run_mpg_ffmpeg.sh')
open(shname,'w').write('#!/usr/bin/env sh\n'+cmd+'\n')
os.system('chmod +x %s' % shname)
os.system('bsub -q hoekstra -o cat-mpg-trans-log "%s"' % shname)
#print cmd
                                                                            
