#!/usr/bin/env python
'''a config.dict file must reside in <tpath>

it must contain a string representation of a dictionary with the following values (at least):
burrow_entrance : (x,y)
ground_anchors : [(x1,y1),(x2,y2), ... ]
hill_bounds : (x_left,x_right) 
'''


import os, sys, Util

#vdir,tpath = sys.argv[1:]
#
#if tpath == '.':
#    tpath = 'all/30fps/'
#tdir = vdir+tpath

## tdir now searched recursively for config.dict
tdir = sys.argv[1]
config_file = Util.file_in_path('config.dict',tdir)

wins =   [30,60,90]
mousez = [9,12,15]
burrowz = [3]
pixavs = [5]
timeavs = [3,5,7]


#o = 9
#u = 3
#p = 5
#t = 5

f = None    #-f None disables video output
#f = 'mp4'    #-f 'mp4' enables video output

config = eval(open(config_file).read())

cmd = 'bsub -o %s -q hoekstra "' % (tdir+'/analyze-antfarm-log')

params = []
for w in wins:
    for o in mousez:
        for u in burrowz:
            for p in pixavs:
                for t in timeavs:
                    params.append((w,o,u,p,t))

for w,o,u,p,t in params: 
    if 'xybounds' in config.keys():
        cmd += r'/n/home08/brantp/code/video_analysis/analyze_antfarm.py -s %s -b \"%s\" -x \"%s\" -g \"%s\" -l \"%s\" -q normal_serial -o %s -u %s -p %s -t %s -n 20 -f %s %s ; ' % (w,config['burrow_entrance'],config['xybounds'],config['ground_anchors'],config['hill_bounds'],o,u,p,t,f,tdir)
    else:
        cmd += r'/n/home08/brantp/code/video_analysis/analyze_antfarm.py -s %s -b \"%s\" -g \"%s\" -l \"%s\" -q normal_serial -o %s -u %s -p %s -t %s -n 20 -f %s %s ; ' % (w,config['burrow_entrance'],config['ground_anchors'],config['hill_bounds'],o,u,p,t,f,tdir)

cmd += '"'

print >> sys.stderr, cmd
os.system(cmd)
