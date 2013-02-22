#!/usr/bin/env python
'''given a .mice file and coords for the fork and two gates,

writes a .mouselocs where:
not found = 0
stem = 1
top fork = 2
bottom fork = 3

'''

out_of_bounds_call = False

import os, sys, re, Util, vidtools,numpy
from glob import glob

(start,stop,segment_summ_dir,splitxy,topxy,bottomxy) = sys.argv[1:7]
(start,stop) = (int(start),int(stop))
splitxy = eval(splitxy)
topxy = eval(topxy)
bottomxy = eval(bottomxy)

print >> sys.stderr, 'classifying mouse coords from files %s to %s from %s.' % (start,stop,segment_summ_dir)
print >> sys.stderr, 'split: %s top gate: %s bottom gate: %s' % (splitxy,topxy,bottomxy)
micefiles = sorted(glob(segment_summ_dir+'/*.mice'))[start:stop]

for mf in micefiles:
    md = eval(open(mf).read())
    mouselocs = {}
    for f,m in md.items():
        if m is not None:
            if m[0] < splitxy[0]-20:
                if m[1] < splitxy[1]:
                    if m[0] > topxy[0] and m[1] >= topxy[1]-10:
                        mouselocs[f] = 2
                    elif out_of_bounds_call:
                        mouselocs[f] = 0
                    else:
                        mouselocs[f] = None
                else:
                    if m[0] > bottomxy[0] and m[1] <= bottomxy[1]+10:
                        mouselocs[f] = 3
                    elif out_of_bounds_call:
                        mouselocs[f] = 0
                    else:
                        mouselocs[f] = None
            elif m[0] > splitxy[0]+20:
                mouselocs[f] = 1
            else:
                mouselocs[f] = 4
        else:
            mouselocs[f] = 0
    outname = mf.rsplit('.',1)[0]+'.mouselocs'
    open(outname,'w').write(mouselocs.__repr__())
    outname = mf.rsplit('.',1)[0]+'.locsumm'
    summ = Util.countdict(mouselocs.values())
    open(outname,'w').write(summ.__repr__())

