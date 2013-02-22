#!/usr/bin/env python
'''given a summary .mouselocs file and number of seconds per segment (e.g. written at the end of analyze_antfarm) draws a final scatter-and-pie summarizing all ground, mouse and activity calls'''

import os, sys, numpy
from video_analysis import viz_vidtools
from glob import glob

locsfile,sps = sys.argv[1:3]

analysis_dir = os.path.dirname(locsfile)
groundfile = locsfile.rsplit('.',1)[0]+'.ground'
if os.path.exists(groundfile):
    grounds = [numpy.fromfile(groundfile,sep='\n')]
else:
    grounds = [numpy.fromfile(g,sep='\n') for g in sorted(glob(analysis_dir+'/*.ground'))]

actoutfile = locsfile.rsplit('.',1)[0]+'.newactout'

if os.path.exists(actoutfile):
    actout = eval(open(actoutfile).read())
else:
    actout = eval(open(sorted(glob(analysis_dir+'/*.preactout'))[-1]).read()) + \
             eval(open(sorted(glob(analysis_dir+'/*.newactout'))[-1]).read())

print >> sys.stderr, 'loaded %s ground lines, %s activity polygons (%s total vertices) from %s' % (len(grounds), len(actout), sum([len(p) for p in actout]),analysis_dir)

viz_vidtools.draw_class_scatter_and_pie(locsfile,sps=float(sps),scatter_lines=grounds,scatter_polys=actout,draw_pie=False)
