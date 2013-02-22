#!/usr/bin/env python

'''given an analysis directory, draws a .png summary of mouse locations, prior and new burrows, and ground line for each segment

NB 20091105 ASSUMES OFF-BY-ONE error where all burrowing information for a given segment has filenames corresponding to the SUBSEQUENT segment, i.e. each .newactout outlines activity from the previous segment
'''

import os, sys, re, Util, pylab
from glob import glob
from video_analysis import viz_vidtools

adir = sys.argv[1]

sps = re.search('([\d+])sec_',adir).groups()[0]

newacts = sorted(glob(adir+'/*-*.newactout'))[:-1]
preacts = sorted(glob(adir+'/*-*.preactout'))[:-1]
actterms = sorted(glob(adir+'/*-*.newactterm'))[:-1]
locs = sorted(glob(adir+'/*-*.mouselocs'))
for i in range(len(newacts)):
    na = filter(None,eval(open(newacts[i]).read()))
    pa = filter(None,eval(open(preacts[i]).read()))
    polys = pa+na
    poly_col = ['y']*len(pa) + ['b']*len(na)
    ml = locs[i+1]
    viz_vidtools.draw_class_scatter_and_pie(ml,'png',sps=float(sps),scatter_polys=polys,poly_col=poly_col,draw_pie=False)
    at = eval(open(actterms[i]).read())
    try:
        x,y = Util.dezip(at[0])
        pylab.plot(x,y,'c')
    except:
        pass
