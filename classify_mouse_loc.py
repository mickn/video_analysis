#!/usr/bin/env python
'''load files in order from start to stop from segment_summ_dir
 (i.e. the directory in which .mice, .ground and .frame files have been deposited)
 
options are:
start,stop,segment_summ_dir,pad_mask,shapey,shapex
OR:
start,stop,segment_summ_dir,pad_mask,shapey,shapex,actzcut

 writes a .mouselocs file for [1:-1]'''

import os, sys, re, Util, vidtools,numpy
from glob import glob

(start,stop,segment_summ_dir,pad_mask) = sys.argv[1:5]
shape = [int(i) for i in sys.argv[5:7]]
pad_mask = int(pad_mask)
if len(sys.argv) == 8:
	actzcut = float(sys.argv[7])
else:
	actzcut = 6

(start,stop) = (int(start),int(stop))

print >> sys.stderr, 'classifying mouse coords from files %s to %s from %s (frames %s). Activity above z score of %s extended by %s' % (start,stop,segment_summ_dir,shape,actzcut,pad_mask)

micefiles = sorted(glob(segment_summ_dir+'/*.mice'))[start:stop]
mice = [eval(open(f).read()) for f in micefiles]
#allmice = reduce(lambda x,y:dict(x.items()+y.items()),mice)

groundfiles = sorted(glob(segment_summ_dir+'/*.ground'))[start:stop]
#print >> sys.stderr, 'grounds: %s' % groundfiles
grounds = [numpy.fromfile(f,sep = '\n') for f in groundfiles]
print >> sys.stderr, 'loaded %s grounds' % len(grounds)
depress_ground = 0

framefiles = sorted(glob(segment_summ_dir+'/*.frame'))[start:stop]
print >> sys.stderr, 'frames: %s' % [(f,os.path.getsize(f)) for f in framefiles]
frames = [numpy.fromfile(f).reshape(shape) for f in framefiles]
print >> sys.stderr, 'loaded %s frames' % len(frames)


for i,f in enumerate(frames[1:-1]): # this throws off numbering--must be i+1 for current index!
	#construct output filenames, check for pre-existing files.  Skip if present.
	filebase = micefiles[i+1].rsplit('.',1)[0]
	actmatf = filebase+'.actmat'
	mouselocsf = filebase+'.mouselocs'
	locsummf = filebase+'.locsumm'
#	if all([os.path.exists(f) for f in (actmatf,mouselocsf,locsummf)]):
#		continue

	mouselocs = {}
	g = grounds[i+1] + depress_ground
	actmat = Util.zscore(frames[i] - frames[i+2])
	actmask = actmat > actzcut
	#output actmask! (use frame output model from summarize)
	print >>sys.stderr,actmatf,len(actmat),len(actmat[0])
	actmat.tofile(actmatf)
	actmask = vidtools.grow_mask(actmask,pad_mask)
	for mk, mv in mice[i+1].items():
		mouselocs[mk] = vidtools.classify_mouse(mv,g,actmask)
	open(mouselocsf,'w').write(mouselocs.__repr__())
	summ = Util.countdict(mouselocs.values())
	open(locsummf,'w').write(summ.__repr__())
