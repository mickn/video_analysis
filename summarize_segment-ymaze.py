#!/usr/bin/env python
'''finds mouse and ground, and stores an averaged frame for frames from start to stop
options are:
maskfile,burrow_maskfile,start,stop,imagedir,outroot
OR:
maskfile,burrow_maskfile,start,stop,imagedir,outroot,mousezcut
OR:
maskfile,burrow_maskfile,start,stop,imagedir,outroot,ybound-low,ybound-high
OR:
maskfile,burrow_maskfile,start,stop,imagedir,outroot,mousezcut,ybound-low,ybound-high
'''

import vidtools,os,sys,re,Util,numpy
from glob import glob

def get_win(i,start,stop,wins):
	return i/((stop-start)/wins)

pixav = 5
timeav = 3

#(maskfile,burrow_maskfile,start,stop,wins,imagedir,outroot) = sys.argv[1:8]
#(start,stop,wins) = (int(start),int(stop),int(wins))
#tlen = stop-start
(maskfile,burrow_maskfile,start,stop,imagedir,outroot) = sys.argv[1:7]
(start,stop) = (int(start),int(stop))


if len(sys.argv) == 8:
	mousezcut = float(sys.argv[7])
	ybounds = None
elif len(sys.argv) == 9:
	mousezcut = None
	ybounds = [int(i) for i in sys.argv[-2:]]
elif len(sys.argv) == 10:
	mousezcut = float(sys.argv[7])
	ybounds = [int(i) for i in sys.argv[-2:]]
else:
	mousezcut = None
	ybounds = None

print >> sys.stderr, 'summarizing frames %s to %s from %s. output to %s. Masked by %s and %s. Movement z score of %s, ybounds of %s' % (start,stop,imagedir,outroot,maskfile,burrow_maskfile,mousezcut,ybounds)

images = sorted(glob(imagedir+'/*.png'))[start:stop]
bkgd_images = sorted(glob(imagedir+'/*.png'))[stop:stop+(stop-start)]

frames = vidtools.load_normed_arrays(images,pixav)

#print >> sys.stderr, 'time averaging invoked, window of %s frames' % (timeav)
#frames = Util.zsmooth_convol(frames,timeav)

#windowed_frames = vidtools.average_frames(frames,num_wins=wins)
#frameav = vidtools.average_frames(windowed_frames)
'''
try:
	frameav = vidtools.average_frames(bkgd_images,pixav)
except IndexError:
	frameav = vidtools.average_frames(frames)
'''
frameav = vidtools.average_frames(frames)

SHAPE = frames[0].shape

print >>sys.stderr,'shape:',SHAPE
if eval(maskfile) is not None:
	mask = numpy.fromfile(maskfile,dtype=bool).reshape(SHAPE)

if ybounds is None:
	ybounds = (0,SHAPE[0])

#mice = dict([(images[i],vidtools.find_mouse(f[ybounds[0]:ybounds[1]],frameav[ybounds[0]:ybounds[1]],zcut=mousezcut,abs_val=True)) for i,f in enumerate(frames)])

mice = []
miceouts = []
for i,f in enumerate(frames):
	m,o = vidtools.find_mouse(f[ybounds[0]:ybounds[1]],frameav[ybounds[0]:ybounds[1]],zcut=mousezcut,abs_val=True,outline_zcut=mousezcut)
	mice.append((images[i],m))
	miceouts.append((images[i],o))
mice = dict(mice)
miceouts = dict(miceouts)

try:
	os.makedirs(outroot)
except:
	pass

frameav.tofile(os.path.join(outroot,'%07d-%07d.frame' % (start,stop)))
open(os.path.join(outroot,'%07d-%07d.miceoutline' % (start,stop)),'w').write(miceouts.__repr__())
open(os.path.join(outroot,'%07d-%07d.mice' % (start,stop)),'w').write(mice.__repr__())
