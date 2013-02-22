#!/usr/bin/env python
'''finds mouse and ground, and stores an averaged frame for frames from start to stop

see optstruct below for options

summarize_segment.py [options] imagedir
'''

import vidtools,os,sys,re,Util,numpy
from glob import glob
from PIL import ImageFilter

def get_win(i,start,stop,wins):
	return i/((stop-start)/wins)


#progname = sys.argv.pop(0)

optstruct = { 
	'start': ('s',int,0,'start of target analysis segment in seconds'),
	'stop' : ('e',int,None,'end of target analysis segment in seconds'),
	'maskfile' : ('m',str,'None','filename of a dimension-matched binary mask file'),
	'burrow_maskfile' : ('k',str,None,'filename of dimension-matched binary burrow mask file [deprecated]'),
	'burrow_entrance_xy' : ('b',eval,(360,240),'x,y tuple of burrow entrance coordinates'),
	'xybounds' : ('x',eval,None,'[(x,y),(x,y)] coords of top-left and bottom-right of valid analysis area'),
	'ground_anchors' : ('g',eval,None,'points to use for groundfinding guides'),
	'skip_pre_erase' : ('i',bool,True,'if true, skip first-pass mouse eraser'),
	'pixav' : ('p',int,5,'number of pixels to average intensities over'),
	'timeav' : ('t',int,0,'number of frames to average intensities at a given pixel over'),
	'mousez' : ('o',float,'6','z-score cutoff for calling mouse location'),
	'burrowz' : ('u',eval,'6','z-score cutoff for calling burrowing activity'),
	'num_batches' : ('n',int,50,'number of batches to bundle LSF jobs into'),
	'outroot' : ('r',str,None,'path to store output')}

opts,args = Util.getopt_long(sys.argv[1:],optstruct,required=['ground_anchors'])

imagedir = args[0]

(maskfile,start,stop,pixav,timeav,outroot,mousezcut) = [opts[v] for v in \
	('maskfile','start','stop','pixav','timeav','outroot','mousez')]

outfile = os.path.join( outroot,'%07d-%07d.mice' % (start,stop) )
miceoutline_outfile = os.path.join( outroot,'%07d-%07d.miceoutline' % (start,stop) )
micez_outfile = os.path.join( outroot,'%07d-%07d.micez' % (start,stop) )
micesize_outfile = os.path.join( outroot,'%07d-%07d.micesize' % (start,stop) )
micelen_outfile = os.path.join( outroot,'%07d-%07d.micelen' % (start,stop) )

if os.path.exists(outfile):
	print >> sys.stderr, 'file %s present; skipping' % outfile
else:
	print >> sys.stderr, 'summarizing frames %s to %s from %s. output to %s. Masked by %s. Movement z score of %s, xybounds of %s. Intensities averaged over %s pixels' % (start,stop,imagedir,outroot,maskfile,mousezcut,opts['xybounds'],pixav)

	#transparent support for supplying a tarball in lieu of a directory; assumes the tarball is only images
	if imagedir.endswith('.tar'):
		import tarfile
		imtar = tarfile.open(imagedir)
		tarcont = sorted(imtar.getnames())[start:stop]
		images = ['%s:%s' % (imagedir,f) for f in tarcont]
		frames = vidtools.load_normed_arrays([imtar.extractfile(f) for f in tarcont],pixav)
	else:
		images = sorted(glob(imagedir+'/*.png'))[start:stop]
		#new toys! image smoothing
		SMOOTH5 = ImageFilter.Kernel( (5, 5) , (1,2,2,2,1,2,3,3,3,2,2,3,5,3,2,2,3,3,3,2,1,2,2,2,1 ) )
		frames = vidtools.load_normed_arrays(images,img_smooth_kernel=SMOOTH5)

	print >> sys.stderr, len(images),'images selected'
	print >> sys.stderr, len(frames),'frames loaded'
	
	exp_frames = int(stop) - int(start)
	if len(frames) != exp_frames:
		raise ValueError, 'expected %s frames, found %s' % (exp_frames,len(frames))

	if maskfile == 'None':
		mask = numpy.zeros(frames[0].shape,dtype=bool)
	else:
		mask = numpy.fromfile(maskfile,dtype=bool).reshape(frames[0].shape)

	if opts['xybounds'] is None:
		opts['xybounds'] = [(0,0),frames[0].shape[::-1]]
		print >> sys.stderr, 'bounds reset to %s' % opts['xybounds']
	else:
		boundmask = numpy.ones(frames[0].shape,dtype=bool)
		[(tx,ty),(bx,by)] = opts['xybounds']
		boundmask[ty:by,tx:bx] = False
		boundmask += mask #piggybacking the mask subtraction for supplied maskfile on groundmask
		print >> sys.stderr, 'masking frames to %s' % opts['xybounds']
		step = len(frames)/10
		for i,f in enumerate(frames):
			frames[i] = Util.subtract_mask(f,boundmask,frames[i][ty:by,tx:bx].min())
			if i % step == 0:
				print >> sys.stderr, 'frame %s done' % i
	

	#consider not replacing frames; keep a non-averaged copy to trace mice on for .micepoly files...
	if timeav:
		print >> sys.stderr, 'time averaging invoked, window of %s frames' % (timeav)
		frames = Util.zsmooth_convol(frames,timeav)

	frameav = vidtools.average_frames(frames)

	
	mouse_grow_by = 0
	mouse_preshrink = 2

	miceli = []
	miceoutli = []
	micez = []
	micesize = []
	micelen = []
	print >>sys.stderr,'processing %s frames' % len(frames)
	tick = int(len(frames)/10)
	if opts['ground_anchors'] is None and opts['burrowz'] is None:
		print >> sys.stderr, 'both ground_anchors and burrowz set None; skipping mouse eraser'
	elif opts['skip_pre_erase']:
		print >> sys.stderr, 'manual set: skipping mouse eraser'
	else:
		fcopy = frameav.copy()
		#first pass removes mice from frame average before finalizing
		print >>sys.stderr,'first pass'
		for i,f in enumerate(frames):
			xy,ol,zsc = vidtools.find_mouse(f,fcopy,zcut=mousezcut,outline_zcut=mousezcut/2.0,grow_by=mouse_grow_by,preshrink=mouse_preshrink)
			for p in ol:
				try:
					vidtools.subtract_outline(p,frameav)
				except:
					print >> sys.stderr, 'failure in frame %s' % i

			if i % tick == 0:
				print >> sys.stderr, 'frame %s done' % i

	fcopy = frameav.copy()
	print >>sys.stderr,'finalize mouse calls'
	for i,f in enumerate(frames):
		#print i
		#get xy coords, outlines of above-cut regions, and mean z-scores in each outline
		xy,ol,zsc = vidtools.find_mouse(f,fcopy,zcut=mousezcut,outline_zcut=mousezcut/2.0,grow_by=mouse_grow_by,preshrink=mouse_preshrink)
		if opts['ground_anchors'] is None and opts['burrowz'] is None:
			pass
		else:
			for p in ol:
				try:
					vidtools.subtract_outline(p,frameav)
				except:
					print >> sys.stderr, 'failure in frame %s' % i
		size = [len(filter(None,vidtools.shrink_mask(vidtools.mask_from_outline(p,f.shape),mouse_grow_by).flat)) for p in ol]
		mlen = [max([vidtools.hypotenuse(p1,p2) for p1 in p for p2 in p]) for p in ol]
		miceli.append( (images[i],xy) )
		miceoutli.append( (images[i],ol) )
		micez.append( (images[i],zsc) )
		micesize.append( (images[i],size) )
		micelen.append( (images[i],mlen) )
		if i % tick == 0:
			print >> sys.stderr, 'frame %s done' % i
	mice = dict(miceli)
	miceout = dict(miceoutli)
	micez = dict(micez)
	micesize = dict(micesize)
	micelen = dict(micelen)

	try:
		os.makedirs(outroot)
	except:
		pass

	#ground.tofile(os.path.join(outroot,'%07d-%07d.ground' % (start,stop)),sep='\n')
	#digs = [(f - windowed_frames[i+1] > 6) for i,f in enumerate(windowed_frames[:-1])]
	open(miceoutline_outfile,'w').write(miceout.__repr__())
	open(micez_outfile,'w').write(micez.__repr__())
	open(micesize_outfile,'w').write(micesize.__repr__())
	open(micelen_outfile,'w').write(micelen.__repr__())
	frameav.tofile(os.path.join(outroot,'%07d-%07d.frame' % (start,stop)))
	open(outfile,'w').write(mice.__repr__())
