from PIL import Image
from glob import glob
import os, sys, re, Util, numpy, subprocess, shutil

def vid_duration(video):
	t = re.search('Duration: (\d+?):(\d+?):(\d+?)\.',subprocess.Popen('ffmpeg -i '+video,stderr=subprocess.PIPE,shell=True).stderr.read()).groups()
	return (int(t[0])*60*60) + (int(t[1])*60) + int(t[2])

def parallel_v2p(vid, fps, tdir=None, global_start=0, global_end=None, num_jobs=40,lock=True):
	'''launches parallelized (lsf) runs to split a video into stills

	returns ids of running jobs
	'''

	if not (global_start or global_end):
		win = 'all'
	elif not global_start and global_end:
		win = 'start-%s' % global_end
	elif global_start and not global_end:
		win = '%s-end' % global_start
	elif global_start and global_end:
		win = '%s-%s' % (global_start,global_end)
	
	if tdir is None:
		tdir = vid.rsplit('.',1)[0]+'/%s/%sfps/' % (win,fps)

	if global_end is None:
		global_end = vid_duration(vid)

	tot = global_end - global_start

	try:
		os.makedirs(tdir)
	except OSError:
		pass
    
	v2p = 'vid2png.py'

	step = tot/num_jobs
	cmds = []
	for i in range(global_start,global_end,step)[:-1]:
		cmds.append('%s %s %s %s %s %s' % (v2p, vid, tdir, i, step, fps))
	if i+step < global_end:
		cmds.append('%s %s %s %s %s %s' % (v2p, vid, tdir, i+step, global_end - (i+step), fps))

	jobids,namedict = Util.lsf_jobs_submit(cmds,tdir+'log','normal_serial',jobname_base='vid2png')
	if lock:
		Util.lsf_wait_for_jobs(jobids,tdir+'restart-log','normal_serial',namedict=namedict)
		return tdir
	else:
		return jobids,tdir

def rename_images_from_zero(imagedir,type='png',digits=7):
	images = sorted(glob(os.path.join(imagedir,'*.'+type)))
	fstr = '/%%0%dd.%%s' % digits
	for i,f in enumerate(images):
		newf = f.rsplit('/',1)[0]+fstr % (i,type)
		shutil.move(f,newf)


def check_contiguous_files(d,ext):
	'''given a directory and an extension, checks to make sure files of the form:
	<d>/start-end.<ext> are sequential,
	i.e. that 0000000-0000900.whatever is followed by 0000900-0001800.whatever
	returned list is tuples, first element is filename AFTER the gap, second is size (in files) of the gap'''
	def get_start_end(s):
		start,end = os.path.split(s)[1].split('.')[0].split('-')
		return start,end
	
	files = sorted(glob(d+'/*.'+ext))
	
	start,last = get_start_end(files[0])
	step = int(last)-int(start)
	gaps = []
	for f in files[1:]:
		start,end = get_start_end(f)
		if last != start:
			gapsize = (int(start)-int(last))/step
			gaps.append((f,gapsize))
		last = end
	return gaps


def timestamp_from_path(imagepath,fps=None):
	'''returns the timestamp (float seconds) of an image, given a pathname'''

	if fps is None:
		match = re.search(r'\/([\d\.]+)fps\/',imagepath)
		if match:
			fps = float(match.groups()[0])
		else:
			raise ValueError, 'no fps spec in path',imagepath
	else:
		fps = float(fps)

	match = re.search(r'\/(\d+)\.[pngj]{3}',imagepath)
	framenum = int(match.groups()[0])

	return framenum/fps

def load_normed_arrays(filelist,pix_av_win=None):
	'''currently hacked to allow tarball:filename indexing, but first element must have this structure!'''
	nars = []
	if '.tar:' in filelist[0]: #assumes a : means file in a tarball; only checks the first element
		import tarfile
		tarf,tarim = filelist[0].split(':')
		print >> sys.stderr, 'filelist contains tar references (".tar:"), loading tarball %s' % tarf
		tarobj = tarfile.open(tarf)
		tar = True
	else:
		tar = False
		
	for i in filelist:
		if tar:
			tarf,tarim = i.split(':')
			if tarf != tarobj.name:
				print >> sys.stderr, 'new tarball referenced, opening %s' % tarf
				tarobj = tarfile.open(tarf)
			im = tarobj.extractfile(tarim)
		else:
			im = i
		ar = numpy.asarray(Image.open(im).convert(mode='L'))
		if ar.any():
			ar = Util.normalize(ar)
			if pix_av_win:
				nars.append(Util.smooth(ar, pix_av_win))
			else:
				nars.append(ar)
	return numpy.array(nars)
		
def average_frames(frames,pix_av_win=None,num_wins=1):
	'''returns a single average of a list of frames
	
	if frames are strings, will treat as filenames, loading with pixel averaging of pix_av_win'''
	
	if num_wins > 1:
		winlen = len(frames) / num_wins
		return [average_frames(frames[i:i+winlen],pix_av_win) for i in range(0,len(frames),winlen)]
	
	if isinstance(frames[0],numpy.ndarray):
		denom = numpy.ndarray(frames[0].shape,dtype=float)
		denom[:,:] = len(frames)
		return reduce(lambda x, y: x+y, frames) / denom
	elif isinstance(frames[0],str):
		first = load_normed_arrays(frames[:1],pix_av_win)[0]
		denom = numpy.ndarray(first.shape,dtype=float)
		denom[:,:] = len(frames)
		running_sum = first
		for f in frames[1:]:
			running_sum += load_normed_arrays([f],pix_av_win)[0]
		return running_sum / denom
		
def find_mouse(frame,background,win=5,zcut=6,origin=(0,0),abs_val=False):
	diff = Util.zscore(Util.smooth(frame,win)-Util.smooth(background,win))
	if abs_val:
		diff = numpy.abs(diff)
	if diff.max() > zcut:
		pix = diff.argmax()
		loc = (pix % diff.shape[1], pix / diff.shape[1])
		return tuple(numpy.array(loc)-numpy.array(origin))
	else:
		return None

def xy_from_idx(idx,shape):
	'''return (x,y) coords for a given index in flattened vector'''
	xy = (idx % shape[1], idx / shape[1])
	return xy

def find_ground(frame,mask=None,burrow_mask=None,ybounds=None,offset=5,smoothby=20,zcut=3):
	'''uses change in pixel intensity to find the ground, given masks for overall problem regions (i.e. support uprights)
	and burrow(s) i.e. most changed pixels
	
	try 
	>>> mask = fullav_smooth > 0.19
	>>> mask10 = vidtools.grow_mask(mask,10)
	guided by pixel intensity histogram
	
	and (to find differences of greater than 2 sigma over mean that are more than 20 pixels underground, and in the top 400 pixels of the frame)
	>>> burrow_mask = vidtools.find_burrow(images,mask10,change_zcut=2,depress_groundmask=20,ybounds=(0,400))
	>>> burrow_mask5 = vidtools.grow_mask(burrow_mask,5)'''
	
	if not ybounds:
		ybounds = (0,len(frame))
		
	ground = []

	if isinstance(mask,str):
		print >> sys.stderr, 'mask supplied as string %s, assuming filename, loading with shape %s' % (mask,frame.shape)
		mask = numpy.fromfile(mask,dtype=bool).reshape(frame.shape)
	
	for i in range(len(frame[0])):
		if mask is not None:
			if mask[ybounds[0]:ybounds[1],i].any():
				ground.append(None)
				continue
		win = frame[ybounds[0]:ybounds[1],i]
		diff = [0]*offset+list(win[offset:] - win[:-1*offset])
		if burrow_mask is not None:
			mask_win = burrow_mask[ybounds[0]:ybounds[1],i]
		else:
			mask_win = numpy.zeros(win.shape)
		diff = Util.zscore(Util.smooth(Util.subtract_mask(numpy.array(diff),mask_win,0),smoothby))
		infl = diff.argmax()
		z = diff.max()
		if z > zcut:
			ground.append(infl)
		else:
			ground.append(None)
		
	return numpy.array(ground).transpose()
	

	
def find_ground2(frame,zcut=3,win=40,top=10,xybounds=None):
	[(tx,ty),(bx,by)] = xybounds
	flip = frame[ty:by,tx:bx].transpose()
	meandiffs = numpy.zeros(flip.shape,dtype=float)
	ground = []
	for i,col in enumerate(flip):
		for j,val in enumerate(col):
			start = max(0,j-win)
			end = min(len(col),j+win)
			above = max(0,col[start:j].mean())
			below = max(0,col[j:end].mean())
			meandiffs[i,j] = below - above
		#ground.append(meandiffs[i][top:].argmax() + top)
	
	for vect in meandiffs:
		zdiffs = Util.zscore(vect[top:])
		infl = zdiffs.argmax() + top
		z = zdiffs.max()
		if z > zcut:
			ground.append(infl+ty)
		else:
			ground.append(None)
			
	#restore bounds
	ground = [None]*tx + ground + [None]*(len(flip)-bx)


	return ground #,meandiffs.transpose()

def find_burrow(images,mask=None,change_zcut=3,smoothby=5,depress_groundmask=10,offsets=None,pct_to_average=0.01,frames_to_average=100,**ground_args):
	if offsets is None:
		pct = int(len(images)*pct_to_average)
		step = pct/frames_to_average
		if step == 0:
			step = 1
		offsets = ((pct,pct*2,step),(-2*pct,-1*pct,step))

	first = Util.smooth(average_frames(images[offsets[0][0]:offsets[0][1]:offsets[0][2]]),smoothby)
	last = Util.smooth(average_frames(images[offsets[1][0]:offsets[1][1]:offsets[1][2]]),smoothby)

	if isinstance(mask,str):
		print >> sys.stderr, 'mask supplied as string %s, assuming filename, loading with shape %s' % (mask,first.shape)
		mask = numpy.fromfile(mask,dtype=bool).reshape(first.shape)

	groundmask = mask_from_vector(Util.smooth(find_ground(first,mask,**ground_args),10,interpolate_nones=True)+depress_groundmask,first.shape)

	change = first - last
	change = Util.zscore(change)
	if mask is not None:
		change = Util.subtract_mask(change,mask,0.0)
	change = Util.subtract_mask(change,groundmask,0.0)
	
	return change > change_zcut

def classify_mouse(mouse,ground,activitymask):
	'''classify a mouse in the following scheme:
	
	0: mouse not found
	1: above ground, no activity (e.g. not digging)
	2: above ground, activity
	3: below ground, no activity
	4: below ground, activity
	given a mouse coord (tuple or None), a ground vector and a mask of recent activity'''
	
	if mouse is None:
		return 0

	if activitymask[mouse[1],mouse[0]]:
		actmod = 1
	else:
		actmod = 0
		
	return ((mouse[1] > ground[mouse[0]]) * 2) + actmod + 1
	
def calculate_cumulative_activity(analysis_dir,zcut,suppress_ground=40,shape=(480,720),be=(360,240),write_files=True,force_all=False):
	'''given a directory containing .actmat and .ground files, iteratively calculates new subterranean activity'''
	actfiles = sorted(glob(os.path.join(analysis_dir,'*.actmat')))
	prevact = {}
	prevtermini = {}
	prevprogress = {}
	for f in actfiles:
		# set up activity filenames
		newactout = os.path.splitext(f)[0]+'.newactout'
		newactterm = os.path.splitext(f)[0]+'.newactterm'
		preactout = os.path.splitext(f)[0]+'.preactout'
		newactprop = os.path.splitext(f)[0]+'.newactprop'
		reqfiles = [newactout,newactterm,preactout,newactprop]

		# skip this iteration (after updating previous activity dicts) if all four outputs are present
		if all([os.path.exists(outfilename) for outfilename in reqfiles]) and not force_all:
			print >> sys.stderr, 'all output present; skipping'
			newout = eval(open(newactout).read())
			newterm = eval(open(newactterm).read())
			#pa_out = eval(open(preactout).read()) #not needed
			prop = eval(open(newactprop).read())

			prevtermini[f] = newterm
			prevact[f] = newout
			prevprogress[f] = prop['progress']
			
			continue
		else:
			print >> sys.stderr, 'checking for files',' '.join(reqfiles)
			for outfilename in reqfiles:
				if not os.path.exists(outfilename):
					print >> sys.stderr, outfilename,'not found'
		#otherwise...

		#pa_flat = filter(None,flatten_points_lists(prevact.values()))
		pa_flat = flatten_points_lists([v for k,v in sorted(prevact.items()) if any(v)])
		pt_flat = flatten_points_lists([v for k,v in sorted(prevtermini.items()) if any(v)])
		#print >>sys.stderr,'flats:',pa_flat,pt_flat

		actmat = numpy.fromfile(f).reshape(shape)
		try: #to find a groundmask
			groundfile = os.path.splitext(f)[0]+'.ground'
			ground = numpy.fromfile(groundfile,sep='\n')
			groundmask = mask_from_vector(ground+suppress_ground,shape)
			actmat = Util.subtract_mask(actmat,groundmask,0)
		except OSError:
			print >>sys.stderr,'No ground file %s found - proceeding unmasked' % groundfile
		
		if pa_flat:
			prevmask = mask_from_outline(pa_flat,actmat.shape)
			newact = Util.subtract_mask(actmat>zcut,prevmask)
		else:
			prevmask = numpy.zeros(actmat.shape,dtype=bool)
			newact = actmat>zcut
		
		if pt_flat:
			newout,newterm = chain_outlines_from_mask(newact,pt_flat[-1],preshrink=1)
		else:
			newout,newterm = chain_outlines_from_mask(newact,be,preshrink=1)

		if any(newout):
			if pa_flat:
				farthest_new = apogee(newout,pa_flat)[0]
				#print >> sys.stderr,'farthest new; pa_flat:', farthest_new,pa_flat
				nearest_old,progress = closest_point(pa_flat,farthest_new,include_dist=True)
			elif newterm:
				nearest_old = newterm[0][0]
				farthest_new = newterm[-1][-1]
				progress = hypotenuse(nearest_old,farthest_new)
			prevtermini[f] = newterm
			prevact[f] = newout
			prevprogress[f] = progress
		else:
			farthest_new,nearest_old,progress = None,None,None
		
		print f
		print 'nearest_old: %s\nfarthest_new: %s\nprogress: %s\n' % (nearest_old,farthest_new,progress)
		
		if write_files:
			print >> sys.stderr, ('**file output invoked.\nSummary outlines of activity prior to this segment in %s' 
				'\nOutlines of activity this segment in %s\nProperties of new activity: %s') \
				% (preactout, newactout, newactprop)
				
			if prevmask.any():
				pa_out,pt_null = chain_outlines_from_mask(prevmask,be,preshrink=1,grow_by=2)
			else:
				pa_out = []

			prop = {}
			prop['nearest_old'] = nearest_old
			prop['farthest_new'] = farthest_new
			prop['progress'] = progress
			prop['area'] = len(flatten_points_lists(points_from_mask(newact)))
			
			open(newactout,'w').write(newout.__repr__())
			open(newactterm,'w').write(newterm.__repr__())
			open(preactout,'w').write(pa_out.__repr__())
			open(newactprop,'w').write(prop.__repr__())

	return prevact,prevtermini,prevprogress

def mask_from_vector(vector,shape):
	'''returns a mask that is false "below" (i.e. higher y values) and true "above" (lower y values)
	the value in each item of the vector (x values from vector indices)'''
	
	mask = numpy.zeros(shape[::-1],dtype=bool)
	for i,k in enumerate(vector):
		mask[i,:k] = True
	return mask.transpose()

def mask_band_around_vector(vector,shape,bandheight,bandmasked=False):
	'''returns a mask around a vector
	default returns true outside of band (bandmasked=False), set bandmasked=True to mask inside the band instead
	'''
	halfheight = bandheight/2
	if bandmasked:
		mask = numpy.zeros(shape[::-1],dtype=bool)
		for i,k in enumerate(vector):
			if k-halfheight < 0:
				top = 0
			else:
				top = k-halfheight
			mask[i,top:k+halfheight] = True
	else:
		mask = numpy.ones(shape[::-1],dtype=bool)
		for i,k in enumerate(vector):
			if k-halfheight < 0:
				top = 0
			else:
				top = k-halfheight
			mask[i,top:k+halfheight] = False
	return mask.transpose()
		

def grow_mask(mask,growby):
	
	values = mask.copy()
	
	for i,r in enumerate(mask):
		for j,k in enumerate(mask[i]):
			if k:
				values[i-growby:i+growby+1,j-growby:j+growby+1] = True
				
	return values
	
def shrink_mask(mask,shrinkby):
	
	values = numpy.zeros(mask.shape,dtype=bool)
	
	for i,r in enumerate(mask):
		for j,k in enumerate(mask[i]):
			if k:
				if not is_edge(mask,(j,i)):
					values[i,j] = True
				
	if shrinkby-1:
		return shrink_mask(values,shrinkby-1)
	else:
		return values

def flatten_points_lists(coords_lists):
	try:
		if isinstance(coords_lists[0],tuple):
			return coords_lists
		elif isinstance(coords_lists[0],list):
			return flatten_points_lists(reduce(lambda x,y: x+y,coords_lists))
	except IndexError:
		return coords_lists

def hypotenuse(p1,p2):
	import math
	x = p1[0] - p2[0]
	y = p1[1] - p2[1]
	return math.sqrt(x**2 + y**2)

def distance_from_point(coords,point):
	'''given a list of coord 2-tuples and a 2-tuple point, returns a list of coords (None for None values in coords)'''

	dists = []
	for c in coords:
		if c is None:
			dists.append(c)
		else:
			dists.append(hypotenuse(c,point))
	return dists
		
def closest_point(coords,point,include_dist=False):
	'''returns the closest (x,y) point from the supplied list'''
	flatcoords = flatten_points_lists(coords)
	idx = numpy.array(distance_from_point(flatcoords,point)).argmin()
	if include_dist:
		return flatcoords[idx],hypotenuse(flatcoords[idx],point)
	else:
		return flatcoords[idx]
	
def farthest_point(coords,point,include_dist=False):
	'''returns the farthest (x,y) point from the supplied list'''
	flatcoords = flatten_points_lists(coords)
	idx = numpy.array(distance_from_point(coords,point)).argmax()
	if include_dist:
		return flatcoords[idx],hypotenuse(flatcoords[idx],point)
	else:
		return flatcoords[idx]
	
def point_distance_matrix(coords1,coords2):
	mat = numpy.zeros((len(coords1),len(coords2)))
	for i,c1 in enumerate(coords1):
		for j,c2 in enumerate(coords2):
			mat[i,j] = hypotenuse(c1,c2)
	return mat
	
def apogee(coords1,coords2,include_dist=False):
	
	#if isinstance(coords1[0],list):
	#	coords1 = reduce(lambda x,y: x+y, coords1)
	#if isinstance(coords2[0],list):
	#	coords2 = reduce(lambda x,y: x+y, coords2)
	coords1 = flatten_points_lists(coords1)
	coords2 = flatten_points_lists(coords2)

	dmat = point_distance_matrix(coords1,coords2)
	c2,c1 = xy_from_idx(dmat.argmax(),dmat.shape) #c2/c1 reversal compensates for x,y <-> i,j mapping reversal
	return coords1[c1],coords2[c2]
	
def perigee(coords1,coords2,include_dist=False):
	
	if isinstance(coords1[0],list):
		coords1 = reduce(lambda x,y: x+y, coords1)
	if isinstance(coords2[0],list):
		coords2 = reduce(lambda x,y: x+y, coords2)

	dmat = point_distance_matrix(coords1,coords2)
	c2,c1 = xy_from_idx(dmat.argmin(),dmat.shape) #c2/c1 reversal compensates for x,y <-> i,j mapping reversal
	return coords1[c1],coords2[c2]
	

def points_from_mask(mask):
	'''returns a list of (x,y) coords lying in "true" mask cells'''
	flatmask = mask.flatten()
	flat_idxs = numpy.arange(0,len(flatmask))[flatmask]
	return [xy_from_idx(i,mask.shape) for i in flat_idxs]
	
def get_adjacent_values(mask,point,skip=[]):
	'''returns a list of values (clockwise from top) adjacent to the specified point'''
	xys = [(0,-1),(1,0),(0,1),(-1,0),(1,-1),(1,1),(-1,1),(-1,-1)]
	return [mask[point[1]+y,point[0]+x] for x,y in xys if (x,y) not in skip]

def get_adjacent_points(point):
	xys = [(0,-1),(1,0),(0,1),(-1,0),(1,-1),(1,1),(-1,1),(-1,-1)]
	return [(point[0]+x,point[1]+y) for x,y in xys]
		

def get_next_edge(mask,point,prev_edges):
	xys = [(0,-1),(1,0),(0,1),(-1,0),(1,-1),(1,1),(-1,1),(-1,-1)]
	if not isinstance(prev_edges,list):
		prev_edges = [prev_edges]
	#print >> sys.stderr,'next point from',point,'avoiding', prev_edges
	skip = [(prev_edge[0]-point[0],prev_edge[1]-point[1]) for prev_edge in prev_edges]
	for i,val in enumerate(get_adjacent_values(mask,point)):
		if val and not xys[i] in skip:
			candidate = (point[0] + xys[i][0], point[1] + xys[i][1])
			#print >> sys.stderr,'%s not in %s, edge?' % (candidate,skip)
			if mask[candidate[1],candidate[0]] and is_edge(mask,candidate):
				#print >> sys.stderr,'yes; keeping'
				return candidate
				
def outline_from_mask(mask,origin=(0,0),grow_by=1,preshrink=0):
	'''given a mask and an origin (point to start closest to) returns an ordered list of (x,y) that trace the mask'''
	if preshrink:
		mask = shrink_mask(mask,preshrink)
		if grow_by:
			grow_by += preshrink
	if grow_by:
		mask = grow_mask(mask,grow_by)
	start = closest_edge(mask,origin)
	#print >> sys.stderr, 'start at %s' % (start,)
	outline = [start]
	next = get_next_edge(mask,start,start)
	while next and next != outline[0]:
		#print >>sys.stderr, 'next point: %s' % (next,)
		outline.append(next)
		try:
			invalid = outline[2:-1]+get_adjacent_points(outline[-3])#+get_adjacent_points(outline[-2])
		except IndexError:
			invalid = outline[-4:-1]
		next = get_next_edge(mask,outline[-1],invalid)
	return outline
	
def chain_outlines_from_mask(mask,origin=(0,0),grow_by=1,preshrink=2):
	'''given a mask and an origin, "hops" from blob to blob chaining an outline'''
	newmask = mask.copy()
	if preshrink:
		newmask = shrink_mask(newmask,preshrink)
	#	if grow_by:
	#		grow_by += preshrink
	#if grow_by:
		newmask = grow_mask(newmask,preshrink)

	outlines = []
	termini = []
	start = origin
	while newmask.any():
		outlines.append(outline_from_mask(newmask,start,grow_by=grow_by))
		end = farthest_point(outlines[-1],start)
		termini.append([outlines[-1][0],end])
		blobmask = mask_from_outline(outlines[-1],newmask.shape)
		newmask = Util.subtract_mask(newmask,blobmask,0)
		
	return outlines,termini
			
	
def mask_from_outline(outline,shape):
	'''given dimensions (shape), generates a bool mask that is True inside shape outline (list of (x,y) coords)'''
	#print >> sys.stderr, "generate mask from outline:",outline
	xsorted = sorted(outline)
	newmask = numpy.zeros(shape,dtype=bool)
	newmask = newmask.transpose()
	
	while xsorted:
		drop = None
		top = xsorted.pop(0)
		last = top[1]
		while xsorted and xsorted[0][0] == top[0] and xsorted[0][1] == last+1:
			drop = xsorted.pop(0)
			last = drop[1]
		if xsorted and xsorted[0][0] == top[0]:				
			bot = xsorted.pop(0)
			while xsorted and xsorted[0][0] == bot[0] and xsorted[0][1] == bot[1]+1:
				bot = xsorted.pop(0)
			newmask[top[0]][top[1]:bot[1]+1] = True
		elif drop:
			newmask[top[0]][top[1]:drop[1]+1] = True
		else:
			newmask[top[0]][top[1]] = True
		
	return newmask.transpose()
	
def is_edge(mask,point):
	'''returns true if the value (x,y) (i.e. mask[y,x]) is true, and an adjacent cell is false'''
	if mask[point[1],point[0]]:
		return not all(get_adjacent_values(mask,point))
	else:
		raise ValueError, '%s in mask is False (this is bad)' % (point,)
	
def closest_edge(mask,point):
	'''finds the nearest edge (true cell adjacent to false cell) to point'''
	cp = closest_point(points_from_mask(mask),point)
	if is_edge(mask,cp):
		return cp
	else:
		return (point[0],numpy.arange(len(mask[:,point[0]]))[mask[:,point[0]]].min())