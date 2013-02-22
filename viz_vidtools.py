'''initial import attempts to load non-interactive backend; 
will fail [semi]gracefully if an interactive backend is already loaded
'''
import matplotlib
matplotlib.use('Agg')

import pylab,os,sys,re,Util,iplot
from PIL import Image
from glob import glob
from video_analysis import vidtools
import numpy

col = {0:'w',1:'r',2:'y',3:'g',4:'b',5:'c'}

def show_keyframe(tdir,frame=0,fignum=1,ext='png'):
	pylab.matshow(pylab.asarray(Image.open(sorted(glob(tdir+'/*.'+ext))[frame]).convert('L')),fignum=fignum)

def show_first_and_last_frames(vid,f1=1,f2=2):
	vidlen = vidtools.vid_duration(vid)
	out = os.path.splitext(vid)[0]
	os.system('vid2png.py %s %s 0 1 1' % (vid,out))
	os.system('vid2png.py %s %s %s 1 1' % (vid,out,vidlen-1))
	m1 = pylab.asarray(Image.open(sorted(glob(out+'/*.png'))[0]).convert('L'))
	m2 = pylab.asarray(Image.open(sorted(glob(out+'/*.png'))[-1]).convert('L'))
	pylab.matshow(m1,fignum=f1)
	pylab.matshow(m2,fignum=f2)
	return(m1.mean()-m2.mean())

def ginput_int(npts):
	pts = pylab.ginput(npts,timeout=0)
	return [(int(x),int(y)) for x,y in pts]
	
def current_view_bounds(fignum,axnum=0):
	'''given a figure number, returns xy coords of upper-left and lower-right extent of the current view'''
	ax = pylab.figure(fignum).axes[axnum]
	return zip([int(x) for x in ax.get_xbound()],[int(y) for y in ax.get_ybound()])

def current_view_crop(fignum=1,shape=(480,720)):
	'''given a figure, gets the current view bounds (origin-zero) and converts to a crop tuple (left, top, right, bottom)
	
	i.e.
	>>> current_view_bounds(1)
	[(47, 16), (701, 357)]
	>>> current_view_crop(1)

	'''
	[(x1,y1),(x2,y2)] = current_view_bounds(fignum)
	cl = x1
	ct = y1
	cr = shape[1] - x2
	cb = shape[0] - y2
	return (cl,ct,cr,cb)


cfg_fn = lambda f: f[:-4]+'-config.dict'
epm_cfg_fn = lambda f: os.path.join(f[:-4],os.path.basename(f[:-4]+'.zones.dict'))

def get_antfarm_config(vid,fignum=1,cfg_fn=cfg_fn):
	config = {}
	pylab.close(fignum)
	fr = vidtools.extract_change_over_vid(vid)
	pylab.matshow(fr,fignum=fignum)
	print >> sys.stderr, 'click burrow entrance'
	config['burrow_entrance'] = ginput_int(1)[0]
	print >> sys.stderr, 'click excavated sand hill bounds'
	config['hill_bounds'] =  [x for x,y in ginput_int(2)]
	pylab.close(fignum)
	fr = vidtools.extract_keyframe(vid)
	pylab.matshow(fr,fignum=fignum)
	print >> sys.stderr, 'click ground anchor points, and press enter when finished'
	config['ground_anchors'] = ginput_int(0)
	print >> sys.stderr, 'click 4 predug burrow corners (top-ground, top-end, bottom-end, bottom-ground), and press enter when finished, or press enter without clicking any points'
	pts = ginput_int(0)
	if len(pts):
		config['predug'] = pts
	cfg = cfg_fn(vid)
	open(cfg,'w').write(config.__repr__())
	pylab.close(fignum)

def get_epm_config(vid,fignum=1,cfg_fn=epm_cfg_fn,close_fig=False):
	pylab.figure(fignum)
	pylab.clf()
	fr = vidtools.extract_keyframe(vid,sec=120)
	pylab.matshow(fr,fignum=fignum)
	print >> sys.stderr, 'click 8 arm corners, proceeding clockwise from the top left corner of the top arm'
	pts = pylab.ginput(8,timeout=0)
	zones = calc_epm_zones(pts)
	
	show_epm_config(pts,zones,fignum)
	pylab.ylim(fr.shape[0],0)
	pylab.xlim(0,fr.shape[1])

	print >> sys.stderr, 'click to revise points or press enter to accept and write configuration'
	change_pt = pylab.ginput(1,timeout=0)
	#change_pt = raw_input('enter the number of an anchor point to change or press enter to save configuration: ')
	while change_pt:
	#	print >> sys.stderr, 'click the revised position for point %s' % change_pt
	#	pts[int(change_pt)] = pylab.ginput(1,timeout=0)[0]
		change_pt = change_pt[0]
		change_idx = min([(vidtools.hypotenuse(change_pt,pt),i) for i,pt in enumerate(pts)])[1]
		pts[change_idx] = change_pt
		zones = calc_epm_zones(pts)
		pylab.clf()
		pylab.matshow(fr,fignum=fignum)
		show_epm_config(pts,zones,fignum)
		pylab.ylim(fr.shape[0],0)
		pylab.xlim(0,fr.shape[1])
		print >> sys.stderr, 'click to revise points or press enter to accept and write configuration'
		change_pt = pylab.ginput(1,timeout=0)
	#	change_pt = raw_input('enter the number of an anchor point to change or press enter to save configuration: ')

	cfg = cfg_fn(vid)
	try:
		os.makedirs(os.path.dirname(cfg))
	except:
		pass
	open(cfg,'w').write(zones.__repr__())
	if close_fig:
		pylab.close(fignum)
	

def show_epm_config(pts,zones,fignum):
       	#display rules
	hatches = {'F':'x','O':'o','C':'.','M':'+'}
	pcols = {'F':'gray','O':'blue','C':'green','M':'yellow'}

	ax = pylab.figure(fignum).axes[0]
	
	X,Y = zip(*pts)
	pylab.scatter(X,Y,c='w',s=50)
	nll = [pylab.text(X[i]+10,Y[i]+10,str(i),color='k',fontsize=18) for i in range(8)]
	for k,v in zones.items():
		ax.add_patch(matplotlib.patches.Polygon(v,fc='none',ec=pcols[k[0]],hatch=hatches[k[0]]))
		pylab.text(v[0][0]+((v[1][0]-v[0][0])/2),v[0][1]+((v[3][1]-v[0][1])/2),k,color='k',fontsize=18)
	pylab.plot()
	

def calc_epm_zones(pts):
    outerpts = []
    m1,b1 = vidtools.line_fx_from_pts(pts[0],pts[1])
    m2,b2 = vidtools.line_fx_from_pts(pts[7],pts[6])
    x = (b2-b1)/(m1-m2)
    y = m2*x+b2
    outerpts.append((x,y))
    m2,b2 = vidtools.line_fx_from_pts(pts[2],pts[3])
    x = (b2-b1)/(m1-m2)
    y = m2*x+b2
    outerpts.append((x,y))
    m1,b1 = vidtools.line_fx_from_pts(pts[4],pts[5])
    x = (b2-b1)/(m1-m2)
    y = m2*x+b2
    outerpts.append((x,y))
    m2,b2 = vidtools.line_fx_from_pts(pts[7],pts[6])
    x = (b2-b1)/(m1-m2)
    y = m2*x+b2
    outerpts.append((x,y))
    #X,Y = zip(*outerpts)
    #scatter(X,Y,c='b',s=50)
    centerpts = []
    m1,b1 = vidtools.line_fx_from_pts(pts[0],pts[5])
    m2,b2 = vidtools.line_fx_from_pts(pts[7],pts[2])
    x = (b2-b1)/(m1-m2)
    y = m2*x+b2
    centerpts.append((x,y))
    m1,b1 = vidtools.line_fx_from_pts(pts[1],pts[4])
    x = (b2-b1)/(m1-m2)
    y = m2*x+b2
    centerpts.append((x,y))
    centerpts
    m2,b2 = vidtools.line_fx_from_pts(pts[6],pts[3])
    x = (b2-b1)/(m1-m2)
    y = m2*x+b2
    centerpts.append((x,y))
    m1,b1 = vidtools.line_fx_from_pts(pts[0],pts[5])
    x = (b2-b1)/(m1-m2)
    y = m2*x+b2
    centerpts.append((x,y))
    #X,Y = zip(*centerpts)
    #scatter(X,Y,c='g',s=50)
    zones = {}
    zones['F1'] = [outerpts[0],pts[0],centerpts[0],pts[7]]
    zones['F2'] = [pts[1],outerpts[1],pts[2],centerpts[1]]
    zones['F3'] = [centerpts[2],pts[3],outerpts[2],pts[4]]
    zones['F4'] = [pts[6],centerpts[3],pts[5],outerpts[3]]
    zones['OT'] = [pts[0],pts[1],centerpts[1],centerpts[0]]
    zones['OB'] = [centerpts[3],centerpts[2],pts[4],pts[5]]
    zones['CR'] = [centerpts[1],pts[2],pts[3],centerpts[2]]
    zones['CL'] = [pts[7],centerpts[0],centerpts[3],pts[6]]
    zones['M'] = centerpts
    return zones

def draw_reanalysis_activity_summary(re_adir,source_adir,fig=1,outformat=None):
	'''given a reanalysis directory and a source directory, generates a figure showing previous and concurrent activity from source
	and activity found in reanalysis

	if format is not None, also outputs to re_adir
	'''

	print >> sys.stderr, 'load:'

	shape = eval(open(re_adir+'shape.tuple').read())

	frames = [numpy.fromfile(f).reshape(shape) for f in sorted(glob(re_adir+'/*.frame'))]

	s,e,fps = re.search('/(\d+)-(\d+)/(\d+)fps/',re_adir).groups()
	source_analysis_window,source_fps =  re.search('/([\d\w-]+)/(\d+)fps/',source_adir).groups()
	offset_fr = 0
	if '-' in source_analysis_window:
		start = source_analysis_window.split('-')[0]
		if start != 'start':
			offset_fr = int(start)*int(source_fps)
	win = [int(fps)*int(n) for n in s,e]

	preactoutfs = sorted(glob(source_adir+'/*.preactout'))
	minactoutfs = sorted(glob(source_adir+'/*.newactout'))
	newacttermfs = sorted(glob(re_adir+'/*.newactterm'))
	preactouts = eval(open(vidtools.get_files_by_win(preactoutfs,win,offset_fr)[0]).read())

	minactouts,term,mask = vidtools.merge_polygons(reduce(lambda x,y:x+y,filter(None,[eval(open(f).read()) for f in vidtools.get_files_by_win(minactoutfs,win,offset_fr)])),shape,(0,0))

	secactouts = [eval(open(f).read()) for f in sorted(glob(re_adir+'*.newactout'))]
	newactterms = [eval(open(f).read()) for f in newacttermfs]
	
	spec = iplot.subspectrum(len(secactouts))


	figobj = pylab.figure(fig,figsize=(9,6))
	figobj.clf()
	print >> sys.stderr, 'render:'


	pylab.matshow(frames[-1]-(frames[0]-frames[-1]),fignum=fig)
	ax = figobj.axes[0]

	ax.set_xticks([])
	ax.set_yticks([])

	
	for p in preactouts:
		if p: ax.add_patch(pylab.matplotlib.patches.Polygon(p,fc='none',ec='k'))
    
	for p in minactouts:
		if p: ax.add_patch(pylab.matplotlib.patches.Polygon(p,fc='none',ec='w',lw=2))

	print >>sys.stderr, 'len spec %s len secactouts %s' % (len(spec), len(secactouts)) 
	for i,poly in enumerate(secactouts):
		for p in poly:
			if p: ax.add_patch(pylab.matplotlib.patches.Polygon(p,fc='none',ec=spec[i]))

	for i, terms in enumerate(newactterms):
		for t in terms:
			x,y = Util.dezip(t)
			pylab.plot(x,y,c=spec[i])


	pylab.gray()
	ax.text(10,20,'%s-%s %s-%s' % (win[0],win[1],Util.hms_from_sec(win[0]/int(fps)),Util.hms_from_sec(win[1]/int(fps))),fontsize=16,color='w')
	pylab.plot()

	if outformat is not None:
		print >> sys.stderr, 'save as '+re_adir+'/summary.'+outformat
		figobj.savefig(re_adir+'/summary.'+outformat)
	

def draw_analysis_activity_summary(adir,fig=1,outformat=None,graph_height=0.2,show_mice=False,dawn_min=None,hill_bounds=None,skip_first=10):
	'''given an analysis directory, generates a figure showing activity and grounds

	if format is not None, also outputs summary figure to adir

	if graph_height is not None, plots a color-coded activity graph on bottom <graph_height> portion of fig
	
	'''

	print >> sys.stderr, 'load:'

	shape = eval(open(adir+'/shape.tuple').read())

	if hill_bounds is None:
		hill_left = 0
		hill_right = shape[1]
	else:
		hill_left,hill_right = hill_bounds


	print >> sys.stderr, '\tframes'
	frames = [numpy.fromfile(f).reshape(shape) for f in sorted(glob(adir+'/*.frame'))]

	actoutfs = sorted(glob(adir+'/*.newactout'))
	acttermfs = sorted(glob(adir+'/*.newactterm'))
	groundfs = sorted(glob(adir+'/*.ground'))

	print >> sys.stderr, '\tactouts'
	actouts = [eval(open(f).read()) for f in actoutfs]
	print >> sys.stderr, '\tactterms'
	actterms = [eval(open(f).read()) for f in acttermfs]
	print >> sys.stderr, '\tgrounds'
	grounds = [Util.smooth(numpy.fromfile(f,sep='\n'),10) for f in groundfs[:-1]]

	hill_area = numpy.array([sum(shape[0]-g[hill_left:hill_right]) for g in grounds])

	grounds = grounds[1:] #drop first ground after calculating differences



	
	spec = iplot.subspectrum(len(actouts))


	figobj = pylab.figure(fig,figsize=(9,6))
	figobj.clf()
	print >> sys.stderr, 'render:'

	pylab.gray()

	if not show_mice:
		bkgd_mat = frames[-1]-(frames[0]-frames[-1])
		if graph_height is not  None:
			h = frames[0].shape[0] * graph_height
			bkgd_mat = numpy.array(list(bkgd_mat) + list(numpy.zeros((h,bkgd_mat.shape[1]))))
		pylab.matshow(bkgd_mat,fignum=fig)
	else:
		print >> sys.stderr, 'show mice invoked; loading mice...',
		mice = [eval(open(f).read()) for f in sorted(glob(adir+'/*.mice'))]
		xco,yco = Util.dezip(reduce(lambda x,y: x+y,[[v for v in m.values() if v] for m in mice]))
		print >> sys.stderr, 'done'
		bkgd_mat = numpy.zeros(shape)
		#raise NotImplementedError
		if graph_height is not  None:
			h = frames[0].shape[0] * graph_height
			bkgd_mat = numpy.array(list(bkgd_mat) + list(numpy.zeros((h,bkgd_mat.shape[1]))))
		pylab.matshow(bkgd_mat,fignum=fig)
		m,x,y = numpy.histogram2d(xco,yco,bins=numpy.array(shape[::-1])/5)
		#the following is nasty: re-center to -0.5 bin width, log transform counts
		pylab.contourf(x[1:]-(x[1]-x[0])/2,y[1:]-(y[1]-y[0])/2,numpy.log(m.transpose()+1),100)

	
	ax = figobj.axes[0]

	ax.set_xticks([])
	ax.set_yticks([])

	
	for i,poly in enumerate(actouts):
		for p in poly:
			if p: ax.add_patch(pylab.matplotlib.patches.Polygon(p,fc='none',ec=spec[i]))

	for i, terms in enumerate(actterms):
		for t in terms:
			x,y = Util.dezip(t)
			pylab.plot(x,y,c=spec[i])

	vinf,ainf = adir.split('/analysis/')

	ax.text(10,20,vinf,fontsize=8,color='w')
	ax.text(10,35,ainf,fontsize=8,color='w')


	if graph_height is not None:
		areas = [sum([len(filter(None,vidtools.mask_from_outline(p,frames[0].shape).flat)) for p in polys]) for polys in actouts]
		progs = [True and sum([vidtools.hypotenuse(*t) for t in terms]) or 0 for terms in actterms]

		hill_increase = []

		
		for v in hill_area[1:] - hill_area[:-1]:
			if 0 < v:
				hill_increase.append(v)
			else:
				hill_increase.append(0)


		hill_increase = numpy.array(hill_increase)
		z3 = (3*hill_increase.std()) + hill_increase.mean()
		#willing to accept that a mouse kicks out at no more than 2x max observed burrow-build
		max_increase = max(2*max(areas),z3)

		print >> sys.stderr, 'for max ground change, z3 = %s; 2*max_areas = %s.  Choose %s' % (z3,2*max(areas),max_increase)
		
		for i,v in enumerate(hill_increase):
			if v > max_increase:
				hill_increase[i] = 0
		
		h = frames[0].shape[0] * graph_height
		base = frames[0].shape[0] + h/2 - 10
		scale = frames[0].shape[1]/float(len(areas))

		x = numpy.arange(0,frames[0].shape[1],scale)[:len(areas)]

		if dawn_min:
			pylab.bar(x[dawn_min],1*h/2,color='w',edgecolor='w',bottom=base)
			pylab.bar(x[dawn_min],-1*h/2,color='w',edgecolor='w',bottom=base)
		hours = []
		for m in range(len(spec)):
			if m%60==0:
				hours.append(1)
			else:
				hours.append(0)
		print len(hours)
		pylab.bar(x,numpy.array(hours)*h/2,color='w',edgecolor='w',alpha=0.4,linewidth=0,bottom=base)
		pylab.bar(x,-1*numpy.array(hours)*h/2,color='w',edgecolor='w',alpha=0.4,linewidth=0,bottom=base)
		
		pylab.bar(x[skip_first:],Util.normalize(numpy.array(hill_increase[skip_first:]))*h/2,color=spec,edgecolor=spec,bottom=base)
		pylab.plot(x[skip_first:],base+Util.normalize(numpy.cumsum(numpy.array(hill_increase[skip_first:])))*h/2,'--w')
		for i,v in zip(x[skip_first:],Util.normalize(numpy.cumsum(numpy.array(hill_increase[skip_first:])))):
			if v > 0.5:
				break
		pylab.plot((i,i),(base,base+(h/2)),'--w',lw=2)
		pylab.bar(x[skip_first:],-1*Util.normalize(numpy.array(areas[skip_first:]))*h/2,color=spec,edgecolor=spec,bottom=base)
		pylab.plot(x[skip_first:],base-Util.normalize(numpy.cumsum(numpy.array(areas[skip_first:])))*h/2,'--w')
		for i,v in zip(x[skip_first:],Util.normalize(numpy.cumsum(numpy.array(areas[skip_first:])))):
			if v > 0.5:
				break
		pylab.plot((i,i),(base,base-(h/2)),'--w',lw=2)
		pylab.text(5,base+h/2,'%0.1f' % max(hill_increase[skip_first:]),color='w')
		pylab.text(5,base-h/2,'%0.1f' % max(areas[skip_first:]),color='w')

	#'''
	spec.reverse() #flip spectrum to draw grounds back-to-front
	for i, g in enumerate(grounds[::-1]):
		pylab.plot(numpy.arange(hill_left,hill_right),g[hill_left:hill_right],c=spec[i])
	spec.reverse() #flip back
	#'''

	pylab.plot()
	pylab.ylim(1.2*shape[0],0)
	pylab.xlim(0,shape[1])

	if outformat is not None:
		print >> sys.stderr, 'save as '+adir+'/activity_summary.'+outformat
		figobj.savefig(adir+'/activity_summary.'+outformat)
	

def draw_class_scatter_and_pie(locsfile,format='pdf',sps=None,scatter_lines=None,scatter_polys=None,draw_pie=True,shape=(480,720),poly_col=None):
    '''given a .mouselocs file
    (and assuming .locsumm, .frame and .mice are present)
    draws a .scatter.[format] and .pie.[format] summarizing the data in that segment

	if .ground or .actmask are present, these will be plotted as lines and polygons respectively
    '''

    def secs(x):
        return '%0.1f sec' % ((x*sps)/100)

    if sps is None:
        match = re.search(r'\/([\d\.]+)sec_',locsfile)
        if match:
            sps = float(match.groups()[0])
        else:
            sps = 1.0

    if scatter_polys is not None and poly_col is None:
        poly_col = ['b']*len(scatter_polys)
    pylab.gray()
    print >> sys.stderr, 'loading mice'
    mice = eval(open(os.path.splitext(locsfile)[0]+'.mice').read())
    print >> sys.stderr, 'loading location summaries'
    locsumm = eval(open(os.path.splitext(locsfile)[0]+'.locsumm').read())
    print >> sys.stderr, 'loading locations'
    mouselocs = eval(open(locsfile).read())
    bkgd_image = sorted(mice.keys())[0] # grabs first frame as background

    sca_out = os.path.splitext(locsfile)[0]+'.scatter.'+format
    pie_out = os.path.splitext(locsfile)[0]+'.pie.'+format
    if draw_pie:
        print >> sys.stderr, 'drawing pie (%s) and scatter (%s) with data from %s (%s frames, %s seconds per segment)' % \
	          (pie_out,sca_out,locsfile,len(mouselocs.keys()),sps)
    else:
        print >> sys.stderr, 'drawing scatter (%s) with data from %s (%s frames, %s seconds per segment)' % \
          (sca_out,locsfile,len(mouselocs.keys()),sps)

    class_locs = Util.invert_dict(mouselocs)

    fig = pylab.figure(1,figsize=(9,6))
    fig.clf()
    print >> sys.stderr, 'render:'
    try:
	    bkgd_mat = vidtools.load_normed_arrays([bkgd_image])[0]
    except:
	    print >> sys.stderr, '\timages unavailable, attempting with .frame'
	    frame = os.path.splitext(locsfile)[0]+'.frame'
	    bkgd_mat = numpy.fromfile(frame).reshape(shape)
    print >> sys.stderr, '\tframe'
    pylab.matshow(bkgd_mat,fignum=1)
    fig.axes[0].set_xticks([])
    fig.axes[0].set_yticks([])

    if os.path.exists(os.path.splitext(locsfile)[0]+'.ground'):
        ground = pylab.fromfile(os.path.splitext(locsfile)[0]+'.ground',sep='\n')
        pylab.plot(ground,'g')

	if os.path.exists(os.path.splitext(locsfile)[0]+'.actpoly'):
		actpoly = eval(open(os.path.splitext(locsfile)[0]+'.actpoly').read())
	#elif os.path.exists(os.path.splitext(locsfile)[0]+'.actmat'):
	#	actmat = numpy.fromfile(os.path.splitext(locsfile)[0]+'.actmat').reshape(bkgd_mat.shape)
	#	if ground:
	#		actmat = Util.subtract_mask(actmat,vidtools.mask_from_vector(ground+10,bkgd_mat.shape))
	#attempt to draw activity polygons from actmasks directly is too much work at this step;
	#supply scatter_polys or make sure an actpoly file is present	

    counts = []
    piecols = []
    
    print >> sys.stderr, '\tlines'
    if isinstance(scatter_lines,list):
        for l in scatter_lines:
            pylab.plot(l)
			
    print >> sys.stderr, '\tpolygons'
    if isinstance(scatter_polys,list):
        for i,p in enumerate(scatter_polys):
	    try:
                fig.axes[0].add_patch(pylab.matplotlib.patches.Polygon(p,fc='none',ec=poly_col[i]))
	    except AssertionError:
	        print >> sys.stderr, 'poly %s improper dimensions' % p

    print >> sys.stderr, '\tscatter'
    for act_class,frames in class_locs.items():
        if act_class is None:
            continue
        coords = filter(None,[mice[f] for f in frames])
        if (coords):
            x,y = Util.dezip(coords)
            #pylab.scatter(x,y,s=10,c=col[act_class])
	    pylab.scatter(x,y,s=3,c='r',edgecolors='none')
        counts.append(len(frames))
        piecols.append(col[act_class])

    pylab.savefig(sca_out)
    if draw_pie:
        print >> sys.stderr, '\tpie'
        fig = pylab.figure(2,figsize=(6,6))
        fig.clf()
        pylab.pie(counts,colors=piecols,autopct=secs )
        pylab.savefig(pie_out)
   
def track_class_transitions(mouselocs,trans,plotfile=None,symm=False,fig=1):
    '''takes a mouselocs dict and records the number of times the mouse transitions between specified classifications

    FUTURE:
    trans are specified as a list of 2-tuples (numerical classification pairs).  Directionality is preserved (i.e. 1->2 != 2->1 given (1,2) != (2,1)
    setting symm = True overrides this (both directions will be counted together)
    NOW:
    trans are a list of numerical classifications, all transitions between these are scored


    if a path string is supplied to plotfile, will attempt to save the plot to that filename

    returns a dict of transition counts'''

    '''FUTURE
    if symm:
        ext = []
        for tup in trans:
            ext.append(tup[::-1])
        trans.extend(ext)

    trans = list(set(trans))
    '''

    transpairs = list(set([(i,j) for i in trans for j in trans if i != j]))
    track = {}.fromkeys(transpairs,0)
    for k in track:
        track[k] = []

    last = None
    for image,cl in sorted(mouselocs.items()):
        if cl in trans:
            if last is not None and cl != last[1]:
                track[(last[1],cl)].append((last[0],image))
            last = (image,cl)

    x = []
    y = []
    labels = []
    for i,(t,v) in enumerate(sorted(track.items())):
        x.append(i)
        y.append(len(v))
        labels.append('%s: %s' % (i,t))

    bar_fig = pylab.figure(fig)
    pylab.bar(x,y,align='center')
    bar_fig.axes[0].set_xticks(x)
    bar_fig.axes[0].set_xlabel('       '.join(labels))
    pylab.plot()

    return track
