import pylab,vidtools,os,sys,re,Util

col = {0:'w',1:'r',2:'y',3:'g',4:'b',5:'c'}

def draw_class_scatter_and_pie(locsfile,format='pdf',sps=None,scatter_lines=None,scatter_polys=None):
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
    
    pylab.gray()

    mice = eval(open(os.path.splitext(locsfile)[0]+'.mice').read())
    locsumm = eval(open(os.path.splitext(locsfile)[0]+'.locsumm').read())
    mouselocs = eval(open(locsfile).read())
    bkgd_image = sorted(mice.keys())[0] # grabs first frame as background

    sca_out = os.path.splitext(locsfile)[0]+'.scatter.'+format
    pie_out = os.path.splitext(locsfile)[0]+'.pie.'+format

    print >> sys.stderr, 'drawing pie (%s) and scatter (%s) with data from %s (%s frames, %s seconds per segment)' % \
          (pie_out,sca_out,locsfile,len(mouselocs.keys()),sps)

    class_locs = Util.invert_dict(mouselocs)

    fig = pylab.figure(1,figsize=(9,6))
    fig.clf()
    bkgd_mat = vidtools.load_normed_arrays([bkgd_image])[0]
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
    
    if isinstance(scatter_lines,list):
        for l in scatter_lines:
            pylab.plot(l)
			
    if isinstance(scatter_polys,list):
        for p in scatter_polys:
            fig.axes[0].add_patch(pylab.matplotlib.patches.Polygon(p,fc='none',ec='b'))

    for act_class,frames in class_locs.items():
        if act_class is None:
            continue
        coords = filter(None,[mice[f] for f in frames])
        if (coords):
            x,y = Util.dezip(coords)
            pylab.scatter(x,y,s=10,c=col[act_class])
        counts.append(len(frames))
        piecols.append(col[act_class])

    pylab.savefig(sca_out)

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
