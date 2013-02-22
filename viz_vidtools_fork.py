import pylab,vidtools,os,sys,re,Util

col = {0:'w',1:'r',2:'y',3:'g',4:'b',5:'c'}

def draw_class_scatter_and_pie(locsfile,format='pdf',sps=None):
    '''given a .mouselocs file
    (and assuming .locsumm, .frame and .mice are present)
    draws a .scatter.[format] and .pie.[format] summarizing the data in that segment
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
    pylab.matshow(vidtools.load_normed_arrays([bkgd_image])[0],fignum=1)
    fig.axes[0].set_xticks([])
    fig.axes[0].set_yticks([])

    counts = []
    piecols = []
    
    for act_class,frames in class_locs.items():
        if act_class is None:
            continue
        coords = filter(None,[mice[f] for f in frames])
        if (coords):
            x,y = Util.dezip(coords)
            pylab.scatter(x,y,c=col[act_class])
        counts.append(len(frames))
        piecols.append(col[act_class])

    pylab.savefig(sca_out)

    fig = pylab.figure(2,figsize=(6,6))
    pylab.pie(counts,colors=piecols,autopct=secs )
    pylab.savefig(pie_out)
    
    
def track_class_transitions(mouselocs,trans,plotfile=None,fig=1):

    transpairs = list(set([(i,j) for i in trans for j in trans if i != j]))
    track = {}.fromkeys(transpairs,0)
    for k in track.keys():
        track[k] = []

    x = []
    y = []
    labels = []
    last = None
    for image,cl in sorted(mouselocs.items()):
        if cl in trans:
            if last is not None and cl != last[1]:
                track[(last[1],cl)].append((last[0],image))
            last = (image,cl)

    for i,(k,v) in enumerate(sorted(track.items(),key = lambda x: len(x[1]))):
        x.append(i)
        y.append(len(v))
        labels.append('%s: %s' % (i,k))

    bar_fig = pylab.figure(fig)
    pylab.bar(x,y,align='center')
    bar_fig.axes[0].set_xticks(x)
    bar_fig.axes[0].set_xlabel('     '.join(labels))
    pylab.plot()
    if plotfile:
        pylab.savefig(plotfile)

    return track
