import iplot
import numpy
import pylab
import tarfile
import Util
from video_analysis import vidtools,submit_summarize_runs
from glob import glob
import os,sys,re
from collections import defaultdict
from scipy import stats

def peak_hour(digs,win=120):
    starts = numpy.arange(0,len(digs)-win,1)
    return sorted([(sum(digs[start:start+win]),start) for start in starts])[-1][1]/float(win)

def regression_scatter(X,Y,meas,fignum=1,subplotnum=(1,1,1)):
    pylab.figure(fignum)
    pylab.subplot(*subplotnum)
    pylab.scatter( X,Y,edgecolor='k',facecolor='none')
    m,b,r,p,se = stats.linregress( X,Y)
    yval = lambda x: m*x+b
    pylab.plot([min(X), max(X)],[yval(min(X)),yval(max(X))],'r')
    pylab.title(meas+' (r^2: %0.2f | p: %0.1e)' % (r,p))

def paired_trials(d,idx1=0,idx2=-1):

    d_by_ind = defaultdict(list)
    for k,v in d.items():
        if -numpy.inf<v<numpy.inf:
            d_by_ind[k.split('-',1)[1]].append(v)
    
    X,Y = zip(*[(fn(v)[idx1],fn(v)[idx2]) for k,v in d_by_ind.items() if len(fn(v))>1])
    return X,Y

def paired_measures(d1,d2):
    fn = lambda x:-numpy.inf<x<numpy.inf
    X,Y = zip(*[(d1[k],d2[k]) for k in d1.keys() if k in d1 and k in d2 and fn(d1.get(k,numpy.inf)) and fn(d2.get(k,numpy.inf))])
    return X,Y

def plot_trial_repeatability(digs,fignum=1):
    for i,(meas,d) in enumerate(digs.items(),1):
        X,Y = paired_trials(dict(d[3]))
        regression_scatter(X,Y,meas,fignum,(2,2,i))

def violins(boxdata,labels=['BW','PO','F1','F2'],fignum=1,subplotnum=(1,1,1),pointsize=10,mediancolors=['r','b']):
    fig = pylab.figure(fignum)
    ax = fig.add_subplot(*subplotnum)
    
    if pointsize:
        [pylab.scatter( \
            ((numpy.random.random(size=len(v))-0.5)*0.2)+i, \
            v, \
            facecolor='none',edgecolor='k',s=pointsize) \
         for i,v in enumerate(boxdata)]
    if mediancolors:
        for c,v in zip(mediancolors,boxdata):
            pylab.plot([-0.5,len(boxdata)-0.5],[numpy.median(v),numpy.median(v)],c+':')
    iplot.violin_plot(ax,boxdata,range(4),bp=0)
    pylab.xticks(range(len(labels)),labels)

def pct_updig(checked,kbouts=2,maxsep=50):
    ad = submit_summarize_runs.analysis_dir_from_donebase(checked.rsplit('.',1)[0])
    #config = eval(open(checked.split('l1800')[0]+'config.dict').read())

    newact_tarf = glob(ad+'/newact_ols_postsummary-*.tar')[0]

    newact_tarh = tarfile.open(newact_tarf)
    newacts = Util.tar2obj_all(newact_tarh)


    SHAPE = eval(open(ad+'/SHAPE').read())
    cm = vidtools.calc_coordMat(SHAPE)
    centroids = [vidtools.centroid(reduce(lambda x,y:x+y, l),SHAPE,cm) for l in newacts if len(l)>0]
    
    #pdy = config['predug'][1][1]
    
    #return len(filter(lambda x:x<pdy, \
    #                  [y for x,y in centroids]))/float(len(centroids))
    k = trial_from_filename(glob(ad+'/*hwc4*hmcdNone*dig_areas.list')[0])
    y_offsets = [p1[1]-p2[1] for p1,p2 in zip(centroids[:-kbouts],centroids[kbouts:]) if vidtools.hypotenuse(p1,p2)<maxsep]

    return k,len(filter(lambda x:x>0,y_offsets))/float(len(y_offsets))

trial_from_filename = lambda f: f.split('/')[-1].split('_')[0]


bounds = {}

bounds['20111102-BW-1192'] = (959,1063)
bounds['20111013-BW-1191'] = (175,177)
bounds['20111013-BW-1192'] = (899,1007)
bounds['20111108-BW-1194'] = (834,1014)
bounds['20091121-BWf783'] = (415,1026)

def make_boxdata(digs):
    filt_fn = { 'udigs':lambda x: x<1, \
                'area_mean':lambda x: 0<x<numpy.inf, \
                'pct':lambda x: 0<x<numpy.inf, \
                'peak_hour':lambda x: 0<x<numpy.inf }

    boxdata = {}
    for k,v in digs.items():
        boxdata[k] = [filter(filt_fn[k],dict(l).values()) for l in v]

    return boxdata

def load_all(ROOT='/n/hoekstrafs2/burrowing/antfarms/data'):
    areas = {}
    checked = glob(os.path.join(ROOT,'*/*/*.checked'))

    for chk in checked:
        ad = submit_summarize_runs.analysis_dir_from_donebase(chk.rsplit('.',1)[0])
        digfs = glob(ad+'/*hwc4*hmcdNone*dig_areas.list')
        areas[digfs[0]] = eval(open(digfs[0]).read())

    digs = {}
    print >> sys.stderr, 'updigs'
    digs['udigs'] = load_udigs(checked)
    print >> sys.stderr, 'pct'
    digs['pct'] = load_digs_pct(areas)
    print >> sys.stderr, 'peak_hour'
    digs['peak_hour'] = load_digs_peak_hour(areas)
    print >> sys.stderr, 'area_mean'
    digs['area_mean'] = load_digs_area_mean(areas)

    return digs

def load_udigs(checked,kbouts=2,maxsep=50):
    BW_udigs = []
    print >> sys.stderr, 'BW'
    for chk in [chk for chk in checked if '0_BW' in chk]:
        try:
            k,v = pct_updig(chk,kbouts,maxsep)
            if v < 1: BW_udigs.append((k,v))
        except:
            pass
        
    PO_udigs = []
    print >> sys.stderr, 'PO'
    for chk in [chk for chk in checked if '0_PO' in chk]:
        try:
            k,v = pct_updig(chk,kbouts,maxsep)
            if v < 1: PO_udigs.append((k,v))
        except:
            pass
        
    F1_udigs = []
    print >> sys.stderr, 'F1'    
    for chk in [chk for chk in checked if '0_F1' in chk]:
        try:
            k,v = pct_updig(chk,kbouts,maxsep)
            if v < 1: F1_udigs.append((k,v))
        except:
            pass
        
    F2_udigs = []
    print >> sys.stderr, 'F2'
    for chk in [chk for chk in checked if '0_F2' in chk]:
        try:
            k,v = pct_updig(chk,kbouts,maxsep)
            if v < 1: F2_udigs.append((k,v))
        except:
            pass
        
    return BW_udigs,PO_udigs,F1_udigs,F2_udigs

def load_digs_peak_hour(areas,after=0.5):
    BW_digs_peak_hour = [( trial_from_filename(k), peak_hour(v)) for k,v in areas.items() if '0_BW' in k and peak_hour(v) > after and len(v)<1500]
    PO_digs_peak_hour = [( trial_from_filename(k), peak_hour(v)) for k,v in areas.items() if '0_PO' in k and peak_hour(v) > after and len(v)<1500]
    F1_digs_peak_hour = [( trial_from_filename(k), peak_hour(v)) for k,v in areas.items() if '0_F1' in k and peak_hour(v) > after and len(v)<1500]
    F2_digs_peak_hour = [( trial_from_filename(k), peak_hour(v)) for k,v in areas.items() if '0_F2' in k and peak_hour(v) > after and len(v)<1500]

    sk = lambda i:i[1]
    sortdrop = lambda x:sorted(x,key=sk)[2:]
    return map(sortdrop,[BW_digs_peak_hour,PO_digs_peak_hour,F1_digs_peak_hour,F2_digs_peak_hour])

def load_digs_pct(areas):
    BW_digs_pct = [( trial_from_filename(k),len(filter(None,v[bounds.get(trial_from_filename(k),(0,-1))[0]:bounds.get(trial_from_filename(k),(0,-1))[1]]))/float(len(v))) for k,v in areas.items() if '0_BW' in k]
    PO_digs_pct = [( trial_from_filename(k),len(filter(None,v[bounds.get(trial_from_filename(k),(0,-1))[0]:bounds.get(trial_from_filename(k),(0,-1))[1]]))/float(len(v))) for k,v in areas.items() if '0_PO' in k]
    F1_digs_pct = [( trial_from_filename(k),len(filter(None,v[bounds.get(trial_from_filename(k),(0,-1))[0]:bounds.get(trial_from_filename(k),(0,-1))[1]]))/float(len(v))) for k,v in areas.items() if '0_F1' in k]
    F2_digs_pct = [( trial_from_filename(k),len(filter(None,v[bounds.get(trial_from_filename(k),(0,-1))[0]:bounds.get(trial_from_filename(k),(0,-1))[1]]))/float(len(v))) for k,v in areas.items() if '0_F2' in k]

    return BW_digs_pct,PO_digs_pct,F1_digs_pct,F2_digs_pct

def load_digs_area_mean(areas):
    BW_digs_area_mean = [( trial_from_filename(k),numpy.mean(filter(None,v[bounds.get(trial_from_filename(k),(0,-1))[0]:bounds.get(trial_from_filename(k),(0,-1))[1]]))) for k,v in areas.items() if '0_BW' in k]
    PO_digs_area_mean = [( trial_from_filename(k),numpy.mean(filter(None,v[bounds.get(trial_from_filename(k),(0,-1))[0]:bounds.get(trial_from_filename(k),(0,-1))[1]]))) for k,v in areas.items() if '0_PO' in k]
    F1_digs_area_mean = [( trial_from_filename(k),numpy.mean(filter(None,v[bounds.get(trial_from_filename(k),(0,-1))[0]:bounds.get(trial_from_filename(k),(0,-1))[1]]))) for k,v in areas.items() if '0_F1' in k]
    F2_digs_area_mean = [( trial_from_filename(k),numpy.mean(filter(None,v[bounds.get(trial_from_filename(k),(0,-1))[0]:bounds.get(trial_from_filename(k),(0,-1))[1]]))) for k,v in areas.items() if '0_F2' in k]

    return BW_digs_area_mean,PO_digs_area_mean,F1_digs_area_mean,F2_digs_area_mean

    
