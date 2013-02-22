#!/usr/bin/env python
'''run rudimentary behavioral coding on Y-maze video

takes as input:
imagedir,seglen,splitxy,topxy,bottomxy,mousez

seglen is the length of time (in seconds) to consider for each submission to summarize_segment.py

xy positions are center of fork, top gate and bottom gate respectively

'''

import vidtools,viz_vidtools,os,sys,re,Util, math,iplot,pylab
from glob import glob

nomask = 'None'

arg = sys.argv[1:]

(imagedir,seglen,splitxy,stemxy,topxy,bottomxy,mousez) = [arg.pop(0) for i in xrange(7)]

OPTJOBS=100
if 'jpg' in imagedir.split('/'):
    FORMAT='jpg'
    print >> sys.stderr, '"jpg" found in path, assuming FORMAT=jpg\n'
else:
    FORMAT='png'

match = re.search(r'\/([\d\.]+)fps\/',imagedir)
if match:
    FPS = match.groups()[0]
else:
    FPS = 30

images = sorted(glob(imagedir+'/*.'+FORMAT))
print >> sys.stderr, len(images),'images found in imagedir\n'
print >> sys.stderr, 'split, top and bottom XY:',splitxy,topxy,bottomxy

SHAPE=vidtools.load_normed_arrays(images[:1])[0].shape

match = re.search('(\d+)fps',imagedir)
if match:
    segment_step = int(seglen)*int(match.groups()[0])
    print >> sys.stderr, 'segment length will be %s frames (%s seconds, %s fps)\n' % (segment_step,seglen,match.groups()[0])
    unit = 'sec'
else:
    segment_step = int(seglen)
    print >> sys.stderr,'fps could not be found in image path, using seglen (%s) as segment_step (i.e. 1 fps)\n' % seglen
    unit = 'frames'

outroot = os.path.join(imagedir,'analysis','%s%s_%smousez' % (seglen,unit,mousez))

try:
    os.makedirs(outroot)
except OSError:
    pass

shapefile = os.path.join(outroot,'shape.tuple')
open(shapefile,'w').write(SHAPE.__repr__())

jobids = {}
namedict = {}
prereq = []
if not glob(outroot+'/*.mice'):
    cmds = []
    for i in range(0,len(images),segment_step):
        cmds.append('summarize_segment-ymaze.py %s %s %d %d %s %s %s'
                    % (nomask,'None',i,i+segment_step,imagedir,outroot,mousez) )
    logfile = os.path.join(outroot,'summarize-segment-log')
    print >> sys.stderr,'.mice not present in %s\n\trunning summary of %s segments, log written to %s\n' % (outroot,len(cmds),logfile)
    jids,ndict = Util.lsf_jobs_submit(cmds,logfile,'short_serial',jobname_base='summarize')
    jobids.update(jids)
    namedict.update(ndict)
    prereq = ndict.values()
    Util.lsf_wait_for_jobs(jobids,os.path.join(outroot,'restarts'),namedict)
    #Util.lsf_wait_for_jobs(jobids,logfile)
else:
    print >> sys.stderr,'.mice files present in %s\n\tmoving right along...\n' % outroot


jobids = {}
namedict = {}
    
exp_mice = int(math.ceil(len(images)/float(segment_step)))

if not glob(outroot+'/*.mouselocs'):
    interval = max(exp_mice / OPTJOBS , 10)
    pad_mask = 5     #pixels to expand the segment activity mask by
    cmds = []
    for i in range(0,exp_mice,interval):
        cmds.append('classify_mouse_loc-ymaze.py %d %d %s \\"%s\\" \\"%s\\" \\"%s\\"'
                    % (i,i+interval,outroot,splitxy,topxy,bottomxy) )
    logfile = os.path.join(outroot,'classify-mouse-loc-log')
    print >> sys.stderr,'.mouselocs not present in %s\n\trunning classification of %s segments in %s jobs, log written to %s\n' % \
          (outroot,exp_mice,len(cmds),logfile)
    jids,ndict = Util.lsf_jobs_submit(cmds,logfile,'normal_serial',jobname_base='classify')
    jobids.update(jids)
    namedict.update(ndict)
    prereq = ndict.values()
    Util.lsf_wait_for_jobs(jobids,os.path.join(outroot,'restarts'),namedict)
    #Util.lsf_wait_for_jobs(jobids,logfile)
else:
    print >> sys.stderr,'.mouselocs present in %s\n\tmoving right along...\n' % outroot

jobids = {}
namedict = {}

exp_locs = exp_mice
movie = os.path.join(outroot,'full_%szmouse.mov' % (mousez))

if not os.path.exists(movie):
    #interval = max(exp_mice / OPTJOBS , 1)
    interval = 1
    cmds = []
    for i in range(0,exp_locs,interval):
        cmds.append('draw_mouse_locations.py %s %d %d' % (outroot,i,i+interval))
    logfile = os.path.join(outroot,'draw-mouse-locations-log')
    print >> sys.stderr,'final video not present in %s\n\trunning rendering of %s segments in %s jobs, log written to %s\n' \
          % (outroot,exp_locs,len(cmds),logfile)
    jids,ndict = Util.lsf_jobs_submit(cmds,logfile,'short_serial',jobname_base='draw')
    jobids.update(jids)
    namedict.update(ndict)
    prereq = ndict.values()
    Util.lsf_wait_for_jobs(jobids,os.path.join(outroot,'restarts'),namedict)

    jobids = {}
    namedict = {}
    
    merge = 'ffmpeg -r %s -i %s/images/%%07d.png -b 1500k %s' % (FPS,outroot,movie)  

    print >> sys.stderr,'rendering movie, log written to %s\n' % (logfile)
    jids,ndict = Util.lsf_jobs_submit([merge],logfile,'long_serial',jobname_base='video')
    jobids.update(jids)
    namedict.update(ndict)    

    #Util.lsf_wait_for_jobs(jobids,os.path.join(outroot,'restarts'),namedict)

micefiles = sorted(glob(outroot+'/*.mice'))
locsfiles = sorted(glob(outroot+'/*.mouselocs'))
locsummfiles = sorted(glob(outroot+'/*.locsumm'))
framefiles = sorted(glob(outroot+'/*.frame'))

print >> sys.stderr,'%s .mice found (%s expected)\n' % (len(micefiles),exp_mice)
print >> sys.stderr,'%s .mouselocs found (%s expected)\n' % (len(locsfiles),exp_locs)

mice = Util.merge_dictlist([eval(open(f).read()) for f in micefiles])
locs = Util.merge_dictlist([eval(open(f).read()) for f in locsfiles])
locsumms = Util.countdict(locs.values())

mice_out = movie.rsplit('.',1)[0]+'.mice'
open(mice_out,'w').write(mice.__repr__())
locs_out = movie.rsplit('.',1)[0]+'.mouselocs'
open(locs_out,'w').write(locs.__repr__())
summ_out = movie.rsplit('.',1)[0]+'.locsumm'
open(summ_out,'w').write(locsumms.__repr__())

sps = float(seglen) * len(locsfiles)

viz_vidtools.draw_class_scatter_and_pie(locs_out,sps=sps)

#histos of distance-to-gate in post-decision locations - class 2 are 'top'; class 3 'bottom'
inv_locs = Util.invert_dict(locs)

stemxy = eval(stemxy)
topxy = eval(topxy)
bottomxy = eval(bottomxy)

stemmice = [mice[f] for f in inv_locs[1] if mice[f][0] > stemxy[0]]
topmice = [mice[f] for f in inv_locs[2]]
bottommice = [mice[f] for f in inv_locs[3]]

stemdists = vidtools.distance_from_point(stemmice,stemxy)
topdists = vidtools.distance_from_point(topmice,topxy)
bottomdists = vidtools.distance_from_point(bottommice,bottomxy)

bins = pylab.arange(0,max(stemdists+topdists+bottomdists),10)

iplot.draw_line_hist(stemdists,bins,color=viz_vidtools.col[1],fig=3)
iplot.draw_line_hist(topdists,bins,color=viz_vidtools.col[2],fig=3)
iplot.draw_line_hist(bottomdists,bins,color=viz_vidtools.col[3],fig=3)

pylab.savefig(movie.rsplit('.',1)[0]+'.hist.pdf')
