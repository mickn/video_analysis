#!/usr/bin/env python
#
# reanalyze_activity_segments.py prev_analysis_dir new_tdir new_anaylsis_interval
#
# given a source dir <prev_analysis_dir> for non-activity data
# (grounds, mice, previous activity, ...)
# recomputes new activity on specified timescale <new_analysis_interval> (in seconds)
# over data in <new_tdir> (e.g. .pngs from interval with observed activity)

import os, sys, re, Util, numpy

from glob import glob
from video_analysis import vidtools,viz_vidtools

prev_adir, new_tdir, new_seglen = sys.argv[1:]
new_seglen = float(new_seglen)

#extract fps from path
match = re.search('([\d\.]+)fps',new_tdir)
if match:
    fps = float(match.groups()[0])
    segment_step = int(new_seglen * fps)
    if segment_step == 0: segment_step = 1
    print >> sys.stderr, 'segment length will be %s frames (%s seconds, %s fps)\n' % (segment_step,new_seglen,fps)
    unit = 'sec'
else:
    print >> sys.stderr,'fps could not be found in image path, please ensure that fps appears in new_tdir (%s given)\n' % new_tdir
    sys.exit(1)

match = re.search('/([\d\w-]+)/([\d\.]+)fps.+?([\d\.]+)zmouse_([\d\.]+)zburrow_([\d\.]+)pixav_([\d\.]+)timeav',prev_adir)
if match:
    source_analysis_window = match.groups()[0]
    source_fps,zmouse,zburrow,pixav,timeav = [int(m) for m in match.groups()[1:]]
    offset_fr = 0
    if '-' in source_analysis_window:
        start = source_analysis_window.split('-')[0]
        if start != 'start':
            offset_fr = int(start)*source_fps
            print >> sys.stderr, 'offset of source %s sec at %s fps; %s frames' % (start,source_fps,offset_fr)
else:
    print >> sys.stderr, 'could not find original analysis parameters in previous analysis dir (%s given)' % prev_adir

print >> sys.stderr, 'loading images...',
new_ims = sorted(glob(new_tdir+'/*.png'))
print >> sys.stderr, '%s images found' % len(new_ims)

#get frame filenames from source <prev_adir>
sourceframes = sorted(glob(prev_adir+'/*.frame'))
sourcespans = [[int(i)+offset_fr for i in os.path.basename(l).split('.')[0].split('-')] for l in sourceframes]

new_outroot = os.path.join(new_tdir,'analysis/%0.1fsec_%szmouse_%szburrow_%spixav_%stimeav/' % (new_seglen,zmouse,zburrow,pixav,timeav) )
if not os.path.exists(new_outroot): os.makedirs(new_outroot)
open(new_outroot+'sourcedir','w').write(prev_adir)

#get, and ref shape
os.system('cp %s/shape.tuple %s' % (prev_adir,new_outroot))
shape = eval(open(new_outroot+'/shape.tuple').read())

currmice = ''

for i in range(0,len(new_ims),segment_step):
    startframe = int(os.path.basename(new_ims[i]).split('.')[0])
    new_base = new_outroot+'%07d-%07d'% (startframe,startframe+segment_step)
    #print >> sys.stderr, 'new base %s, %s in ' % (new_base,startframe)
    for j,(s,e) in enumerate(sourcespans):
        if startframe in range(s,e):
            break
    source_base = os.path.splitext(sourceframes[j])[0]
    #print >> sys.stderr, source_base

    if i == 0: # grab required files: reqfiles = [newactout,newactterm,preactout,newactprop] while we're here
        nextbase = new_outroot+'%07d-%07d'% (startframe+segment_step,startframe+2*segment_step)
        for ext in ['.newactout','.newactterm','.preactout','.newactprop']:
            #os.system('cp %s %s' % (source_base+ext,new_base+ext))
            os.system('cp %s %s' % (source_base+ext,nextbase+ext))

    # get source .mice
    micefile = source_base+'.mice'
    if micefile != currmice and currmice != 'all':
        print >> sys.stderr, 'load mice from %s ...' % micefile,
        mice = eval(open(micefile).read())
        # HAX grabs next .mice, in case we "run over"
        try:
            mice.update(eval(open(os.path.splitext(sourceframes[j+1])[0]+'.mice').read()))
        except IndexError:
            pass
        mkbase = os.path.dirname(mice.keys()[0])
        print >> sys.stderr, 'done'
        currmice = micefile

    # if we need to write mice:
    if not os.path.exists(new_base+'.mice'):
        try:
            thesemice = dict([(os.path.basename(k),mice[os.path.join(mkbase,os.path.basename(k))]) for k in new_ims[i:i+segment_step]])
        except KeyError:
            try:
                if currmice != 'all':
                    allmicef = glob(os.path.dirname(source_base)+'/*sec*.mice')[0]
                    print >>sys.stderr, 'key %s not found; loading ALL mice from %s ...' % (os.path.join(mkbase,os.path.basename(k)),allmicef),
                    mice = eval(open(allmicef).read())
                    print >> sys.stderr, 'done'
                    currmice = 'all'
                thesemice = dict([(os.path.basename(k),mice[os.path.join(mkbase,os.path.basename(k))]) for k in new_ims[i:i+segment_step]])
            except KeyError:
                print >>sys.stderr, 'key %s not found; currently loaded .mice is: %s\nswitching to frame-by-frame' % (os.path.join(mkbase,os.path.basename(k)),currmice)
                thesemice = {}
                for k in new_ims[i:i+segment_step]:
                    try:
                        thesemice[os.path.basename(k)] = mice[os.path.join(mkbase,os.path.basename(k))]
                    except KeyError:
                        print >>sys.stderr, 'key %s not found; skip' % os.path.join(mkbase,os.path.basename(k))
                
        #print >> sys.stderr, 'write %s mice to %s' % (len(thesemice),new_base+'.mice')
        open(new_base+'.mice','w').write(thesemice.__repr__())

    #if we need to link ground:
    if not os.path.exists(new_base+'.ground'):
        #print >> sys.stderr, 'link %s from %s' % (new_base+'.ground',source_base+'.ground')
        os.system('ln -s %s %s' % (os.path.join(os.getcwd(),source_base+'.ground'), os.path.join(os.getcwd(),new_base+'.ground') ) )

    #if we need to compute frame avg:
    if not os.path.exists(new_base+'.frame'):
        frameav = vidtools.average_frames(vidtools.load_normed_arrays(new_ims[i:i+segment_step],pixav))
        frameav.tofile(new_base+'.frame')

frameavs = sorted(glob(new_outroot+'/*.frame'))
print >> sys.stderr, 'compute activity matrices from frame averages (%s frame averages found)' % len(frameavs)
for i,f in enumerate(frameavs):
    actmatf = os.path.splitext(f)[0]+'.actmat'
    if not os.path.exists(actmatf) and i>0 and i<len(frameavs)-1:
        actmat = Util.zscore(numpy.fromfile(frameavs[i-1]).reshape(shape) - numpy.fromfile(frameavs[i+1]).reshape(shape))
        actmat.tofile(actmatf)
    




# switches cumulative activity calc to activity below an artificial "lowest ground" - should fix squirrely groundfinding

allgrounds = numpy.array([numpy.fromfile(f,sep='\n') for f in glob(prev_adir+'/*.ground')])
low_ground = numpy.array([max(col) for col in allgrounds.transpose()])

#zburrow = 3
#for now, force_all is set manually; might migrate to opts eventually
activity,termini,progress = vidtools.calculate_cumulative_activity(new_outroot,float(zburrow),shape=shape,be=None,suppress_ground=5,force_all=False,use_ground=low_ground)

viz_vidtools.draw_reanalysis_activity_summary(new_outroot,prev_adir,outformat='pdf')
