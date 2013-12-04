#!/usr/bin/env python
import vidtools

# Not necessary; done in vidtools
#import matplotlib
#matplotlib.use('Agg')

import os,sys,re,numpy
import time, datetime, itertools, tarfile
from glob import glob
from collections import defaultdict

from PIL import ImageFilter,Image
import cv
import vidtools,Util,iplot
import pylab


def get_win(i,start,stop,wins):
    return i/((stop-start)/wins)

def init_objects(stream,frames,currsum,denom,seglen,cutoff,frames_offset,SHAPE,size_h,size_bins,fol_h,fol_bins,transform='',outline_engine='homebrew'):
    hsl = seglen/2
    min_arc_score = (2*max(size_h))+max(fol_h) #score of a "perfect" object arc of length 2
    ols = []
    last_frames = []
    ols_offset = frames_offset + hsl
    c = itertools.cycle(['|','/','-','\\'])
    t = time.time()
    print >> sys.stderr, 'analyze %s frames:' % seglen
    for i in xrange(seglen):
        last_frames.append(frames[0])
	frames_offset += 1
	mm = vidtools.shift_frames_return_diff(stream,frames,currsum,denom,seglen,transform=transform)
        if outline_engine == 'homebrew':
            ol = vidtools.chain_outlines_from_mask(mm>cutoff,preshrink=1,debug=False,return_termini=False,order_points=True,sort_outlines=False) #order_points should be True
        elif outline_engine == 'shapely':
            ol = vidtools.chain_outlines_from_mask_shapely(mm>cutoff,preshrink=1)
        else:
            print >> sys.stderr, 'outline_engine must be one of %s' % (['homebrew','shapely'])
            raise ValueError
	ols.append(ol)
	print >> sys.stderr, '\r%s %s ' % (i,c.next()),

    print >> sys.stderr, 'done in', str(datetime.timedelta(seconds=int(time.time() - t)))
    objs = {}
    splits = defaultdict(list)
    objs_sizes = {}
    objs_fols = {}
    
    print >> sys.stderr, '\nperform initial object arc tracking on %s frames' % len(ols)
    #one round of object tracking; assumes next object tracking call will start a single frame from now
    to_retire_objs, to_retire_objs_sizes, to_retire_objs_fols = vidtools.find_objs_progressive(ols, ols_offset, ols_offset, ols_offset+1, SHAPE, objs, splits, \
                                                                                               objs_sizes, objs_fols, size_h,size_bins,fol_h,fol_bins)

    prelast_avg = vidtools.average_frames(last_frames[hsl:])
    prelast_mm = vidtools.mousemask_from_object_arcs(frames_offset-hsl,frames_offset,min_arc_score,ols,ols_offset, \
                                                     Util.merge_dictlist([objs,to_retire_objs]), \
                                                     Util.merge_dictlist([objs_sizes,to_retire_objs_sizes]), \
                                                     Util.merge_dictlist([objs_fols,to_retire_objs_fols]), \
                                                     size_h, size_bins, fol_h, fol_bins,SHAPE) 

    print >> sys.stderr, 'object initialization complete in', str(datetime.timedelta(seconds=int(time.time() - t)))
    return ols,ols_offset,frames_offset,objs,splits,objs_sizes,objs_fols,prelast_avg,prelast_mm,to_retire_objs,to_retire_objs_sizes,to_retire_objs_fols

def advance_analysis(ols,ols_offset,objs,splits,objs_sizes,objs_fols,to_retire_objs,to_retire_objs_sizes,to_retire_objs_fols,last_frames,stream,frames,currsum,denom,seglen,cutoff,frames_offset,SHAPE,size_h,size_bins,fol_h,fol_bins,transform='',outline_engine='homebrew'):
    '''
    given current state objects, proceeds a single frame with all steps
    (advances frames, finds movement, extends object arcs)
    '''

    #shift frames
    frames_offset += 1
    last_frames.append(frames[0])
    mm = vidtools.shift_frames_return_diff(stream,frames,currsum,denom,seglen,transform=transform)

    #find blobs
    if outline_engine == 'homebrew':
        ol = vidtools.chain_outlines_from_mask(mm>cutoff,preshrink=1,debug=False,return_termini=False,order_points=True,sort_outlines=False)
    elif outline_engine == 'shapely':
        ol = vidtools.chain_outlines_from_mask_shapely(mm>cutoff,preshrink=1)
    else:
        print >> sys.stderr, 'outline_engine must be one of %s' % (['homebrew','shapely'])
        raise ValueError

    
    #shift blob outlines
    ols_offset += 1
    ols.pop(0)
    ols.append(ol)
    
    #advance object_arc tracking
    start_frame = ols_offset + len(ols) - 1
    ntr_objs, ntr_objs_sizes, ntr_objs_fols = vidtools.find_objs_progressive(ols, ols_offset, start_frame, ols_offset+1, SHAPE, objs, splits, \
                                                                             objs_sizes, objs_fols, size_h,size_bins,fol_h,fol_bins)
    to_retire_objs.update(ntr_objs)
    to_retire_objs_sizes.update(ntr_objs_sizes)
    to_retire_objs_fols.update(ntr_objs_fols)
    
    return ols_offset,frames_offset

def new_ground(g1,frame,hill_bounds,ground_change_max = 10, improvement = 0.01, window = 10):
    accept = 1+improvement
    g = []
    for col,gv in enumerate(g1):
        # go go gadget maybe-less-crappy...
        gpos = sorted([ ( numpy.mean(frame.transpose()[col-int(window/2):col+int(window/2),i:]) / \
                          numpy.mean(frame.transpose()[col-int(window/2):col+int(window/2),:i]) , i ) \
                        for i in range(int(gv-ground_change_max),int(gv+1))],reverse=True)
        gthis = numpy.mean(frame.transpose()[col-int(window/2):col+int(window/2),gv:])/numpy.mean(frame.transpose()[col-int(window/2):col+int(window/2),:gv])
        if gpos[0][0]/gthis > accept and hill_bounds[0] <= col <= hill_bounds[1]:
            g.append(gpos[0][1])
        else:
            g.append(gv)
    return g

def adjust_ground_by_mousemask(ground,mousemask,ground_change_max=10):
    adj_ground = []
    gmask = vidtools.mask_from_vector(ground,mousemask.shape)
    pts = vidtools.points_from_mask(mousemask&gmask)
    for i,last_gpt in enumerate(ground):
        above = sorted([y for x,y in pts if x==i and y <= last_gpt])
        if len(above) > 0 and last_gpt-ground_change_max <= above[-1] < last_gpt:
            adj_ground.append(above[-1])
        else:
            adj_ground.append(last_gpt)
    return adj_ground

def retire_objs(retire_before,to_retire_objs,to_retire_objs_sizes,to_retire_objs_fols, \
		retired_objs,retired_objs_sizes,retired_objs_fols):
    retire_keys = [k for k in to_retire_objs if max([fr for fr,ob in to_retire_objs[k]]) < retire_before]
    for k in retire_keys:
        retired_objs[k] = to_retire_objs.pop(k)
        retired_objs_sizes[k] = to_retire_objs_sizes.pop(k)
        retired_objs_fols[k] = to_retire_objs_fols.pop(k)


if __name__ == "__main__":
    
    import argparse
    
    ds =  ' [%(default)s]'
    #command parser
    parser = argparse.ArgumentParser(description='performs segmentation on the requested video interval at supplied thresholds; optionally emit frame averages for intervals (i.e. for ground finding) and burrow activity candidates (non-mouse feature change below ground)')
    
    parser.add_argument('-s','--start',default=0,type=int,help='start of target analysis segment in seconds'+ds)
    parser.add_argument('-e','--stop',default=None,type=int,help='end of target analysis segment in seconds'+ds)
    parser.add_argument('-f','--fps',default=None,type=float,help='force framerate in frames-per-second'+ds)

    parser.add_argument('-mt','--mouse_threshold',default=None,type=float,help='intensity difference threshold for mouse tracking'+ds)

    parser.add_argument('-nf','--nframes',default=300,type=int,help='number of frames to analyze in each segment during parameter fitting (only if -mt not set)'+ds)
    parser.add_argument('-np','--nparts',default=100,type=int,help='number of segments to analyze during parameter fitting (only if -mt not set)'+ds)
    parser.add_argument('-ns','--nstep',default=1,type=int,help='number of segments to space parameter fit parts by during parameter fitting (only if -mt not set)'+ds)
    parser.add_argument('-q','--queue',default='normal_serial',type=str,help='LSF queue for parameter fitting (only if -mt not set)'+ds)
    parser.add_argument('-tc','--threshold_coeff',default=1.0,type=float,help='multiply best fit mouse threshold by threshold_coeff (e.g. set to 1.2 to increase stringency of mouse tracking by 20%)'+ds)
    
    parser.add_argument('-l','--seglen',default=900,type=int,help='number of frames to average for object tracking background subtraction; see vidtools.init_frames()'+ds)
    
    parser.add_argument('-ac','--antfarm_config',default=None,type=str,help='config file for antfarm params (ground_anchors, hill_bounds, burrow_entrance)'+ds)
    parser.add_argument('-gi','--ground_improvement',default=0.05,type=float,help='percent improvement a new ground must make for acceptance (only relevant if antfarm_config used)'+ds)
    parser.add_argument('-gs','--ground_suppress',default=20.0,type=float,help='number of pixels below ground to extend "above ground" mask (burrows only scored below this mask; only relevant if antfarm_config used)'+ds)
    parser.add_argument('-fm','--former_mm',default=3,type=int,help='number of pre-last mosuemasks to apply in masking for ground segmentation (burrows only scored below this mask; only relevant if antfarm_config used)'+ds)

    parser.add_argument('-zc','--zone_config',default=None,type=str,help='config file for tracked zones (named locations in frame)'+ds)
    
    parser.add_argument('-tr','--transform',default='',type=str,choices=['','invert','absval'],help='transform frame intensities (tracks dark object on light background)'+ds)

    parser.add_argument('-oe','--outline_engine',default='homebrew',type=str,choices=['homebrew','shapely'],help='switch between homebrew and shapely chain_outlines calls (shapely should be faster and more accurate, but requires working shapely python library and libgeos_c)'+ds)
    parser.add_argument('--max_itertime',default=1800,type=int,help='maximum average interation time in seconds to permit after 5 interations (if average segment loop time exceeds this value, terminate)'+ds)
    
    parser.add_argument('-vs','--video_suffix',default=None,type=str,help='write summary video with this suffix if supplied'+ds)
    
    parser.add_argument('vid',help='video file to process')
    
    opts = parser.parse_args()


    if opts.antfarm_config:
	config = eval(open(opts.antfarm_config).read())
	    
    hsl = opts.seglen/2
    
    #load openCV stream
    stream = cv.CaptureFromFile(opts.vid)
    #get video framerate
    if opts.fps is None:
        fps = float(cv.GetCaptureProperty(stream,cv.CV_CAP_PROP_FPS))
    else:
    	fps = opts.fps
    
    min_start_sec = int(opts.seglen/fps)
    if opts.start < min_start_sec:
        print >> sys.stderr, '--start (%s) target places analysis before 0; earliest available start is %s' % (opts.start,min_start_sec)
	raise ValueError
    else:
	target_frame_start = int(opts.start * fps)
		
    #if specified stop is "None" or greater than video length,
    # set to video length
    dur = vidtools.vid_duration(opts.vid)
    if opts.stop is None or opts.stop > dur:
	print >> sys.stderr, '--stop (%s) is None or exceeds video length (%s); set to length' % (opts.stop, dur-opts.seglen)
	opts.stop = dur-opts.seglen

    target_frame_stop = int(opts.stop * fps)


    # find optimal cutoff; pull scoring distributions
    if opts.mouse_threshold is None:
        cut_step = 0.001
        totframes = target_frame_stop - target_frame_start
        trainframes = opts.nparts * opts.seglen * opts.nstep
        if trainframes > totframes:
            errstr = 'training duration exceeds video duration (train: %s frames, total: %s frames, max nparts at nstep = %s: %s)' % (trainframes,totframes,opts.nstep,totframes/(opts.seglen * opts.nstep))
            raise ValueError, errstr
        scores,dists = vidtools.run_mousezopt(opts.vid,opts.seglen,opts.nframes,opts.nstep,opts.nparts,cut_step,transform=opts.transform,pass1queue=opts.queue,start_offset=target_frame_start)
        cutoff_rank,cutoff = vidtools.choose_cutoff(scores,cut_step) #or cut_step*2
        size_h,size_bins,fol_h,fol_bins = dists[cutoff]
        min_arc_score = (2*max(size_h))+max(fol_h)
        if opts.threshold_coeff == 1:
            thresh_str = 'auto-thresh-%0.3f' % cutoff
            print >> sys.stderr, 'mouse intensity threshold %0.3f chosen' % cutoff
        else:
            orig_cutoff = cutoff
            cutoff *= opts.threshold_coeff
            thresh_str = 'coeff-%0.3f-thresh-%0.3f' % (opts.threshold_coeff,cutoff)
            print >> sys.stderr, 'mouse intensity threshold %0.3f chosen after %s threshold coefficient (original: %s)' % (cutoff,opts.threshold_coeff,orig_cutoff)
    else:
        cutoff = opts.mouse_threshold
        thresh_str = 'man-thresh-%0.3f' % cutoff
        size_h = [1]
        size_bins = [0,numpy.inf]
        fol_h = [1]
        fol_bins = [0,numpy.inf]
        min_arc_score = 0

    #actual analysis window must be one full <seglen> longer in both directions than the desired target window
    frame_start = target_frame_start - opts.seglen
    frame_stop = target_frame_stop + opts.seglen
    
    analyze_hsls = int((frame_stop-target_frame_start)/hsl) # i THINK you go init (=1 seg); analyze until target+1seg

    print >> sys.stderr, 'target_frame_start: %s\nframe_start: %s\ntarget_frame_stop: %s\nframe_stop: %s\nhsl: %s\nanalyze_hsls: %s ' % (target_frame_start,frame_start,target_frame_stop,frame_stop,hsl,analyze_hsls)

    vidtools.seek_in_stream(stream,frame_start)
    frames_offset = frame_start

    #init frames
    print >> sys.stderr, 'initialize frames ...',
    frames,currsum,denom = vidtools.init_frames(stream,opts.seglen)
    print >> sys.stderr, 'done'

    SHAPE = frames[0].shape

    #SKIP IF NOT WRITING VIDEO
    if opts.video_suffix:
        pixdim = tuple(reversed(SHAPE)) #flip for numpy -> opencv
        pixdim = tuple(numpy.array(pixdim)*2) #double video size

    # init empty receivers
    ols = []
    objs = {}
    objs_sizes = {}
    objs_fols = {}
    to_retire_objs = {}
    to_retire_objs_sizes = {}
    to_retire_objs_fols = {}
    retired_objs = {}
    retired_objs_sizes = {}
    retired_objs_fols = {}

    #init object arcs
    ols,ols_offset,frames_offset,objs,splits,objs_sizes,objs_fols,prelast_avg,prelast_mm,to_retire_objs,to_retire_objs_sizes,to_retire_objs_fols = init_objects(stream,frames,currsum,denom,opts.seglen,cutoff,frames_offset,SHAPE,size_h,size_bins,fol_h,fol_bins,transform=opts.transform,outline_engine=opts.outline_engine)

    if opts.antfarm_config:
        prelast_masked = prelast_avg.copy()
        prelast_masked[prelast_mm] = numpy.mean(prelast_avg[:50,:50])

        print >> sys.stderr, 'find initial ground ...',
        g1 = vidtools.find_ground4(prelast_masked,config['ground_anchors'],be=config['burrow_entrance'])
        print >> sys.stderr, 'done'
        
        groundmask = vidtools.mask_from_vector(numpy.array(g1)+opts.ground_suppress,SHAPE)
        
        grounds = [new_ground(g1,prelast_masked,config['hill_bounds'],improvement=opts.ground_improvement,window=10)]
        digdiffs = []
        newactols = []
        prevactmask = numpy.zeros(SHAPE,dtype=bool)
        
    mousemasks = []
    segavgs = []
    segmasked = []

    i=0

    param_str = '%s-%s_seg%s_%s-%s' % (target_frame_start,target_frame_stop,opts.seglen,thresh_str,opts.video_suffix)
    if opts.video_suffix:
        vidout = opts.vid[:-4]+'_%s.avi' % (param_str)
        print >> sys.stderr, 'write output video to %s' % vidout
        try:
            os.unlink(vidout)
        except:
            pass
        pylab.gray()
        vidwriter = cv.CreateVideoWriter(vidout , cv.FOURCC('x','v','i','d'), fps, pixdim,1)


    analysis_root = os.path.join(opts.vid[:-4],'analysis',param_str)
    print >> sys.stderr, 'write analysis results to %s' % analysis_root
    try:
        os.makedirs(analysis_root)
    except:
        pass

    #prepare output tarfile names
    tarfiles = dict([(fname,os.path.join(analysis_root,fname+'.tar')) \
                     for fname in ['miceols','objs','objs_fols','objs_sizes','prevact_ols','newact_ols','grounds','mousemasks','digdiffs','segavgs']])

    open(os.path.join(analysis_root,'SHAPE'),'w').write(SHAPE.__repr__())
    times = []
    
    while i <= analyze_hsls:
        last_miceols_file = '%07d-%07d-mice_ols.list' % (frames_offset,frames_offset+hsl)
        retired_objs_file = '%07d-%07d-retired_objs.dict' % (frames_offset,frames_offset+hsl)
        retired_objs_fols_file = '%07d-%07d-retired_objs_fols.dict' % (frames_offset,frames_offset+hsl)
        retired_objs_sizes_file = '%07d-%07d-retired_objs_sizes.dict' % (frames_offset,frames_offset+hsl)
        last_mousemask_file = '%07d-%07d-mousemask.mat' % (frames_offset,frames_offset+hsl)
        last_segavg_file = '%07d-%07d-segavg.mat' % (frames_offset,frames_offset+hsl)
        if opts.antfarm_config:
            last_prevactols_file = '%07d-%07d-prevact_ols.list' % (frames_offset,frames_offset+hsl)
            last_newactols_file = '%07d-%07d-newact_ols.list' % (frames_offset,frames_offset+hsl)
            last_ground_file = '%07d-%07d-ground.list' % (frames_offset,frames_offset+hsl)
            last_digdiff_file = '%07d-%07d-digdiff.mat' % (frames_offset,frames_offset+hsl)

        t = time.time()
        #eventually, choice to reload previous analysis goes here
        last_frames = []
        while len(last_frames) < hsl:
            ols_offset, frames_offset = advance_analysis(ols,ols_offset,objs,splits,objs_sizes,objs_fols, \
                                                         to_retire_objs,to_retire_objs_sizes,to_retire_objs_fols, \
                                                         last_frames,stream,frames,currsum,denom,opts.seglen,cutoff,frames_offset, \
                                                         SHAPE,size_h,size_bins,fol_h,fol_bins,transform=opts.transform,outline_engine=opts.outline_engine)
        last_avg = vidtools.average_frames(last_frames)
        last_mm = vidtools.mousemask_from_object_arcs(frames_offset-len(last_frames),frames_offset,min_arc_score,ols,ols_offset, \
                                                        Util.merge_dictlist([objs,to_retire_objs]), \
                                                        Util.merge_dictlist([objs_sizes,to_retire_objs_sizes]), \
                                                        Util.merge_dictlist([objs_fols,to_retire_objs_fols]), \
                                                        size_h, size_bins, fol_h, fol_bins,SHAPE)
        this_avg = vidtools.average_frames(frames[:hsl])
        this_mm = vidtools.mousemask_from_object_arcs(frames_offset,frames_offset+hsl,min_arc_score,ols,ols_offset, \
                                                        Util.merge_dictlist([objs,to_retire_objs]), \
                                                        Util.merge_dictlist([objs_sizes,to_retire_objs_sizes]), \
                                                        Util.merge_dictlist([objs_fols,to_retire_objs_fols]), \
                                                        size_h, size_bins, fol_h, fol_bins,SHAPE)
        prelast_masked = prelast_avg.copy()
        prelast_masked[prelast_mm] = numpy.mean(prelast_avg[:50,:50])
        last_masked = last_avg.copy()
        last_masked[last_mm] = numpy.mean(last_avg[:50,:50])
        this_masked = this_avg.copy()
        this_masked[this_mm] = numpy.mean(this_avg[:50,:50])

        if opts.antfarm_config: #antfarm-specific analysis steps
            digdiffs.append(prelast_masked-this_masked)
            if len(digdiffs) > 3:
                nll = digdiffs.pop(0)
            if newactols and newactols[-1]:
                prevactmask += reduce(lambda x,y:x+y, [vidtools.shrink_mask(vidtools.mask_from_outline(ol,this_masked.shape),1) for ol in newactols[-1]])
            if len(digdiffs) > 1:
                m = digdiffs[-2] > 0.2 #ARBITRARY
                m[last_mm] = True #add mousemask to burrow area
                m[vidtools.grow_mask(vidtools.shrink_mask(prevactmask,1),1)] = False #MASK PREVIOUS DIGGING
                m[groundmask] = False #AND EVERYTHING ABOVE FIRST GROUND
                if opts.outline_engine == 'homebrew':
                    newactols.append(vidtools.chain_outlines_from_mask(m,preshrink=1,grow_by=1,debug=False,return_termini=False,order_points=True,sort_outlines=False))
                    prevol = vidtools.chain_outlines_from_mask(prevactmask,preshrink=1,grow_by=1,debug=False,return_termini=False,order_points=True,sort_outlines=False)
                elif opts.outline_engine == 'shapely':
                    newactols.append(vidtools.chain_outlines_from_mask_shapely(m,preshrink=1,grow_by=1))
                    prevol = vidtools.chain_outlines_from_mask_shapely(prevactmask,preshrink=1,grow_by=1)
                else:
                    print >> sys.stderr, 'outline_engine must be one of %s' % (['homebrew','shapely'])
                    raise ValueError
                digol = newactols[-1]
            else:
                digol = []
                prevol = []

            groundtrack_masked = last_masked.copy()
            for former_mm in mousemasks[-opts.former_mm:]:
                groundtrack_masked[former_mm] = numpy.mean(last_avg[:50,:50])
            last_ground = new_ground(grounds[-1],groundtrack_masked,config['hill_bounds'],improvement=opts.ground_improvement,window=10) #ARBITRARY
            if len(digdiffs) > 1:
                ddmat = digdiffs[-2] #NOTE -2 INDEX; aligns with this activity segment
            else:
                ddmat = numpy.zeros(SHAPE,dtype='float')

        #write current lasts
        Util.append_obj2tar(ols[:hsl], last_miceols_file, tarfiles['miceols'])
        Util.append_ar2tar(last_mm, last_mousemask_file, tarfiles['mousemasks'])
        Util.append_ar2tar(last_avg, last_segavg_file, tarfiles['segavgs'])
        #Util.append_obj2tar(to_retire_objs, retired_objs_file, tarfiles['objs'])
        #Util.append_obj2tar(to_retire_objs_fols, retired_objs_fols_file, tarfiles['objs_fols'])
        #Util.append_obj2tar(to_retire_objs_sizes, retired_objs_sizes_file, tarfiles['objs_sizes'])
        if opts.antfarm_config:
            Util.append_obj2tar(prevol, last_prevactols_file, tarfiles['prevact_ols'])
            Util.append_obj2tar(digol, last_newactols_file, tarfiles['newact_ols'])
            Util.append_obj2tar(last_ground, last_ground_file, tarfiles['grounds'])
            Util.append_ar2tar(ddmat, last_digdiff_file, tarfiles['digdiffs'])
        #done writing files
        #handle shifting stored current data
        mousemasks.append(last_mm)
        if len(mousemasks) > 3:
            nll = mousemasks.pop(0)
        segavgs.append(last_avg)
        if len(segavgs) > 3:
            nll = segavgs.pop(0)
        segmasked.append(last_masked)
        if len(segmasked) > 3:
            nll = segmasked.pop(0)
        prelast_avg = last_avg
        prelast_mm = prelast_mm + last_mm
        if opts.antfarm_config:
            grounds.append(last_ground)
        #done handling data
        #DRAW FRAME RESULTS
        if opts.video_suffix:
            if opts.antfarm_config:
                this_color = iplot.subspectrum(analyze_hsls+1)[i]
                if i > 0:
                    g_col_li = [('k',zip(*list(enumerate(Util.smooth(grounds[0],10)))))]
                else:
                    g_col_li = []
                g_col_li.append((this_color,zip(*list(enumerate(Util.smooth(grounds[-1],10))))))

            last_frames_objs_ols = vidtools.ols_in_interval(frames_offset-hsl-1,frames_offset,min_arc_score,ols,ols_offset, \
                                                            Util.merge_dictlist([objs,to_retire_objs]), \
                                                            Util.merge_dictlist([objs_sizes,to_retire_objs_sizes]), \
                                                            Util.merge_dictlist([objs_fols,to_retire_objs_fols]), \
                                                            size_h, size_bins, fol_h, fol_bins,True)
            
            for vfi,m in enumerate(last_frames):
                if opts.antfarm_config:
                    cv_im = vidtools.mat_polys2cv(m,zip(iplot.subspectrum(6)[1:],list(reversed(last_frames_objs_ols[vfi][:5])))+zip(['k']*len(prevol),prevol),zip([this_color]*len(digol),['\\']*len(digol),digol),g_col_li,80,2)
                else:
                    cv_im = vidtools.mat_polys2cv(m,zip(iplot.subspectrum(6)[1:],list(reversed(last_frames_objs_ols[vfi][:5]))),[],[],80,2)
                nll = cv.WriteFrame(vidwriter,cv_im)
                print >> sys.stderr, '\r\twrite video frame %s' % vfi,

        retire_objs(ols_offset-hsl,to_retire_objs,to_retire_objs_sizes,to_retire_objs_fols,retired_objs,retired_objs_sizes,retired_objs_fols)
        i+=1
        this_time = int(time.time() - t)
        times.append(this_time)
        avg_time = int(numpy.mean(times))
        print >> sys.stderr, i, '/', analyze_hsls, 'done in',str(datetime.timedelta(seconds=this_time)),'avg loop time',str(datetime.timedelta(seconds=avg_time)),'done in',str(datetime.timedelta(seconds=avg_time*(analyze_hsls-i)))
        if i > 5 and avg_time > opts.max_itertime:
            errstr = 'mean iteration time %s after %s rounds exceeds max %s' % (avg_time,i+1,opts.max_itertime)
            raise ValueError, errstr


    last_miceols_file = '%07d-%07d-mice_ols.list' % (frames_offset,frames_offset+hsl)
    retired_objs_file = '%07d-%07d-retired_objs.dict' % (frames_offset,frames_offset+hsl)
    retired_objs_fols_file = '%07d-%07d-retired_objs_fols.dict' % (frames_offset,frames_offset+hsl)
    retired_objs_sizes_file = '%07d-%07d-retired_objs_sizes.dict' % (frames_offset,frames_offset+hsl)
    last_mousemask_file = '%07d-%07d-mousemask.mat' % (frames_offset,frames_offset+hsl)
    last_segavg_file = '%07d-%07d-segavg.mat' % (frames_offset,frames_offset+hsl)
    if opts.antfarm_config:
        last_prevactols_file = '%07d-%07d-prevact_ols.list' % (frames_offset,frames_offset+hsl)
        last_newactols_file = '%07d-%07d-newact_ols.list' % (frames_offset,frames_offset+hsl)
        last_ground_file = '%07d-%07d-ground.list' % (frames_offset,frames_offset+hsl)
        last_digdiff_file = '%07d-%07d-digdiff.mat' % (frames_offset,frames_offset+hsl)

    #write current lasts
    Util.append_obj2tar(ols[:hsl], last_miceols_file, tarfiles['miceols'])
    Util.append_ar2tar(last_mm, last_mousemask_file, tarfiles['mousemasks'])
    Util.append_ar2tar(last_avg, last_segavg_file, tarfiles['segavgs'])
    #Util.append_obj2tar(to_retire_objs, retired_objs_file, tarfiles['objs'])
    #Util.append_obj2tar(to_retire_objs_fols, retired_objs_fols_file, tarfiles['objs_fols'])
    #Util.append_obj2tar(to_retire_objs_sizes, retired_objs_sizes_file, tarfiles['objs_sizes'])
    if opts.antfarm_config:
        Util.append_obj2tar(prevol, last_prevactols_file, tarfiles['prevact_ols'])
        Util.append_obj2tar(digol, last_newactols_file, tarfiles['newact_ols'])
        Util.append_obj2tar(last_ground, last_ground_file, tarfiles['grounds'])
        Util.append_ar2tar(ddmat, last_digdiff_file, tarfiles['digdiffs'])

    open(os.path.join(analysis_root,'objs.dict'),'w').write(dict(retired_objs.items()+to_retire_objs.items()+objs.items()).__repr__())
    open(os.path.join(analysis_root,'objs_sizes.dict'),'w').write(dict(retired_objs_sizes.items()+to_retire_objs_sizes.items()+objs_sizes.items()).__repr__())
    open(os.path.join(analysis_root,'objs_fols.dict'),'w').write(dict(retired_objs_fols.items()+to_retire_objs_fols.items()+objs_fols.items()).__repr__())
    
    print >> sys.stderr, 'DONE'
