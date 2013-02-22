#!/usr/bin/env python

import os, sys, Util, numpy
from video_analysis import vidtools
import cv
from collections import defaultdict

def generate_outfile_names(vid,seglen,offset,nframes,maxint,minint,stepint):
    vidbase,idx,analysis_win = os.path.splitext(vid)[0].rsplit('_',2)
    mousezopt_out = os.path.join(os.path.dirname(vidbase),idx,analysis_win,'mousezopt')

    if None in [maxint,minint,stepint]:
        scoresfile = os.path.join(mousezopt_out,'seg%s_nframes%s_off%s_pass1-scores.dict' \
                              % (seglen,nframes,offset))
        olsfile = os.path.join(mousezopt_out,'seg%s_nframes%s_off%s_pass1-ols.dict' \
                              % (seglen,nframes,offset))
    else:
        scoresfile = os.path.join(mousezopt_out,'seg%s_nframes%s_off%s_max%0.3f_min%0.3f_step%0.3f-scores.dict' \
                              % (seglen,nframes,offset,maxint,minint,stepint))
        olsfile = os.path.join(mousezopt_out,'seg%s_nframes%s_off%s_max%0.3f_min%0.3f_step%0.3f-ols.dict' \
                              % (seglen,nframes,offset,maxint,minint,stepint))
    
    return scoresfile,olsfile

def score_segment_by_cutoff(mms,maxint,minint,stepint,maxblobs = 20,size_binw=50,fol_binw=0.05):
    
    import time
    ols_by_cut = {}
    scores = {}
    times = []

    for cutoff in numpy.arange(maxint,minint,-stepint):
        print >> sys.stderr, '\nCUTOFF',cutoff
        ols_this = []
        t1 = time.time()
        stop_seg = False
        for mm in mms:
            ols_this.append(vidtools.chain_outlines_from_mask(mm>cutoff, \
                                                              preshrink=1,debug=False,return_termini=False, \
                                                              order_points=False,sort_outlines=False))
            if len(ols_this[-1]) > maxblobs:
                print >> sys.stderr, 'segmentation result (%s) exceeds maximum number of blobs per frame (%s); break' \
                      % (max([len(ol) for ol in ols_this]),maxblobs)
                stop_seg = True
                break
        
        t2 = time.time()
        print >> sys.stderr, 'find blobs %s sec' % (t2-t1)

        if stop_seg:
            scores[cutoff] = 0
            break

        ols_by_cut[cutoff] = ols_this

        if fol_binw is None or size_binw is None:
            scores[cutoff] = 0
            times.append(t2-t1)
            continue
        
        nblobs = len(reduce(lambda x,y:x+y,[[]]+[[vidtools.size_of_polygon(p) for p in ol if len(p) > 0] for ol in ols_this if len(ol) > 0]))
        if nblobs == 0:
            print >> sys.stderr, 'no blobs; skip'
            scores[cutoff] = 0
            continue
        
        t3 = time.time()
        objs_this,splits_this,joins_this =  vidtools.find_objs(ols_by_cut[cutoff],mm.shape)
        t4 = time.time()
        print >> sys.stderr, 'track objects %s sec' % (t4-t3)
        t5 = time.time()
        size_h,size_bins,fol_h,fol_bins = vidtools.get_object_arc_param_dists(ols_by_cut[cutoff],mm.shape,n_obj,size_binw=size_binw,fol_binw=fol_binw)
        t6 = time.time()
        if size_h is None:
            print >> sys.stderr, 'no object arcs; skip'
            scores[cutoff] = 0
            continue
        
        print >> sys.stderr, 'calculate scoring distributions %s sec' % (t6-t5)
        keep,drop = vidtools.greedy_objs_filter(objs_this,ols_by_cut[cutoff],size_h,size_bins,fol_h,fol_bins,mm.shape)
        sscore = sum([vidtools.score_object_arc_size(o,ols_by_cut[cutoff],size_h,size_bins) for o in keep])
        fscore = sum([vidtools.score_object_arc_fol(o,ols_by_cut[cutoff],fol_h,fol_bins,mm.shape) for o in keep])
        t7 = time.time()
        print >> sys.stderr, cutoff, nblobs, sscore, fscore, sscore+fscore
        scores[cutoff] = sscore+fscore
        times.append(t7-t1)

    print >> sys.stderr, "total time %s minutes" % (sum(times)/60.0)

    return scores,ols_by_cut

if __name__ == '__main__':


    #two run modes:
    # if vid,seglen,offset,nframes supplied at run:
    #  run first-pass, calculate cutoffs, local scores and store
    # if vid,seglen,offset,nframes,maxint,minint,stepint supplied at run:
    #  run second-pass, use supplied cutoffs to write blob outlines dict, do not score
    if len(sys.argv) == 7:
        vid,seglen,offset,nframes,n_obj,transform = sys.argv[1:]
        maxint = None
        minint = 0.0
        stepint = None
        size_binw = 100
        fol_binw = 0.1
    else:
        vid,seglen,offset,nframes,n_obj,transform,maxint,minint,stepint = sys.argv[1:]
        maxint = float(maxint)
        minint = float(minint)
        stepint = float(stepint)
        size_binw = None
        fol_binw = None

    offset = int(offset)
    nframes = int(nframes)
    seglen = int(seglen)
    n_obj = int(n_obj)

    scoresfile,olsfile = generate_outfile_names(vid,seglen,offset,nframes,maxint,minint,stepint)
    if transform:
        print >> sys.stderr, 'invoke image transformation: %s' % transform

    #outline_z = 4
    #max_z = 24
    #min_z = 3
    #step_z = 0.5


    #maxint = 0.05
    #minint = 0.000
    #stepint = 0.001
    nsteps = 50

    stream = cv.CaptureFromFile(vid)

    if offset:
        print >> sys.stderr, 'seek %s frames ...' % offset,
        vidtools.seek_in_stream(stream,offset)
        print >> sys.stderr, 'done'

    print >> sys.stderr, 'initialize %s frames ...' % seglen,
    frames,currsum,denom = vidtools.init_frames(stream,seglen)
    print >> sys.stderr, 'done'

    match_int = []
    best_int = []
    best_sizes = defaultdict(list)
    #nticks = 20
    #if nframes < nticks:
    #    tickon = 1
    #else:
    #    tickon = nframes/nticks


    print >> sys.stderr, 'load %s frames ...' % nframes,
    mms = []
    for fritr in xrange(nframes):
        #mm = Util.zscore(vidtools.shift_frames_return_diff(stream,frames,currsum,denom,seglen))
        #ol,term = vidtools.chain_outlines_from_mask(mm>outline_z,grow_by=0,preshrink=1)
        #olmax = numpy.array([mm[vidtools.mask_from_outline(p,mm.shape)].max() for p in ol])

        #get movement matrices
        mm = vidtools.shift_frames_return_diff(stream,frames,currsum,denom,seglen,transform=transform)
        mms.append(mm)
    print >> sys.stderr, 'done'
    
    if maxint is None:
        maxint = max([mm.max() for mm in mms])
        print >> sys.stderr, 'maximum observed intensity:',maxint
    else:
        maxint = float(maxint)
        print >> sys.stderr, 'maximum analysis intesity set to:',maxint

    if stepint is None:
        stepint = ((maxint-minint)/float(nsteps))
    else:
        stepint = float(stepint)

    scores, ols_by_cut = score_segment_by_cutoff(mms,maxint,minint,stepint,size_binw=size_binw,fol_binw=fol_binw)

    open(olsfile,'w').write(ols_by_cut.__repr__())
    open(scoresfile,'w').write(scores.__repr__())

    print >> sys.stderr, 'DONE'
    

    #old parameter optimization follows
    '''
    for mm in mms:
        best = (None,0)
        for i in numpy.arange(maxint,minint,-1*((maxint-minint)/20.0)):
            ol = vidtools.chain_outlines_from_mask(mm>i,preshrink=1,debug=False,return_termini=False,order_points=False,sort_outlines=False)
            #ol,term = vidtools.chain_outlines_from_mask(mm>i,grow_by=0,preshrink=1)
            if len(ol) > 3: break
            if len(ol) == 1:
                s = vidtools.size_of_polygon(ol[0])
                if s > best[1]:
                    best = (round(i,3),s)
            #print >> sys.stderr, i, len(ol)
        maxint = best[0]
        print >> sys.stderr, fritr,round((fritr/float(nframes))*100),'%','PASS 1',best
        if best[0] is None:
            continue
        #print >> sys.stderr, maxint, minint,stepint,numpy.arange(maxint,minint,-stepint)
        for i in numpy.arange(maxint,minint,-stepint):
            #print >> sys.stderr, i
            ol = vidtools.chain_outlines_from_mask(mm>i,preshrink=1,debug=False,return_termini=False,order_points=False,sort_outlines=False)
            #ol,term = vidtools.chain_outlines_from_mask(mm>i,grow_by=0,preshrink=1)
            if len(ol) > 3: break
            if len(ol) == 1:
                s = vidtools.size_of_polygon(ol[0])
                match_int.append(round(i,3))
                if s > best[1]:
                    best = (round(i,3),s)
            #print >> sys.stderr, i, len(ol)
        print >> sys.stderr, fritr,round((fritr/float(nframes))*100),'%','PASS 2',best

        best_int.append(best[0])
        best_sizes[best[0]].append(best[1])

        #if fritr % tickon == 0: print >> sys.stderr, fritr, (fritr/float(nframes))*100

    best_int_d = Util.countdict(best_int)

    for k,v in sorted(Util.countdict(match_int).items()):
        if len(best_sizes[k]) == 0:
            meansize = 0
        else:
            meansize = numpy.mean(best_sizes[k])
        print '%s\t%s\t%s\t%s' % (k,v,best_int_d.get(k,0),meansize)
    '''

