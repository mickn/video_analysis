#!/usr/bin/env python

import os,sys,Util,cv,numpy
from video_analysis import vidtools,mousezopt

def generate_outfile_names(vid,seglen,step,nframes,nreps,stepint,cutoff):
    outroot = os.path.dirname(mousezopt.generate_outfile_names(vid,seglen,0,nframes,None,None,None)[1])
    scorefile = os.path.join(outroot,'seg%s_nframes%s_step%s_nreps%s_stepint%s_cutoff%s-score.dict' % (seglen,nframes,step,nreps,stepint,cutoff) )
    distsfile = os.path.join(outroot,'seg%s_nframes%s_step%s_nreps%s_stepint%s_cutoff%s-model_dists.dict' % (seglen,nframes,step,nreps,stepint,cutoff) )

    return scorefile,distsfile

if __name__ == "__main__":
    vid,seglen,step,nframes,nreps,start_offset,cutoff,peak_max,peak_min,target_int_step,n_obj = sys.argv[1:]

    seglen = int(seglen)
    step = int(step)
    nreps = int(nreps)
    start_offset = int(start_offset)
    cutoff = float(cutoff)
    peak_max = float(peak_max)
    peak_min = float(peak_min)
    target_int_step = float(target_int_step)
    n_obj = int(n_obj)

    #get a dummy frame for pixel dimensions
    stream = cv.CaptureFromFile(vid)
    SHAPE = vidtools.array_from_stream(stream).shape

    obj_train_arcs_d = {}
    obj_arcs_d = {}
    size_train_values_d = {}
    size_values_d = {}
    fol_train_values_d = {}
    fol_values_d = {}
    print >> sys.stderr, 'load second pass blob outlines; calculate object properties',
    for i in range(nreps):
        offset = (i*seglen*step)+start_offset
        source_f = mousezopt.generate_outfile_names(vid,seglen,offset,nframes,peak_max,peak_min,target_int_step)[1]
        ols_d = eval(open(source_f).read())
        ols = ([v for k,v in ols_d.items() if numpy.abs(cutoff-k) < target_int_step/2]+[[]])[0]
        objs_train = [o for o in vidtools.find_objs(ols,SHAPE,n_obj,n_obj)[0] if len(o) > 0]
        obj_train_arcs_d[offset] = objs_train
        size_train_values_d[offset] = [[vidtools.size_of_polygon(ols[i][j]) for i,j in o] for o in objs_train]
        fol_train_values_d[offset] = [vidtools.fol_from_obj_arc(o,ols,SHAPE) for o in objs_train]
        objs = [o for o in vidtools.find_objs(ols,SHAPE)[0] if len(o) > 0]
        obj_arcs_d[offset] = objs
        size_values_d[offset] = [[vidtools.size_of_polygon(ols[i][j]) for i,j in o] for o in objs]
        fol_values_d[offset] = [vidtools.fol_from_obj_arc(o,ols,SHAPE) for o in objs]
        print >> sys.stderr, '.',
    print >> sys.stderr, 'done'

    if len(reduce(lambda x,y:x+y,size_train_values_d.values())) > 0 and len(reduce(lambda x,y:x+y,fol_train_values_d.values())) > 0:
        print >> sys.stderr, 'calculate scoring distributions ...' ,
        size_h,size_bins,fol_h,fol_bins = vidtools.get_object_arc_param_dists_from_values_dict(size_train_values_d,fol_train_values_d,size_binw=20,fol_binw=0.01)
        print >> sys.stderr, 'done'
        print >> sys.stderr, 'score objects ...' ,
        #sscore = sum([vidtools.score_object_arc_size(o,ols_cat,size_h,size_bins) for o in keep])
        #fscore = sum([vidtools.score_object_arc_fol(o,ols_cat,fol_h,fol_bins,SHAPE) for o in keep])
        scores_all = []
        for offset in size_values_d.keys():
            keep,drop,keep_scores,drop_scores = vidtools.greedy_objs_filter_from_values(obj_arcs_d[offset],size_values_d[offset],fol_values_d[offset],size_h,size_bins,fol_h,fol_bins)
            scores_all.append(sum(keep_scores))
        print >> sys.stderr, 'done'
        print >> sys.stderr, 'cutoff %s score: %s' % (cutoff, sum(scores_all))
        pass2_scores = {cutoff:sum(scores_all)}
        pass2_scoring_dists = {cutoff:(list(size_h),list(size_bins),list(fol_h),list(fol_bins))}
    else:
        print >> sys.stderr, 'scoring dicts empty at cutoff %s' % cutoff
        pass2_scores = {cutoff:0}
        pass2_scoring_dists = {cutoff:(None,None,None,None)}




    scorefile,distsfile = generate_outfile_names(vid,seglen,step,nframes,nreps,target_int_step,cutoff)

    print >> sys.stderr, 'write files:\n%s\n%s' % (distsfile,scorefile) 
    open(distsfile,'w').write(pass2_scoring_dists.__repr__())
    open(scorefile,'w').write(pass2_scores.__repr__())




         
