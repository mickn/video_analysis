#!/usr/bin/env python
import time
start_sec = time.time()
'''run behavioral coding on antfarm video

takes as input:
now see optstruct below for format, briefly:
analyze_antfarm.py [options as per optstruct] imagedir

seglen is the length of time (in seconds) to consider for each submission to summarize_segment.py

wishlist:
split and synchronize antfarm video (front and back),

'''

optstruct = { 
    'seglen': ('s',int,30,'length of analysis segment in seconds'),
    'mask' : ('m',str,'None','filename of a dimension-matched binary mask file'),
    'burrow_mask' : ('k',str,None,'filename of dimension-matched binary burrow mask file [deprecated]'),
    'burrow_entrance_xy' : ('b',str,'"(360,240)"','x,y tuple of burrow entrance coordinates'),
    'xybounds' : ('x',str,'None','[(x,y),(x,y)] coords of top-left and bottom-right of valid analysis area'),
    'ground_anchors' : ('g',eval,None,'list of guide points for groundfinding'),
    'pixav' : ('p',int,3,'number of pixels to average intensities over'),
    'timeav' : ('t',int,10,'number of frames to average intensities at a given pixel over'),
    'mousez' : ('o',str,'12','z-score cutoff for calling mouse location'),
    'burrowz' : ('u',str,'12','z-score cutoff for calling burrowing activity'),
    'num_batches' : ('n',int,0,'number of batches to bundle LSF jobs into'),
    'vid_bitrate' : ('v',str,'6000k','bitrate passed to png2vid.py'),
    'vid_format' : ('f',str,'mp4','format for summary video.  if this is None, neither video nor output frames will be generated'),
    'summarize_queue' : ('q', str, 'short_serial','queue to execute summarize_segment on'),
    'hill_bounds' : ('l',str,'"(0,240)"','string of the left-right bounds of hill for analysis') }


import os,sys,re,Util, math,numpy,LSF
from video_analysis import vidtools
from video_analysis import viz_vidtools
from numpy import nan
from glob import glob

ground_change_max = 10


opts,args = Util.getopt_long(sys.argv[1:],optstruct,required=['ground_anchors'])

imagedir = os.path.join(args[0],'png')

(seglen,mask,burrow_entrance_xy,pixav,timeav,mousez,burrowz,hill_bounds) = [opts[v] for v in ['seglen','mask','burrow_entrance_xy','pixav','timeav','mousez','burrowz','hill_bounds']]

OPTJOBS=100
if opts['num_batches']:
    num_batches=opts['num_batches']
else:
    num_batches=None
nomask = '/n/home08/brantp/code/video_analysis/nomask'

if 'jpg' in imagedir.split('/'):
    FORMAT='jpg'
    print >> sys.stderr, '"jpg" found in path, assuming FORMAT=jpg\n'
else:
    FORMAT='png'

if imagedir.endswith('.tar'):
    import tarfile
    imtar = tarfile.open(imagedir)
    tarcont = sorted(imtar.getnames())
    images = ['%s:%s' % (imagedir,f) for f in tarcont]
else:
    images = sorted(glob(imagedir+'/*.'+FORMAT))
	
print >> sys.stderr, len(images),'images found in imagedir'

if not images:
    raise ValueError, 'no images found in '+imagedir


if mask == 'None':
    try:
        mask = Util.file_in_path('mask',imagedir)
        print >> sys.stderr, 'mask in path (%s) exists; using' % mask
    except ValueError:
        print >> sys.stderr, 'no mask in path'
else:
    print >> sys.stderr, 'mask found? %s\n' % os.path.exists(mask) 
        

SHAPE=vidtools.load_normed_arrays(images[:1])[0].shape

#print >> sys.stderr, opts

print >> sys.stderr,'xybounds: %s\nburrow entrance at: %s\nground anchors at: %s\nhill_bounds: %s' \
      % (opts['xybounds'],opts['burrow_entrance_xy'],opts['ground_anchors'],opts['hill_bounds'])

match = re.search('([\d\.]+)fps',imagedir)
if match:
    fps = float(match.groups()[0])
    segment_step = int(seglen)* fps
    print >> sys.stderr, 'segment length will be %s frames (%s seconds, %s fps)\n' % (segment_step,seglen,fps)
    unit = 'sec'
else:
    fps = 1.0
    segment_step = int(seglen)
    print >> sys.stderr,'fps could not be found in image path, using seglen (%s) as segment_step (i.e. 1 fps)\n' % seglen
    unit = 'frames'

print >> sys.stderr,'2D averaging of %s pixels; time averaging of %s frames (%0.2f sec)' % (pixav,timeav,float(timeav)/fps)

outroot = os.path.join(args[0],'analysis','%s%s_%szmouse_%szburrow_%spixav_%stimeav' % (seglen,unit,mousez,burrowz,pixav,timeav))

try:
    os.makedirs(outroot)
    or_present = False
except OSError:
    or_present = True

########
# check for previous analyses with matching segment summary params; link *.frame and *.mice* files (should skip summarize)

sourceroots = glob(os.path.join(args[0],'analysis','%s%s_%szmouse_%szburrow_%spixav_%stimeav' % (seglen,unit,mousez,'*',pixav,timeav)))
if sourceroots and (not or_present):
    print >> sys.stderr, 'for current run parameters, segment summaries present in %s' % sourceroots[0]
    print >> sys.stderr, 'get frames ...',
    sourceframes = glob(os.path.join(sourceroots[0],'*.frame'))
    print >> sys.stderr, 'found %s' % len(sourceframes)
    print >> sys.stderr, 'get mice* ...',
    sourcemice = glob(os.path.join(sourceroots[0],'*.mice*'))
    print >> sys.stderr, 'found %s' % len(sourcemice)
    if len(sourcemice) == len(sourceframes)*5:
        currdir = os.getcwd()
        os.chdir(outroot)
        print >> sys.stderr, '%s mice consistent with expectation (%s); link' % (len(sourcemice),len(sourceframes)*5)
        print >> sys.stderr, 'frames ...',
        for f in sourceframes:
            ret = os.system('ln -s %s .' % (os.path.join(currdir,f)))
        print >> sys.stderr, 'done'
        print >> sys.stderr, 'mice ...',
        for f in sourcemice:
            ret = os.system('ln -s %s .' % (os.path.join(currdir,f)))
        print >> sys.stderr, 'done'
        os.chdir(currdir)
    else:
        print >> sys.stderr, 'file counts inconsistent; proceed with segment summaries'

shapefile = os.path.join(outroot,'shape.tuple')
open(shapefile,'w').write(SHAPE.__repr__())



jobids = {}
namedict = {}
prereq = []


rerun = True
restart_z = 12
while rerun:
    cmds = []
    for i in range(0,len(images),int(segment_step)):
        this_out = os.path.join( outroot,'%07d-%07d.mice' % (i,i+segment_step))
        if not os.path.exists(this_out):
            cmd = 'summarize_segment.py -m %s -s %d -e %d -p %d -t %d -r %s -o %s -x \\"%s\\" -b \\"%s\\" -g \\"%s\\" %s' \
                  % (mask,i,i+segment_step,pixav,timeav,outroot,mousez,opts['xybounds'],opts['burrow_entrance_xy'],opts['ground_anchors'],imagedir) 
            cmds.append(cmd)

    # drop last command (incomplete segment)
    dropcmd = cmds.pop()
    logfile = os.path.join(outroot,'summarize-segment-log')
    print >> sys.stderr,'running summary of %s segments, log written to %s\n' % (len(cmds),logfile)
    print >> sys.stderr,'bundle into %s batches' % num_batches
    jids,ndict = LSF.lsf_jobs_submit(cmds,logfile,opts['summarize_queue'],jobname_base='summarize',num_batches=num_batches,bsub_flags='-R "select[mem > 30000]"')
    jobids.update(jids)
    namedict.update(ndict)
    if glob(outroot+'/*.mice'):
        restart_z = None
    else:
        restart_z = 12
    LSF.lsf_wait_for_jobs(jobids,os.path.join(outroot,'restarts'),namedict=namedict,restart_z=restart_z) #restart_z=None) 
    jobids = {}
    namedict = {}

    #remove .mice corresponding to mis-sized frames
    for f in glob(outroot+'/*.frame'):
        fgrp = os.path.splitext(f)[0]
        if os.path.getsize(fgrp+'.frame') != 8 * SHAPE[0] * SHAPE[1]:
            print >> sys.stderr, 'remove missized file %s (obs: %s exp %s)' % (f, os.path.getsize(fgrp+'.frame'), 8 * SHAPE[0] * SHAPE[1])
            os.unlink(fgrp+'.mice')

        
    gaps = vidtools.check_contiguous_files(outroot,'mice')
    if any(gaps):
        print >> sys.stderr,'rerunning due to gaps:',gaps
        #restart_z = None
    else:
        print >> sys.stderr,'.mice pass contiguity check'
        rerun = False

exp_mice = int(math.ceil(len(images)/float(segment_step)))
exp_locs = exp_mice - 2


# parallel classify_mouse_locs and ground correction removed
# present in 20011013 video_analysis package archive

if opts['ground_anchors'] is not None:
    framefiles = sorted(glob(outroot+'/*.frame'))
    print >> sys.stderr, 'load %s frames ...' % len(framefiles),
    groundfiles = [os.path.splitext(f)[0]+'.ground' for f in framefiles]
    frames = [numpy.fromfile(f).reshape(SHAPE) for f in framefiles]
    print >> sys.stderr, 'done'

    if all([os.path.exists(gf) for gf in groundfiles]):
        print >> sys.stderr, '%s ground files found; loading ...' % (len(groundfiles)),
        grounds = [numpy.fromfile(gf,sep='\n') for gf in groundfiles]
        print >> sys.stderr, 'done; %s ground vectors loaded' % len(grounds)
    else:
        print >> sys.stderr, 'Run ground finding'
        ##### ground-finding goes here; serial modification within a bounded range after first ground
        grounds = []
        g1 = vidtools.find_ground4(frames[0],opts['ground_anchors'],xybounds=opts['xybounds'],be=opts['burrow_entrance_xy'])
        grounds.append(g1)
        g1.tofile(os.path.join(groundfiles[0]),sep='\n')

        print >> sys.stderr, 'process ground in segments'
        for itr,(frame,gfile) in enumerate(zip(frames[1:],groundfiles[1:])):
            if itr % 50 == 0:
                print >> sys.stderr, '%5d/%5d %0.1f%%' % (itr,len(frames),float(itr)/len(frames)*100)
            g = []
            for gv,fc in zip(grounds[-1],frame.transpose()):
                # go go gadget maybe-less-crappy...
                gpos = sorted([( numpy.mean(fc[i:])/numpy.mean(fc[:i]),i) for i in range(int(gv-ground_change_max),int(gv+1))],reverse=True)
                gthis = numpy.mean(fc[gv:])/numpy.mean(fc[:gv])
                if gpos[0][0]/gthis > 1.01:
                    g.append(gpos[0][1])
                else:
                    g.append(gv)

                #g.append(sorted([( numpy.mean(fc[i:])/numpy.mean(fc[:i]),i) for i in range(int(gv-ground_change_max),int(gv+1))],reverse=True)[0][1])
            grounds.append(numpy.array(g))
            grounds[-1].tofile(os.path.join(gfile),sep='\n')


if opts['ground_anchors'] is not None and eval(opts['burrowz']) is not None:
    print >> sys.stderr, 'compute activity masks'
    ##### logic from classify_mouse_locs follows
    for i,f in enumerate(frames[1:-1]): # this throws off numbering--must be i+1 for current index!
        filebase = groundfiles[i+1].rsplit('.',1)[0]
        actmatf = filebase+'.actmat'
        actmaskf = filebase+'.actmask'
        
        g = grounds[i+1]
        actmat = Util.zscore(frames[i] - frames[i+2])
        actzcut = float(opts['burrowz'])
        actmask = actmat > actzcut

        #print >>sys.stderr,actmatf,len(actmat),len(actmat[0])
        actmat.tofile(actmatf)
        actmask.tofile(actmaskf)
    ##### end classify_mouse_locs logic
    
if opts['ground_anchors'] is not None and eval(opts['burrowz']) is not None:
    print >> sys.stderr, 'calculating burrowing activity over segments'


    ###################
    # added 20091030
    # switches cumulative activity calc to activity below an artificial "lowest ground" - should fix squirrely groundfinding

    allgrounds = numpy.array([numpy.fromfile(f,sep='\n') for f in glob(outroot+'/*.ground')])
    low_ground = numpy.array([max(col) for col in allgrounds.transpose()])

    #for now, force_all is set manually; might migrate to opts eventually
    activity,termini,progress = vidtools.calculate_cumulative_activity(outroot,float(burrowz),shape=SHAPE,be=eval(opts['burrow_entrance_xy']), \
                                                                       suppress_ground=5,force_all=False,use_ground=low_ground)

    # end 20091030 addition
    ###################


#Goal for final analysis movie: split into segments in order to control files-per-directory
if opts['vid_format'] != 'None':
    movie = os.path.join(outroot,'%s%s_%szmouse_%szburrow_%spixav_%stimeav_%sbit' % (seglen,unit,mousez,burrowz,pixav,timeav,opts['vid_bitrate']))
if opts['vid_format'] != 'None' and not os.path.exists('%s.%s' % (movie,opts['vid_format'])):

    interval = 1

    print >> sys.stderr, 'render analysis-markup video as %s' % movie
    
    print >> sys.stderr, 'get source image count ...',
    source_image_count = len(images)
    print >> sys.stderr, '%s' % source_image_count

    print >> sys.stderr, 'get current image count ...',
    current_image_count = len(glob(outroot+'/images/*.png'))
    print >> sys.stderr, '%s' % current_image_count

    print >> sys.stderr, 'target image count: %s' % ((source_image_count*0.98) -  (segment_step * 3))
    if current_image_count + (segment_step * 3) < source_image_count*0.98:
        cmds = []
        for i in range(0,exp_locs,interval):
            cmds.append('draw_mouse_locations.py %s %d %d' % (outroot,i,i+interval))
        logfile = os.path.join(outroot,'draw-mouse-locations-log')
        print >> sys.stderr,'final video not present in %s\n\trunning rendering of %s segments in %s jobs, log written to %s\n' \
              % (outroot,exp_locs,len(cmds),logfile)
        print >> sys.stderr,'bundle into %s batches' % num_batches

        print >> sys.stderr,'run frame-by-frame image generation' #gaps:',gaps
        rerun = True
        restart_z = 12
        curr_run = 0
        max_run = 5        
    else:
        print >> sys.stderr,'output frame .png files pass contiguity check'
        rerun = False

    while rerun:
        jids,ndict = LSF.lsf_jobs_submit(cmds,logfile,'normal_serial',jobname_base='draw',num_batches=min(64,num_batches),bsub_flags='-R "select[mem > 30000]"')
        jobids.update(jids)
        namedict.update(ndict)

        LSF.lsf_wait_for_jobs(jobids,os.path.join(outroot,'restarts'),namedict=namedict)
        jobids = {}
        namedict = {}
		
        #remove zero-size images
        for f in glob(outroot+'/images/*.png'):
            if os.path.getsize(f) == 0:
                os.unlink(f)
		
        #gaps = vidtools.check_sequential_files(outroot+'/images','png')
        #if any(gaps):

        
        print >> sys.stderr, 'get current image count ...',
        current_image_count = len(glob(outroot+'/images/*.png'))
        print >> sys.stderr, '%s' % current_image_count
        
        print >> sys.stderr, 'target image count: %s' % ((source_image_count*0.98) -  (segment_step * 3))
        if current_image_count + (segment_step * 3) < source_image_count*0.98:
            print >> sys.stderr,'rerunning due to missing images' #gaps:',gaps
            restart_z=None
        else:
            print >> sys.stderr,'output frame .png files pass contiguity check'
            rerun = False


    merge = 'png2vid.py -c -i %s -v %s -n %s -b %s -f %s %s/images/' % (FORMAT,opts['vid_format'],movie,opts['vid_bitrate'],fps,outroot)
    print >> sys.stderr,'re-zeroing %s output images' % len(glob(outroot+'/images/*.png'))
    vidtools.rename_images_from_zero(outroot+'/images')
    print >> sys.stderr,'rendering movie, log written to %s\n' % (logfile)
    jids,ndict = LSF.lsf_jobs_submit([merge],logfile,'hoekstra',jobname_base='video')
    jobids.update(jids)
    namedict.update(ndict)    


micefiles = sorted(glob(outroot+'/*-*.mice'))
#locsfiles = sorted(glob(outroot+'/*-*.mouselocs'))
#locsummfiles = sorted(glob(outroot+'/*-*.locsumm'))
framefiles = sorted(glob(outroot+'/*-*.frame'))
#miceoutlinefiles = sorted(glob(outroot+'/*-*.miceoutline'))
micezfiles = sorted(glob(outroot+'/*-*.micez'))
micesizefiles = sorted(glob(outroot+'/*-*.micesize'))

if opts['ground_anchors'] is not None:
    groundfiles = sorted(glob(outroot+'/*-*.ground'))


mice = []
print >> sys.stderr,'%s .mice found (%s expected)\n\tloading...' % (len(micefiles),exp_mice)
for i,f in enumerate(micefiles):
	if i % 100 == 0:
		print >> sys.stderr, '\t%s' % i
	mice.append(eval(open(f).read()))
print >> sys.stderr,'\tmerging...'
mice = Util.merge_dictlist(mice,verbose=True)
print >> sys.stderr,'\tdone\n'

'''
locs = []
print >> sys.stderr,'%s .mouselocs found (%s expected)\n\tloading...' % (len(locsfiles),exp_locs)
for i,f in enumerate(locsfiles):
	if i % 100 == 0:
		print >> sys.stderr, '\t%s' % i
	locs.append(eval(open(f).read()))
print >> sys.stderr,'\tmerging...'
locs = Util.merge_dictlist(locs,verbose=True)
print >> sys.stderr,'\tdone'

locsumms = Util.countdict(locs)
'''

'''miceoutlines are a little too big to work with in single summary files
miceoutlines = []
print >> sys.stderr,'%s .miceoutlines found\n\tloading...' % (len(miceoutlinefiles))
for i,f in enumerate(miceoutlinefiles):
	if i % 100 == 0:
		print >> sys.stderr, '\t%s' % i
	miceoutlines.append(eval(open(f).read()))
print >> sys.stderr,'\tmerging...'
miceoutlines = Util.merge_dictlist(miceoutlines,verbose=True)
print >> sys.stderr,'\tdone'
'''

micezsc = []
print >> sys.stderr,'%s .micez found\n\tloading...' % (len(micezfiles))
for i,f in enumerate(micezfiles):
	if i % 100 == 0:
		print >> sys.stderr, '\t%s' % i
	micezsc.append(eval(open(f).read()))
print >> sys.stderr,'\tmerging...'
micezsc = Util.merge_dictlist(micezsc,verbose=True)
print >> sys.stderr,'\tdone'

micesize = []
print >> sys.stderr,'%s .micesize found\n\tloading...' % (len(micesizefiles))
for i,f in enumerate(micesizefiles):
	if i % 100 == 0:
		print >> sys.stderr, '\t%s' % i
	micesize.append(eval(open(f).read()))
print >> sys.stderr,'\tmerging...'
micesize = Util.merge_dictlist(micesize,verbose=True)
print >> sys.stderr,'\tdone'

#grounds = [numpy.fromfile(g,sep='\n') for g in groundfiles]

#[(xl,yl),(xh,yh)] = eval(opts['xybounds'].strip('"'))
#bbox = [(xl,yl),(xh,yl),(xh,yh),(xl,yh)]

#scatter_polys = [bbox]

#final_actout = eval(open(sorted(glob(outroot+'/*.preactout'))[-1]).read())
#scatter_polys += final_actout

#locs_out = movie.rsplit('.',1)[0]+'.mouselocs'
#open(locs_out,'w').write(locs.__repr__())

#summ_out = movie.rsplit('.',1)[0]+'.locsumm'
#open(summ_out,'w').write(locsumms.__repr__())

#miceoutlines too big to summarize!
#miceoutlines_out = movie.rsplit('.',1)[0]+'.miceoutline'
#open(miceoutlines_out,'w').write(miceoutlines.__repr__())

try:
    mice_out = movie.rsplit('.',1)[0]+'.mice'
    open(mice_out,'w').write(mice.__repr__())

    micezsc_out = movie.rsplit('.',1)[0]+'.miceoutline'
    open(micezsc_out,'w').write(micezsc.__repr__())

    micesizes_out = movie.rsplit('.',1)[0]+'.miceoutline'
    open(micesizes_out,'w').write(micesize.__repr__())
except:
    pass

sps = float(seglen)

#viz_vidtools.draw_class_scatter_and_pie(locs_out,sps=sps,scatter_lines=grounds,scatter_polys=scatter_polys)
#draw_summ = 'draw_summary_fig.py %s %s' % (locs_out, sps)
#jids,ndict = LSF.lsf_jobs_submit([draw_summ],logfile,'normal_serial',jobname_base='draw_summary')

if eval(opts['burrowz']) is not None and opts['ground_anchors'] is not None:
    viz_vidtools.draw_analysis_activity_summary(outroot,1,'pdf',hill_bounds=eval(hill_bounds),show_mice=True)
end_sec = time.time()

t = int(end_sec-start_sec)
print >> sys.stderr, 'total runtime %dh %dm %ds' % (t/60/60, t/60%60, t%60)
