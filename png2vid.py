#!/usr/bin/env python
'''
ping2vid.py [options] imagedir

accepts a directory containing images of specified type (default: png)
generates a single video in the specified format (default: mp4)

if first file is not zero, will re-zero all files (calls vidtools.rename_images_from_zero())

can also accept a frame range, in which case copies are made before zeroing
'''

optstruct = { 
	'startframe' : ('s',int,None,'optional start offset'),
	'endframe' : ('e',int,None,'optional end frame'),
	'imageformat' : ('i',str,'png','format of input images'),
	'vidformat' : ('v',str,'mp4','format for output video'),
	'vidname' : ('n',str,None,'name for output video'),
	'bitrate' : ('b',str,'1500k','encoding bitrate for video'),
	'fps' : ('f',str,'29.97','encoding framerate for video'),
	'resolution' : ('r',str,None,'target resolution for output video'),
	'cleanup' : ('c','flag',False,'delete images when finished.  removes whole image tree (i.e. rm -rf imagedir) so use with caution!')
	}
	
import Util,os,sys,re,shutil
from video_analysis import vidtools
from glob import glob	

opts,args = Util.getopt_long(sys.argv[1:],optstruct)

imagedir = args[0]

print imagedir+'/*.%s' % opts['imageformat']

image1 = sorted(glob(imagedir+'/*.%s' % opts['imageformat']))[0]
iname = os.path.split(image1)[1]
inum = os.path.splitext(iname)[0]
digits = len(inum)

#check if re-zeroing is required:
if int(inum) != 0:
	print >> sys.stderr, 'first image (%s) does not start at zero; reindexing' % image1
	vidtools.rename_images_from_zero(imagedir,type=opts['imageformat'],digits=digits)
	image1 = sorted(glob(imagedir+'/*.%s' % opts['imageformat']))[0]
	iname = os.path.split(image1)[1]
	inum = os.path.splitext(iname)[0]
	if int(inum) == 0:
		print >> sys.stderr, 'reindexing complete'
	else:
		raise ValueError, 'Re-indexing called, but first frame is %s (%s) - aborting' % (inum,image1)

if opts.get('resolution',None):
	res_str = '-s %s' % opts['resolution']
else:
	res_str = ''

if opts.get('startframe',None) and opts.get('endframe',None):
	startsecs = opts['startframe']/float(opts['fps'])
	durframes = opts['endframe'] - opts['startframe']
	dursecs = durframes/float(opts['fps'])
	vidname = opts.get(
		'vidname',
		os.path.join(imagedir,'%s-%s_%sfps_%sbit' % \
			(opts['startframe'],opts['endframe'],opts['fps'],opts['bitrate']) ))
	print >> sys.stderr, 'start,end of %s, %s invoked.  %s frames @ %s fps; start %0.2f, %0.2f seconds total' % \
		(opts['startframe'], opts['endframe'], durframes, opts['fps'], startsecs, dursecs)
	cmd = 'ffmpeg -r %s -i %s/%%0%dd.%s -ss %0.2f -t %0.2f -b %s %s -y %s.%s' % \
		(opts['fps'],imagedir,digits,opts['imageformat'],startsecs,dursecs,opts['bitrate'],res_str,vidname,opts['vidformat'])
else:
	vidname = opts.get(
		'vidname',
		os.path.join(imagedir,'all_%sfps_%sbit' % (opts['fps'],opts['bitrate']) ))
	cmd = 'ffmpeg -r %s -i %s/%%0%dd.%s -b %s %s -y %s.%s' % \
		(opts['fps'],imagedir,digits,opts['imageformat'],opts['bitrate'],res_str,vidname,opts['vidformat'])

print >> sys.stderr, 'running: %s' % cmd
os.system(cmd)
if opts.get('cleanup',None):
	numframes = len(glob(imagedir+'/*.%s' % opts['imageformat']))
	exp_vidlen = numframes / float(opts['fps'])
	print >> sys.stderr, 'vidname %s.%s from %s frames expect %s sec;' % (vidname,opts['vidformat'],numframes,exp_vidlen)
	obs_vidlen = vidtools.vid_duration('%s.%s' % (vidname,opts['vidformat']))
	if obs_vidlen < exp_vidlen*0.95:
		print >> sys.stderr, 'observed length (%s) less than expected (%s), skipping cleanup' % (obs_vidlen,exp_vidlen)
	else:
		print >> sys.stderr, 'observed length (%s). cleanup...' % (obs_vidlen)
		os.system('rm -rf %s' % imagedir.rstrip('/'))
