#!/usr/bin/env python
import os,sys,re,shutil
from glob import glob
from video_analysis import vidtools
from PIL import Image

(vid,tdir,offset,dur,fps,cropstr) = sys.argv[1:7]

if cropstr == '.':
	cropstr = ''

try:
	foffset = int(sys.argv[7]) + int(int(offset)*float(fps))
except IndexError:
	foffset = int(int(offset)*float(fps))

if dur == 0:
	dur = vidtools.vid_duration(vid) - offset

####
# experimental: switch to node scratch

#outroot = os.path.join(tdir,str(foffset))
outroot = os.path.join('/scratch/brantp/vid2png/',str(foffset))

#
####

try:
	os.system('rm -rf %s' % outroot)
except:
	print >> sys.stderr, "couldn't remove %s" % outroot 
try:
	os.makedirs(outroot)
	print >> sys.stderr, 'makedirs done for %s' % outroot
except OSError:
	print >> sys.stderr, 'makedirs failed for %s' % outroot
	outroot = os.path.join(tdir,str(foffset))
	print >> sys.stderr, 'switched to %s' % outroot
	os.system('rm -rf %s' % outroot)
	os.makedirs(outroot)
	

rerun = True
scratchruns = 1
maxruns = 4
nrun = 0
#execstr = 'ffmpeg -ss %s -t %s -i %s -r %s -y %s ' % (offset,dur,vid,fps,outstr)
last_empties = []
while rerun:
	outstr = os.path.join(outroot,'%07d.png')
	execstr = 'ffmpeg -ss %s -t %s -i %s -r %s -y %s %s 2> /dev/null' % (offset,dur,vid,fps,cropstr,outstr)
	print >> sys.stderr, 'execute %s\nrunning %s' % (nrun,execstr)
	os.system(execstr)
	nrun += 1
	#empties = [f for f in sorted(glob(outroot+'/*.png')) if os.path.getsize(f) < 5000]
	empties = vidtools.get_bad_images(outroot,'png')
	if len(empties) > 0:
		#if set(last_empties) == set(empties): #REMOVE TWO-STRIKES; replace w/ switch off scratch
		#	print >> sys.stderr, '%s empty twice; re-zeroing and moving on' % empties
		#	vidtools.rename_images_from_zero(outroot,clear_below=5000)
		#	rerun = False
		print >> sys.stderr, '%s empty files, clear output:\n%s' % (len(empties),outroot)
		os.system('find %s -name "*.png" -delete' % outroot)
		if nrun >= maxruns:
			raise OSError, '%s attempts failed; quitting' % nrun
		elif nrun >= scratchruns and outroot.startswith('/scratch'):
			outroot = os.path.join(tdir,str(foffset))
			try:
				os.system('rm -rf %s' % outroot)
			except:
				print >> sys.stderr, "couldn't remove %s" % outroot 
       			try:
				os.makedirs(outroot)
				print >> sys.stderr, 'makedirs done for %s' % outroot
			except OSError:
				print >> sys.stderr, '%s already exists' % outroot

			print >> sys.stderr, '%s scratch attempts failed; switched to %s' % (nrun,outroot)
		else:
			print >> sys.stderr, 'rerunning'
			last_empties = empties
	else:
		print >> sys.stderr, 'no empty files, proceeding'
		rerun = False
	

pngs = glob(os.path.join(outroot,'*.png'))
print >> sys.stderr, '%s output images successfully created' % len(pngs)

print >> sys.stderr, 'transferring...',
for f in pngs:
	n = os.path.split(f)[1].rsplit('.',1)[0]
	actual = int(n) + foffset
	newname = os.path.join(tdir,'%07d.png' % actual)
	shutil.move(f,newname)
print >> sys.stderr, 'clearing scratch...',	
os.system('rm -rf %s' % outroot)
print >> sys.stderr, 'done.'
