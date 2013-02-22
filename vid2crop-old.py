#!/usr/bin/env python
import os,sys,re,shutil
from glob import glob
from video_analysis import vidtools
from PIL import Image
import LSF

if len(sys.argv) == 3:
	(vid,cropsdictfile) = sys.argv[1:3]
	dur = 0
	offset = 0
else:
	(vid,offset,dur,cropsdictfile) = sys.argv[1:5]

cropsdict = eval(open(cropsdictfile).read())

offset = int(offset)
dur = int(dur)

if dur == 0:
	dur = vidtools.vid_duration(vid) - offset

cmds = []
rerun = True
while rerun:
	for clab,crops in cropsdict.items():
		outbase,outext = os.path.splitext(vid)
		outvid = '%s_%s_%s-%s%s' % (outbase,clab,offset,dur,outext)
		if os.path.exists(outvid) and ( vidtools.vid_duration(outvid) == dur ):
			print >> sys.stderr, '%s present and expected size, skip' % outvid
		else:
			cropstr = '-vf crop=in_w-%s:in_h-%s:%s:%s' % (crops[0]+crops[2],crops[1]+crops[3],crops[0],crops[1])
			cmd = 'ffmpeg -ss %s -t %s -i %s -y %s -b 20000k %s' % (offset,dur,vid,cropstr,outvid)
			cmds.append(cmd)

	
	logfile = os.path.join(os.path.dirname(vid),'crop-log')
	jobids,namedict = LSF.lsf_jobs_submit(cmds,logfile,'normal_serial',jobname_base='vid2crop')
	LSF.lsf_wait_for_jobs(jobids,logfile,'normal_serial',namedict=namedict)
	
	cmds = LSF.lsf_no_success_from_log(logfile)
	if len(cmds) == 0:
		rerun = False
		
sys.exit()
rerun = True

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
