#!/usr/bin/env python
import os,sys,re,shutil
from glob import glob
from video_analysis import vidtools
from PIL import Image
import LSF,run_safe

job_ram = 30000
MAX_RETRY = 3
queue = 'normal_serial'

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

to_run_dict = {}
for clab,crops in cropsdict.items():
	outbase,outext = os.path.splitext(vid)
	outvid = '%s_%s_%s-%s%s' % (outbase,clab,offset,dur,outext)
	if os.path.exists(outvid) and ( vidtools.vid_duration(outvid) == dur ):
		print >> sys.stderr, '%s present and expected size, skip' % outvid
	else:
		cropstr = '-vf crop=in_w-%s:in_h-%s:%s:%s' % (crops[0]+crops[2],crops[1]+crops[3],crops[0],crops[1])
		cmd = 'ffmpeg -ss %s -t %s -i %s -y %s -r 29.97 -b 20000k %s' % (offset,dur,vid,cropstr,outvid)
		to_run_dict[outvid] = run_safe.safe_script(cmd,outvid)

logfile = os.path.join(os.path.dirname(vid),'logs','crop-log')
LSF.lsf_run_until_done(to_run_dict,logfile,queue,'-R "select[mem>%s]"' % job_ram, 'crop-ffmpeg',10, MAX_RETRY)
   

#cmds = []
#rerun = True
#while rerun:
#	for clab,crops in cropsdict.items():
#		outbase,outext = os.path.splitext(vid)
#		outvid = '%s_%s_%s-%s%s' % (outbase,clab,offset,dur,outext)
#		if os.path.exists(outvid) and ( vidtools.vid_duration(outvid) == dur ):
#			print >> sys.stderr, '%s present and expected size, skip' % outvid
#		else:
#			cropstr = '-vf crop=in_w-%s:in_h-%s:%s:%s' % (crops[0]+crops[2],crops[1]+crops[3],crops[0],crops[1])
#			cmd = 'ffmpeg -ss %s -t %s -i %s -y %s -b 20000k %s' % (offset,dur,vid,cropstr,outvid)
#			cmds.append(cmd)
#
#	
#	logfile = os.path.join(os.path.dirname(vid),'crop-log')
#	jobids,namedict = LSF.lsf_jobs_submit(cmds,logfile,'normal_serial',jobname_base='vid2crop')
#	LSF.lsf_wait_for_jobs(jobids,logfile,'normal_serial',namedict=namedict)
#	
#	cmds = LSF.lsf_no_success_from_log(logfile)
#	if len(cmds) == 0:
#		rerun = False
		
print >> sys.stderr, 'DONE'
