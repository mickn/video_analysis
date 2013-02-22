#!/usr/bin/env python

from video_analysis import vidtools
import sys,os

vid = sys.argv[1]
out = os.path.splitext(vid)[0]+'.diff'
open(out,'w').write('%s' % vidtools.diff_first_and_last_frames(vid))