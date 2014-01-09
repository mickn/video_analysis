#!/usr/bin/env python

GROUND_STEP_COEFF = 1.5 #multiple of mouse radius 
MAX_PIX_MULT = 3
MAX_OBJ = 5
MAX_DIST_MULT = 10
HILL_WIN_COEFF = 4

import os,sys
import viz_vidtools

analysis_dir = sys.argv[1]

viz_vidtools.draw_rainbowtron_opencv(analysis_dir,hill_bounds=None,set_hill_max=200,set_dig_max=200,ground_step_coeff=GROUND_STEP_COEFF,max_pix_mult=MAX_PIX_MULT,max_obj=MAX_OBJ,max_dist_mult=MAX_DIST_MULT,hill_win_coeff=HILL_WIN_COEFF)
