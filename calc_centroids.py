#!/usr/bin/env python

'''given a mouse outlines tarfile, runs vidtools.centroids_from_mouseols_tar() and saves result to micecentroids.npy in same directory
'''

import os,sys
import vidtools

tarf = sys.argv[1]

centfile = os.path.join(os.path.dirname(tarf),'micecentroids.npy')

centroids = vidtools.centroids_from_mouseols_tar(tarf,centfile)
