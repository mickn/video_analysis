#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')

import vidtools,viz_vidtools
import os, sys, re, Util, numpy, pylab
from glob import glob

pylab.gray()
col = {1:'r',2:'y',3:'g',4:'b',5:'c'}

hres = 720

(output_dir,start,stop) = sys.argv[1:]

an_im_dir = os.path.join(output_dir,'images')
try:
	os.makedirs(an_im_dir)
except OSError:
	pass

micefiles = sorted(glob(output_dir+'/*.mice'))
lastfind = None
startn = re.search('(\d+?)-\d+?\.mice',micefiles[0]).groups()[0]
startn = int(startn)
#print >> sys.stderr, 'all frames offset to -%s' % start
for i,f in enumerate(micefiles[int(start):int(stop)]):
	print >> sys.stderr, 'processing %s (%s of %s)' % (f,i+1, len(micefiles[int(start):int(stop)]))
	segbase = os.path.splitext(f)[0]
	try:
		g = numpy.fromfile(os.path.splitext(f)[0]+'.ground',sep='\n')
	except:
		pass
	mice = eval(open(os.path.splitext(f)[0]+'.mice').read())
	miceout = eval(open(os.path.splitext(f)[0]+'.miceoutline').read())
	micesize = eval(open(os.path.splitext(f)[0]+'.micesize').read())
	micez = eval(open(os.path.splitext(f)[0]+'.micez').read().replace('nan','0.0'))
	#locsumm = eval(open(os.path.splitext(f)[0]+'.locsumm').read())
	#mouselocs = eval(open(f).read())
	tickon = len(mice)/20
	for cnt,image in enumerate(mice.keys()):
		if cnt%tickon == 0:
			print >> sys.stderr, '%6d/%6d' % (cnt,len(mice))
		im = os.path.split(image)[1]
		n,ext = os.path.splitext(im)
		n = int(n)
		outf = os.path.join(an_im_dir,'%07d.png' % (n-startn))
                #print >> sys.stderr, '\tdraw %s...' % outf,

		if os.path.exists(outf):
                	#print >> sys.stderr, 'skip'
			continue

		#pylab.matshow(vidtools.load_normed_arrays([image])[0],fignum=1)
                fig = pylab.figure(1)
                fr = vidtools.load_normed_arrays([image])[0]
                aspect = fr.shape[1]/float(fr.shape[0])
                pylab.imshow(fr)
                ax = pylab.figure(1).axes[0]

		try:
			pylab.plot(g,'g')
		except:
			pass

		# out for now; if "mouseloc" concept returns, uncomment
		#if mouselocs[image] and mice[image] is not None:
                #	lastcoord = Util.dezip([mice[image]])
		#	lastcol = col[mouselocs[image]]
		#	pylab.scatter(c=lastcol,*lastcoord)

		if mice[image] is not None:
                	lastcoord = Util.dezip([mice[image]])
			pylab.scatter(c='r',*lastcoord)


                if miceout[image] is not None:
                	for o in miceout[image]:
                        	ax.add_patch(pylab.matplotlib.patches.Polygon(o,ec='y',fill=False))
                        
                        
		pylab.text(50,200,'%s\n%s\n%s' \
                   % tuple([os.path.split(source)[1] for source in [image, f, '/z: %s; %s pix' % (['%0.2f' % d for d in micez[image]],micesize[image])]]), color='white')

		#look for, load and draw burrow details:
                # previous activity
                # current activity; ruler line
                # len and area stats
                                
                thisf = micefiles[int(start)+i]
                actsegbase = os.path.splitext(thisf)[0]
		if os.path.exists(actsegbase+'.newactout'):
			newactout = eval(open(actsegbase+'.newactout').read())
			for o in newactout:
				ax.add_patch(pylab.matplotlib.patches.Polygon(o,ec='b',fill=False))

		if os.path.exists(actsegbase+'.preactout'):
			preactout = eval(open(actsegbase+'.preactout').read())
			for o in preactout:
				if len(o) > 2:
					ax.add_patch(pylab.matplotlib.patches.Polygon(o,ec='r',fill=False))
		if os.path.exists(actsegbase+'.newactprop'):
			prop = eval(open(actsegbase+'.newactprop').read())
			if prop['farthest_new'] is not None and prop['nearest_old'] is not None:
				x,y = Util.dezip([prop['farthest_new'],prop['nearest_old']])
				pylab.plot(x,y,marker='+',mew=2,color='k')
				x,y = prop['farthest_new']
				pylab.text(x,y,'%s, %s' % (prop['progress'],prop['area']))
                ax.set_yticks([])
                ax.set_xticks([])

                fig.subplots_adjust(0,0,1,1)                
                fig.set_size_inches((hres/100.0*aspect,hres/100.0+0.48),forward=True)

		pylab.savefig(outf)
		pylab.close(fig)
                #print >> sys.stderr, 'done'
	#commented 20111013 - revisit after class_scatter_and_pie reviewed for classify-less pipeline
	#viz_vidtools.draw_class_scatter_and_pie(f)
