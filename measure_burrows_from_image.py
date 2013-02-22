'''
functionality for loading a measuring an image (photo) of antfarm tunnels
'''

import numpy, pylab, sys, os
import vidtools, iplot, Util
from PIL import Image

def load_image(im_file):
	ar = numpy.asarray(Image.open(im).convert(mode='L'))
	return ar

def get_calibration(fignum = 1, pct_wide = 0.2, pct_high = 0.33, tunneldicts=None):
	ax = pylab.figure(fignum).axes[0]
	ymax = round(ax.get_ybound()[1])
	xmax = round(ax.get_xbound()[1])
	pix_wide = ymax*pct_wide
	pix_high = ymax*pct_high
	print >> sys.stderr, 'image is %s x %s; windows will be %s, %s' % (xmax,ymax,pix_wide,pix_high)
	cm_dists = []
	print >> sys.stderr, 'click 0, 1, 10 cm marks'
	pylab.xlim(0,pix_high)
	pylab.ylim(ymax,ymax-pix_wide)
	p0,p1,p10 = pylab.ginput(3,0)
	cm_dists.append(vidtools.hypotenuse(p0,p1))
	cm_dists.append(vidtools.hypotenuse(p0,p10)/10.0)
	print >> sys.stderr, 'click 0, 1, 10 cm marks'
	pylab.xlim(xmax-pix_wide,xmax)
	pylab.ylim(pix_high,0)
	p0,p1,p10 = pylab.ginput(3,0)
	cm_dists.append(vidtools.hypotenuse(p0,p1))
	cm_dists.append(vidtools.hypotenuse(p0,p10)/10.0)
	print >> sys.stderr, 'click bolt 1'
	pylab.xlim(0,pix_wide)
	pylab.ylim(ymax,ymax-pix_wide)
	horiz = [pylab.ginput(1,0)[0]]
	print >> sys.stderr, 'click bolt 2'
	pylab.xlim(xmax-pix_wide,xmax)
	pylab.ylim(ymax,ymax-pix_wide)
	horiz.append(pylab.ginput(1,0)[0])
	pylab.xlim(0,xmax)
	pylab.ylim(ymax,0)
	if tunneldicts is not None:
		tunneldicts['cm_pts'] = cm_dists
		tunneldicts['horiz'] = horiz
	else:
		return cm_dists,horiz

def add_straight_tunnel(tunnels,tunnel_oris,horiz,this_name,parent_name=None,update_plot=1,**kwargs):
	'''
	specify start and end of straight line.  if parent_name is None, treat as hillside
	if update_plot, draws all tunnels in figure specified (e.g. update_plot == 1; plot in figure 1)
	'''
	tunnels[this_name] = pylab.ginput(2,0)
	if parent_name is not None:
		tunnel_oris[this_name] = parent_name
	if update_plot:
		plot_tunnels(tunnels,horiz,update_plot)



def add_curved_tunnel(tunnels,tunnel_oris,horiz,this_name,parent_name=None,update_plot=1,**kwargs):
	'''
	specify 5 points of curved tunnel
	if update_plot, draws all tunnels in figure specified (e.g. update_plot == 1; plot in figure 1)
	'''
	print >> sys.stderr, \
	'''specify 5 points of curved tunnel:
		pt1 starts tunnel
		pt1 -> pt2 : plane of start of tunnel (angles calculated will use this segment)
		pt3 is curve inflection
		pt4 -> pt5 : plane of end of tunnel
		pt5 ends tunnel'''

	tunnels[this_name] = pylab.ginput(5,0)
	if parent_name is not None:
		tunnel_oris[this_name] = parent_name
	if update_plot:
		plot_tunnels(tunnels,horiz,update_plot)


def tunnel_len(tun,cm_factor=1):
	p1 = tun[0]
	tot = 0                  
	for p in tun[1:]:
		tot += vidtools.hypotenuse(p1,p)
		p1 = p                                                        
	return tot/cm_factor

def nearest_point_on_line(lpt1,lpt2,qpt):
	h = vidtools.hypotenuse(lpt1,lpt2)
	xdiff = lpt2[0] - lpt1[0]
	m,b = vidtools.line_fx_from_pts(lpt1,lpt2)
	xstep = xdiff/h                                                            
	lpoly = [(x,(m*x+b)) for x in numpy.arange(lpt1[0],lpt2[0],xstep)]
	return vidtools.perigee(lpoly,[qpt])

def nearest_point_on_curve(curve,qpt):
    lseg_dists = []                   
    p1 = curve[0]            
    for i,p in enumerate(curve[1:]):
        lseg_dists.append((vidtools.hypotenuse(*nearest_point_on_line(p1,p,qpt)),(i,nearest_point_on_line(p1,p,qpt)[0])))
        p1 = p                                                        
    return min(lseg_dists)[-1]

def slope(p1,p2):          
    return (p2[1]-p1[1]) / (p2[0]-p1[0])

def plot_tunnels(tunnels,horiz,fignum=1,keep_lim=True,clear_lines=True,**kwargs):
	'''plot current tunnels in specified figure (axes[0])
	re-zoom to limits at time of call after plotting if keep_lim
	clear previous plotted lines in axes[0] if clear_lines
	'''
	ax = pylab.figure(fignum).axes[0]
	prev_ylim = ax.get_ybound()
	prev_xlim = ax.get_xbound()

	ax.lines = []

	x,y = Util.dezip(horiz)
	pylab.plot(x,y,'k--',lw=2)
	for c,k in iplot.subspec_enum(sorted(tunnels.keys())):
		x,y = Util.dezip(tunnels[k])
		if not '.' in k:
		    pylab.plot(x,y,'w:',lw=2)
		else:
		    pylab.plot(x,y,c=c,lw=2)

	pylab.xlim(*prev_xlim)
	pylab.ylim(*prev_ylim[::-1])

def calc_tunnel_H(this_name,tunnels,horiz,cm_factor=1):
	Hstart = vidtools.hypotenuse(*nearest_point_on_line(horiz[0],horiz[1],tunnels[this_name][0]))/ cm_factor
	Hend = vidtools.hypotenuse(*nearest_point_on_line(horiz[0],horiz[1],tunnels[this_name][-1]))/ cm_factor

	return Hstart, Hend

def calc_tunnel_angles(this_name,tunnels,tunnel_oris,horiz):
	if len(tunnels[tunnel_oris[this_name]]) == 2:
		prev_deg = numpy.degrees(numpy.arctan(slope(*tunnels[tunnel_oris[this_name]])))
	else:
		prev_seg,prev_pt = nearest_point_on_curve(tunnels[tunnel_oris[this_name]],tunnels[this_name][0])
		prev_deg = numpy.degrees(numpy.arctan(slope(*tunnels[tunnel_oris[this_name]][prev_seg:prev_seg+2])))

	prev_rightward = tunnels[tunnel_oris[this_name]][0][0] < tunnels[tunnel_oris[this_name]][-1][0]
	this_rightward = tunnels[this_name][0][0] < tunnels[this_name][-1][0]
	#print >> sys.stderr, 'prev rt',prev_rightward,'this rt', this_rightward

	this_deg = numpy.degrees(numpy.arctan(slope(*tunnels[this_name][:2])))
	horiz_deg = numpy.degrees(numpy.arctan(slope(*horiz)))

	if prev_rightward != this_rightward:
		this_deg = 180 - this_deg

	if not prev_rightward: #flip reference frame, mouse is "headed left"
		prev_deg *= -1
		horiz_deg *= -1
	if not this_rightward:
		this_deg *= -1

	angle_to_prev = this_deg - prev_deg
	angle_to_horiz = this_deg - horiz_deg

	return angle_to_prev,angle_to_horiz

def calc_chord_stats(tun,cm_factor=1):
    h = vidtools.hypotenuse(tun[0],tun[-1])
    xdiff = tun[-1][0] - tun[0][0]
    m,b = vidtools.line_fx_from_pts(tun[0],tun[-1])
    xstep = xdiff/h                                                                                     
    lpoly = [(x,(m*x+b)) for x in numpy.arange(tun[0][0],tun[-1][0],xstep)]
    on_chord_pt,on_curve_pt = max([(vidtools.hypotenuse(*vidtools.perigee(lpoly,[p])),vidtools.perigee(lpoly,[p])) for p in tun])[1]

    chord_len = vidtools.hypotenuse(tun[0],tun[-1])/cm_factor
    chord_pt = vidtools.hypotenuse(tun[0],on_chord_pt)/cm_factor
    chord_dp = vidtools.hypotenuse(on_chord_pt,on_curve_pt)/cm_factor
    return chord_len, chord_pt, chord_dp

def calc_ori_pt(this_name,tunnels,tunnel_oris,cm_factor=1):
	i,near_pt = nearest_point_on_curve(tunnels[tunnel_oris[this_name]],tunnels[this_name][0])
	ori = (tunnel_len(tunnels[tunnel_oris[this_name]][:i+1]) + vidtools.hypotenuse(tunnels[tunnel_oris[this_name]][i],near_pt)) / cm_factor
	return ori

def calc_tunnel_properties(this_name,tunnels,tunnel_oris,horiz,cm_factor=1,**kwargs):
	if kwargs.has_key('cm_pts'):
		cm_factor = numpy.mean(kwargs['cm_pts'])

	if isinstance(cm_factor,list):
		cm_factor = numpy.mean(cm_factor)
	tunnel_prop = {}
	tunnel_prop['Hstart'], tunnel_prop['Hend'] = calc_tunnel_H(this_name,tunnels,horiz,cm_factor)
	tunnel_prop['L'] = tunnel_len(tunnels[this_name],cm_factor)
	tunnel_prop['angle_to_prev'],tunnel_prop['angle_to_horiz'] = calc_tunnel_angles(this_name,tunnels,tunnel_oris,horiz)

	if len(tunnels[this_name]) > 2:
		tunnel_prop['chord_len'],tunnel_prop['chord_pt'],tunnel_prop['chord_dp'] = calc_chord_stats(tunnels[this_name],cm_factor)

	if '.' in tunnel_oris[this_name]:
		tunnel_prop['origin'] = calc_ori_pt(this_name,tunnels,tunnel_oris,cm_factor)

	return tunnel_prop

def save_tunnels(im,tunnels,tunnel_oris,horiz,cm_pts):
	outfile = os.path.splitext(im)[0] + '.tunneldicts'
	outdict = {'tunnels':tunnels,'tunnel_oris':tunnel_oris,'horiz':horiz,'cm_pts':cm_pts}
	open(outfile,'w').write(outdict.__repr__())

def load_tunnels(tunneldicts,asdict=False):
	if not tunneldicts.endswith('.tunneldicts'):
		tunneldicts = os.path.splitext(tunneldicts)[0] + '.tunneldicts'
	tdict = eval(open(tunneldicts).read())
	if asdict:
		return tdict
	else:
		return [tdict[k] for k in ['tunnels','tunnel_oris','horiz','cm_pts']]


