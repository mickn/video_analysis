
im = '20120319/af8_PO-22.jpg'
ar = numpy.asarray(Image.open(im).convert(mode='L'))
from PIL import Image
ar = numpy.asarray(Image.open(im).convert(mode='L'))
matshow(ar)
fig = plt.figure(1)
def onclick(event):
    print 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(
        event.button, event.x, event.y, event.xdata, event.ydata)
cid = fig.canvas.mpl_connect('button_press_event', onclick)

fig.canvas.mpl_disconnect(cid)
pylab.ginput(2,0)
p1,p2 = _
from video_analysis import vidtools
from video_analysis import vidtools, viz_vidtools
vidtools.hypotenuse(p1,p2)
p0,p1,p10 = ginput(3,0
)
cm_dists = {}
cm_dists['bot1'] = vidtools.hypotenuse(p0,p1)/1
cm_dists['bot10'] = vidtools.hypotenuse(p0,p10)/1
cm_dists
p0,p1,p10 = ginput(3,0
)
cm_dists['rt1'] = vidtools.hypotenuse(p0,p1)/1
cm_dists['rt10'] = vidtools.hypotenuse(p0,p10)/1
cm_dists
mean([cm_dists['bot1'],cm_dists['bot1'],cm_dists['bot1'],)
cm_dists['bot10'] = vidtools.hypotenuse(p0,p10)/10
cm_dists['rt10'] = vidtools.hypotenuse(p0,p10)/10
cm_dists
p0,p1,p10 = ginput(3,0
)
cm_dists['rt1'] = vidtools.hypotenuse(p0,p1)/1
cm_dists['rt10'] = vidtools.hypotenuse(p0,p10)/10
cm_dists
mean(cm_dists.values())
tunnels = {}
tunnels['MR1.1'] = {}
tunnels['MR1.1']['
horiz
horiz = ginput(2,0)
horiz = ginput(2,0)
horiz
def slope(p1,p2):
    return (p2[1]-p1[1]) / (p2[0]-p1[0])
slope(*horiz)
tunnels['MLhill'] = []
tunnels['MLhill'] = {}
tunnels['MLhill'] = ginpu(2,0)
tunnels['MLhill'] = ginput(2,0)
tan(slope(*horiz))
tan(slope(*horiz))
tan(1)
degrees(tan(1))
#?degrees
rad(45)
radiansd(45)
radians(45)
#?tan
radians([45])
tan(1)
pi/tan(1)
pi/arctan(1)
degrees(arctan(1))
degrees(tan(1))
degrees(arctan(slope(*horiz))))
degrees(arctan(slope(*horiz)))
tunnels
degrees(arctan(slope(*tunnels['MLhill'])))
slope(*tunnels['MLhill'])
tunnels
tunnels['MR1.1'] = ginput(2,0)
slope(*tunnels['MR1.1'])
degrees(arctan(slope(*tunnels['MR1.1'])))
degrees(arctan(slope(*tunnels['MR1.1']))) - degrees(arctan(slope(*tunnels['MLhill'])))
degrees(arctan(slope(*tunnels['MR1.1']))) - degrees(arctan(slope(*horiz)))
#?raw_input
gray()
import Util
x,y = Util.dezip(tunnels['MLhill'])
x
y
plot(x,y,'w--',lw=2)
x,y = Util.dezip(tunnels['MR1.1'])
import iplot
spec = iplot(10)
spec = iplot.subspectrum(10)
plot(x,y,c=spec[0],lw=2)
tunnels['MR1.2'] = ginput(5,0)
x,y = Util.dezip(tunnels['MR1.2'])
plot(x,y,c=spec[1],lw=2)
degrees(arctan(slope(*tunnels['MR1.2'][:2]))) - degrees(arctan(slope(*tunnels['MR1.1'])))
degrees(arctan(slope(*tunnels['MR1.2'][:2]))) - degrees(arctan(slope(*tunnels['MR1.1'])))
degrees(arctan(slope(*tunnels['MR1.1']))) - degrees(arctan(slope(*horiz)))
degrees(arctan(slope(*tunnels['MR1.2'][:2]))) - degrees(arctan(slope(*horiz)))
tunnels['MR1.3'] = ginput(2,0)
tunnels['MR1.4'] = ginput(4,0)
tunnels['MR1.5'] = ginput(2,0)
x,y = Util.dezip(tunnels['MR1.3'])
plot(x,y,c=spec[2],lw=2)
x,y = Util.dezip(tunnels['MR1.4'])
plot(x,y,c=spec[3],lw=2)
x,y = Util.dezip(tunnels['MR1.5'])
plot(x,y,c=spec[4],lw=2)
def tunnel_len(tun):
    p1 = tun[0]
    tot = 0
    for p in tun[1:]:
        tot += vidtools.hypotenuse(p1,p)
        p1 = p
        
    return tot

tunnel_len(tunnels['MR1.1'])
tunnel_len(tunnels['MR1.2'])
cm_factor = mean(cm_dists.values())
tunnel_len(tunnels['MR1.1']) / cm_factor
tunnel_len(tunnels['MR1.2']) / cm_factor
tunnel_len(tunnels['MR1.3']) / cm_factor
tunnel_len(tunnels['MR1.4']) / cm_factor
tunnel_len(tunnels['MR1.5']) / cm_factor
degrees(arctan(slope(*tunnels['MR1.5'])))
slope(*tunnels['MR1.5'])
tunnels['MRhill'] = tunnels['MLhill']
tunnels['MLhill'] = ginput(2,0)
x,y = Util.dezip(tunnels['MLhill'])
plot(x,y,'w--',lw=2)
tunnels['Ml1.1'] = ginput(5,0)
x,y = Util.dezip(tunnels['Ml1.1'])
plot(x,y,c=spec[7],lw=2)
tunnels['MR1.2']
vidtools.hypotenuse(tunnels['MR1.2'][0],tunnels['MR1.2'][-1])
#?vidtools.line_fx_from_pts
vidtools.line_fx_from_pts(tunnels['MR1.2'][0],tunnels['MR1.2'][-1])
slope(tunnels['MR1.2'][0],tunnels['MR1.2'][-1])
tunnels['MR1.2'][0][0],tunnels['MR1.2'][-1][0]
max(tunnels['MR1.2'][0][0],tunnels['MR1.2'][-1][0],tunnels['MR1.2'][0][0],tunnels['MR1.2'][-1][0]) - min(tunnels['MR1.2'][0][0],tunnels['MR1.2'][-1][0],tunnels['MR1.2'][0][0],tunnels['MR1.2'][-1][0])
h = vidtools.hypotenuse(tunnels['MR1.2'][0],tunnels['MR1.2'][-1])
xdiff = max(tunnels['MR1.2'][0][0],tunnels['MR1.2'][-1][0],tunnels['MR1.2'][0][0],tunnels['MR1.2'][-1][0]) - min(tunnels['MR1.2'][0][0],tunnels['MR1.2'][-1][0],tunnels['MR1.2'][0][0],tunnels['MR1.2'][-1][0])
xstep = xdiff/h
xstep
slope(tunnels['MR1.5'][0],tunnels['MR1.5'][-1])
m,b = vidtools.line_fx_from_pts(tunnels['MR1.2'][0],tunnels['MR1.2'][-1])
lpoly = [(x,(m*x+b)) for x in xrange(tunnels['MR1.2'][0][0],tunnels['MR1.2'][-1][0],xstep)]
x
lpoly = [(x,(m*x+b)) for x in arange(tunnels['MR1.2'][0][0],tunnels['MR1.2'][-1][0],xstep)]
figure(2)
x,y = Util.dezip(lpoly)
scatter(x,y,s=1)
figure(2)
scatter(x,y,s=0.5,c=spec[1])
clf()
scatter(x,y,s=0.5,col=spec[1])
scatter(x,y,s=0.5,ec='none',fc=spec[1])
scatter(x,y,s=0.5,ec='none',facecolor=spec[1])
scatter(x,y,s=0.5,edgecolor='none',facecolor=spec[1])
scatter(x,y,s=0.5,edgecolor='none',facecolor=spec[1])
vidtools.apogee(lpoly,tunnels['MR1.2'])
[vidtools.apogee(lpoly,[p]) for p in tunnels['MR1.2']]
[(vidtools.hypotenuse(*vidtools.apogee(lpoly,[p])),vidtools.apogee(lpoly,[p])) for p in tunnels['MR1.2']]
[(vidtools.hypotenuse(*vidtools.perigee(lpoly,[p])),vidtools.perigee(lpoly,[p])) for p in tunnels['MR1.2']]
max([(vidtools.hypotenuse(*vidtools.perigee(lpoly,[p])),vidtools.perigee(lpoly,[p])) for p in tunnels['MR1.2']])
max([(vidtools.hypotenuse(*vidtools.perigee(lpoly,[p])),vidtools.perigee(lpoly,[p])) for p in tunnels['MR1.2']])[1]
chord_pt,curve_pt = max([(vidtools.hypotenuse(*vidtools.perigee(lpoly,[p])),vidtools.perigee(lpoly,[p])) for p in tunnels['MR1.2']])[1]
x,y = Util.dezip([chord_pt,curve_pt])
plot(x,y,c=spec[1],ls='--',lw=0.5)
def calc_chord_stats(tun):
    h = vidtools.hypotenuse(tun[0],tun[-1])
    xdiff = tun[-1][0] - tun[0][0]
    m,b = vidtools.line_fx_from_pts(tun[0],tun[-1])
    xstep = xdiff/h
    lpoly = [(x,(m*x+b)) for x in numpy.arange(tunnels['MR1.2'][0][0],tunnels['MR1.2'][-1][0],xstep)]
    chord_pt,curve_pt = max([(vidtools.hypotenuse(*vidtools.perigee(lpoly,[p])),vidtools.perigee(lpoly,[p])) for p in tun])[1]
    return vidtools.hypotenuse(tun[0],tun[-1]), vidtools.hypotenuse(tun[0],chord_pt), vidtools.hypotenuse(chord_pt,curve_pt)

calc_chord_stats(tunnels['MR1.2'])
def calc_chord_stats(tun,cm_factor):
    h = vidtools.hypotenuse(tun[0],tun[-1])
    xdiff = tun[-1][0] - tun[0][0]
    m,b = vidtools.line_fx_from_pts(tun[0],tun[-1])
    xstep = xdiff/h
    lpoly = [(x,(m*x+b)) for x in numpy.arange(tunnels['MR1.2'][0][0],tunnels['MR1.2'][-1][0],xstep)]
    chord_pt,curve_pt = max([(vidtools.hypotenuse(*vidtools.perigee(lpoly,[p])),vidtools.perigee(lpoly,[p])) for p in tun])[1]
    return vidtools.hypotenuse(tun[0],tun[-1])/cm_factor, vidtools.hypotenuse(tun[0],chord_pt)/cm_factor, vidtools.hypotenuse(chord_pt,curve_pt)/cm_factor

calc_chord_stats(tunnels['MR1.2'])
calc_chord_stats(tunnels['MR1.2'],cm_factor)
matshow(ar)
tun = ginput(5,0)
calc_chord_stats(tun,cm_factor)
x,y = Util.dezip(tun)
plot(x,y)
def nearest_point_on_line(lpt1,lpt2,qpt):
    h = vidtools.hypotenuse(lpt1,lpt2)
    xdiff = lpt2[0] - lpt1[0]
    m,b = vidtools.line_fx_from_pts(lpt1,lpt2)
    xstep = xdiff/h
    lpoly = [(x,(m*x+b)) for x in numpy.arange(lpt1[0],lpt2[0],xstep)]
    return vidtools.perigee(lpoly,[qpt])

def calc_chord_stats(tun,cm_factor):
    h = vidtools.hypotenuse(tun[0],tun[-1])
    xdiff = tun[-1][0] - tun[0][0]
    m,b = vidtools.line_fx_from_pts(tun[0],tun[-1])
    xstep = xdiff/h
    lpoly = [(x,(m*x+b)) for x in numpy.arange(tun[0][0],tun[-1][0],xstep)]
    chord_pt,curve_pt = max([(vidtools.hypotenuse(*vidtools.perigee(lpoly,[p])),vidtools.perigee(lpoly,[p])) for p in tun])[1]
    return vidtools.hypotenuse(tun[0],tun[-1])/cm_factor, vidtools.hypotenuse(tun[0],chord_pt)/cm_factor, vidtools.hypotenuse(chord_pt,curve_pt)/cm_factor

tunnels['MR1.2']
lpt1,lpt2 = tunnels['MR1.2'][2:4]
lpt1
lpt2
nearest_point_on_line(lpt1,lpt2,tunnels['MR1.3'][0])
def nearest_point_on_line(lpt1,lpt2,qpt):
    h = vidtools.hypotenuse(lpt1,lpt2)
    xdiff = lpt2[0] - lpt1[0]
    m,b = vidtools.line_fx_from_pts(lpt1,lpt2)
    xstep = xdiff/h
    lpoly = [(x,(m*x+b)) for x in numpy.arange(lpt1[0],lpt2[0],xstep)]
    return vidtools.apogee(lpoly,[qpt])
nearest_point_on_line(lpt1,lpt2,tunnels['MR1.3'][0])
h = vidtools.hypotenuse(lpt1,lpt2)
h
xdiff = lpt2[0] - lpt1[0]
xdiff
m,b = vidtools.line_fx_from_pts(lpt1,lpt2)
m
b
xstep = xdiff/h
xstep
lpoly = [(x,(m*x+b)) for x in numpy.arange(lpt1[0],lpt2[0],xstep)]
len(lpoly)
vidtools.apogee(lpoly,[tunnels['MR1.3'][0]])
def nearest_point_on_line(lpt1,lpt2,qpt):
    h = vidtools.hypotenuse(lpt1,lpt2)
    xdiff = lpt2[0] - lpt1[0]
    m,b = vidtools.line_fx_from_pts(lpt1,lpt2)
    xstep = xdiff/h
    lpoly = [(x,(m*x+b)) for x in numpy.arange(lpt1[0],lpt2[0],xstep)]
    return vidtools.perigee(lpoly,[qpt])

close(1)
nearest_point_on_line(lpt1,lpt2,tunnels['MR1.4'][0])
def nearest_point_on_curve(curve,qpt):
    lseg_dists = []
    p1 = curve[0]
    for p in curve[1:]:
        lseg_dists.append((nearest_point_on_line(p1,p,qpt),(p1,p)))
        p1 = p
    return lseg_dists

nearest_point_on_curve(tunnels['MR1.2'],tunnels['MR1.4'][0])
def nearest_point_on_curve(curve,qpt):
    lseg_dists = []
    p1 = curve[0]
    for p in curve[1:]:
        lseg_dists.append((vidtools.hypotenuse(*nearest_point_on_line(p1,p,qpt)),(p1,p)))
        p1 = p
    return lseg_dists

nearest_point_on_curve(tunnels['MR1.2'],tunnels['MR1.4'][0])
def nearest_point_on_curve(curve,qpt):
    lseg_dists = []
    p1 = curve[0]
    for p in curve[1:]:
        lseg_dists.append((vidtools.hypotenuse(*nearest_point_on_line(p1,p,qpt)),(p1,p)))
        p1 = p
    return min(lseg_dists)

nearest_point_on_curve(tunnels['MR1.2'],tunnels['MR1.4'][0])
def nearest_point_on_curve(curve,qpt):
    lseg_dists = []
    p1 = curve[0]
    for p in curve[1:]:
        lseg_dists.append((vidtools.hypotenuse(*nearest_point_on_line(p1,p,qpt)),(p1,p)))
        p1 = p
    return min(lseg_dists)[-1]

nearest_point_on_curve(tunnels['MR1.2'],tunnels['MR1.4'][0])
def nearest_point_on_curve(curve,qpt):
    lseg_dists = []
    p1 = curve[0]
    for p in curve[1:]:
        lseg_dists.append((vidtools.hypotenuse(*nearest_point_on_line(p1,p,qpt)),((p1,p),nearest_point_on_line(p1,p,qpt)[0])))
        p1 = p
    return min(lseg_dists)[-1]
def nearest_point_on_line(lpt1,lpt2,qpt):
    h = vidtools.hypotenuse(lpt1,lpt2)
    xdiff = lpt2[0] - lpt1[0]
    m,b = vidtools.line_fx_from_pts(lpt1,lpt2)
    xstep = xdiff/h
    lpoly = [(x,(m*x+b)) for x in numpy.arange(lpt1[0],lpt2[0],xstep)]
    return vidtools.perigee(lpoly,[qpt])
nearest_point_on_curve(tunnels['MR1.2'],tunnels['MR1.4'][0])
def nearest_point_on_curve(curve,qpt):
    lseg_dists = []
    p1 = curve[0]
    for p in curve[1:]:
        lseg_dists.append((vidtools.hypotenuse(*nearest_point_on_line(p1,p,qpt)),((p1,p),nearest_point_on_line(p1,p,qpt)[0])))
        p1 = p
    return min(lseg_dists)[-1]

nearest_point_on_curve(tunnels['MR1.2'],tunnels['MR1.4'][0])
def nearest_point_on_curve(curve,qpt):
    lseg_dists = []
    p1 = curve[0]
    for i,p in enumerate(curve[1:]):
        lseg_dists.append((vidtools.hypotenuse(*nearest_point_on_line(p1,p,qpt)),(i,nearest_point_on_line(p1,p,qpt)[0])))
        p1 = p
    return min(lseg_dists)[-1]

nearest_point_on_curve(tunnels['MR1.2'],tunnels['MR1.4'][0])
tunnels['MR1.2'][2:4]
def tunnel_len(tun):
    p1 = tun[0]
    tot = 0
    for p in tun[1:]:
        tot += vidtools.hypotenuse(p1,p)
        p1 = p
    return tot
tunnel_len(tunnels['MR1.2'][:3]) + vidtools.hypotenuse(tunnels['MR1.2'][2],(1296.195241349966, 391.82083234730317))
(tunnel_len(tunnels['MR1.2'][:3]) + vidtools.hypotenuse(tunnels['MR1.2'][2],(1296.195241349966, 391.82083234730317))) / cm_factor
nearest_point_on_curve(tunnels['MR1.5'],[(1220,320)])
def nearest_point_on_line(lpt1,lpt2,qpt):
    h = vidtools.hypotenuse(lpt1,lpt2)
    xdiff = lpt2[0] - lpt1[0]
    m,b = vidtools.line_fx_from_pts(lpt1,lpt2)
    xstep = xdiff/h
    lpoly = [(x,(m*x+b)) for x in numpy.arange(lpt1[0],lpt2[0],xstep)]
    return vidtools.perigee(lpoly,[qpt])
m,b = vidtools.line_fx_from_pts(*tunnels['MR1.5'])
m.b
m,b
lpt1,lpt2 = tunnels['MR1.5']
h = vidtools.hypotenuse(lpt1,lpt2)
xdiff = lpt2[0] - lpt1[0]
m,b = vidtools.line_fx_from_pts(lpt1,lpt2)
xstep = xdiff/h 
lpoly = [(x,(m*x+b)) for x in numpy.arange(lpt1[0],lpt2[0],xstep)]
h
xdiff
xstep
m
b
matshow(ar)
x,y = Util.dezip(lpoly)
scatter(x,y,s=1,edgecolor='none',facecolor=spec[1])
nearest_point_on_curve(tunnels['MR1.2'],tunnels['MR1.4'][0])
i,near_pt = nearest_point_on_curve(tunnels['MR1.2'],tunnels['MR1.4'][0])
degrees(arctan(slope(*tunnels['MR1.2'][i:i+2]))) - degrees(arctan(slope(*tunnels['MR1.4'][:2])))
degrees(arctan(slope(*tunnels['MR1.4'][:2]))) - degrees(arctan(slope(*tunnels['MR1.2'][i:i+2])))
i,near_pt = nearest_point_on_curve(tunnels['MR1.2'],tunnels['MR1.3'][0])
i
near_pt
tot = len(BW_lines)*len(BW_lines)
line
tunnel_len(tunnels['MR1.2'][:i+1]) + vidtools.hypotenuse(tunnels['MR1.2'][i],near_pt)
tunnel_len(tunnels['MR1.2'][:i+1]) + vidtools.hypotenuse(tunnels['MR1.2'][i],near_pt) / cm_factor
(tunnel_len(tunnels['MR1.2'][:i+1]) + vidtools.hypotenuse(tunnels['MR1.2'][i],near_pt)) / cm_factor
degrees(arctan(slope(*tunnels['MR1.3'][:2]))) - degrees(arctan(slope(*tunnels['MR1.2'][i:i+2])))
degrees(arctan(slope(*tunnels['MR1.3'][:2]))) - degrees(arctan(slope(*horiz)))
ar = numpy.asarray(Image.open(im).convert(mode='L'))
def slope(p1,p2):
    return (p2[1]-p1[1]) / (p2[0]-p1[0])
def get_calibration():
    raw_input('zoom to first scale bar, press any key then click 0, 1, 10 cm marks')
    p0,p1,p10 = pylab.ginput(3,0)
    
cm_dists['rt10'] = vidtools.hypotenuse(p0,p10)/10
def get_calibration():
    cm_dists = []
    raw_input('zoom to first scale bar, press any key then click 0, 1, 10 cm marks')
    p0,p1,p10 = pylab.ginput(3,0)
    cm_dists.append(vidtools.hypotenuse(p0,p1))
    cm_dists.append(vidtools.hypotenuse(p0,p10)/10.0)
    raw_input('zoom to second scale bar, press any key then click 0, 1, 10 cm marks')
    p0,p1,p10 = pylab.ginput(3,0)
    cm_dists.append(vidtools.hypotenuse(p0,p1))
    cm_dists.append(vidtools.hypotenuse(p0,p10)/10.0)
    return cm_dists
matshow(ar)
get_calibration 
get_calibration()
xlim(0,400)
ylim(800,960)
ylim(960,800)
def get_calibration():
    cm_dists = []
    print >> sys.stderr, 'click 0, 1, 10 cm marks'
    xlim(0,400)
    ylim(960,800)
    p0,p1,p10 = pylab.ginput(3,0)
    cm_dists.append(vidtools.hypotenuse(p0,p1))
    cm_dists.append(vidtools.hypotenuse(p0,p10)/10.0)
    print >> sys.stderr, 'click 0, 1, 10 cm marks'
    xlim(0,400)
    ylim(960,800)
    p0,p1,p10 = pylab.ginput(3,0)
    cm_dists.append(vidtools.hypotenuse(p0,p1))
    cm_dists.append(vidtools.hypotenuse(p0,p10)/10.0)
    return cm_dists
def get_calibration():
    cm_dists = []
    print >> sys.stderr, 'click 0, 1, 10 cm marks'
    xlim(0,400)
    ylim(960,800)
    p0,p1,p10 = pylab.ginput(3,0)
    cm_dists.append(vidtools.hypotenuse(p0,p1))
    cm_dists.append(vidtools.hypotenuse(p0,p10)/10.0)
    print >> sys.stderr, 'click 0, 1, 10 cm marks'
    xlim(0,400)
    ylim(960,800)
    p0,p1,p10 = pylab.ginput(3,0)
    cm_dists.append(vidtools.hypotenuse(p0,p1))
    cm_dists.append(vidtools.hypotenuse(p0,p10)/10.0)
    return cm_dists
close(2)
matshow(ar)
get_calibration()
def get_calibration():
    cm_dists = []
    print >> sys.stderr, 'click 0, 1, 10 cm marks'
    xlim(0,400)
    ylim(960,800)
    p0,p1,p10 = pylab.ginput(3,0)
    cm_dists.append(vidtools.hypotenuse(p0,p1))
    cm_dists.append(vidtools.hypotenuse(p0,p10)/10.0)
    print >> sys.stderr, 'click 0, 1, 10 cm marks'
    xlim(0,400)
    ylim(960,800)
    p0,p1,p10 = pylab.ginput(3,0)
    cm_dists.append(vidtools.hypotenuse(p0,p1))
    cm_dists.append(vidtools.hypotenuse(p0,p10)/10.0)
    return cm_dists
def get_calibration():
    cm_dists = []
    print >> sys.stderr, 'click 0, 1, 10 cm marks'
    xlim(0,400)
    ylim(960,800)
    p0,p1,p10 = pylab.ginput(3,0)
    cm_dists.append(vidtools.hypotenuse(p0,p1))
    cm_dists.append(vidtools.hypotenuse(p0,p10)/10.0)
    print >> sys.stderr, 'click 0, 1, 10 cm marks'
    xlim(0,400)
    ylim(960,800)
    p0,p1,p10 = pylab.ginput(3,0)
    cm_dists.append(vidtools.hypotenuse(p0,p1))
    cm_dists.append(vidtools.hypotenuse(p0,p10)/10.0)
    return cm_dists
matshow(ar)
def get_calibration():
    cm_dists = []
    print >> sys.stderr, 'click 0, 1, 10 cm marks'
    xlim(0,400)
    ylim(960,800)
    p0,p1,p10 = pylab.ginput(3,0)
    cm_dists.append(vidtools.hypotenuse(p0,p1))
    cm_dists.append(vidtools.hypotenuse(p0,p10)/10.0)
    print >> sys.stderr, 'click 0, 1, 10 cm marks'
    xlim(1450,1600)
    ylim(300,0)
    p0,p1,p10 = pylab.ginput(3,0)
    cm_dists.append(vidtools.hypotenuse(p0,p1))
    cm_dists.append(vidtools.hypotenuse(p0,p10)/10.0)
    return cm_dists

def get_calibration():
    cm_dists = []
    print >> sys.stderr, 'click 0, 1, 10 cm marks'
    xlim(0,400)
    ylim(960,800)
    p0,p1,p10 = pylab.ginput(3,0)
    cm_dists.append(vidtools.hypotenuse(p0,p1))
    cm_dists.append(vidtools.hypotenuse(p0,p10)/10.0)
    print >> sys.stderr, 'click 0, 1, 10 cm marks'
    xlim(1450,1600)
    ylim(300,0)
    p0,p1,p10 = pylab.ginput(3,0)
    cm_dists.append(vidtools.hypotenuse(p0,p1))
    cm_dists.append(vidtools.hypotenuse(p0,p10)/10.0)
    xlim(0,1600)
    ylim(960,0)
    return cm_dists
get_calibration()
def get_calibration():
    cm_dists = []
    print >> sys.stderr, 'click 0, 1, 10 cm marks'
    xlim(0,400)
    ylim(960,800)
    p0,p1,p10 = pylab.ginput(3,0)
    cm_dists.append(vidtools.hypotenuse(p0,p1))
    cm_dists.append(vidtools.hypotenuse(p0,p10)/10.0)
    print >> sys.stderr, 'click 0, 1, 10 cm marks'
    xlim(1450,1600)
    ylim(300,0)
    p0,p1,p10 = pylab.ginput(3,0)
    cm_dists.append(vidtools.hypotenuse(p0,p1))
    cm_dists.append(vidtools.hypotenuse(p0,p10)/10.0)
    xlim(0,1600)
    ylim(960,0)
    print >> sys.stderr, 'click bolt'
    xlim(0,200)
    ylim(960,800)
    print >> sys.stderr, 'click bolt 2'
    xlim(1450,1600)
    ylim(960,800)
    
def get_calibration():
    cm_dists = []
    print >> sys.stderr, 'click 0, 1, 10 cm marks'
    xlim(0,400)
    ylim(960,800)
    p0,p1,p10 = pylab.ginput(3,0)
    cm_dists.append(vidtools.hypotenuse(p0,p1))
    cm_dists.append(vidtools.hypotenuse(p0,p10)/10.0)
    print >> sys.stderr, 'click 0, 1, 10 cm marks'
    xlim(1450,1600)
    ylim(300,0)
    p0,p1,p10 = pylab.ginput(3,0)
    cm_dists.append(vidtools.hypotenuse(p0,p1))
    cm_dists.append(vidtools.hypotenuse(p0,p10)/10.0)
    xlim(0,1600)
    ylim(960,0)
    print >> sys.stderr, 'click bolt'
    xlim(0,200)
    ylim(960,800)
    horiz = [ginput(1,0)]
    print >> sys.stderr, 'click bolt 2'
    xlim(1450,1600)
    ylim(960,800)
    horiz.append(ginput(1,0))
    xlim(0,1600)
    ylim(960,0)
    return cm_dists,horiz

def get_calibration():
    cm_dists = []
    print >> sys.stderr, 'click 0, 1, 10 cm marks'
    xlim(0,400)
    ylim(960,800)
    p0,p1,p10 = pylab.ginput(3,0)
    cm_dists.append(vidtools.hypotenuse(p0,p1))
    cm_dists.append(vidtools.hypotenuse(p0,p10)/10.0)
    print >> sys.stderr, 'click 0, 1, 10 cm marks'
    xlim(1450,1600)
    ylim(300,0)
    p0,p1,p10 = pylab.ginput(3,0)
    cm_dists.append(vidtools.hypotenuse(p0,p1))
    cm_dists.append(vidtools.hypotenuse(p0,p10)/10.0)
    print >> sys.stderr, 'click bolt 1'
    xlim(0,200)
    ylim(960,800)
    horiz = [ginput(1,0)]
    print >> sys.stderr, 'click bolt 2'
    xlim(1450,1600)
    ylim(960,800)
    horiz.append(ginput(1,0))
    xlim(0,1600)
    ylim(960,0)
    return cm_dists,horiz
get_calibration()
get_calibration()
degrees(arctan(slope(*tunnels['MR1.1'][:2]))) - degrees(arctan(slope(*horiz)))
degrees(arctan(slope(*tunnels['MR1.1'][:2]))) - degrees(arctan(slope(*tunnels['MRhill'])))
im = '20120319/af5_PO-31.jpg'
ar = numpy.asarray(Image.open(im).convert(mode='L'))
matshow(ar)
get_calibration()
get_calibration()
cm_pts = _[0]
cm_pts
mean(cm_pts)
std(cm_pts)
def get_calibration(pct_wide,pct_high):
    cm_dists = []
    print >> sys.stderr, 'click 0, 1, 10 cm marks'
    xlim(0,400)
    ylim(960,800)
    p0,p1,p10 = pylab.ginput(3,0)
    cm_dists.append(vidtools.hypotenuse(p0,p1))
    cm_dists.append(vidtools.hypotenuse(p0,p10)/10.0)
    print >> sys.stderr, 'click 0, 1, 10 cm marks'
    xlim(1450,1600)
    ylim(300,0)
    p0,p1,p10 = pylab.ginput(3,0)
    cm_dists.append(vidtools.hypotenuse(p0,p1))
    cm_dists.append(vidtools.hypotenuse(p0,p10)/10.0)
    print >> sys.stderr, 'click bolt 1'
    xlim(0,200)
    ylim(960,800)
    horiz = [ginput(1,0)]
    print >> sys.stderr, 'click bolt 2'
    xlim(1450,1600)
    ylim(960,800)
    horiz.append(ginput(1,0))
    xlim(0,1600)
    ylim(960,0)
    return cm_dists,horiz
def get_calibration(pct_wide,pct_high):
    cm_dists = []
    print >> sys.stderr, 'click 0, 1, 10 cm marks'
    xlim(0,400)
    ylim(960,800)
    p0,p1,p10 = pylab.ginput(3,0)
    cm_dists.append(vidtools.hypotenuse(p0,p1))
    cm_dists.append(vidtools.hypotenuse(p0,p10)/10.0)
    print >> sys.stderr, 'click 0, 1, 10 cm marks'
    xlim(1450,1600)
    ylim(300,0)
    p0,p1,p10 = pylab.ginput(3,0)
    cm_dists.append(vidtools.hypotenuse(p0,p1))
    cm_dists.append(vidtools.hypotenuse(p0,p10)/10.0)
    print >> sys.stderr, 'click bolt 1'
    xlim(0,200)
    ylim(960,800)
    horiz = [ginput(1,0)]
    print >> sys.stderr, 'click bolt 2'
    xlim(1450,1600)
    ylim(960,800)
    horiz.append(ginput(1,0))
    xlim(0,1600)
    ylim(960,0)
    return cm_dists,horiz
960/4
960-240
def get_calibration(pct_wide,pct_high):
    cm_dists = []
    print >> sys.stderr, 'click 0, 1, 10 cm marks'
    xlim(0,400)
    ylim(960,800)
    p0,p1,p10 = pylab.ginput(3,0)
    cm_dists.append(vidtools.hypotenuse(p0,p1))
    cm_dists.append(vidtools.hypotenuse(p0,p10)/10.0)
    print >> sys.stderr, 'click 0, 1, 10 cm marks'
    xlim(1450,1600)
    ylim(300,0)
    p0,p1,p10 = pylab.ginput(3,0)
    cm_dists.append(vidtools.hypotenuse(p0,p1))
    cm_dists.append(vidtools.hypotenuse(p0,p10)/10.0)
    print >> sys.stderr, 'click bolt 1'
    xlim(0,200)
    ylim(960,800)
    horiz = [ginput(1,0)]
    print >> sys.stderr, 'click bolt 2'
    xlim(1450,1600)
    ylim(960,800)
    horiz.append(ginput(1,0))
    xlim(0,1600)
    ylim(960,0)
    return cm_dists,horiz
960*0.15
matshow(ar)
ylim()
xlim()
[round(v) for v in ylim()]
[round(v) for v in ylim()]
ymax()
close(2)
close(3)
close(4)
ax = matshow(ar)
ax.get_size()
ax = figure().axes[0]
ax = figure(2).axes[0]
ax.get_size()
len(figure(2).axes[0])
len(figure(2).axes)
figure(2)
figure(2).axes
figure(2).axes[0]
ax = figure(2).axes[0]
ax.get_ylim 
ax.get_ylim()
ax.get_ybound()
round(ax.get_ybound()[1])
#?round
round(ax.get_xbound()[1])
from video_analysis import measure_burrows_from_image
from video_analysis import measure_burrows_from_image
measure_burrows_from_image(2)
measure_burrows_from_image.get_calibration(2)
reload(measure_burrows_from_image)
measure_burrows_from_image.get_calibration(2)
reload(measure_burrows_from_image)
measure_burrows_from_image.get_calibration(2)
reload(measure_burrows_from_image)
measure_burrows_from_image.get_calibration(2)
matshow(ar)
measure_burrows_from_image.get_calibration(2)
reload(measure_burrows_from_image)
cm_pts, horiz = _624
horiz
x,y = Util.dezip(horiz)
reload(measure_burrows_from_image)
cm_pts,horiz = measure_burrows_from_image.get_calibration(2)
x,y = Util.dezip(horiz)
plot(x,y,'k--')
tunnels2 = {}
reload(measure_burrows_from_image)
reload(measure_burrows_from_image)
tunnel_oris = {}
measure_burrows_from_image.add_straight_tunnel(tunnels2,tunnel_oris,'MRhill')
measure_burrows_from_image.add_straight_tunnel(tunnels2,tunnel_oris,'MR1.1','MRhill')
measure_burrows_from_image.add_curved_tunnel(tunnels2,tunnel_oris,'MR1.2','MR1.1')
measure_burrows_from_image.add_curved_tunnel(tunnels2,tunnel_oris,'MR1.2','MR1.1')
measure_burrows_from_image.add_straight_tunnel(tunnels2,tunnel_oris,'MR1.3','MR1.2')
tunnel_oris
tunnels2
for c,k in iplot.subspec_enum(tunnels2):
    x,y = Util.dezip(tunnels2[k])
    if not '.' in k:
        plot(x,y,'w--',lw=2)
    else:
        plot(x,y,c=c,lw=2)
        

figure(2).axes[0].lines
figure(2).axes[0].lines = []
for c,k in iplot.subspec_enum(tunnels2):
    x,y = Util.dezip(tunnels2[k])
    if not '.' in k:
        plot(x,y,'w--',lw=2)
    else:
        plot(x,y,c=c,lw=2)
        

measure_burrows_from_image.add_curved_tunnel(tunnels2,tunnel_oris,'MR1.2','MR1.1')
figure(2).axes[0].lines = []
for c,k in iplot.subspec_enum(tunnels2):
    x,y = Util.dezip(tunnels2[k])
    if not '.' in k:
        plot(x,y,'w--',lw=2)
    else:
        plot(x,y,c=c,lw=2)
        

measure_burrows_from_image.add_straight_tunnel(tunnels2,tunnel_oris,'MLhill')
measure_burrows_from_image.add_straight_tunnel(tunnels2,tunnel_oris,'ML1.1','MLhill')
measure_burrows_from_image.add_straight_tunnel(tunnels2,tunnel_oris,'HLhill')
measure_burrows_from_image.add_curved_tunnel(tunnels2,tunnel_oris,'HL1.1','HLhill')
measure_burrows_from_image.add_straight_tunnel(tunnels2,tunnel_oris,'HL1.2','HL1.1')
figure(2).axes[0].lines = []
for c,k in iplot.subspec_enum(tunnels2):
    x,y = Util.dezip(tunnels2[k])
    if not '.' in k:
        plot(x,y,'w--',lw=2)
    else:
        plot(x,y,c=c,lw=2)
        

reload(measure_burrows_from_image)
figure(2).axes[0].lines = []
measure_burrows_from_image.plot_tunnels(tunnels2,horiz)
measure_burrows_from_image.add_straight_tunnel(tunnels2,tunnel_oris,'HRhill')
measure_burrows_from_image.add_straight_tunnel(tunnels2,tunnel_oris,'HR1.1','HRhill')
figure(2).axes[0].lines = []
measure_burrows_from_image.plot_tunnels(tunnels2,horiz)
reload(measure_burrows_from_image)
reload(measure_burrows_from_image)
figure(2).axes[0].lines = []
measure_burrows_from_image.plot_tunnels(tunnels2,horiz)
measure_burrows_from_image.nearest_point_on_line(horiz[0],horiz[1],tunnels2['MR1.1'][0])
x,y = Util.dezip(measure_burrows_from_image.nearest_point_on_line(horiz[0],horiz[1],tunnels2['MR1.1'][0]))
plot(x,y,c='lightgray',ls='--'
)

plot(x,y,c='gray',ls='--')
plot()
figure(2).axes[0].lines = []
plot()
close(1)
close(2)
matshow(ar)
measure_burrows_from_image.plot_tunnels(tunnels2,horiz)
reload(measure_burrows_from_image)
close(1)
matshow(ar)
measure_burrows_from_image.plot_tunnels(tunnels2,horiz)
x,y = Util.dezip(measure_burrows_from_image.nearest_point_on_line(horiz[0],horiz[1],tunnels2['MR1.1'][0]))
plot(x,y,c='gray',ls='--')
x
y
nearest_point_on_curve(tunnels['MR1.2'],tunnels['MR1.4'][0])
measure_burrows_from_image.nearest_point_on_line(horiz[0],horiz[1],tunnels2['MR1.1'][0])
x,y = Util.dezip(list(measure_burrows_from_image.nearest_point_on_line(horiz[0],horiz[1],tunnels2['MR1.1'][0])))
x
plot(x,y,c='gray',ls='--')
reload(measure_burrows_from_image)
figure(1).axes[0].lines = []
measure_burrows_from_image.plot_tunnels(tunnels2,horiz)
plot(x,y,c='gray',ls='--')
plot(x,y,c='ltgray',ls='--')

plot(x,y,c='darkgray',ls='--')
clf()
close(1)
matshow(ar)
measure_burrows_from_image.plot_tunnels(tunnels2,horiz)
plot(x,y,c='darkgray',ls='--')
x,y = Util.dezip(list(measure_burrows_from_image.nearest_point_on_line(horiz[0],horiz[1],tunnels2['MR1.1'][0])))
measure_burrows_from_image.nearest_point_on_line(horiz[0],horiz[1],tunnels2['MR1.1'][0])
vidtools.hypotenuse(*measure_burrows_from_image.nearest_point_on_line(horiz[0],horiz[1],tunnels2['MR1.1'][0]))
vidtools.hypotenuse(*measure_burrows_from_image.nearest_point_on_line(horiz[0],horiz[1],tunnels2['MR1.1'][0]))/
cm
cm_factor = mean(cm_pts)
vidtools.hypotenuse(*measure_burrows_from_image.nearest_point_on_line(horiz[0],horiz[1],tunnels2['MR1.1'][0]))/ cm_factor
vidtools.hypotenuse(*measure_burrows_from_image.nearest_point_on_line(horiz[0],horiz[1],tunnels2['MR1.1'][1]))/ cm_factor
vidtools.hypotenuse(*measure_burrows_from_image.nearest_point_on_line(horiz[0],horiz[1],tunnels2['MR1.1'][-1]))/ cm_factor
round(vidtools.hypotenuse(*measure_burrows_from_image.nearest_point_on_line(horiz[0],horiz[1],tunnels2['MR1.1'][-1]))/ cm_factor,2)
round(vidtools.hypotenuse(*measure_burrows_from_image.nearest_point_on_line(horiz[0],horiz[1],tunnels2['MR1.1'][-1]))/ cm_factor,1)
reload(measure_burrows_from_image)
reload(measure_burrows_from_image)
measure_burrows_from_image.calc_tunnel_H(tunnels2,this_name,cm_factor)
measure_burrows_from_image.calc_tunnel_H(tunnels2,'MR1.1',cm_factor)
reload(measure_burrows_from_image)
measure_burrows_from_image.calc_tunnel_H(tunnels2,'MR1.1',horiz,cm_factor)
measure_burrows_from_image.calc_tunnel_H('MR1.1',tunnels2,horiz,cm_factor)
reload(measure_burrows_from_image)
measure_burrows_from_image.calc_tunnel_H('MR1.1',tunnels2,horiz,cm_factor)
measure_burrows_from_image.calc_tunnel_H('MR1.2',tunnels2,horiz,cm_factor)
measure_burrows_from_image.calc_tunnel_H('MR1.3',tunnels2,horiz,cm_factor)
measure_burrows_from_image.calc_tunnel_H('MR1.4',tunnels2,horiz,cm_factor)
degrees(arctan(slope(*tunnels['MR1.1'][:2]))) - degrees(arctan(slope(*tunnels['MRhill'])))
reload(measure_burrows_from_image)
measure_burrows_from_image.calc_tunnel_angles('MR1.1',tunnels2,tunnel_oris,horiz)
measure_burrows_from_image.calc_tunnel_angles('MR1.2',tunnels2,tunnel_oris,horiz)
reload(measure_burrows_from_image)
measure_burrows_from_image.calc_tunnel_angles('MR1.2',tunnels2,tunnel_oris,horiz)
measure_burrows_from_image.calc_tunnel_angles('MR1.3',tunnels2,tunnel_oris,horiz)
tunnel_oris
#?measure_burrows_from_image.nearest_point_on_curve
measure_burrows_from_image.nearest_point_on_curve(tunnels2[tunnel_oris['MR1.3']],tunnels2['MR1.3'])
measure_burrows_from_image.nearest_point_on_curve(tunnels2[tunnel_oris['MR1.3']],tunnels2['MR1.3'][0])
tunnels2['MR1.3']
tunnels2['MR1.3'][0]
reload(measure_burrows_from_image)
measure_burrows_from_image.calc_tunnel_angles('MR1.3',tunnels2,tunnel_oris,horiz)
def
calc_chord_stats(tunnels2['MR1.2'],cm_factor)

(tunnel_len(tunnels['MR1.2'][:i+1]) + vidtools.hypotenuse(tunnels['MR1.2'][i],near_pt)) / cm_factor
nearest_point_on_line(lpt1,lpt2,tunnels['MR1.3'][0])
i,near_pt = nearest_point_on_curve(tunnels['MR1.2'],tunnels['MR1.3'][0])

reload(measure_burrows_from_image)
reload(measure_burrows_from_image)
measure_burrows_from_image.calc_ori_pt('MR1.2',tunnels2,tunnel_oris,cm_factor)
measure_burrows_from_image.tunnel_len(tunnels2['MR1.1'],cm_factor)
measure_burrows_from_image.calc_ori_pt('MR1.3',tunnels2,tunnel_oris,cm_factor)
measure_burrows_from_image.tunnel_len(tunnels2['MR1.2'],cm_factor)
reload(measure_burrows_from_image)
measure_burrows_from_image('MR1.1',tunnels2,tunnel_oris,horiz,cm_factor)
measure_burrows_from_image.calc_tunnel_properties('MR1.1',tunnels2,tunnel_oris,horiz,cm_factor)
reload(measure_burrows_from_image)
measure_burrows_from_image.calc_tunnel_properties('MR1.1',tunnels2,tunnel_oris,horiz,cm_factor)
measure_burrows_from_image.calc_tunnel_properties('MR1.2',tunnels2,tunnel_oris,horiz,cm_factor)
reload(measure_burrows_from_image)
measure_burrows_from_image.calc_tunnel_properties('MR1.2',tunnels2,tunnel_oris,horiz,cm_factor)
measure_burrows_from_image.calc_tunnel_properties('MR1.3',tunnels2,tunnel_oris,horiz,cm_factor)
reload(measure_burrows_from_image)
measure_burrows_from_image.calc_tunnel_properties('MR1.1',tunnels2,tunnel_oris,horiz,cm_factor)
measure_burrows_from_image.calc_tunnel_properties('MR1.2',tunnels2,tunnel_oris,horiz,cm_factor)
measure_burrows_from_image.calc_tunnel_properties('MR1.3',tunnels2,tunnel_oris,horiz,cm_factor)
measure_burrows_from_image.calc_tunnel_properties('ML1.1',tunnels2,tunnel_oris,horiz,cm_factor)
measure_burrows_from_image.calc_tunnel_properties('HL1.1',tunnels2,tunnel_oris,horiz,cm_factor)
tunnels2
horiz
reload(measure_burrows_from_image)
measure_burrows_from_image.calc_tunnel_properties('HL1.1',tunnels2,tunnel_oris,horiz,cm_factor)
measure_burrows_from_image.calc_tunnel_properties('ML1.1',tunnels2,tunnel_oris,horiz,cm_factor)
measure_burrows_from_image.calc_tunnel_properties('MR1.1',tunnels2,tunnel_oris,horiz,cm_factor)
tunnels
measure_burrows_from_image.calc_tunnel_properties('MR1.1',tunnels2,tunnel_oris,horiz,cm_factor)
tunnel_oris_fake = {'MR1.5': 'MR1.2'}
measure_burrows_from_image.calc_tunnel_properties('MR1.5',tunnels,tunnel_oris_fake,horiz,cm_factor)
#?measure_burrows_from_image.add_straight_tunnel
measure_burrows_from_image.add_straight_tunnel(tunnels2,tunnel_oris,'MR1.4fake','MR1.2')
measure_burrows_from_image.calc_tunnel_properties('MR1.4fake',tunnels2,tunnel_oris,horiz,cm_factor)
degrees(arctan(slope(*horiz)))
degrees(arctan(slope(*reversed(horiz))))
reload(measure_burrows_from_image)
measure_burrows_from_image.calc_tunnel_properties('MR1.1',tunnels2,tunnel_oris,horiz,cm_factor)
measure_burrows_from_image.calc_tunnel_properties('MR1.2',tunnels2,tunnel_oris,horiz,cm_factor)
measure_burrows_from_image.calc_tunnel_properties('MR1.3',tunnels2,tunnel_oris,horiz,cm_factor)
measure_burrows_from_image.calc_tunnel_properties('MR1.4fake',tunnels2,tunnel_oris,horiz,cm_factor)
measure_burrows_from_image.plot_tunnels(tunnels2,horiz)
measure_burrows_from_image.calc_tunnel_properties('ML1.1',tunnels2,tunnel_oris,horiz,cm_factor)
measure_burrows_from_image.calc_tunnel_properties('HL1.1',tunnels2,tunnel_oris,horiz,cm_factor)
measure_burrows_from_image.calc_tunnel_properties('HL1.2',tunnels2,tunnel_oris,horiz,cm_factor)
im = '20120319/af6_PO-66.jpg'
matshow(ar)
close(2)
ar = numpy.asarray(Image.open(im).convert(mode='L'))
matshow(ar)
measure_burrows_from_image.calc_tunnel_properties('HL1.2',tunnels2,tunnel_oris,horiz,cm_factor)
#?measure_burrows_from_image.get_calibration
measure_burrows_from_image.get_calibration(2)
cm_pts,horiz = _
cm_factor = mean(cm_pts)
tunnels = {}
tunnel_oris = {}
#?measure_burrows_from_image.add_straight_tunnel
measure_burrows_from_image.add_straight_tunnel(tunnels,tunnel_oris,'MRhill')
measure_burrows_from_image.add_straight_tunnel(tunnels,tunnel_oris,'MR1.1','MRhill')
measure_burrows_from_image.add_curved_tunnel(tunnels,tunnel_oris,'MR1.2','MR1.1')
measure_burrows_from_image.add_straight_tunnel(tunnels,tunnel_oris,'MR1.3','MR1.2')
measure_burrows_from_image.plot_tunnels(tunnels,horiz)
#?measure_burrows_from_image.plot_tunnels
measure_burrows_from_image.plot_tunnels(tunnels,horiz,2)
measure_burrows_from_image.add_straight_tunnel(tunnels,tunnel_oris,'MR1.1','MRhill')
measure_burrows_from_image.add_curved_tunnel(tunnels,tunnel_oris,'MR1.2','MR1.1')
figure(2).axes[0].lines = []
measure_burrows_from_image.plot_tunnels(tunnels,horiz,2)
measure_burrows_from_image.calc_tunnel_properties('MR1.1',tunnels,tunnel_oris,horiz,cm_factor)
measure_burrows_from_image.calc_tunnel_properties('MR1.2',tunnels,tunnel_oris,horiz,cm_factor)
measure_burrows_from_image.add_straight_tunnel(tunnels,tunnel_oris,'MLhill')
measure_burrows_from_image.add_straight_tunnel(tunnels,tunnel_oris,'ML1.1','MLhill')
measure_burrows_from_image.add_curved_tunnel(tunnels,tunnel_oris,'ML1.2','ML1.1')
measure_burrows_from_image.add_straight_tunnel(tunnels,tunnel_oris,'ML2.1','MLhill')
measure_burrows_from_image.add_straight_tunnel(tunnels,tunnel_oris,'ML2.1','MLhill')
measure_burrows_from_image.add_curved_tunnel(tunnels,tunnel_oris,'ML2.1','MLhill')
measure_burrows_from_image.plot_tunnels(tunnels,horiz,2)
[(k,measure_burrows_from_image.calc_tunnel_properties(k,tunnels,tunnel_oris,horiz,cm_factor)) for k in tunnels if '.' in k]
[(k,measure_burrows_from_image.calc_tunnel_properties(k,tunnels,tunnel_oris,horiz,cm_factor)) for k in sorted(tunnels) if '.' in k]
im = '20120319/af7_PO-04.jpg'
ar = numpy.asarray(Image.open(im).convert(mode='L'))
close(1)
matshow(ar)
close(3)
close(2)
matshow(ar)
measure_burrows_from_image.get_calibration()
cm_pts,horiz = _
cm_factor = mean(cm_pts)
measure_burrows_from_image.add_curved_tunnel(tunnels,tunnel_oris,'MRhill')

measure_burrows_from_image.add_straight_tunnel(tunnels,tunnel_oris,'MRhill')
measure_burrows_from_image.add_straight_tunnel(tunnels,tunnel_oris,'MR1.1','MRhill')
measure_burrows_from_image.add_curved_tunnel(tunnels,tunnel_oris,'MR1.2','MR1.1')
measure_burrows_from_image.add_straight_tunnel(tunnels,tunnel_oris,'MR1.3','MR1.2')
measure_burrows_from_image.add_straight_tunnel(tunnels,tunnel_oris,'MR1.4','MR1.2')
measure_burrows_from_image.plot_tunnels(tunnels,horiz,2)
measure_burrows_from_image.plot_tunnels(tunnels,horiz,1)
[(k,measure_burrows_from_image.calc_tunnel_properties(k,tunnels,tunnel_oris,horiz,cm_factor)) for k in sorted(tunnels) if '.' in k and 'R' in k]
[(k,measure_burrows_from_image.calc_tunnel_properties(k,tunnels,tunnel_oris,horiz,cm_factor)) for k in sorted(tunnels) if '.' in k]
im = '20120319/af6_PO-66.jpg'
ar = numpy.asarray(Image.open(im).convert(mode='L'))
matshow(ar, fignum=1)
from pretty import pprint
import gdata.spreadsheet.service
import gdata_tools
import gdata_tools
key,gd_client = gdata_tools.get_spreadsheet_key('antfarm_trials')
close(1)
close(2)
im = '20120319/af5_PO-31.jpg'
ar = numpy.asarray(Image.open(im).convert(mode='L'))
get_calibration()
#?measure_burrows_from_image.get_calibration
measure_burrows_from_image.get_calibration()
matshow(ar, fignum=1)
measure_burrows_from_image.get_calibration()
cm_pts,horiz = _
reload(measure_burrows_from_image)
cm_factor = mean(cm_pts)
tunnels = {}
tunnel_oris = {}
#?measure_burrows_from_image.add_straight_tunnel
reload(measure_burrows_from_image)
#?measure_burrows_from_image.add_straight_tunnel
measure_burrows_from_image.add_straight_tunnel(tunnels, tunnel_oris, horiz, 'HLhill')
measure_burrows_from_image.add_straight_tunnel(tunnels, tunnel_oris, horiz, 'MLhill')
measure_burrows_from_image.add_straight_tunnel(tunnels, tunnel_oris, horiz, 'MLhill')
measure_burrows_from_image.add_straight_tunnel(tunnels, tunnel_oris, horiz, 'ML1.1','MLhill')
measure_burrows_from_image.add_straight_tunnel(tunnels, tunnel_oris, horiz, 'HL1.1','HLhill')
measure_burrows_from_image.add_curved_tunnel(tunnels, tunnel_oris, horiz, 'HL1.1','HLhill')
reload(measure_burrows_from_image)
measure_burrows_from_image.add_curved_tunnel(tunnels, tunnel_oris, horiz, 'HL1.1','HLhill')
measure_burrows_from_image.add_curved_tunnel(tunnels, tunnel_oris, horiz, 'HL1.2','HL1.1')
measure_burrows_from_image.add_straight_tunnel(tunnels, tunnel_oris, horiz, 'HL1.2','HL1.1')
measure_burrows_from_image.add_straight_tunnel(tunnels, tunnel_oris, horiz, 'MRhill')
measure_burrows_from_image.add_straight_tunnel(tunnels, tunnel_oris, horiz, 'MRhill')
measure_burrows_from_image.add_straight_tunnel(tunnels, tunnel_oris, horiz, 'MR1.1','MRhill')
measure_burrows_from_image.add_curved_tunnel(tunnels, tunnel_oris, horiz, 'MR1.2','MR1.1')
measure_burrows_from_image.add_curved_tunnel(tunnels, tunnel_oris, horiz, 'MR1.2','MR1.1')
measure_burrows_from_image.add_straight_tunnel(tunnels, tunnel_oris, horiz, 'MR1.3','MR1.2')
measure_burrows_from_image.add_straight_tunnel(tunnels, tunnel_oris, horiz, 'HRhill')
measure_burrows_from_image.add_straight_tunnel(tunnels, tunnel_oris, horiz, 'HR1.1','HRhill')
reload(measure_burrows_from_image)
reload(measure_burrows_from_image)
#?measure_burrows_from_image.calc_tunnel_properties
measure_burrows_from_image.calc_tunnel_properties('HR1.1', tunnels, tunnel_oris, horiz, cm_factor)
cm_pts
measure_burrows_from_image.calc_tunnel_properties('HR1.1', tunnels, tunnel_oris, horiz, cm_factor)
measure_burrows_from_image.calc_tunnel_properties('HR1.1', tunnels, tunnel_oris, horiz, cm_pts)
reload(measure_burrows_from_image)
cm_pts
measure_burrows_from_image.calc_tunnel_properties('HR1.1', tunnels, tunnel_oris, horiz, cm_pts)
cm_pts
im
measure_burrows_from_image.save_tunnels(im,tunnels,tunnel_oris,cm_pts)
measure_burrows_from_image.save_tunnels(im,tunnels,tunnel_oris,horiz,cm_pts)
reload(measure_burrows_from_image)
measure_burrows_from_image.save_tunnels(im,tunnels,tunnel_oris,horiz,cm_pts)
reload(measure_burrows_from_image)
measure_burrows_from_image.save_tunnels(im,tunnels,tunnel_oris,horiz,cm_pts)
tdict = measure_burrows_from_image.load_tunnels(im,True)
reload(measure_burrows_from_image)
tdict = measure_burrows_from_image.load_tunnels(im,True)
measure_burrows_from_image.calc_tunnel_properties('HR1.1', **tdict
)
reload(measure_burrows_from_image)
measure_burrows_from_image.calc_tunnel_properties('HR1.1', **tdict)
tund = {'tunnels':{},'tunnel_oris':{}}
cm_pts,horiz = _
cm_pts,horiz
tund['cm_pts'],tund['horiz'] = _
tund
im2 = '20120319/af6_PO-66.jpg'
ar2 = numpy.asarray(Image.open(im2).convert(mode='L'))
matshow(ar2,fignum=2)
tund['cm_pts'],tund['horiz'] = measure_burrows_from_image.get_calibration(fignum=2)
tund
#?measure_burrows_from_image.add_straight_tunnel
measure_burrows_from_image.add_straight_tunnel(this_name='MRhill',update_plot=2,**tund)
tund
reload(measure_burrows_from_image)
reload(measure_burrows_from_image)
measure_burrows_from_image.add_straight_tunnel(this_name='MRhill',update_plot=2,**tund)
measure_burrows_from_image.plot_tunnels(fignum=2,**tund)
measure_burrows_from_image.plot_tunnels(fignum=1,**tdict)
measure_burrows_from_image.add_straight_tunnel(this_name='MR1.1',parent_name='MRhill',update_plot=2,**tund)

figure(2)
measure_burrows_from_image.add_straight_tunnel(this_name='MR1.1',parent_name='MRhill',update_plot=2,**tund)
tund
measure_burrows_from_image.calc_tunnel_properties('MR1.1',**tund)
#?measure_burrows_from_image.calc_tunnel_properties
measure_burrows_from_image.calc_tunnel_properties('MR1.1',**tund)
tund.keys()
measure_burrows_from_image.calc_tunnel_properties('MR1.1',**tund)
#?measure_burrows_from_image.add_curved_tunnel
#?measure_burrows_from_image.add_curved_tunnel
measure_burrows_from_image.add
reload(measure_burrows_from_image)
close(2)
ar2 = numpy.asarray(Image.open(im2).convert(mode='L'))
matshow(ar2,fignum=2)
tund = {'tunnels':{},'tunnel_oris':{}}
measure_burrows_from_image.get_calibration(fignum=2,tunneldicts=tund)
tund
#?measure_burrows_from_image.add_curved_tunnel
(this_name='ML1.1',parent
measure_burrows_from_image.add_curved_tunnel(this_name='ML1.1',parent_name='MLhill',update_plot=2,**tund)
tund
close(2)
[measure_burrows_from_image.calc_tunnel_properties(k,**tdict) for k in tdict['tunnels'].keys()]
[measure_burrows_from_image.calc_tunnel_properties(k,**tdict) for k in tdict['tunnels'].keys() if '.' in k]
[measure_burrows_from_image.calc_tunnel_properties(k,**tdict)['angle_to_prev'] for k in tdict['tunnels'].keys() if '.' in k]
[measure_burrows_from_image.calc_tunnel_properties(k,**tdict)['angle_to_prev'] for k in tdict['tunnels'].keys() if '1.1' in k]
[(k,measure_burrows_from_image.calc_tunnel_properties(k,**tdict)['angle_to_prev']) for k in tdict['tunnels'].keys() if '1.1' in k]
[(k,measure_burrows_from_image.calc_tunnel_properties(k,**tdict)['angle_to_horiz']) for k in tdict['tunnels'].keys() if '1.1' in k]
[(k,measure_burrows_from_image.calc_tunnel_properties(k,**tdict)['angle_to_prev']) for k in tdict['tunnels'].keys() if '1.2' in k]
[(k,measure_burrows_from_image.calc_tunnel_properties(k,**tdict)['angle_to_prev']) for k in tdict['tunnels'].keys() if '1.3' in k]
im = '20120323/af7_predug.jpg'
ar2 = numpy.asarray(Image.open(im2).convert(mode='L'))
matshow(ar2,fignum=3)
close(3)
im2 = '20120323/af7_predug.jpg'
ar2 = numpy.asarray(Image.open(im2).convert(mode='L'))
matshow(ar2,fignum=3)
tund_pre = {'tunnels':{},'tunnel_oris':{}}
get_calibration()
measure_burrows_from_image.get_calibration(fignum=3,tunneldicts=tund_pre)
measure_burrows_from_image.add_straight_tunnel(this_name='MLhill',update_plot=3,**tund_pre)
measure_burrows_from_image.add_straight_tunnel(this_name='ML1.1pre',parent_name='MLhill',update_plot=3,**tund_pre)
measure_burrows_from_image.add_straight_tunnel(this_name='MRhill',update_plot=3,**tund_pre)
measure_burrows_from_image.add_straight_tunnel(this_name='MR1.1pre',parent_name='MRhill',update_plot=3,**tund_pre)
measure_burrows_from_image.calc_tunnel_properties('MR1.1pre',**tund)
measure_burrows_from_image.calc_tunnel_properties('MR1.1pre',**tund_pre)
measure_burrows_from_image.add_straight_tunnel(this_name='MR1.1pre',parent_name='MRhill',update_plot=3,**tund_pre)
measure_burrows_from_image.calc_tunnel_properties('MR1.1pre',**tund_pre)
measure_burrows_from_image.calc_tunnel_properties('ML1.1pre',**tund_pre)
measure_burrows_from_image.add_straight_tunnel(this_name='ML1.1pre',parent_name='MLhill',update_plot=3,**tund_pre)
measure_burrows_from_image.calc_tunnel_properties('ML1.1pre',**tund_pre)
