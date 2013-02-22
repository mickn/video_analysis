
import matplotlib
matplotlib.use('Agg')
import pylab
import cv
vid = '20100818-1_PO32-2_PO34-3_PO17/merge6mbit_720_PO32_0-29443.mp4'
pixdim = (522, 246)
_ip.system("rm 20100818-1_PO32-2_PO34-3_PO17/test_opencv_write.avi")
vidwriter = cv.CreateVideoWriter('/n/hoekstrafs1/burrowing/antfarms/data/_2011fall/20100818-1_PO32-2_PO34-3_PO17/test_opencv_write.avi',cv.FOURCC('x','v','i','d'), 29.97, pixdim,1)
from video_analysis import vidtools

m = vidtools.array_from_stream(stream)
stream = cv.CaptureFromFile(vid)
m = vidtools.array_from_stream(stream)
def mat2cv(m):
fig = pylab.figure(1)
_ip.magic("timeit fig = pylab.figure(1)")

ax = pylab.matshow(m,fignum=1)
_ip.magic("timeit ax = pylab.matshow(m,fignum=1)")
def mat2cv(m):
    fig = plt.figure(1)
    ax = plt.matshow(m,fignum=1)
    buf = io.BytesIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    pi = Image.open(buf)
    cv_im = cv.CreateImageHeader(pi.size, cv.IPL_DEPTH_8U, 1)
    cv.SetData(cv_im, pi.tostring())
def mat2cv(m):
    fig = pylab.figure(1)
    ax = pylab.matshow(m,fignum=1)
    buf = io.BytesIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    pi = Image.open(buf)
    cv_im = cv.CreateImageHeader(pi.size, cv.IPL_DEPTH_8U, 1)
    cv.SetData(cv_im, pi.tostring())
    

import io
buf = io.BytesIO()
_ip.magic("timeit buf = io.BytesIO()")
fig.savefig(buf,format='png')

_ip.magic("timeit fig.savefig(buf,format='png')")
def save2ram():
    buf = io.BytesIO()
    fig.savefig(buf,format='png')
    return buf
buf = save2ram()
buf.close()
buf = save2ram()
pylab.close(1)
fig = pylab.figure(1)
ax = pylab.matshow(m,fignum=1)
buf = save2ram()
_ip.magic("timeit buf = save2ram()")

buf.seek(0)
pi = Image.open(buf)
from PIL import Image
pi = Image.open(buf)
def buf2im(buf):
    buf.seek(0)
    pi = Image.open(buf)
    
def buf2im(buf):
    buf.seek(0)
    pi = Image.open(buf)
    return pi
def buf2im(buf):
    buf.seek(0)
    pi = Image.open(buf)
    return pi

_ip.magic("timeit pi = buf2im(save2ram())")
pi.show()
pi.size
pi.palette
pi.getcolors 
vidwriter = cv.CreateVideoWriter('/n/hoekstrafs1/burrowing/antfarms/data/_2011fall/20100818-1_PO32-2_PO34-3_PO17/test_opencv_write.avi',cv.FOURCC('x','v','i','d'), 29.97, pi.size,1)
def mat2cv(m):
    fig = pylab.figure(1)
    ax = pylab.matshow(m,fignum=1)
    buf = io.BytesIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    pi = Image.open(buf)
    cv_im = cv.CreateImageHeader(pi.size, cv.IPL_DEPTH_8U, 1)
    cv.SetData(cv_im, pi.tostring())
    

for i in xrange(600):
    m = vidtools.array_from_stream(stream)
    cv_im = mat2cv(m)
    retval = cv.WriteFrame(vidwriter,cv_im)
    

def mat2cv(m):
    fig = pylab.figure(1)
    ax = pylab.matshow(m,fignum=1)
    buf = io.BytesIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    pi = Image.open(buf)
    cv_im = cv.CreateImageHeader(pi.size, cv.IPL_DEPTH_8U, 1)
    cv.SetData(cv_im, pi.tostring())
    return cv_im
def mat2cv(m):
    fig = pylab.figure(1)
    ax = pylab.matshow(m,fignum=1)
    buf = io.BytesIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    pi = Image.open(buf)
    cv_im = cv.CreateImageHeader(pi.size, cv.IPL_DEPTH_8U, 1)
    cv.SetData(cv_im, pi.tostring())
    return cv_im

vidwriter = cv.CreateVideoWriter('/n/hoekstrafs1/burrowing/antfarms/data/_2011fall/20100818-1_PO32-2_PO34-3_PO17/test_opencv_write.avi',cv.FOURCC('x','v','i','d'), 29.97, pi.size,1)
for i in xrange(600):
    m = vidtools.array_from_stream(stream)
    cv_im = mat2cv(m)
    retval = cv.WriteFrame(vidwriter,cv_im)
    

retval
pixdim = (522, 246)
vidwriter = cv.CreateVideoWriter('/n/hoekstrafs1/burrowing/antfarms/data/_2011fall/20100818-1_PO32-2_PO34-3_PO17/test_opencv_write.avi',cv.FOURCC('x','v','i','d'), 29.97, pixdim,1)
for i in xrange(600):
    img = cv.QueryFrame(stream)
    cv.WriteFrame(vidwriter,img)
    

img
cv_im
cv_im = cv.CreateImageHeader(pi.size, cv.IPL_DEPTH_8U, 3)
cv.SetData(cv_im, pi.tostring())
cv_im = cv.CreateImageHeader(pi.size, cv.IPL_DEPTH_8U, 3)
cv.SetData(cv_im, pi.tostring())
cv_im
def mat2cv(m):
    fig = pylab.figure(1)
    ax = pylab.matshow(m,fignum=1)
    buf = io.BytesIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    pi = Image.open(buf)
    cv_im = cv.CreateImageHeader(pi.size, cv.IPL_DEPTH_8U, 3)
    cv.SetData(cv_im, pi.tostring())
    return cv_im

pi.size
vidwriter = cv.CreateVideoWriter('/n/hoekstrafs1/burrowing/antfarms/data/_2011fall/20100818-1_PO32-2_PO34-3_PO17/test_opencv_write.avi',cv.FOURCC('x','v','i','d'), 29.97, pi.size,1)
_ip.system("rm 20100818-1_PO32-2_PO34-3_PO17/test_opencv_write.avi")
vidwriter = cv.CreateVideoWriter('/n/hoekstrafs1/burrowing/antfarms/data/_2011fall/20100818-1_PO32-2_PO34-3_PO17/test_opencv_write.avi',cv.FOURCC('x','v','i','d'), 29.97, pi.size,1)
for i in xrange(600):
    m = vidtools.array_from_stream(stream)
    cv_im = mat2cv(m)
    cv.WriteFrame(vidwriter,cv_im)
    


#?cv.CreateImageHeader
def array2cv(a):
  dtype2depth = {
        'uint8':   cv.IPL_DEPTH_8U,
        'int8':    cv.IPL_DEPTH_8S,
        'uint16':  cv.IPL_DEPTH_16U,
        'int16':   cv.IPL_DEPTH_16S,
        'int32':   cv.IPL_DEPTH_32S,
        'float32': cv.IPL_DEPTH_32F,
        'float64': cv.IPL_DEPTH_64F,
    }
  try:
    nChannels = a.shape[2]
  except:
    nChannels = 1
  cv_im = cv.CreateImageHeader((a.shape[1],a.shape[0]),
          dtype2depth[str(a.dtype)],
          nChannels)
  cv.SetData(cv_im, a.tostring(),
             a.dtype.itemsize*nChannels*a.shape[1])
  return cv_im
cv_im = array2cv(m)
cv_im
from cv import adaptors
from opencv import adaptors
import adaptors
cv_im = cv.CreateImageHeader(pi.size, cv.IPL_DEPTH_8U, 3)
cv.SetData(cv_im, pi.tostring(),pi.size[0]*3)
cv_im
def mat2cv(m):
    fig = pylab.figure(1)
    ax = pylab.matshow(m,fignum=1)
    buf = io.BytesIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    pi = Image.open(buf)
    cv_im = cv.CreateImageHeader(pi.size, cv.IPL_DEPTH_8U, 3)
    cv.SetData(cv_im, pi.tostring(),pi.size[0]*3)
    return cv_im
cv_im = mat2cv(m)
_ip.magic("timeit cv_im = mat2cv(m)")
_ip.magic("timeit cv_im = mat2cv(m)")
def mat2cv(m):
    fig = pylab.figure(1)
    ax = pylab.matshow(m,fignum=1)
    buf = io.BytesIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    pi = Image.open(buf)
    buf.close()
    cv_im = cv.CreateImageHeader(pi.size, cv.IPL_DEPTH_8U, 3)
    cv.SetData(cv_im, pi.tostring(),pi.size[0]*3)
    return cv_im
def mat2cv(m):
    fig = pylab.figure(1)
    ax = pylab.matshow(m,fignum=1)
    buf = io.BytesIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    pi = Image.open(buf)
    buf.close()
    cv_im = cv.CreateImageHeader(pi.size, cv.IPL_DEPTH_8U, 3)
    cv.SetData(cv_im, pi.tostring(),pi.size[0]*3)
    return cv_im

_ip.magic("timeit cv_im = mat2cv(m)")


def mat2cv(m):
    fig = pylab.figure(1)
    ax = pylab.matshow(m,fignum=1)
    buf = io.BytesIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    pi = Image.open(buf)
    cv_im = cv.CreateImageHeader(pi.size, cv.IPL_DEPTH_8U, 3)
    cv.SetData(cv_im, pi.tostring(),pi.size[0]*3)
    buf.close()
    return cv_im
def mat2cv(m):
    fig = pylab.figure(1)
    ax = pylab.matshow(m,fignum=1)
    buf = io.BytesIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    pi = Image.open(buf)
    cv_im = cv.CreateImageHeader(pi.size, cv.IPL_DEPTH_8U, 3)
    cv.SetData(cv_im, pi.tostring(),pi.size[0]*3)
    buf.close()
    return cv_im

_ip.magic("timeit cv_im = mat2cv(m)")
#?io
import StringIO
#?StringIO
def mat2cv(m):
    fig = pylab.figure(1)
    ax = pylab.matshow(m,fignum=1)
    buf = StringIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    pi = Image.open(buf)
    cv_im = cv.CreateImageHeader(pi.size, cv.IPL_DEPTH_8U, 3)
    cv.SetData(cv_im, pi.tostring(),pi.size[0]*3)
    buf.close()
    return cv_im

_ip.magic("timeit cv_im = mat2cv(m)")
def mat2cv(m):
    fig = pylab.figure(1)
    ax = pylab.matshow(m,fignum=1)
    buf = StringIO.StringIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    pi = Image.open(buf)
    cv_im = cv.CreateImageHeader(pi.size, cv.IPL_DEPTH_8U, 3)
    cv.SetData(cv_im, pi.tostring(),pi.size[0]*3)
    buf.close()
    return cv_im

_ip.magic("timeit cv_im = mat2cv(m)")

def mat2cv(m):
    fig = pylab.figure(1)
    ax = pylab.matshow(m,fignum=1)
    buf = io.BytesIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    pi = Image.open(buf)
    cv_im = cv.CreateImageHeader(pi.size, cv.IPL_DEPTH_8U, 3)
    cv.SetData(cv_im, pi.tostring(),pi.size[0]*3)
    buf.close()
    return cv_im


import time
def mat2cv(m):
    t=[]
    t.append(time.time())
def mat2cv(m):
  t=[time.time()]
  fig = pylab.figure(1)
  t.append(time.time())
  ax = pylab.matshow(m,fignum=1)
  t.append(time.time())
  buf = io.BytesIO()
  t.append(time.time())
  fig.savefig(buf,format='png')
  t.append(time.time())
  buf.seek(0)
  t.append(time.time())
  pi = Image.open(buf)
  t.append(time.time())
  cv_im = cv.CreateImageHeader(pi.size, cv.IPL_DEPTH_8U, 3)
  t.append(time.time())
  cv.SetData(cv_im, pi.tostring(),pi.size[0]*3)
  t.append(time.time())
  buf.close()
  t.append(time.time())
  return cv_im,t
cv_im,t = mat2cv(m)
[t[i]-t[i-1] for i in range(1,len(t))]
tdiffs = [t[i]-t[i-1] for i in range(1,len(t))]
[td/sum(tdiffs) for td in tdiffs]
[int(100*(td/sum(tdiffs))) for td in tdiffs]
def mat2cv(m):
  t=[time.time()]
  pylab.close(1)
  t.append(time.time())
  fig = pylab.figure(1)
  t.append(time.time())
  ax = pylab.matshow(m,fignum=1)
  t.append(time.time())
  buf = io.BytesIO()
  t.append(time.time())
  fig.savefig(buf,format='png')
  t.append(time.time())
  buf.seek(0)
  t.append(time.time())
  pi = Image.open(buf)
  t.append(time.time())
  cv_im = cv.CreateImageHeader(pi.size, cv.IPL_DEPTH_8U, 3)
  t.append(time.time())
  cv.SetData(cv_im, pi.tostring(),pi.size[0]*3)
  t.append(time.time())
  buf.close()
  t.append(time.time())
  return cv_im,t
cv_im,t = mat2cv(m)
tdiffs = [t[i]-t[i-1] for i in range(1,len(t))]
[td/sum(tdiffs) for td in tdiffs]
[int(100*(td/sum(tdiffs))) for td in tdiffs]
sum(tdiffs)
_ip.magic("timeit cv_im = mat2cv(m)")
vidwriter = cv.CreateVideoWriter('/n/hoekstrafs1/burrowing/antfarms/data/_2011fall/20100818-1_PO32-2_PO34-3_PO17/test_opencv_write.avi',cv.FOURCC('x','v','i','d'), 29.97, pi.size,1)
def mat2cv(m):
    pylab.close(1)
    fig = pylab.figure(1)
    ax = pylab.matshow(m,fignum=1)
    buf = io.BytesIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    pi = Image.open(buf)
    cv_im = cv.CreateImageHeader(pi.size, cv.IPL_DEPTH_8U, 3)
    cv.SetData(cv_im, pi.tostring(),pi.size[0]*3)
    buf.close()
    return cv_im

def mat2cv(m):
    pylab.close(1)
    fig = pylab.figure(1)
    ax = pylab.matshow(m,fignum=1)
    buf = io.BytesIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    pi = Image.open(buf)
    cv_im = cv.CreateImageHeader(pi.size, cv.IPL_DEPTH_8U, 3)
    cv.SetData(cv_im, pi.tostring(),pi.size[0]*3)
    buf.close()
    return cv_im

vidwriter = cv.CreateVideoWriter('/n/hoekstrafs1/burrowing/antfarms/data/_2011fall/20100818-1_PO32-2_PO34-3_PO17/test_opencv_write.avi',cv.FOURCC('x','v','i','d'), 29.97, pi.size,1)
for i in xrange(600):
    m = vidtools.array_from_stream(stream)
    cv_im = mat2cv(m)
    cv.WriteFrame(vidwriter,cv_im)
    

img = cv.QueryFrame(stream)
img
1566/522
1566/522.
pi.show()
m.shape
pi.mode
p2 = pi.convert('RGB')
p2.mode
def mat2cv(m):
    pylab.close(1)
    fig = pylab.figure(1)
    ax = pylab.matshow(m,fignum=1)
    buf = io.BytesIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    pi = Image.open(buf).convert('RGB')
    cv_im = cv.CreateImageHeader(pi.size, cv.IPL_DEPTH_8U, 3)
    cv.SetData(cv_im, pi.tostring(),pi.size[0]*3)
    buf.close()
    return cv_im
vidwriter = cv.CreateVideoWriter('/n/hoekstrafs1/burrowing/antfarms/data/_2011fall/20100818-1_PO32-2_PO34-3_PO17/test_opencv_write.avi',cv.FOURCC('x','v','i','d'), 29.97, pi.size,1)
for i in xrange(30):
    m = vidtools.array_from_stream(stream)
    cv_im = mat2cv(m)
    cv.WriteFrame(vidwriter,cv_im)
    

_ip.system("rm 20100818-1_PO32-2_PO34-3_PO17/test_opencv_write.avi")
vidwriter = cv.CreateVideoWriter('/n/hoekstrafs1/burrowing/antfarms/data/_2011fall/20100818-1_PO32-2_PO34-3_PO17/test_opencv_write.avi',cv.FOURCC('x','v','i','d'), 29.97, pi.size,1)
for i in xrange(30):
    m = vidtools.array_from_stream(stream)
    cv_im = mat2cv(m)
    cv.WriteFrame(vidwriter,cv_im)
    

_ip.system("rm 20100818-1_PO32-2_PO34-3_PO17/test_opencv_write.avi")
vidwriter = cv.CreateVideoWriter('/n/hoekstrafs1/burrowing/antfarms/data/_2011fall/20100818-1_PO32-2_PO34-3_PO17/test_opencv_write.avi',cv.FOURCC('x','v','i','d'), 29.97, pi.size,1)
for i in xrange(600):
    m = vidtools.array_from_stream(stream,normed=False)
    cv_im = mat2cv(m)
    cv.WriteFrame(vidwriter,cv_im)
    


m.shape
[i/80. for i in m.shape]
reversed([i/80. for i in m.shape])
list(reversed([i/80. for i in m.shape]))
tuple(reversed([i/80. for i in m.shape]))
def mat2cv(m,dpi=80):
    pylab.close(1)
    fig = pylab.figure(1,figsize=tuple(reversed([i/80. for i in m.shape])),dpi=dpi)
    ax = pylab.matshow(m,fignum=1)
    buf = io.BytesIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    pi = Image.open(buf).convert('RGB')
    cv_im = cv.CreateImageHeader(pi.size, cv.IPL_DEPTH_8U, 3)
    cv.SetData(cv_im, pi.tostring(),pi.size[0]*3)
    buf.close()
    return cv_im
pylab.close(1)
dpi = 80
fig = pylab.figure(1,figsize=tuple(reversed([i/80. for i in m.shape])),dpi=dpi)
ax = pylab.matshow(m,fignum=1)
buf = io.BytesIO()
fig.savefig(buf,format='png')
buf.seek(0)
pi = Image.open(buf).convert('RGB')
pi.show()

fig.subplots_adjust(bottom=0.05)
fig.subplots_adjust(left=0.05)
fig.subplots_adjust(top=0.95)
fig.subplots_adjust(right=0.95)
buf = io.BytesIO()
fig.savefig(buf,format='png')
pi = Image.open(buf).convert('RGB')
buf.seek(0)
pi = Image.open(buf).convert('RGB')
pi.show()
#?fig.get_figheight
fig.get_figheight()
fig.



#?fig.subplots_adjust
fig.subplots_adjust(0.05,0.05.0.95,0.95)
fig.subplots_adjust(0.05,0.05,0.95,0.95)
buf = io.BytesIO()
fig.savefig(buf,format='png')
buf.seek(0)

pi = Image.open(buf).convert('RGB')
pi.show()
pylab.figure(1).subplots_adjust(0.05,0.05,0.95,0.95)
buf = io.BytesIO()
fig.savefig(buf,format='png')
buf.seek(0)

pi = Image.open(buf).convert('RGB')
pi.show()
fig.plot()
pylab.plot()
_ip.magic("history -n")
buf = io.BytesIO()
fig.savefig(buf,format='png')
buf.seek(0)

pi = Image.open(buf).convert('RGB')
pi.show()
show()
pylab.show()
pylab.close(1)
pylab.figure(1,figsize=tuple(reversed([i/80. for i in m.shape])),dpi=dpi)
pylab.matshow(m,fignum=1)
pylab.subplots_adjust(0.05,0.05,0.95,0.95)
buf = io.BytesIO()
pylab.savefig(buf,format='png')
buf.seek(0)
pi = Image.open(buf).convert('RGB')
pi.show()
pylab.subplots_adjust(0.2,0.2,0.8,0.8)
buf = io.BytesIO()
pylab.savefig(buf,format='png')
buf.seek(0)
pi = Image.open(buf).convert('RGB')
pi.show()
dpi = 150
pylab.close(1)
pylab.figure(1,figsize=tuple(reversed([i/80. for i in m.shape])),dpi=dpi)
pylab.matshow(m,fignum=1)
buf = io.BytesIO()
pylab.savefig(buf,format='png')
pi = Image.open(buf).convert('RGB')
buf.seek(0)
pi = Image.open(buf).convert('RGB')
pi.show()
buf.close()
buf.seek(0)
buf = io.BytesIO()
dpi = 300
tuple(reversed([i/dpi. for i in m.shape]))
tuple(reversed([i/dpi for i in m.shape]))
def mat2cv(m,dpi=80,scale=1):
    pylab.close(1)
    fig = pylab.figure(1,figsize=tuple(reversed([float(i/dpi)*scale for i in m.shape])),dpi=dpi)
    ax = pylab.matshow(m,fignum=1)
    buf = io.BytesIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    pi = Image.open(buf).convert('RGB')
    cv_im = cv.CreateImageHeader(pi.size, cv.IPL_DEPTH_8U, 3)
    cv.SetData(cv_im, pi.tostring(),pi.size[0]*3)
    buf.close()
    return cv_im
vidwriter = cv.CreateVideoWriter('/n/hoekstrafs1/burrowing/antfarms/data/_2011fall/20100818-1_PO32-2_PO34-3_PO17/test_opencv_write.avi',cv.FOURCC('x','v','i','d'), 29.97, pi.size,1)
_ip.system("rm 20100818-1_PO32-2_PO34-3_PO17/test_opencv_write.avi")
vidwriter = cv.CreateVideoWriter('/n/hoekstrafs1/burrowing/antfarms/data/_2011fall/20100818-1_PO32-2_PO34-3_PO17/test_opencv_write.avi',cv.FOURCC('x','v','i','d'), 29.97, pi.size,1)
for i in xrange(300):
    m = vidtools.array_from_stream(stream,normed=False)
    cv_im = mat2cv(m)
    cv.WriteFrame(vidwriter,cv_im)
    

pi.size
vidwriter = cv.CreateVideoWriter('/n/hoekstrafs1/burrowing/antfarms/data/_2011fall/20100818-1_PO32-2_PO34-3_PO17/test_opencv_write.avi',cv.FOURCC('x','v','i','d'), 29.97, pi.size,1)
_ip.system("rm 20100818-1_PO32-2_PO34-3_PO17/test_opencv_write.avi")
vidwriter = cv.CreateVideoWriter('/n/hoekstrafs1/burrowing/antfarms/data/_2011fall/20100818-1_PO32-2_PO34-3_PO17/test_opencv_write.avi',cv.FOURCC('x','v','i','d'), 29.97, pi.size,1)
def mat2cv(m,dpi=80,scale=1):
    pylab.close(1)
    fig = pylab.figure(1,figsize=tuple(reversed([float(i/dpi)*scale for i in m.shape])),dpi=dpi)
    ax = pylab.matshow(m,fignum=1)
    buf = io.BytesIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    pi = Image.open(buf).convert('RGB')
    cv_im = cv.CreateImageHeader(pi.size, cv.IPL_DEPTH_8U, 3)
    cv.SetData(cv_im, pi.tostring(),pi.size[0]*3)
    buf.close()
    return cv_im
for i in xrange(300):
    m = vidtools.array_from_stream(stream,normed=False)
    cv_im = mat2cv(m)
    cv.WriteFrame(vidwriter,cv_im)
    

_ip.system("rm 20100818-1_PO32-2_PO34-3_PO17/test_opencv_write.avi")
cv_im = mat2cv(m)
cv_im.height
cv_im.width
cv_im
def mat2cv(m,dpi=80,scale=1):
    pylab.close(1)
    fig = pylab.figure(1,figsize=tuple(reversed([(float(i)/dpi)*scale for i in m.shape])),dpi=dpi)
    ax = pylab.matshow(m,fignum=1)
    buf = io.BytesIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    pi = Image.open(buf).convert('RGB')
    cv_im = cv.CreateImageHeader(pi.size, cv.IPL_DEPTH_8U, 3)
    cv.SetData(cv_im, pi.tostring(),pi.size[0]*3)
    buf.close()
    return cv_im
cv_im = mat2cv(m)
cv_im
cv_im = mat2cv(m,150,2)
cv_im
cv_im.width,cv_im.height
pixdim = (cv_im.width,cv_im.height)
_ip.system("rm 20100818-1_PO32-2_PO34-3_PO17/test_opencv_write.avi")
vidwriter = cv.CreateVideoWriter('/n/hoekstrafs1/burrowing/antfarms/data/_2011fall/20100818-1_PO32-2_PO34-3_PO17/test_opencv_write.avi',cv.FOURCC('x','v','i','d'), 29.97, pixdim,1)
for i in xrange(300):
    m = vidtools.array_from_stream(stream,normed=False)
    cv_im = mat2cv(m,150,2)
    cv.WriteFrame(vidwriter,cv_im)
    

cv_im = mat2cv(m,150,4)
pixdim = (cv_im.width,cv_im.height)
_ip.system("rm 20100818-1_PO32-2_PO34-3_PO17/test_opencv_write.avi")
vidwriter = cv.CreateVideoWriter('/n/hoekstrafs1/burrowing/antfarms/data/_2011fall/20100818-1_PO32-2_PO34-3_PO17/test_opencv_write.avi',cv.FOURCC('x','v','i','d'), 29.97, pixdim,1)
for i in xrange(300):
    m = vidtools.array_from_stream(stream,normed=False)
    cv_im = mat2cv(m,150,4)
    cv.WriteFrame(vidwriter,cv_im)
    

seglen = 900
import cv
from video_analysis import vidtools,summarize_segment_opencv
import Util,time,datetime

# get started
seglen = 900
#vid = '20110916-1_BW-1143_2_BW-1145_3_BW-1148_4_BW-1142/merge6mbit_720_BW-1142_0-46860.mp4'

#good early PO dig
vid = '20100818-1_PO32-2_PO34-3_PO17/merge6mbit_720_PO32_0-29443.mp4'
start_offset = 27000

#vid = '20100818-1_PO32-2_PO34-3_PO17/merge6mbit_720_PO17_0-29443.mp4'
#start_offset = 435166

# find optimal cutoff; pull scoring distributions
nframes = 180
nparts = 180
cut_step = 0.001
scores,dists = vidtools.run_mousezopt(vid,seglen,nframes,2,nparts,cut_step)
cutoff_rank,cutoff = vidtools.choose_cutoff(scores,cut_step) #or cut_step*2
size_h,size_bins,fol_h,fol_bins = dists[cutoff]
min_arc_score = (2*max(size_h))+max(fol_h)

#video stream
stream = cv.CaptureFromFile(vid)

# fake offset to test things out
#start_offset = 36000
frames_offset = start_offset - seglen
vidtools.seek_in_stream(stream,frames_offset)

#init frames
frames,currsum,denom = vidtools.init_frames(stream,seglen)
SHAPE = frames[0].shape

cutoff
tream,seglen)

In [532]: SHAPE = frames[0].shape

In [533]: 

In [534]: cutoff
tream,seglen)

In [532]: SHAPE = frames[0].shape

In [533]: 

In [534]: cutoff
# init empty receivers 
hsl = seglen/2
ols = []
objs = {}
objs_sizes = {}
objs_fols = {}
to_retire_objs = {}
to_retire_objs_sizes = {}
to_retire_objs_fols = {}


#init object arcs
ols,ols_offset,frames_offset,objs,splits,objs_sizes,objs_fols,prelast_avg,prelast_mm,to_retire_objs,to_retire_objs_sizes,to_retire_objs_fols = summarize_segment_opencv.init_objects(stream,frames,currsum,denom,seglen,cutoff,frames_offset,SHAPE,size_h,size_bins,fol_h,fol_bins)
ols_offset
frames_offset
frames_offset - ols_offset
len(last_frames)
len(ols)
frames.shape
frames[0].shape

ols[0]
import matplotlib
_ip.system("rm 20100818-1_PO32-2_PO34-3_PO17/test_opencv_write.avi")
vidwriter = cv.CreateVideoWriter('/n/hoekstrafs1/burrowing/antfarms/data/_2011fall/20100818-1_PO32-2_PO34-3_PO17/test_opencv_write.avi',cv.FOURCC('x','v','i','d'), 29.97, pixdim,1)
for ol,fr in zip(ols[450:],frames[:450]):

def mat2cv(m,polys,dpi=80,scale=1):
    pylab.close(1)
    fig = pylab.figure(1,figsize=tuple(reversed([(float(i)/dpi)*scale for i in m.shape])),dpi=dpi)
    ax = pylab.matshow(m,fignum=1)
    for c,p in zip(iplot.subspec(len(polys)),polys):
        ax.add_patch(matplotlib.patches.Polygon(p,fc='none',ec=c))
    pylab.plot()
def mat2cv(m,polys,dpi=80,scale=1):
    pylab.close(1)
    fig = pylab.figure(1,figsize=tuple(reversed([(float(i)/dpi)*scale for i in m.shape])),dpi=dpi)
    ax = pylab.matshow(m,fignum=1)
    for c,p in zip(iplot.subspec(len(polys)),polys):
        ax.add_patch(matplotlib.patches.Polygon(p,fc='none',ec=c))
    pylab.plot()
    buf = io.BytesIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    pi = Image.open(buf).convert('RGB')
    cv_im = cv.CreateImageHeader(pi.size, cv.IPL_DEPTH_8U, 3)
    cv.SetData(cv_im, pi.tostring(),pi.size[0]*3)
    buf.close()
    return cv_im
def mat_polys2cv(m,polys,dpi=80,scale=1):
    pylab.close(1)
    fig = pylab.figure(1,figsize=tuple(reversed([(float(i)/dpi)*scale for i in m.shape])),dpi=dpi)
    ax = pylab.matshow(m,fignum=1)
    for c,p in zip(iplot.subspec(len(polys)),polys):
        ax.add_patch(matplotlib.patches.Polygon(p,fc='none',ec=c))
    pylab.plot()
    buf = io.BytesIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    pi = Image.open(buf).convert('RGB')
    cv_im = cv.CreateImageHeader(pi.size, cv.IPL_DEPTH_8U, 3)
    cv.SetData(cv_im, pi.tostring(),pi.size[0]*3)
    buf.close()
    return cv_im
pylab.gray()
_ip.system("rm 20100818-1_PO32-2_PO34-3_PO17/test_opencv_write.avi")
vidwriter = cv.CreateVideoWriter('/n/hoekstrafs1/burrowing/antfarms/data/_2011fall/20100818-1_PO32-2_PO34-3_PO17/test_opencv_write.avi',cv.FOURCC('x','v','i','d'), 29.97, pixdim,1)

for ol,fr in zip(ols[450:],frames[:450]):
    cv_im = mat_polys2cv(m,ol,150,4)
    cv.WriteFrame(vidwriter,cv_im)
    

import iplot
_ip.system("rm 20100818-1_PO32-2_PO34-3_PO17/test_opencv_write.avi")
vidwriter = cv.CreateVideoWriter('/n/hoekstrafs1/burrowing/antfarms/data/_2011fall/20100818-1_PO32-2_PO34-3_PO17/test_opencv_write.avi',cv.FOURCC('x','v','i','d'), 29.97, pixdim,1)
for ol,fr in zip(ols[450:],frames[:450]):
    cv_im = mat_polys2cv(m,ol,150,4)
    cv.WriteFrame(vidwriter,cv_im)
    

def mat_polys2cv(m,polys,dpi=80,scale=1):
    pylab.close(1)
    fig = pylab.figure(1,figsize=tuple(reversed([(float(i)/dpi)*scale for i in m.shape])),dpi=dpi)
    ax = pylab.matshow(m,fignum=1)
    for c,p in zip(iplot.subspectrum(len(polys)),polys):
        ax.add_patch(matplotlib.patches.Polygon(p,fc='none',ec=c))
    pylab.plot()
    buf = io.BytesIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    pi = Image.open(buf).convert('RGB')
    cv_im = cv.CreateImageHeader(pi.size, cv.IPL_DEPTH_8U, 3)
    cv.SetData(cv_im, pi.tostring(),pi.size[0]*3)
    buf.close()
    return cv_im
_ip.system("rm 20100818-1_PO32-2_PO34-3_PO17/test_opencv_write.avi")
vidwriter = cv.CreateVideoWriter('/n/hoekstrafs1/burrowing/antfarms/data/_2011fall/20100818-1_PO32-2_PO34-3_PO17/test_opencv_write.avi',cv.FOURCC('x','v','i','d'), 29.97, pixdim,1)
for ol,fr in zip(ols[450:],frames[:450]):
    cv_im = mat_polys2cv(m,ol,150,4)
    cv.WriteFrame(vidwriter,cv_im)
    

def mat_polys2cv(m,polys,dpi=80,scale=1):
    pylab.close(1)
    fig = pylab.figure(1,figsize=tuple(reversed([(float(i)/dpi)*scale for i in m.shape])),dpi=dpi)
    ax = pylab.matshow(m,fignum=1)
    ax = pylab.figure(1).axes[0]
    for c,p in zip(iplot.subspectrum(len(polys)),polys):
        ax.add_patch(matplotlib.patches.Polygon(p,fc='none',ec=c))
    pylab.plot()
    buf = io.BytesIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    pi = Image.open(buf).convert('RGB')
    cv_im = cv.CreateImageHeader(pi.size, cv.IPL_DEPTH_8U, 3)
    cv.SetData(cv_im, pi.tostring(),pi.size[0]*3)
    buf.close()
    return cv_im
_ip.system("rm 20100818-1_PO32-2_PO34-3_PO17/test_opencv_write.avi")
vidwriter = cv.CreateVideoWriter('/n/hoekstrafs1/burrowing/antfarms/data/_2011fall/20100818-1_PO32-2_PO34-3_PO17/test_opencv_write.avi',cv.FOURCC('x','v','i','d'), 29.97, pixdim,1)
for ol,fr in zip(ols[450:],frames[:450]):
    cv_im = mat_polys2cv(m,ol,150,4)
    cv.WriteFrame(vidwriter,cv_im)
    

_ip.system("rm 20100818-1_PO32-2_PO34-3_PO17/test_opencv_write.avi")
vidwriter = cv.CreateVideoWriter('/n/hoekstrafs1/burrowing/antfarms/data/_2011fall/20100818-1_PO32-2_PO34-3_PO17/test_opencv_write.avi',cv.FOURCC('x','v','i','d'), 29.97, pixdim,1)
for ol,fr in zip(ols[450:],frames[:450]):
    cv_im = mat_polys2cv(fr,ol,150,4)
    cv.WriteFrame(vidwriter,cv_im)
    

def mat_polys2cv(m,polys,dpi=80,scale=1):
    pylab.close(1)
    fig = pylab.figure(1,figsize=tuple(reversed([(float(i)/dpi)*scale for i in m.shape])),dpi=dpi)
    ax = pylab.matshow(m,fignum=1)
    ax = pylab.figure(1).axes[0]
    for c,p in zip(iplot.subspectrum(len(polys)),polys):
        ax.add_patch(matplotlib.patches.Polygon(p,fc='none',ec=c))
    pylab.plot()
    pylab.ylim(0,m[0])
    pylab.xlim(0,m[1])
    buf = io.BytesIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    pi = Image.open(buf).convert('RGB')
    cv_im = cv.CreateImageHeader(pi.size, cv.IPL_DEPTH_8U, 3)
    cv.SetData(cv_im, pi.tostring(),pi.size[0]*3)
    buf.close()
    return cv_im
def mat_polys2cv(m,polys,dpi=80,scale=1):
    pylab.close(1)
    fig = pylab.figure(1,figsize=tuple(reversed([(float(i)/dpi)*scale for i in m.shape])),dpi=dpi)
    ax = pylab.matshow(m,fignum=1)
    ax = pylab.figure(1).axes[0]
    for c,p in zip(iplot.subspectrum(len(polys)),polys):
        ax.add_patch(matplotlib.patches.Polygon(p,fc='none',ec=c))
    pylab.plot()
    pylab.ylim(0,m[0])
    pylab.xlim(0,m[1])
    buf = io.BytesIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    pi = Image.open(buf).convert('RGB')
    cv_im = cv.CreateImageHeader(pi.size, cv.IPL_DEPTH_8U, 3)
    cv.SetData(cv_im, pi.tostring(),pi.size[0]*3)
    buf.close()
    return cv_im
_ip.system("rm 20100818-1_PO32-2_PO34-3_PO17/test_opencv_write.avi")
vidwriter = cv.CreateVideoWriter('/n/hoekstrafs1/burrowing/antfarms/data/_2011fall/20100818-1_PO32-2_PO34-3_PO17/test_opencv_write.avi',cv.FOURCC('x','v','i','d'), 29.97, pixdim,1)
for ol,fr in zip(ols[450:],frames[:450]):
    cv_im = mat_polys2cv(fr,ol,150,4)
    cv.WriteFrame(vidwriter,cv_im)
    
for ol,fr in zip(ols[450:],frames[:450]):
    cv_im = mat_polys2cv(fr,ol,150,4)
    nll = cv.WriteFrame(vidwriter,cv_im)
    

def mat_polys2cv(m,polys,dpi=80,scale=1):
    pylab.close(1)
    fig = pylab.figure(1,figsize=tuple(reversed([(float(i)/dpi)*scale for i in m.shape])),dpi=dpi)
    ax = pylab.matshow(m,fignum=1)
    ax = pylab.figure(1).axes[0]
    for c,p in zip(iplot.subspectrum(len(polys)),polys):
        ax.add_patch(matplotlib.patches.Polygon(p,fc='none',ec=c))
    pylab.plot()
    pylab.ylim(0,m.shape[0])
    pylab.xlim(0,m.shape[1])
    buf = io.BytesIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    pi = Image.open(buf).convert('RGB')
    cv_im = cv.CreateImageHeader(pi.size, cv.IPL_DEPTH_8U, 3)
    cv.SetData(cv_im, pi.tostring(),pi.size[0]*3)
    buf.close()
    return cv_im
_ip.system("rm 20100818-1_PO32-2_PO34-3_PO17/test_opencv_write.avi")
vidwriter = cv.CreateVideoWriter('/n/hoekstrafs1/burrowing/antfarms/data/_2011fall/20100818-1_PO32-2_PO34-3_PO17/test_opencv_write.avi',cv.FOURCC('x','v','i','d'), 29.97, pixdim,1)
for ol,fr in zip(ols[450:],frames[:450]):
    cv_im = mat_polys2cv(fr,ol,150,4)
    nll = cv.WriteFrame(vidwriter,cv_im)
    

cv_im = mat_polys2cv(m,[],150,2)
cv_im
pixdim = (cv_im.width,cv_im.height)
def mat_polys2cv(m,polys,dpi=80,scale=1):
    pylab.close(1)
    fig = pylab.figure(1,figsize=tuple(reversed([(float(i)/dpi)*scale for i in m.shape])),dpi=dpi)
    ax = pylab.matshow(m,fignum=1)
    ax = pylab.figure(1).axes[0]
    for c,p in zip(iplot.subspectrum(len(polys)),polys):
        ax.add_patch(matplotlib.patches.Polygon(p,fc='none',ec=c,lw=2))
    pylab.plot()
    pylab.ylim(m.shape[0],0)
    pylab.xlim(0,m.shape[1])
    buf = io.BytesIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    pi = Image.open(buf).convert('RGB')
    cv_im = cv.CreateImageHeader(pi.size, cv.IPL_DEPTH_8U, 3)
    cv.SetData(cv_im, pi.tostring(),pi.size[0]*3)
    buf.close()
    return cv_im

_ip.system("rm 20100818-1_PO32-2_PO34-3_PO17/test_opencv_write.avi")
vidwriter = cv.CreateVideoWriter('/n/hoekstrafs1/burrowing/antfarms/data/_2011fall/20100818-1_PO32-2_PO34-3_PO17/test_opencv_write.avi',cv.FOURCC('x','v','i','d'), 29.97, pixdim,1)
cv_im = mat_polys2cv(m,[],150,1)
pixdim = (cv_im.width,cv_im.height)
_ip.system("rm 20100818-1_PO32-2_PO34-3_PO17/test_opencv_write.avi")
vidwriter = cv.CreateVideoWriter('/n/hoekstrafs1/burrowing/antfarms/data/_2011fall/20100818-1_PO32-2_PO34-3_PO17/test_opencv_write.avi',cv.FOURCC('x','v','i','d'), 29.97, pixdim,1)
cv_im = mat_polys2cv(m,[],80,1)
_ip.system("rm 20100818-1_PO32-2_PO34-3_PO17/test_opencv_write.avi")
vidwriter = cv.CreateVideoWriter('/n/hoekstrafs1/burrowing/antfarms/data/_2011fall/20100818-1_PO32-2_PO34-3_PO17/test_opencv_write.avi',cv.FOURCC('x','v','i','d'), 29.97, pixdim,1)
pixdim = (cv_im.width,cv_im.height)
_ip.system("rm 20100818-1_PO32-2_PO34-3_PO17/test_opencv_write.avi")
vidwriter = cv.CreateVideoWriter('/n/hoekstrafs1/burrowing/antfarms/data/_2011fall/20100818-1_PO32-2_PO34-3_PO17/test_opencv_write.avi',cv.FOURCC('x','v','i','d'), 29.97, pixdim,1)
for ol,fr in zip(ols[450:],frames[:450]):
    cv_im = mat_polys2cv(fr,ol,80,1)
    nll = cv.WriteFrame(vidwriter,cv_im)
    


#?summarize_segment_opencv.advance_analysis
#?summarize_segment_opencv.advance_analysis?
for i xrange(600):
for i in xrange(600):
#?summarize_segment_opencv.advance_analysis?
for i in xrange(600):
    mm = vidtools.shift_frames_return_diff(stream,frames,currsum,denom,seglen)
#?summarize_segment_opencv.advance_analysis?
for i in xrange(600):
    mm = vidtools.shift_frames_return_diff(stream,frames,currsum,denom,seglen)
    ol = vidtools.chain_outlines_from_mask(mm>cutoff,preshrink=1,debug=False,return_termini=False,order_points=True,sort_outlines=False)
    cv_im = mat_polys2cv(frames[(seglen/2)],ol,80,1)
    nll = cv.WriteFrame(vidwriter,cv_im)
for i in xrange(600):
    mm = vidtools.shift_frames_return_diff(stream,frames,currsum,denom,seglen)
    ol = vidtools.chain_outlines_from_mask(mm>cutoff,preshrink=1,debug=False,return_termini=False,order_points=True,sort_outlines=False)
    cv_im = mat_polys2cv(frames[(seglen/2)],ol,80,1)
    nll = cv.WriteFrame(vidwriter,cv_im)
    

for i in xrange(600):
    mm = vidtools.shift_frames_return_diff(stream,frames,currsum,denom,seglen)
    ol = vidtools.chain_outlines_from_mask(mm>cutoff,preshrink=1,debug=False,return_termini=False,order_points=True,sort_outlines=False)
    cv_im = mat_polys2cv(frames[(seglen/2)],ol,80,1)
    nll = cv.WriteFrame(vidwriter,cv_im)
    

for i in xrange(600):
    mm = vidtools.shift_frames_return_diff(stream,frames,currsum,denom,seglen)
    ol = vidtools.chain_outlines_from_mask(mm>cutoff,preshrink=1,debug=False,return_termini=False,order_points=True,sort_outlines=False)
    cv_im = mat_polys2cv(frames[(seglen/2)],ol,80,1)
    nll = cv.WriteFrame(vidwriter,cv_im)
    


vid = '20110916-1_BW-1143_2_BW-1145_3_BW-1148_4_BW-1142/merge6mbit_720_BW-1148_0-46860.mp4'
stream = cv.CaptureFromFile(vid)
nframes = 180
nparts = 180
cut_step = 0.001
scores,dists = vidtools.run_mousezopt(vid,seglen,nframes,2,nparts,cut_step)
cutoff_rank,cutoff = vidtools.choose_cutoff(scores,cut_step) #or cut_step*2
size_h,size_bins,fol_h,fol_bins = dists[cutoff]
min_arc_score = (2*max(size_h))+max(fol_h)

#video stream
stream = cv.CaptureFromFile(vid)

cutoff
frames,currsum,denom = vidtools.init_frames(stream,seglen)
SHAPE = frames[0].shape


# init empty receivers 
hsl = seglen/2
ols = []
objs = {}
objs_sizes = {}
objs_fols = {}
to_retire_objs = {}
to_retire_objs_sizes = {}
to_retire_objs_fols = {}


#init object arcs
ols,ols_offset,frames_offset,objs,splits,objs_sizes,objs_fols,prelast_avg,prelast_mm,to_retire_objs,to_retire_objs_sizes,to_retire_objs_fols = summarize_segment_opencv.init_objects(stream,frames,currsum,denom,seglen,cutoff,frames_offset,SHAPE,size_h,size_bins,fol_h,fol_bins)
cv_im = mat_polys2cv(frames[0],[],80,1)
cv_im
pixdim = (cv_im.width,cv_im.height)
vidwriter = cv.CreateVideoWriter(vid[:-4]+'-mousetrack.avi' , cv.FOURCC('x','v','i','d'), 29.97, pixdim,1)
for ol,fr in zip(ols[450:],frames[:450]):
    cv_im = mat_polys2cv(fr,ol,80,1)
    nll = cv.WriteFrame(vidwriter,cv_im)
    

for i in xrange(6000):
    mm = vidtools.shift_frames_return_diff(stream,frames,currsum,denom,seglen)
    ol = vidtools.chain_outlines_from_mask(mm>cutoff,preshrink=1,debug=False,return_termini=False,order_points=True,sort_outlines=False)
    cv_im = mat_polys2cv(frames[(seglen/2)],ol,80,1)
    nll = cv.WriteFrame(vidwriter,cv_im)
    

_ip.magic("history -n 300")

vid = '20110916-1_BW-1143_2_BW-1145_3_BW-1148_4_BW-1142/merge6mbit_720_BW-1142_0-46860.mp4'
nframes = 180
nparts = 180
cut_step = 0.001
scores,dists = vidtools.run_mousezopt(vid,seglen,nframes,2,nparts,cut_step)
cutoff_rank,cutoff = vidtools.choose_cutoff(scores,cut_step) #or cut_step*2
size_h,size_bins,fol_h,fol_bins = dists[cutoff]
min_arc_score = (2*max(size_h))+max(fol_h)

#video stream
stream = cv.CaptureFromFile(vid)

cutoff
frames,currsum,denom = vidtools.init_frames(stream,seglen)
SHAPE = frames[0].shape


# init empty receivers 
hsl = seglen/2
ols = []
objs = {}
objs_sizes = {}
objs_fols = {}
to_retire_objs = {}
to_retire_objs_sizes = {}
to_retire_objs_fols = {}


#init object arcs
ols,ols_offset,frames_offset,objs,splits,objs_sizes,objs_fols,prelast_avg,prelast_mm,to_retire_objs,to_retire_objs_sizes,to_retire_objs_fo
ls = summarize_segment_opencv.init_objects(stream,frames,currsum,denom,seglen,cutoff,frames_offset,SHAPE,size_h,size_bins,fol_h,fol_bins)
cv_im = mat_polys2cv(frames[0],[],80,1)
cv_im
pixdim = (cv_im.width,cv_im.height)
vidwriter = cv.CreateVideoWriter(vid[:-4]+'-mousetrack.avi' , cv.FOURCC('x','v','i','d'), 29.97, pixdim,1)
for ol,fr in zip(ols[450:],frames[:450]):
    cv_im = mat_polys2cv(fr,ol,80,1)
    nll = cv.WriteFrame(vidwriter,cv_im)
    
for i in xrange(6000):
    mm = vidtools.shift_frames_return_diff(stream,frames,currsum,denom,seglen)
    ol = vidtools.chain_outlines_from_mask(mm>cutoff,preshrink=1,debug=False,return_termini=False,order_points=True,sort_outlines=False)
    cv_im = mat_polys2cv(frames[(seglen/2)],ol,80,1)
    nll = cv.WriteFrame(vidwriter,cv_im)
ols,ols_offset,frames_offset,objs,splits,objs_sizes,objs_fols,prelast_avg,prelast_mm,to_retire_objs,to_retire_objs_sizes,to_retire_objs_fols = ls
vidwriter = cv.CreateVideoWriter(vid[:-4]+'-mousetrack.avi' , cv.FOURCC('x','v','i','d'), 29.97, pixdim,1)
for ol,fr in zip(ols[450:],frames[:450]):
    cv_im = mat_polys2cv(fr,ol,80,1)
    nll = cv.WriteFrame(vidwriter,cv_im)
    

for i in xrange(6000):
    mm = vidtools.shift_frames_return_diff(stream,frames,currsum,denom,seglen)
    ol = vidtools.chain_outlines_from_mask(mm>cutoff,preshrink=1,debug=False,return_termini=False,order_points=True,sort_outlines=False)
    cv_im = mat_polys2cv(frames[(seglen/2)],ol,80,1)
    nll = cv.WriteFrame(vidwriter,cv_im)
    

plot(size_h)
pylab.plot(size_h)
show()
pylab.show()
_ip.magic("history -n")
#?history
_ip.magic("history -n -f /n/home08/brantp/code/video_analysis/ipy_hist-write_vid_in_realtime.py")
_ip.system("more /n/home08/brantp/code/video_analysis/ipy_hist-write_vid_in_realtime.py")
_ip.system("more /n/home08/brantp/code/video_analysis/ipy_hist-write_vid_in_realtime.py")
_ip.magic("history -n 0,900 -f /n/home08/brantp/code/video_analysis/ipy_hist-write_vid_in_realtime.py")
#?history
_ip.magic("history 0 900 -n -f /n/home08/brantp/code/video_analysis/ipy_hist-write_vid_in_realtime.py")
_ip.magic("history 0 874 -n -f /n/home08/brantp/code/video_analysis/ipy_hist-write_vid_in_realtime.py")
_ip.magic("history -n -f /n/home08/brantp/code/video_analysis/ipy_hist-write_vid_in_realtime.py 0 900")
