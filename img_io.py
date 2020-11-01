import numpy as np
import scipy.misc
import OpenEXR, Imath
import cv2 as cv

class IOException(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

# Read and prepare 8-bit image in a specified resolution
def readLDR(file, sz, clip=True, sc=1.0):
    try:
        x_buffer = scipy.misc.imread(file)

        # Clip image, so that ratio is not changed by image resize
        if clip:
            sz_in = [float(x) for x in x_buffer.shape]
            sz_out = [float(x) for x in sz]

            r_in = sz_in[1]/sz_in[0]
            r_out = sz_out[1]/sz_out[0]

            if r_out / r_in > 1.0:
                sx = sz_in[1]
                sy = sx/r_out
            else:
                sy = sz_in[0]
                sx = sy*r_out

            yo = np.maximum(0.0, (sz_in[0]-sy)/2.0)
            xo = np.maximum(0.0, (sz_in[1]-sx)/2.0)

            x_buffer = x_buffer[int(yo):int(yo+sy),int(xo):int(xo+sx),:]

        # Image resize and conversion to float
        x_buffer = scipy.misc.imresize(x_buffer, sz)
        x_buffer = x_buffer.astype(np.float32)/255.0

        # Scaling and clipping
        if sc > 1.0:
            x_buffer = np.minimum(1.0, sc*x_buffer)

        x_buffer = x_buffer[np.newaxis,:,:,:]

        return x_buffer
            
    except Exception as e:
        raise IOException("Failed reading LDR image: %s"%e)

# Write exposure compensated 8-bit image
def writeLDR(img, file, exposure=0):
    
    # Convert exposure fstop in linear domain to scaling factor on display values
    sc = np.power(np.power(2.0, exposure), 0.5)

    try:
        scipy.misc.toimage(sc*np.squeeze(img), cmin=0.0, cmax=1.0).save(file)
    except Exception as e:
        raise IOException("Failed writing LDR image: %s"%e)

# Write HDR image using OpenEXR
def writeEXR(img, file):
    try:
        img = np.squeeze(img)
        sz = img.shape
        header = OpenEXR.Header(sz[1], sz[0])
        half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
        header['channels'] = dict([(c, half_chan) for c in "RGB"])
        out = OpenEXR.OutputFile(file, header)
        R = (img[:,:,0]).astype(np.float16).tostring()
        G = (img[:,:,1]).astype(np.float16).tostring()
        B = (img[:,:,2]).astype(np.float16).tostring()
        out.writePixels({'R' : R, 'G' : G, 'B' : B})
        out.close()
    except Exception as e:
        raise IOException("Failed writing EXR: %s"%e)


# Read training data (HDR ground truth and LDR JPEG images)
def load_training_pair(name_hdr, name_jpg):

    # data = np.fromfile(name_hdr, dtype=np.float32)
    # ss = len(data)
    
    # if ss < 3:
    #     return (False,0,0)

    # sz = np.floor(data[0:3]).astype(int)
    # npix = sz[0]*sz[1]*sz[2]
    # meta_length = ss - npix

    # # Read binary HDR ground truth
    # y = np.reshape(data[meta_length:meta_length+npix], (sz[0], sz[1], sz[2]))
    y = cv.imread(name_hdr, -1)
    y = cv.cvtColor(y, cv.COLOR_BGR2RGB)
    y = y[:320,:320,:]
    # Read JPEG LDR image
    x = scipy.misc.imread(name_jpg).astype(np.float32)/255.0
    x = x[:320,:320,:]

    return (True,x,y)