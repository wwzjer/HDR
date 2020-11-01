import os
import gzip
import pickle
import numpy as np
import argparse
import lie_learn.spaces.S2 as S2
from torchvision import datasets
import torch
import matplotlib.pyplot as plt
import cv2

NORTHPOLE_EPSILON = 1e-3


def rand_rotation_matrix(deflection=1.0, randnums=None):
    """
    Creates a random rotation matrix.

    deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
    rotation. Small deflection => small perturbation.
    randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.

    # http://blog.lostinmyterminal.com/python/2015/05/12/random-rotation-matrix.html
    """

    if randnums is None:
        randnums = np.random.uniform(size=(3,))

    theta, phi, z = randnums

    theta = theta * 2.0*deflection*np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0*np.pi  # For direction of pole deflection.
    z = z * 2.0*deflection  # For magnitude of pole deflection.

    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.

    r = np.sqrt(z)
    V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
    )

    st = np.sin(theta)
    ct = np.cos(theta)

    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

    # Construct the rotation matrix  ( V Transpose(V) - I ) R.

    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M


def rotate_grid(rot, grid):
    x, y, z = grid
    xyz = np.array((x, y, z))
    x_r, y_r, z_r = np.einsum('ij,jab->iab', rot, xyz)
    return x_r, y_r, z_r


def get_projection_grid(b, grid_type="Driscoll-Healy"):
    ''' returns the spherical grid in euclidean
    coordinates, where the sphere's center is moved
    to (0, 0, 1)'''
    #equidistribution#
    theta, phi = S2.meshgrid(b=b, grid_type=grid_type)
    x_ = np.sin(theta) * np.cos(phi)
    y_ = np.sin(theta) * np.sin(phi)
    z_ = np.cos(theta)
    return x_, y_, z_


def project_sphere_on_xy_plane(grid, projection_origin):
    ''' returns xy coordinates on the plane
    obtained from projecting each point of
    the spherical grid along the ray from
    the projection origin through the sphere '''

    sx, sy, sz = projection_origin
    x, y, z = grid
    z = z.copy() + 1

    t = -z / (z - sz)
    qx = t * (x - sx) + x
    qy = t * (y - sy) + y

    xmin = 1/2 * (-1 - sx) + -1
    ymin = 1/2 * (-1 - sy) + -1

    # ensure that plane projection
    # ends up on southern hemisphere
    rx = (qx - xmin) / (2 * np.abs(xmin))
    ry = (qy - ymin) / (2 * np.abs(ymin))

    return rx, ry


def sample_within_bounds(signal, x, y, bounds):
    ''' '''
    xmin, xmax, ymin, ymax = bounds

    idxs = (xmin <= x) & (x < xmax) & (ymin <= y) & (y < ymax)

    if len(signal.shape) > 2:
        sample = np.zeros((signal.shape[0], x.shape[0], x.shape[1]))
        sample[:, idxs] = signal[:, x[idxs], y[idxs]]
    else:
        sample = np.zeros((x.shape[0], x.shape[1]))
        sample[idxs] = signal[x[idxs], y[idxs]]
    return sample


def sample_bilinear(signal, rx, ry):
    ''' '''

    signal_dim_x = signal.shape[1]
    signal_dim_y = signal.shape[2]

    rx *= signal_dim_x
    ry *= signal_dim_y

    # discretize sample position
    ix = rx.astype(int)
    iy = ry.astype(int)

    # obtain four sample coordinates
    ix0 = ix - 1
    iy0 = iy - 1
    ix1 = ix + 1
    iy1 = iy + 1

    bounds = (0, signal_dim_x, 0, signal_dim_y)

    # sample signal at each four positions
    signal_00 = sample_within_bounds(signal, ix0, iy0, bounds)
    signal_10 = sample_within_bounds(signal, ix1, iy0, bounds)
    signal_01 = sample_within_bounds(signal, ix0, iy1, bounds)
    signal_11 = sample_within_bounds(signal, ix1, iy1, bounds)

    # linear interpolation in x-direction
    fx1 = (ix1-rx) * signal_00 + (rx-ix0) * signal_10
    fx2 = (ix1-rx) * signal_01 + (rx-ix0) * signal_11

    # linear interpolation in y-direction
    return (iy1 - ry) * fx1 + (ry - iy0) * fx2


def project_sphere_new(grid, projection_origin):
    H = 160
    W = 320
    rx = np.zeros((60,60))
    ry = np.zeros((60,60))

    for i in range(60):
        for j in range(60):
            x = grid[0][i,j]
            y = grid[1][i,j]
            z = grid[2][i,j]
            theta = np.arccos(z)
            phi = np.arctan2(y,x)
            if phi < 0:
                phi += 2*np.pi

            ry[i,j] = theta / np.pi
            rx[i,j] = phi / (np.pi*2)
    return rx, ry

def project_2d_on_sphere(signal, grid, projection_origin=None):
    ''' '''
    if projection_origin is None:
        projection_origin = (0, 0, 2 + NORTHPOLE_EPSILON)
        
#    signal[:,240:480,:] = 0
    
#    rx, ry = project_sphere_on_xy_plane(grid, projection_origin)
    rx, ry = project_sphere_new(grid, projection_origin)  # rx [60,60] ry [60,60]
     
    sample = sample_bilinear(signal, rx, ry)     # [500,60,60]

#    # ensure that only north hemisphere gets projected
    sample *= (grid[1] >= 0).astype(np.float64)

    # # rescale signal to [0,1]
    # sample_min = sample.min(axis=(1, 2)).reshape(-1, 1, 1)
    # sample_max = sample.max(axis=(1, 2)).reshape(-1, 1, 1)

    # sample = (sample - sample_min) / (sample_max - sample_min)
    # sample *= 255
    # sample = sample.astype(np.uint8)
    
    
    for k in range(signal.shape[0]):
        original = signal[k,:,:]
        mask = np.zeros(original.shape)
        for i in range(60):
            for j in range(60):
                mask[int(rx[i][j])][int(ry[i][j])] = 1
    

    return sample


def main():
    ''' '''
    parser = argparse.ArgumentParser()

    parser.add_argument("--bandwidth",
                        help="the bandwidth of the S2 signal",
                        type=int,
                        default=30,
                        required=False)
    parser.add_argument("--noise",
                        help="the rotational noise applied on the sphere",
                        type=float,
                        default=1.0,
                        required=False)
    parser.add_argument("--chunk_size",
                        help="size of image chunk with same rotation",
                        type=int,
                        default=500,
                        required=False)
    parser.add_argument("--mnist_data_folder",
                        help="folder for saving the mnist data",
                        type=str,
                        default="MNIST_data",
                        required=False)
    parser.add_argument("--output_file",
                        help="file for saving the data output (.gz file)",
                        type=str,
                        default="126_north_NRNR.gz",
                        required=False)
    parser.add_argument("--no_rotate_train",
                        help="do not rotate train set",
                        dest='no_rotate_train', action='store_true')
    parser.add_argument("--no_rotate_test",
                        help="do not rotate test set",
                        dest='no_rotate_test', action='store_true')

    args = parser.parse_args()

    """
    preparing the HDR data, input: [658, 60, 480, 960] label: [658]
    """

    file_list = "dataset/360hdr/"
    path_list = os.listdir(file_list)
    path_list.sort()
#    path_list = path_list[:40]
    
    num = len(path_list)
    height = 160
    width = 320
    channel = 1
    
    data = np.zeros([num,channel,height,width])
    label = np.zeros([num])
    
    index = 0
    for file_name in path_list:
        print(os.path.join(file_name))
        list = file_name.split('_')
        
        x = cv2.imread(os.path.join(file_list, file_name), -1)[:,:,::-1]
        x = cv2.resize(x, (320, 160), interpolation=cv2.INTER_CUBIC)
        x = np.transpose(x,(2,0,1))
        y = int(list[2][:-4])
        # from rgb to luminance (brightness)
        data[index,0,:,:] = 179 * (x[0,:,:] * 0.2651 + x[1,:,:] * 0.6701 + x[2,:,:] * 0.0648) / 255.0
        label[index] = y
        index += 1

#    for i in range(data.shape[0]):
#        for j in range(data.shape[1]):
#            plt.imsave('verify/'+str(i)+'_'+str(j)+'.jpg', data[i,j,:,:], cmap='gray')
    
    print('data_size', data.shape)
    print('label_size', label.shape)

    HDR_train = {}
    split = 350
    HDR_train['images'] = data[0:split,:,:,:]
    HDR_train['labels'] = label[0:split]
#    rand_sample = np.arange(data.shape[0])
#    np.random.seed(111)
#    np.random.shuffle(rand_sample)
#    print(rand_sample)
#    HDR_train['images'] = data[rand_sample[0:split],:,:,:]
#    HDR_train['labels'] = label[rand_sample[0:split]]
    
    HDR_test = {}
    HDR_test['images'] = data[split:,:,:,:]
    HDR_test['labels'] = label[split:]
#    HDR_test['images'] = data[rand_sample[split:],:,:,:]
#    HDR_test['labels'] = label[rand_sample[split:]]
            
    """
    generate the project dataset
    """
    grid = get_projection_grid(b=args.bandwidth)  # x,y,z coordinates, each 60*60

    # result
    dataset = {}

    no_rotate = {"train": args.no_rotate_train, "test": args.no_rotate_test}
    augment = 0
    
#    for j in range(100):
#        plt.imsave('verify/real'+str(j)+'.jpg', HDR_train['images'][j,0,:,:]/255., cmap='gray')
#
#    for j in range(100):
#        plt.imsave('verify/'+str(j)+'.jpg', np.mean(HDR_train['images'][j,0:3,:,:],axis=0)/255., cmap='gray')
    

    for label, data in zip(["train", "test"], [HDR_train, HDR_test]):

        print("projecting {0} data set".format(label))
        current = 0
        signals = data['images'].astype(np.float32)

        print('done')
        n_signals = signals.shape[0]
        
        if label == 'train':
            projections = np.ndarray(
                ((augment+1) * signals.shape[0], channel, 2 * args.bandwidth, 2 * args.bandwidth),
                dtype=np.uint8)
        else:
            projections = np.ndarray(
                (signals.shape[0], channel, 2 * args.bandwidth, 2 * args.bandwidth),
                dtype=np.uint8)

        while current < n_signals:
            if not no_rotate[label]:
                rot = rand_rotation_matrix(deflection=args.noise)
                rotated_grid = rotate_grid(rot, grid)
            else:
                rotated_grid = grid

            idxs = np.arange(current, min(n_signals, current + args.chunk_size))
            idxs_augment = np.arange(current, min(n_signals*(augment+1), current + args.chunk_size*(1+augment)))
            if label == 'train':
                chunk = signals[idxs]
                chunk_augment = np.zeros((chunk.shape))
                chunk_total = np.zeros((chunk.shape[0]*(augment+1), channel, height, width))
                chunk_total[:chunk.shape[0],:,:,:] = chunk
                for i in range(augment):
                    id = int(np.random.choice(width//2, 1))
                    tmp = chunk[:,:,0:id]
                    chunk_augment[:,:,0:width-id] = chunk[:,:,id:]
                    chunk_augment[:,:,width-id:] = tmp
                    chunk_total[(i+1)*chunk.shape[0]:(i+2)*chunk.shape[0],:,:] = chunk_augment
                chunk = chunk_total
            else:
                chunk = signals[idxs]
            
            if label == 'train':
                for i in range(channel):
                    projections[idxs_augment,i,:,:] = project_2d_on_sphere(chunk[:,i,:,:], rotated_grid)
            else:
                for i in range(channel):
                    projections[idxs,i,:,:] = project_2d_on_sphere(chunk[:,i,:,:], rotated_grid)
                
            if label == 'train':
                current += args.chunk_size * (augment+1)
            else:
                current += args.chunk_size
                
            print("\r{0}/{1}".format(current, n_signals), end="")
        
        print("")
        if label == 'train':
            dataset['train'] = {'images': projections,
                                'labels': np.tile(data['labels'], augment+1)}
        else:
            dataset['test'] = {'images': projections,
                               'labels': data['labels']}
    

    print("writing pickle")
    print(dataset['train']['images'].shape)
    print(dataset['train']['labels'].shape)
    print(dataset['test']['images'].shape)
    print(dataset['test']['labels'].shape)
    
    with gzip.open(args.output_file, 'wb') as f:
        pickle.dump(dataset, f)
    print("done")


if __name__ == '__main__':
    main()
