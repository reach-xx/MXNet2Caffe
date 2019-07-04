# import numpy as np
# import caffe
# import sys
#
# import numpy as np
#
# MEAN_NPY_PATH = 'mean.npy'
# mean = np.ones([3, 112, 112], dtype=np.float)
# mean[0, :, :] = 127.5
# mean[1, :, :] = 127.5
# mean[2, :, :] = 127.5
#
# np.save(MEAN_NPY_PATH, mean)
# meanProtoPath = 'mean.binaryproto'
# blob = caffe.proto.caffe_pb2.BlobProto()
# with open(MEAN_NPY_PATH, 'rb') as f:
#     mean = np.load(f)
# blob.channels= 3
# blob.height=mean.shape[0]
# blob.width = mean.shape[1]
# blob.data.extend(mean.astype(float).flat)
# binaryprotoFile = open(meanProtoPath, 'wb')
# binaryprotoFile.write(blob.SerializeToString())
# binaryprotoFile.close()
from __future__ import division
import sklearn.preprocessing as preprocessing
import numpy as np



#
# X = [[ 1., -1.,  2.],[ 2.,  0.,  0.],[ 0.,  1., -1.]]

# X = [[1, 2, 3, -2]]
#
# X_normalized = preprocessing.normalize(X, norm='l2')
#
# print(X_normalized)
#
# a = np.zeros((1, 512), dtype=np.float32)
# with open("liu_face_rgb.ff", "r") as r:
#     lines = r.readlines()
#     i = 0
#     for line in lines:
#         a[0][i] = float(line)
#         i += 1
# a_embedding = preprocessing.normalize(a).flatten()
# print(a_embedding.shape)
#
# e = np.square(a)
# print(e.shape)
# value = np.sum(e, axis=1)
# sqrt = np.sqrt(value)
#
# f = a/sqrt
# e_embedding = f.flatten()
# dist = np.sum(np.square(a_embedding-e_embedding))
# print("a - e distance: ", dist)
# sim = np.dot(a_embedding, e_embedding.T)
# print("dot: ", sim)



# # print(a_embedding)
#
# b = np.zeros((1, 512), dtype=np.float32)
# with open("liu_face2_rgb.ff", "r") as r:
#     lines = r.readlines()
#     i = 0
#     for line in lines:
#         b[0][i] = float(line)
#         i += 1
# b_embedding = preprocessing.normalize(b).flatten()
# # print(b_embedding)
#
# c = np.zeros((1, 512), dtype=np.float32)
# with open("liu_face50_rgb.ff", "r") as r:
#     lines = r.readlines()
#     i = 0
#     for line in lines:
#         c[0][i] = float(line)
#         i += 1
# c_embedding = preprocessing.normalize(c).flatten()
#
#
# d = np.zeros((1, 512), dtype=np.float32)
# with open("liu_face_no_rgb.ff", "r") as r:
#     lines = r.readlines()
#     i = 0
#     for line in lines:
#         d[0][i] = float(line)
#         i += 1
# d_embedding = preprocessing.normalize(d).flatten()
#
# dist = np.sum(np.square(b_embedding-a_embedding))
# print("b -a distance: ", dist)
# sim = np.dot(b_embedding, a_embedding.T)
# print("dot: ", sim)
#
# dist = np.sum(np.square(c_embedding-a_embedding))
# print("c -a distance: ", dist)
# sim = np.dot(c_embedding, a_embedding.T)
# print("dot: ", sim)
#
# dist = np.sum(np.square(c_embedding-b_embedding))
# print("c - b distance: ", dist)
# sim = np.dot(c_embedding, b_embedding.T)
# print("dot: ", sim)
#
# dist = np.sum(np.square(b_embedding-d_embedding))
# print("b - d distance: ", dist)
# sim = np.dot(b_embedding, d_embedding.T)
# print("dot: ", sim)
#

import math
import numpy as np
import cv2





class SimilarityTransform(object):
    """2D similarity transformation.
    Has the following form::
        X = a0 * x - b0 * y + a1 =
          = s * x * cos(rotation) - s * y * sin(rotation) + a1
        Y = b0 * x + a0 * y + b1 =
          = s * x * sin(rotation) + s * y * cos(rotation) + b1
    where ``s`` is a scale factor and the homogeneous transformation matrix is::
        [[a0  b0  a1]
         [b0  a0  b1]
         [0   0    1]]
    The similarity transformation extends the Euclidean transformation with a
    single scaling factor in addition to the rotation and translation
    parameters.
    Parameters
    ----------
    matrix : (3, 3) array, optional
        Homogeneous transformation matrix.
    scale : float, optional
        Scale factor.
    rotation : float, optional
        Rotation angle in counter-clockwise direction as radians.
    translation : (tx, ty) as array, list or tuple, optional
        x, y translation parameters.
    Attributes
    ----------
    params : (3, 3) array
        Homogeneous transformation matrix.
    """

    def __init__(self, matrix=None, scale=None, rotation=None,
                 translation=None):
        params = any(param is not None
                     for param in (scale, rotation, translation))

        if params and matrix is not None:
            raise ValueError("You cannot specify the transformation matrix and"
                             " the implicit parameters at the same time.")
        elif matrix is not None:
            if matrix.shape != (3, 3):
                raise ValueError("Invalid shape of transformation matrix.")
            self.params = matrix
        elif params:
            if scale is None:
                scale = 1
            if rotation is None:
                rotation = 0
            if translation is None:
                translation = (0, 0)

            self.params = np.array([
                [math.cos(rotation), - math.sin(rotation), 0],
                [math.sin(rotation),   math.cos(rotation), 0],
                [                 0,                    0, 1]
            ])
            self.params[0:2, 0:2] *= scale
            self.params[0:2, 2] = translation
        else:
            # default to an identity transform
            self.params = np.eye(3)

    def ___umeyama(self, src, dst, estimate_scale):
        """Estimate N-D similarity transformation with or without scaling.
        Parameters
        ----------
        src : (M, N) array
            Source coordinates.
        dst : (M, N) array
            Destination coordinates.
        estimate_scale : bool
            Whether to estimate scaling factor.
        Returns
        -------
        T : (N + 1, N + 1)
            The homogeneous similarity transformation matrix. The matrix contains
            NaN values only if the problem is not well-conditioned.
        References
        ----------
        .. [1] "Least-squares estimation of transformation parameters between two
                point patterns", Shinji Umeyama, PAMI 1991, :DOI:`10.1109/34.88573`
        """

        num = src.shape[0]
        dim = src.shape[1]

        # Compute mean of src and dst.
        src_mean = src.mean(axis=0)
        print("src_mean: ", src_mean)
        dst_mean = dst.mean(axis=0)
        print("src_mean: ", dst_mean)

        # Subtract mean from src and dst.
        src_demean = src - src_mean
        dst_demean = dst - dst_mean
        print("src_demean: ", src_demean)
        print("dst_demean: ", dst_demean)

        # Eq. (38).
        A = dst_demean.T @ src_demean / num
        print("num: ", num, dim)
        print("A:  ", A)
        print("B:  ", np.dot(dst_demean.T, src_demean)/num)

        # Eq. (39).
        d = np.ones((dim,), dtype=np.double)
        if np.linalg.det(A) < 0:
            d[dim - 1] = -1

        T = np.eye(dim + 1, dtype=np.double)

        U, S, V = np.linalg.svd(A)
        # U, S, V = cv2.SVD(A)

        # Eq. (40) and (43).
        rank = np.linalg.matrix_rank(A)
        print("rank:", rank)
        if rank == 0:
            return np.nan * T
        elif rank == dim - 1:
            if np.linalg.det(U) * np.linalg.det(V) > 0:
                T[:dim, :dim] = U @ V
            else:
                s = d[dim - 1]
                d[dim - 1] = -1
                T[:dim, :dim] = U @ np.diag(d) @ V
                d[dim - 1] = s
        else:
            T[:dim, :dim] = U @ np.diag(d) @ V

        if estimate_scale:
            # Eq. (41) and (42).
            scale = 1.0 / src_demean.var(axis=0).sum() * (S @ d)
        else:
            scale = 1.0

        T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean.T)
        T[:dim, :dim] *= scale

        return T


    def estimate(self, src, dst):
        """Estimate the transformation from a set of corresponding points.
        You can determine the over-, well- and under-determined parameters
        with the total least-squares method.
        Number of source and destination coordinates must match.
        Parameters
        ----------
        src : (N, 2) array
            Source coordinates.
        dst : (N, 2) array
            Destination coordinates.
        Returns
        -------
        success : bool
            True, if model estimation succeeds.
        """

        self.params = self.___umeyama(src, dst, True)

        return True

    @property
    def scale(self):
        if abs(math.cos(self.rotation)) < np.spacing(1):
            # sin(self.rotation) == 1
            scale = self.params[1, 0]
        else:
            scale = self.params[0, 0] / math.cos(self.rotation)
        return scale

src = np.array([
  [30.2946, 51.6963],
  [65.5318, 51.5014],
  [48.0252, 71.7366],
  [33.5493, 92.3655],
  [62.7299, 92.2041] ], dtype=np.float32 )

src[:,0] += 8.0
print('-'*20)
print(src)

dst = np.array([[39., 47.],
                [77., 43.],
                [62., 72.],
                [41., 78.],
                [78., 80.]], dtype=np.float32)
print(dst)
print("-"*20)

tran = SimilarityTransform()
tran.estimate(dst, src)
M = tran.params[0:2, :]
# a = dst.reshape(1, 5, 2)
# print(a)
# M = cv2.estimateRigidTransform(dst.reshape(1, 5, 2), src.reshape(1, 5, 2), True)
print("M: ", M)
