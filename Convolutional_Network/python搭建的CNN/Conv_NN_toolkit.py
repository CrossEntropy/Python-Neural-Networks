import numpy as np


def zero_pad(X, pad):
    """
    对X进行pad
     (m, n_H, n_W, n_C) --> (m, n_H + 2 * pad, n_W + 2 * pad, n_C)
    :param X: numpy数组， shape=(m, n_H, n_W, n_C)
    :param pad: 标量, 填充的圈数
    :return X_pad: 填充之后的numpy 数组, shape=(m, n_H + 2 * pad, n_W + 2 * pad, n_C)
    """

    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), "constant", consant_value=0)
    return X_pad


def conv_single_step(a_slice, w, b):
    """
    对单个filter、单个a的切片进行卷积操作
    :param a_slice: 对a进行slice, 是一个numpy的三维数组, a的shape = (f, f, n_C)
    :param w: numpy三维数组, w.shape = (f, f, n_C)
    :param b: numpy的三维数组, b.shape = (1, 1, 1)
    :return z: scaler
    """

    assert a_slice.shape == w.shape  # 保证 a_slice的shape和w的shape相同
    assert b.ndim == 3  #保证b是一个三维数组

    z = np.sum(a_slice * w)
    z = z + np.squeeze(b)  # 将b变成一个scalar
    return z


def conv_forward(A_prev, W, b, hparameters):

    # retrieve (m, n_H_prev, n_W_prev, n_C_prev) fron A_prev.shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # retrieve (f, f, n_C_prev, n_C) from W.shape
    (f, f, n_C_prev, n_C) = W.shape

    # retrieve stride and pad fron hparameters
    stride = hparameters["stride"]
    pad = hparameters["pad"]

    # 对A_prev 进行填充
    A_pad_prev = zero_pad(A_prev, pad)

    # 获取A的dimensions
    n_H = (n_H_prev - f + 2*pad)//stride + 1
    n_W = (n_W_prev - f + 2*pad)//stride + 1

    # initialize the dimensions of Z
    Z = np.zeros(shape=(m, n_H, n_W, n_C))

    for i in range(m):  # loop over the training examples
        a_pad_prev = A_pad_prev[i]  # select single training example
        for h in range(n_H): # loop over the vertical axis
            for w in range(n_W): # loop over the horizontal axis
                for c in range(n_C): # loop over all filters

                    # found the "corners" of a_slicce
                    vert_start = h * stride
                    vert_end = h * stride + f
                    horiz_start = w * stride
                    horiz_end = w * stride + f

                    # 对 a_pad_prev 进行切片, 是一个三维数组
                    a_slice = a_pad_prev[vert_start: vert_end, horiz_start:horiz_end, ]

                    # 进行卷积
                    z  = conv_single_step(a_slice, W[:, :, :, c], b[:, :, :, c])

                    # 对输出对应的位置赋值
                    Z[i, h, w, c] = z

    # Save information in "cache" for the backprop
    cache = (A_prev, W, b, hparameters)

    return Z, cache


def pool_forward(A_prev, hparameters, mode="max"):

    # retrieve (m, n_H_pev, n_W_prev, n_C_prev) from A_prev.shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # retrieve (f, stride) from hparameters
    f = hparameters["f"]
    stride = hparameters["stride"]

    # 获取A的dimensions
    n_H = (n_H_prev - f) // stride + 1
    n_W = (n_W_prev - f) // stride + 1

    # initialize the dimensions of A
    A = np.zeros(shape=(m, n_H, n_W, n_C_prev))

    for i in range(m):   # loop over the training examples
        a_prev = A_prev[i]  # select single training example
        for h in range(n_H): # loop over the vertical axis
            for w in range(n_W): # loop over the horizontal axis
                for c in range(n_C_prev):  # loop over the channels of A_prev

                    # find the "corners" of a_slice
                    vert_start = h * stride
                    vert_end = h * stride + f
                    horiz_start = w * stride
                    horiz_end = w * stride + f

                    if mode == "max":
                        A[i, h, w, c] = np.max(a_prev[vert_start: vert_end, horiz_start: horiz_end, c])
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_prev[vert_start: vert_end, horiz_start: horiz_end, c])

    # save information in "cache" for the backprop
    cache = (A_prev, hparameters)
    return A, cache


def conv_backward(dZ, cache):

    # retrieve (A_prev, W, b, hparameters) from cache
    (A_prev, W, b, hparameters) = cache

    # retrieve (m, n_H, n_W, n_C) from dZ.shape
    (m, n_H, n_W, n_C) = dZ.shape

    # retrieve (m, n_H_prev, n_W_prev, n_C_prev) from A_prev.shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # retrieve stride and pad from hparameters
    stride = hparameters["stride"]
    pad = hparameters["pad"]

    # retrieve (f, f, n_C_prev, n_C) from W.shape
    (f, f, n_C_prev, n_C) = W.shape

    # 对A_prev进行pad
    A_pad_prev = zero_pad(A_prev, pad)

    # initialize the dimensions of dA_prev, dA_padd_prev
    dA_prev = np.zeros(shape=A_prev.shape)
    dA_padd_prev = np.zeros(shape=A_pad_prev.shape)

    # initilazie the dimensions of dW, db
    dW = np.zeros(shape=W.shape)
    db = np.zeros(shape=b.shape)

    for i in range(m):  # loop over the training examples
        a_pad_prev = A_pad_prev[i]  # loop over single example
        for h in range(n_H):    # loop over the  vertiac axis
            for w in range(n_W):  # loop over the horizontal axis
                for c in range(n_C):  # loop over the channels of dZ

                    # find the "corners" for a_slice
                    vert_start = h * stride
                    vert_end = h * stride + f
                    horiz_start = w * stride + f
                    horiz_end = w * stride + f

                    # 找出a_slice
                    a_slice = a_pad_prev[vert_start: vert_end, horiz_start: horiz_end, :]

                    # 求出dw, db
                    dW[:, :, :, c] += dZ[i, h, w, c] * a_slice
                    db[:, :, :, c] += dZ[i, h, w, c]

                    # 求出dA_pad_prev
                    dA_padd_prev[i, vert_start: vert_end, horiz_start: horiz_end, :] += dZ[i, h, w, c] * W[:, :, :, c]
        # 将dA_pad_prev--->dA_prev
        dA_prev[i, :, :, :] = dA_padd_prev[i, pad:-pad, pad:-pad, :]

    return dA_prev


def create_mask_from_window(A_slice_prev):
    mask = (A_slice_prev == np.max(A_slice_prev))
    return mask


def distribute_value(dZ, shape):

    value = dZ * np.ones(shape) /(shape[0]**2)
    return value


def pool_backward(dA, cache, mode="max"):

    (m, n_H, n_W, n_C) = dA.shape
    (A_prev, hparameters) = cache
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    stride = hparameters["stride"]
    f = hparameters["f"]

    dA_prev = np.zeros(shape=A_prev.shape)

    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):

                    vert_start = h * stride
                    vert_end = h *stride + f
                    horiz_start = w * stride
                    horiz_end = w * stride + f
                    a_slice = A_prev[i, vert_start: vert_end, horiz_start: horiz_end, c]

                    if mode == "average":
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] = distribute_value(dA[i, h, w, c], (f, f))

                    elif mode == "max":
                        a_slice = A_prev[i, vert_start: vert_end, horiz_start: horiz_end, c]
                        mask = create_mask_from_window(a_slice)
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] = dA[i, h, w, c] * mask

    return dA_prev


if __name__ == "__main__":
    np.random.seed(1)
    A_prev = np.random.randn(5, 5, 3, 2)
    hparameters = {"stride": 1, "f": 2}
    A, cache = pool_forward(A_prev, hparameters)
    dA = np.random.randn(5, 4, 2, 2)

    dA_prev = pool_backward(dA, cache, mode="max")
    print("mode = max")
    print('mean of dA = ', np.mean(dA))
    print('dA_prev[1,1] = ', dA_prev[1, 1])
    print()
    dA_prev = pool_backward(dA, cache, mode="average")
    print("mode = average")
    print('mean of dA = ', np.mean(dA))
    print('dA_prev[1,1] = ', dA_prev[1, 1])









