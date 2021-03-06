import numpy as np
from CLASS_4.week1.Convolutional_step_by_step import padding
from CLASS_4.week1.Convolutional_step_by_step.Conv_forward import conv_forward


def conv_backward(dZ, cache):
    """
    Implement the backward propagation  for a convolution function

    Arguments:
    dZ -- gradient of the cost with respect to the output of the conv layer(Z), numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward(), output of conv_forward()

    Returns:
        dA_prev -- gradient of the cost with respect to the input of the conv layer(A_prev)
                    numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
        dW -- gradient of the cost with respect to the Weights of the conv layer(W)
                    numpy array of shape (f, f, n_C_prev, n_C)
        db -- gradient of the cost with respect to the bias of the conv layer(b)
                   numpy array of shape (1, 1, 1, n_C)
    """

    # Retrieve information from cache
    (A_prev, W, b, hparameters) = cache

    # Retrieve dimensions from A_prev
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # Retrieve dimensions from dZ
    (m, n_H, n_W, n_C) = dZ.shape

    # Retrieve dimensions from W's shape
    (f, f, n_C_prev, n_C) = W.shape

    # Retrieve information from hparameters
    stride = hparameters["stride"]
    pad = hparameters["pad"]

    # Initialize dA_prev, dW, db with correct shape
    dA_prev = np.zeros(shape=A_prev.shape)
    dW = np.zeros(shape=W.shape)
    db = np.zeros(shape=b.shape)

    # pad A_prev and dA_prev
    A_prev_pad = padding.zero_pad(A_prev, pad)
    dA_prev_pad = padding.zero_pad(dA_prev, pad)

    for i in range(m):  # loop over the training examples
        # select ith training example from A_prev_pad and dA_prev_pad
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]
        for h in range(n_H):        # loop over vertical axis of the output volume
            for w in range(n_W):    # loop over horizontal axis of the output volume
                for c in range(n_C):   # loop over the channels of the output volume

                    # Find the corners of the current "slice"
                    vert_start = h * stride
                    vert_end = h * stride + f
                    horiz_start = w * stride
                    horiz_end = w * stride + f

                    # Update gradients for the window and the filter's parameters using the code formulas given above
                    da_prev_pad[vert_start: vert_end, horiz_start: horiz_end, :] += dZ[h, w] * W[:, :, :, c]
                    dW[:, :, :, c] += dZ[h, w] * a_prev_pad[vert_start: vert_end, horiz_start: horiz_end, :]
                    db[:, :, :, c] += dZ[h, w]

        # Set the ith training example's dA_prev to the unpaded da_prev_pad (Hint: use X[pad:-pad, pad:-pad, :])
        dA_prev[i, :, :, :] = dA_prev_pad[i, pad:-pad, pad:-pad, :]

    # Making sure your output shape is correct
    assert dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev)

    return dA_prev, dW, db



if __name__ == "__main__":
    np.random.seed(1)
    A_prev = np.random.randn(10, 4, 4, 3)
    W = np.random.randn(2, 2, 3, 8)
    b = np.random.randn(1, 1, 1, 8)
    hparameters = {"pad": 2,
                   "stride": 2}

    Z, cache_conv = conv_forward(A_prev, W, b, hparameters)
    print("Z's mean =", np.mean(Z))
    print("Z[3,2,1] =", Z[3, 2, 1])
    print("cache_conv[0][1][2][3] =", cache_conv[0][1][2][3])









