from CLASS_4.week1.Convolutional_step_by_step.Conv import conv_single_step
from CLASS_4.week1.Convolutional_step_by_step.padding import zero_pad
import numpy as np


def conv_forward(A_prev, W, b, hparameters):
    """
    Implements the forward propagation for a convolution function

    Arguments:
    A_prev -- output activations of the previous layer, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
    b -- Biases, numpy array of shape (1, 1, 1, n_C)
    hparameters -- python dictionary containing "stride" and "pad"

    Returns:
    Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward() function
    """

    # Retrieve dimensions from A_prev's shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # Retrive dimensions from W's shape
    (f, f, n_C_prev, n_C) = W.shape

    # Retrive stride and pad from hparameters
    stride = hparameters["stride"]
    pad = hparameters["pad"]

    # Padding the A_prev
    A_prev_pad = zero_pad(A_prev, pad)
    # initialize Z
    n_H = (n_H_prev - f + 2 * pad) // stride + 1   # round down
    n_W = (n_W_prev - f + 2 * pad) // stride + 1
    Z = np.zeros(shape=(m, n_H, n_W, n_C))

    for i in range(m):  # loop over the training examples
        a_prev_pad = A_prev_pad[i]  #  Select ith training example's padded activation
        for h in range(n_H):        # loop over vertical axis of output volume
            for w in range(n_W):    # loop over horizontal axis of output volume
                for c in range(n_C):  # loop over channels op output volume

                    # Find the corners of the current "slice"
                    vert_start = h * stride
                    vert_end = h * stride + f
                    horiz_start = w * stride
                    horiz_end = w * stride + f

                    # Use the corners to define the (3D) slice of a_prev_pad
                    a_slice_prev_pad = a_prev_pad[vert_start: vert_end, horiz_start: horiz_end, :]

                    #  Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron.
                    Z[i, h, w, c] = conv_single_step(a_slice_prev_pad, W[:, :, :, c], b[:, :, :, c])


    # Making sure your output shape is correct
    assert (Z.shape == (m, n_H, n_W, n_C))

    # Save information in "cache" for the backprop
    cache = (A_prev, W, b, hparameters)

    return Z, cache


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