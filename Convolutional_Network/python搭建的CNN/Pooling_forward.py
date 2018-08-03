import numpy as np


def pool_forward(A_prev, hyparameters, mode="max"):
    """
        Implements the forward pass of the pooling layer

        Arguments:
        A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
        hparameters -- python dictionary containing "f" and "stride"
        mode -- the pooling mode you would like to use, defined as a string ("max" or "average")

        Returns:
        A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
        cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters
        """

    # Retrieve the dimensions from A_prev.shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # Retrieve f and stride from hparameters
    f = hyparameters["f"]
    stride = hyparameters["stride"]

    # Define the dimensions of output
    n_H = (n_H_prev - f) // stride + 1
    n_W = (n_W_prev - f) // stride + 1

    # Initialize output matrix A
    A = np.zeros(shape=(m, n_H, n_W, n_C_prev))

    for i in range(m):   # loop over the training examples
        a_slice_prev = A_prev[i]  # Select the ith training examples activation
        for h in range(n_H):   # loop over the vertical axis of output volume
            for w in range(n_W):  # loop over the horizontal axis of output volume
                for c in range(n_C_prev):   # loop over the channel of output volume

                    # Find the corners of current "slice"
                    vert_start = h * stride
                    vert_end = h * stride + f
                    horiz_start = w * stride
                    horiz_end = w * stride + f

                    if mode == "max":
                        A[i, h, w, c] = np.max(a_slice_prev[vert_start: vert_end, horiz_start: horiz_end, c])
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_slice_prev[vert_start: vert_end, horiz_start: horiz_end, c])

    # cache contains the input and hparameter
    cache = (A_prev, hyparameters)

    return A, cache


if __name__ == "__main__":
    np.random.seed(1)

    A_prev = np.random.randn(2, 4, 4, 3)
    hparameters = {"stride": 2, "f": 3}

    A, cache = pool_forward(A_prev, hparameters)
    print("mode = max")
    print("A =", A)
    print()
    A, cache = pool_forward(A_prev, hparameters, mode="average")
    print("mode = average")
    print("A =", A)