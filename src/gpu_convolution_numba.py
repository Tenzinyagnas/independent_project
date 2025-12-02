# gpu_convolution_numba.py
import numpy as np
import time

def numba_cuda_available():
    try:
        import numba.cuda  # noqa: F401
        return True
    except Exception:
        return False

def gpu_gaussian_blur_numba(gray_img: np.ndarray, kernel: np.ndarray):
    """
    Numba CUDA kernel: each thread computes one output pixel (grayscale).
    Input: gray_img (2D uint8), kernel (2D float32)
    Returns: out (2D uint8), elapsed_seconds (float)
    """
    import numba
    from numba import cuda

    img = gray_img.astype(np.float32)
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2

    H, W = img.shape
    # pad image
    img_pad = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    out = np.zeros_like(img, dtype=np.float32)

    # transfer to device
    d_img = cuda.to_device(img_pad)
    d_out = cuda.to_device(out)
    d_kernel = cuda.to_device(kernel.astype(np.float32))

    # kernel launcher
    threadsperblock = (16, 16)
    blockspergrid_x = (W + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (H + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    @cuda.jit
    def conv_kernel(img_pad_dev, kernel_dev, out_dev, H, W, kh, kw, pad_h, pad_w):
        x, y = cuda.grid(2)
        if x >= W or y >= H:
            return
        # compute convolution centered at (y, x) in output domain
        acc = 0.0
        for i in range(kh):
            for j in range(kw):
                yy = y + i
                xx = x + j
                acc += img_pad_dev[yy, xx] * kernel_dev[i, j]
        out_dev[y, x] = acc

    t0 = time.perf_counter()
    # launch
    conv_kernel[blockspergrid, threadsperblock](d_img, d_kernel, d_out, H, W, kh, kw, pad_h, pad_w)
    # wait for completion
    cuda.synchronize()
    t1 = time.perf_counter()

    out = d_out.copy_to_host()
    out = np.clip(out, 0, 255).astype('uint8')
    return out, t1 - t0
