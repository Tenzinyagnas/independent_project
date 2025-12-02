# gpu_convolution_cupy.py
import time
import numpy as np

def cupy_available():
    try:
        import cupy  # noqa: F401
        return True
    except Exception:
        return False

def gpu_gaussian_blur_cupy(gray_img: np.ndarray, kernel: np.ndarray):
    """
    GPU convolution using CuPy.
    Tries cupyx.scipy.signal.convolve2d; falls back to naive element-wise if not available.
    Inputs: gray_img (2D uint8), kernel (2D float32)
    Returns: out (2D uint8), elapsed_seconds (float)
    """
    import cupy as cp
    t0 = time.perf_counter()

    img_gpu = cp.asarray(gray_img.astype('float32'))
    kern_gpu = cp.asarray(kernel.astype('float32'))

    # try to use cupyx.scipy.signal.convolve2d for best performance
    try:
        import cupyx.scipy.signal as csignal
        # 'same' returns same shape
        out_gpu = csignal.convolve2d(img_gpu, kern_gpu, mode='same', boundary='symm')
    except Exception:
        # naive separable approach: if kernel is separable, use two 1D convolutions.
        # For general case, do simple sliding-window (inefficient but works).
        kh, kw = kernel.shape
        pad_h = kh // 2
        pad_w = kw // 2
        img_pad = cp.pad(img_gpu, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
        H, W = gray_img.shape
        out_gpu = cp.zeros_like(img_gpu)
        # naive nested loops on GPU are slow if done in python; do convolution with strided trick:
        # create sliding windows via as_strided equivalent using cupy.lib.stride_tricks.sliding_window_view if available
        try:
            from cupy.lib.stride_tricks import sliding_window_view
            windows = sliding_window_view(img_pad, (kh, kw))
            # windows shape: (H, W, kh, kw)
            out_gpu = (windows * kern_gpu).sum(axis=(2,3))
        except Exception:
            # as last resort: move to CPU (fallback)
            out = _cupy_naive_fallback_cpu(gray_img, kernel)
            t1 = time.perf_counter()
            return out, t1 - t0

    out = cp.asnumpy(out_gpu)
    # normalize & convert to uint8 (clip)
    out = np.clip(out, 0, 255).astype('uint8')
    t1 = time.perf_counter()
    return out, t1 - t0

def _cupy_naive_fallback_cpu(gray_img, kernel):
    """
    Extremely conservative fallback to CPU using numpy if cupy sliding windows not available.
    """
    import numpy as np
    H, W = gray_img.shape
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2
    img_pad = np.pad(gray_img.astype('float32'), ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    out = np.zeros_like(gray_img, dtype='float32')
    for i in range(H):
        for j in range(W):
            patch = img_pad[i:i+kh, j:j+kw]
            out[i, j] = (patch * kernel).sum()
    return np.clip(out, 0, 255).astype('uint8')
