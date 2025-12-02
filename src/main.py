# main.py
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

from cpu_convolution import cpu_gaussian_blur
from gpu_convolution_cupy import cupy_available, gpu_gaussian_blur_cupy
from gpu_convolution_numba import numba_cuda_available, gpu_gaussian_blur_numba

# Paths
ROOT = os.path.dirname(os.path.dirname(__file__))  # project/src/.. -> project/
INPUT_DIR = os.path.join(ROOT, "data", "input")
OUTPUT_DIR = os.path.join(ROOT, "output")
PLOTS_DIR = os.path.join(ROOT, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Gaussian kernel generator
def gaussian_kernel(k=5, sigma=1.0):
    """Return a k x k gaussian kernel (float32) normalized to sum=1."""
    ax = np.arange(-(k//2), k//2 + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel.astype('float32')

def to_gray_uint8(img):
    if img is None:
        return None
    if len(img.shape) == 2:
        return img.astype('uint8')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray.astype('uint8')

def main():
    kernel = gaussian_kernel(k=7, sigma=1.5)  # 7x7 Gaussian
    img_files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png','.jpg','.jpeg','.tif','.tiff'))])
    if len(img_files) == 0:
        print("No input images found in", INPUT_DIR)
        return

    results = {'cpu': [], 'cupy': [], 'numba': []}
    cupy_ok = cupy_available()
    numba_ok = numba_cuda_available()
    print("CuPy available:", cupy_ok, "Numba CUDA available:", numba_ok)

    for fname in img_files:
        path = os.path.join(INPUT_DIR, fname)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        gray = to_gray_uint8(img)
        if gray is None:
            print("Skipping:", fname)
            continue

        # CPU
        out_cpu, t_cpu = cpu_gaussian_blur(gray, kernel)
        cpu_out_path = os.path.join(OUTPUT_DIR, f"cpu_{fname}")
        cv2.imwrite(cpu_out_path, out_cpu)
        results['cpu'].append(t_cpu)
        print(f"{fname} CPU time: {t_cpu:.6f}s")

        # CuPy
        if cupy_ok:
            try:
                out_cupy, t_cupy = gpu_gaussian_blur_cupy(gray, kernel)
                cupy_out_path = os.path.join(OUTPUT_DIR, f"cupy_{fname}")
                cv2.imwrite(cupy_out_path, out_cupy)
                results['cupy'].append(t_cupy)
                print(f"{fname} CuPy time: {t_cupy:.6f}s")
            except Exception as e:
                print("CuPy processing failed:", e)
                results['cupy'].append(None)
        else:
            results['cupy'].append(None)

        # Numba CUDA
        if numba_ok:
            try:
                out_numba, t_numba = gpu_gaussian_blur_numba(gray, kernel)
                numba_out_path = os.path.join(OUTPUT_DIR, f"numba_{fname}")
                cv2.imwrite(numba_out_path, out_numba)
                results['numba'].append(t_numba)
                print(f"{fname} Numba-CUDA time: {t_numba:.6f}s")
            except Exception as e:
                print("Numba CUDA processing failed:", e)
                results['numba'].append(None)
        else:
            results['numba'].append(None)

    # Prepare plot of average runtimes (ignore None entries)
    cpu_times = [t for t in results['cpu'] if t is not None]
    cupy_times = [t for t in results['cupy'] if t is not None]
    numba_times = [t for t in results['numba'] if t is not None]

    labels = []
    avgs = []
    if cpu_times:
        labels.append('CPU (OpenCV)')
        avgs.append(np.mean(cpu_times))
    if cupy_times:
        labels.append('CuPy GPU')
        avgs.append(np.mean(cupy_times))
    if numba_times:
        labels.append('Numba-CUDA')
        avgs.append(np.mean(numba_times))

    if len(labels) > 0:
        plt.figure(figsize=(6,4))
        plt.bar(labels, avgs)
        plt.ylabel('Average runtime (s)')
        plt.title('Average runtime per image (lower is better)')
        plot_path = os.path.join(PLOTS_DIR, 'runtime_comparison.png')
        plt.savefig(plot_path, bbox_inches='tight', dpi=150)
        print("Saved runtime plot to", plot_path)
    else:
        print("No timing data to plot.")

if __name__ == "__main__":
    main()
