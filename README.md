### CPU (OpenCV) vs CuPy GPU vs Numba-CUDA

This project implements **Gaussian blur image processing** using three different execution paths:

1. **CPU implementation using OpenCV**  
2. **GPU implementation using CuPy**  
3. **GPU implementation using a custom Numba-CUDA kernel**

The goal is to understand how different computational models perform on the same workload, and to explore practical aspects of GPU acceleration for real-world image tasks.

---

## 📌 Project Motivation

I wanted to explore how GPU acceleration affects image-processing workloads and how different GPU programming approaches compare.  
This project demonstrates:

- How a **CPU-optimized library** (OpenCV) can outperform GPU code for small workloads  
- How **GPU overhead** affects runtime  
- How **naïve GPU kernels** differ from optimized CUDA implementations  
- How libraries like **CuPy** wrap CUDA and behave internally  

## 🚀 How to Run the Project

### 1. Create and activate a virtual environment
```bash
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
python src/main.py
