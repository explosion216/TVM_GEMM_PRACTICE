# TVM_GEMM_PRACTICE
![](https://img.shields.io/badge/cuda-11.6-blue) ![](https://img.shields.io/badge/nvidia-RTX3060-blue)

 Leverage TVM's AutoTVM and Ansor frameworks to perform automated optimization of GEMM operations on an RTX 3060 GPU.

## AutoTVM
Size: 16384 × 16384 × 16384
| CUBLAS | Kernel | Kernel/CUBLAS(%) |
| ------ | ------ | ---------------- |
| 1096.73 ms 8020.27 GFLOPS | 1114.63 ms 7891.48 GFLOPS | 98.3942 |

## Ansor

| No | Size   | CUBLAS | Kernel | Kernel/CUBLAS(%) |
| - | ---------- | ----------------------- | -------- | ------------------------ |
| 1 | 8192 × 8192 × 8192 | 113.235657 ms 9709.941725 GFLOPS | 144.596045 ms 7604.022838 GFLOPS | 78.311725 |
| 2 | 8192 × 9216 × 9216 | 149.826868 ms 9287.849544 GFLOPS | 186.836377 ms, 7448.064595 GFLOPS | 80.191486 |
| 3 | 8192 × 9216 × 36864 | 769.071533 ms 7237.659145 GFLOPS | 725.734570 ms 7669.853199 GFLOPS | 105.971462 |
| 4 | 8192 × 36864 × 9216 | 594.789160 ms 9358.404606 GFLOPS | 721.469043 ms 7715.199523 GFLOPS | 82.441397 |
| 5 | 8192 × 14336 × 14336 | 353.609326 ms 9522.527014 GFLOPS | 466.157373 ms 7223.428299 GFLOPS | 75.856212 |
| 6 | 16384 × 16384 × 16384 | 911.825098 ms 9646.688872 GFLOPS | 1128.783789 ms 7792.540173 GFLOPS | 80.779429 |
| 7 | 8192 × 8192 × 28672 | 400.652808 ms 9605.051117 GFLOPS | 459.296973 ms 8378.654611 GFLOPS | 87.231755 |

## Comprehensive Guides to Building TVM from Source
[Building TVM GPU 0.18.0 with Docker Image](https://blog.csdn.net/m0_74408076/article/details/142164801?spm=1001.2014.3001.5501)  
[Installing TVM with GPU Support on Windows 11](https://blog.csdn.net/m0_74408076/article/details/141940144?spm=1001.2014.3001.5501)