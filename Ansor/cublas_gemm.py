# 0.cublas_gemm
import os
import numpy as np
import tvm
from tvm import te, auto_scheduler, topi
from tvm.topi.testing import conv2d_nchw_python
from tvm.contrib import cublas
target = tvm.target.Target('cuda')

def random_initial(M,N,K,dtype):
    # M, N, K = nn, nn, nn
    a_np = np.random.rand(M, K).astype(dtype)
    b_np = np.random.rand(K, N).astype(dtype)
    return a_np,b_np


def test_cublas_gemm(M,N,K,_dtype,data_np,weight_np,num_runs,answer):
    
    # assert data_np.shape == (M, K), f"data_np 的形状应该是 ({M}, {K}), 但实际为 {data_np.shape}"
    # assert weight_np.shape == (K, N), f"weight_np 的形状应该是 ({K}, {N}), 但实际为 {weight_np.shape}"

    A = te.placeholder((M, K), name='data', dtype=_dtype)
    B = te.placeholder((K, N), name='kernel', dtype=_dtype)
    C = cublas.matmul(A, B, False, False, dtype=_dtype)

    sch = te.create_schedule(C.op)
    args = [A, B, C]
    func = tvm.build(sch, args, target)

    # Check correctness
    # out_np = np.matmul(data_np, weight_np.T)

    ctx = tvm.cuda()
    data_tvm = tvm.nd.array(data_np, ctx)
    weight_tvm = tvm.nd.array(weight_np, ctx)
    out_tvm = tvm.nd.array(np.zeros((M, N), dtype=C.dtype), ctx)
    func(data_tvm, weight_tvm, out_tvm)
    ctx.sync()
    
    # Check results
    np.testing.assert_allclose(answer, out_tvm.asnumpy(), rtol=1e1)

    # Evaluate execution time
    # num_runs=10
    evaluator = func.time_evaluator(func.entry_name, ctx, number=num_runs)
    time = np.median(evaluator(data_tvm, weight_tvm, out_tvm).results)
    GFLOPS=2 * (M*N*K) / time / 1e9
    # print("shape", data_np.shape, weight_np.shape)
    # print("Execution time of this operator: %.3f ms" % (time * 1000))
    # print("Speed: %g GFLOPS" % (2 * (M*N*K) / time / 1e9))
    # print("average time cost of %d runs = %g ms, %g GFLOPS." %
    #         (num_runs, time * 1e3, GFLOPS))
    return time,GFLOPS
    
    
if __name__ == "__main__":
    M=8192
    N=36864
    K=9216
    dtype = "float32"
    a_np,b_np = random_initial(M,N,K,dtype)
    num_runs = 10
    out_np = np.matmul(a_np,b_np)
    time,GFLOPS = test_cublas_gemm(M,N,K,dtype,a_np,b_np,num_runs,out_np)
    print("average time cost of %d runs = %g ms, %g GFLOPS." %
        (num_runs, time * 1e3, GFLOPS))