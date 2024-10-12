# 0.cublas_gemm
import os
import numpy as np
import tvm
from tvm import te, auto_scheduler, topi
from tvm.topi.testing import conv2d_nchw_python
from tvm.contrib import cublas
_dtype = "float32"
target = tvm.target.Target('cuda')

def random_initial(nn):
    n, m, l = nn, nn, nn
    a_np = np.random.uniform(size=(n, l)).astype(_dtype)
    b_np = np.random.uniform(size=(m, l)).astype(_dtype)
    return a_np,b_np


def test_cublas_gemm(nn,data_np,weight_np,num_runs):
    m = nn
    n = nn
    l = nn
    A = te.placeholder((l, m), name='data', dtype='float32')
    B = te.placeholder((l, n), name='kernel', dtype='float32')
    C = cublas.matmul(A, B, False, True, dtype='float32')

    sch = te.create_schedule(C.op)
    args = [A, B, C]
    func = tvm.build(sch, args, target)

    # Check correctness
    out_np = np.matmul(data_np, weight_np.T)

    ctx = tvm.cuda()
    data_tvm = tvm.nd.array(data_np, ctx)
    weight_tvm = tvm.nd.array(weight_np, ctx)
    out_tvm = tvm.nd.array(np.zeros((m, n), dtype=C.dtype), ctx)
    func(data_tvm, weight_tvm, out_tvm)
    ctx.sync()
    
    # Check results
    np.testing.assert_allclose(out_np, out_tvm.asnumpy(), rtol=1e-3)

    # Evaluate execution time
    # num_runs=10
    evaluator = func.time_evaluator(func.entry_name, ctx, number=num_runs)
    time = np.median(evaluator(data_tvm, weight_tvm, out_tvm).results)
    GFLOPS=2 * (m*n*l) / time / 1e9
    # print("shape", data_np.shape, weight_np.shape)
    # print("Execution time of this operator: %.3f ms" % (time * 1000))
    # print("Speed: %g GFLOPS" % (2 * (M*N*K) / time / 1e9))
    # print("average time cost of %d runs = %g ms, %g GFLOPS." %
    #         (num_runs, time * 1e3, GFLOPS))
    return time,GFLOPS
    
    
if __name__ == "__main__":
    nn=16384
    a_np,b_np = random_initial(nn)
    num_runs = 10
    time,GFLOPS = test_cublas_gemm(nn,a_np,b_np,num_runs)
    print("average time cost of %d runs = %g ms, %g GFLOPS." %
        (num_runs, time * 1e3, GFLOPS))