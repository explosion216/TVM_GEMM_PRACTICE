import tvm
from tvm import te
import os
from tvm.contrib import nvcc
from tvm.contrib import spirv
import numpy as np
from tvm import autotvm
import sys
import logging
from tvm import rpc
from tvm.contrib import utils
import tvm.testing
from cublas_gemm import test_cublas_gemm

@autotvm.template('gemm_vectorize')
def gemm_autotune(mm, nn, ll, _dtype='float32'):
    # graph
    m, n, l = te.var('m'), te.var('n'), te.var('l')
    m, n, l = tvm.runtime.convert(mm), tvm.runtime.convert(nn), tvm.runtime.convert(ll)
    A = te.placeholder((l, n), dtype=_dtype, name="A")
    B = te.placeholder((l, m), dtype=_dtype, name="B")
    k = te.reduce_axis((0, l), name="k")
    C = te.compute((m, n), lambda ii, jj: te.sum(
        A[k, jj] * B[k, ii], axis=k), name="C")

    # schedule
    s = te.create_schedule(C.op)

    AA = s.cache_read(A, "shared", [C])
    BB = s.cache_read(B, "shared", [C])
    AL = s.cache_read(AA, "local", [C])
    BL = s.cache_read(BB, "local", [C])
    CC = s.cache_write(C, "local")

    cfg = autotvm.get_config()
    cfg.define_knob('tile_partsx', [1, 2, 4, 8, 16])
    cfg.define_knob('tile_partsy', [1, 2, 4, 8, 16])
    cfg.define_knob('tile_num_threadx', [1, 2, 4, 8, 16, 32, 64, 128])
    cfg.define_knob('tile_num_thready', [1, 2, 4, 8, 16, 32, 64, 128])
    cfg.define_knob('tile_num_blockx', [1, 2, 4, 8, 16, 32, 64, 128])
    cfg.define_knob('tile_num_blocky', [1, 2, 4, 8, 16, 32, 64, 128])
    # scale = 8
    # num_thread = 8
    partsx = cfg['tile_partsx'].val
    partsy = cfg['tile_partsy'].val
    num_threadx = cfg['tile_num_threadx'].val
    num_thready = cfg['tile_num_thready'].val
    block_factorx = cfg['tile_num_blockx'].val
    block_factory = cfg['tile_num_blocky'].val
    # grid_size 16384
    # block_size 256

    # block_x = te.thread_axis("blockIdx.x")
    # block_y = te.thread_axis("blockIdx.y")
    # thread_x = te.thread_axis("threadIdx.x")
    # thread_y = te.thread_axis("threadIdx.y")
    # thread_xz = te.thread_axis("vthread", name="vx")
    # thread_yz = te.thread_axis("vthread", name="vy")
    
    block_x = te.thread_axis("blockIdx.x")
    thread_x = te.thread_axis((0, num_threadx), "threadIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    thread_y = te.thread_axis((0, num_thready), "threadIdx.y")
    thread_xz = te.thread_axis("vthread", name="vx")
    thread_yz = te.thread_axis("vthread", name="vy")
    # thread_xz = te.thread_axis((0, 2), "vthread", name="vx")
    # thread_yz = te.thread_axis((0, 2), "vthread", name="vy")

    Grid_Size_X = 128
    Grid_Size_Y = 128
    Block_Size_X = 16
    Block_Size_Y = 16

    BK = 16

    # bx, xi = s[C].split(C.op.axis[0], nparts=Grid_Size_X)
    # by, yi = s[C].split(C.op.axis[1], nparts=Grid_Size_Y)
    # s[C].bind(by, block_y)
    # s[C].bind(bx, block_x)
    # s[C].reorder(by, bx, xi, yi)
    
    by, yi = s[C].split(C.op.axis[0], factor=block_factory)
    bx, xi = s[C].split(C.op.axis[1], factor=block_factorx)
    s[C].bind(by, block_y)
    s[C].bind(bx, block_x)
    s[C].reorder(by, bx, yi, xi)

    # tyz, yi = s[C].split(yi, nparts=2)
    # ty, yi = s[C].split(yi, nparts=Block_Size_Y)
    # txz, xi = s[C].split(xi, nparts=2)
    # tx, xi = s[C].split(xi, nparts=Block_Size_X)
    # s[C].bind(tyz, thread_yz)
    # s[C].bind(txz, thread_xz)
    # s[C].bind(ty, thread_y)
    # s[C].bind(tx, thread_x)
    # s[C].reorder(tyz, txz, ty, tx, xi, yi)
    # s[CC].compute_at(s[C], tx)
    
    tyz, yi = s[C].split(yi, nparts=partsy)
    ty, yi = s[C].split(yi, nparts=num_thready)
    txz, xi = s[C].split(xi, nparts=partsx)
    tx, xi = s[C].split(xi, nparts=num_threadx)
    s[C].bind(tyz, thread_yz)
    s[C].bind(txz, thread_xz)
    s[C].bind(ty, thread_y)
    s[C].bind(tx, thread_x)
    s[C].reorder(tyz, txz, ty, tx, yi, xi)
    s[CC].compute_at(s[C], tx)

    # thread block tiling
    ko, ki = s[CC].split(k, factor=BK)
    yc, xc = s[CC].op.axis
    s[CC].reorder(ko, ki, yc, xc)
    
    s[AA].compute_at(s[CC], ko)
    s[BB].compute_at(s[CC], ko)
    s[AL].compute_at(s[CC], ki)
    s[BL].compute_at(s[CC], ki)

    aa_yi, aa_ty = s[AA].split(s[AA].op.axis[0], factor=num_thready)
    aa_xi, aa_tx = s[AA].split(s[AA].op.axis[1], factor=num_threadx * 4)
    aa_tx, aa_vi = s[AA].split(aa_tx, nparts=num_threadx)
    s[AA].reorder(aa_ty, aa_tx, aa_yi, aa_xi, aa_vi)
    s[AA].bind(aa_ty, thread_y)
    s[AA].bind(aa_tx, thread_x)
    s[AA].vectorize(aa_vi)

    bb_yi, bb_ty = s[BB].split(s[BB].op.axis[0], factor=num_thready)
    bb_xi, bb_tx = s[BB].split(s[BB].op.axis[1], factor=num_threadx * 4)
    bb_tx, bb_vi = s[BB].split(bb_tx, nparts=num_threadx)
    s[BB].reorder(bb_ty, bb_tx, bb_yi, bb_xi, bb_vi)
    s[BB].bind(bb_ty, thread_y)
    s[BB].bind(bb_tx, thread_x)
    s[BB].vectorize(bb_vi)


    al_yi, al_xi = s[AL].op.axis
    s[AL].vectorize(al_xi)
    bl_yi, bl_xi = s[BL].op.axis
    s[BL].vectorize(bl_xi)

    return s, [A, B, C]

def test_gemm(mm, nn, ll):
    # correctness
    m, n, l = mm, nn, ll
    dtype = 'float32'

    logging.getLogger('autotvm').setLevel(logging.DEBUG)
    logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))
    log_file = 'gemm_vectorize.log'

    task = autotvm.task.create('gemm_vectorize',
                               args = (m, n, l), target='cuda')
    print(task.config_space)


    measure_option = autotvm.measure_option(
        builder = autotvm.LocalBuilder(timeout=30),
        runner = autotvm.LocalRunner(repeat=2, min_repeat_ms=100, timeout=60)
    )
    tuner = autotvm.tuner.XGBTuner(task, feature_type='knob')
    tuner.tune(n_trial=50,
               measure_option = measure_option,
               callbacks = [autotvm.callback.log_to_file(log_file)])

    dispatch_context = autotvm.apply_history_best(log_file)
    best_config = dispatch_context.query(task.target, task.workload)
    print('\nBest config:')
    print(best_config)

    with autotvm.apply_history_best(log_file):
        with tvm.target.create('cuda'):
            s, arg_bufs = gemm_autotune(m, n, l)
            f = tvm.build(s, arg_bufs)
    # launch the kernel.
    # a_np = np.random.uniform(size=(n, l)).astype(A.dtype)
    # b_np = np.random.uniform(size=(m, l)).astype(B.dtype)
    ctx = tvm.cuda(0)
    a_np = np.random.uniform(size=(l, n)).astype(dtype)
    b_np = np.random.uniform(size=(l, m)).astype(dtype)
    a = tvm.nd.array(a_np, ctx)
    b = tvm.nd.array(b_np, ctx)
    c = tvm.nd.array(np.zeros((m, n), dtype=dtype), ctx)
    for i in range(2):
        f(a, b, c)
    print('function called')
    tvm.testing.assert_allclose(
        c.asnumpy(), np.dot(b_np.T, a_np), rtol=1e1)

    num_flops = 2 * nn * mm * ll
    num_runs = 10
    timer_f = f.time_evaluator(f.entry_name, ctx, number=num_runs)
    t = timer_f(a, b, c).mean
    GFLOPS = num_flops / t / 1e9
    print("average time cost of %d runs = %g ms, %g GFLOPS." % (num_runs, t * 1e3, GFLOPS))
    
def test(m,n,l):
    log_file = 'gemm_vectorize.log'
    dtype='float32'
    with autotvm.apply_history_best(log_file):
            with tvm.target.Target('cuda'):
                s, arg_bufs = gemm_autotune(m, n, l)
                f = tvm.build(s, arg_bufs)
    # launch the kernel.
    # a_np = np.random.uniform(size=(n, l)).astype(A.dtype)
    # b_np = np.random.uniform(size=(m, l)).astype(B.dtype)
    a_np = np.random.uniform(size=(l, n)).astype(dtype)
    b_np = np.random.uniform(size=(l, m)).astype(dtype)
    nn = 16384
    num_runs = 10
    time,GFLOPS = test_cublas_gemm(nn,a_np,b_np,num_runs)
    print("average time cost of cublas_gemm %d runs = %g ms, %g GFLOPS." %
        (num_runs, time * 1e3, GFLOPS))
    mm, nn, ll = m, n, l 
    num_flops = 2 * nn * mm * ll
    
    ctx = tvm.cuda(0)
    a = tvm.nd.array(a_np, ctx)
    b = tvm.nd.array(b_np, ctx)
    c = tvm.nd.array(np.zeros((m, n), dtype=dtype), ctx)
    for i in range(2):
        f(a, b, c)
    ctx.sync()
    # print('function called')
    tvm.testing.assert_allclose(
        c.asnumpy(), np.dot(b_np.T, a_np), rtol=1e1)


    num_runs = 10
    timer_f = f.time_evaluator(f.entry_name, ctx, number=num_runs)
    t = timer_f(a, b, c).mean
    GFLOPS1 = num_flops / t / 1e9
    print("average time cost of auto_vectorize_gemm %d runs = %g ms, %g GFLOPS." %
        (num_runs, t * 1e3, GFLOPS1))
    print("%g %% of cublas_gemm." %(time/t*1e2))

if __name__ == "__main__":
    # test_gemm(16384, 16384, 16384)
    test(16384, 16384, 16384)