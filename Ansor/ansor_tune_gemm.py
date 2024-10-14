import tvm
import tvm.testing
from tvm import te, auto_scheduler
import numpy
import timeit
import os
from cublas_gemm import test_cublas_gemm

os.environ['TVM_NUM_THREADS']=str(1)


target = "cuda"
dev = tvm.device(target, 0)

EVAL_REPEAT_TIME = 10

# 计算C(M, N) = A(M, K) x B(K, N)
@auto_scheduler.register_workload
def matmul(M, N, K, dtype):
    A = te.placeholder((M, K), name="A", dtype=dtype)
    B = te.placeholder((K, N), name="B", dtype=dtype)

    k = te.reduce_axis((0, K), name="k")
    C = te.compute(
        (M, N),
        lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
        name="C",
        attrs={"layout_free_placeholders": [B]},  # enable automatic layout transform for tensor B
    )

    return [A, B, C]

def autotune(M, N, K, dtype, target_name, log_file, builder_timeout, runner_timeout, trial_time):
    print(target_name)
    target = tvm.target.Target(target_name)
    task = tvm.auto_scheduler.SearchTask(func=matmul, args=(M, N, K, dtype), target=target)

    # Inspect the computational graph
    print("Computational DAG:")
    print(task.compute_dag)

    builder = auto_scheduler.LocalBuilder(timeout=builder_timeout)
    runner = auto_scheduler.LocalRunner(repeat=2, min_repeat_ms=300, timeout=runner_timeout)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=trial_time,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=2,
        builder=builder,
        runner=runner
    )

    task.tune(tune_option)
    # sch, args = task.apply_best(log_file)
    # tuner.tune(n_trial=50,
    # measure_option = tune_option)

# 检查矩阵乘法结果是否正确，并返回乘法函数
def get_matmul_func(M, N, K, dtype, target_name, log_file):
    a = tvm.nd.array(numpy.random.rand(M, K).astype(dtype), dev)
    b = tvm.nd.array(numpy.random.rand(K, N).astype(dtype), dev)

    answer = numpy.dot(a.numpy(), b.numpy())
    
    target = tvm.target.Target(target_name)
    task = tvm.auto_scheduler.SearchTask(func=matmul, args=(M, N, K, dtype), target=target)
    sch, args = task.apply_best(log_file)
    func = tvm.build(sch, args, target=target, name="matmul")
    assert func

    # print(tvm.lower(sch, args, simple_mode=True))
    # print(func.get_source("asm"))
    # func.export_library("tvm_autoscheduler.so")

    c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)
    func(a, b, c)
    tvm.testing.assert_allclose(c.numpy(), answer, rtol=1e1)
    
    return func

def benchmark(matmul_func, dtype, M, N, K):
    # Random generated tensor for testing
    a = tvm.nd.array(numpy.random.rand(M, K).astype(dtype), dev)
    b = tvm.nd.array(numpy.random.rand(K, N).astype(dtype), dev)

    np_repeat = EVAL_REPEAT_TIME
    np_runing_time = timeit.timeit(
        setup="import numpy\n"
        "M = " + str(M) + "\n"
        "K = " + str(K) + "\n"
        "N = " + str(N) + "\n"
        'dtype = "' + str(dtype) + '"\n'
        "a = numpy.random.rand(M, K).astype(dtype)\n"
        "b = numpy.random.rand(K, N).astype(dtype)\n",
        stmt="answer = numpy.dot(a, b)",
        number=np_repeat,
    )
    print("Numpy running time: %f" % (np_runing_time / np_repeat))

    answer = numpy.dot(a.numpy(), b.numpy())

    time0,GFLOPS0=test_cublas_gemm(M,N,K,dtype,a,b,np_repeat,answer)
    print("CuBlas running time: %f ms, %f GFLOPS." %
        (time0 * 1e3, GFLOPS0))
    
    c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)
    matmul_func(a, b, c)
    dev.sync()
    tvm.testing.assert_allclose(c.numpy(), answer, rtol=1e1)

    evaluator = matmul_func.time_evaluator(matmul_func.entry_name, dev, number=EVAL_REPEAT_TIME)
    tvm_time = evaluator(a, b, c).mean
    # print("TVM autoscheduler tuned: %f" % tvm_time)
    GFLOPS = 2 * M * K * N * 1e-9
    tvm_glops = (GFLOPS / tvm_time)
    # print(f'TVM autoscheduler tuned GFLOPS: {tvm_glops}')
    print("TVM running time: %f ms, %f GFLOPS." %
        (tvm_time * 1e3, tvm_glops))
    print("%f %% of CuBlas" %(time0/tvm_time*100))

def tuner(dtype, M, N, K, size_name, builder_timeout, runnert_timeout, trial_time):
    if not os.path.exists(size_name):
        os.makedirs(size_name)
    if dtype == "float32":
        log_file = os.path.join(size_name, "matmul_autoscheduler_32_" + size_name + ".json")
    elif dtype == "float64":
        log_file = os.path.join(size_name, "matmul_autoscheduler_64_" + size_name + ".json")
        
    os.makedirs(size_name, exist_ok=True)
    with open(log_file, 'a') as f:
        pass

    autotune(M, N, K, dtype, target, log_file, builder_timeout, runnert_timeout, trial_time) 
    
    func = get_matmul_func(M, N, K, dtype, target, log_file)
    benchmark(func, dtype, M, N, K)
    

if __name__ == '__main__':
    dtype = "float32"
    M = 8192
    N = 14336
    K = 14336
    builder_timeout = 60
    runner_timeout = 120
    trial_time = 1000
    size_name = str(M)+"*"+str(N)+"*"+str(K)
    tuner(dtype, M, N, K, size_name, builder_timeout, runner_timeout, trial_time)