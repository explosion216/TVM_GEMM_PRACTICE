from ansor_tune_gemm import *
if __name__ == '__main__':
    print("start")
    
    # No.1 8192*8192*8192
    dtype = "float32"
    nn = 8192
    M = nn
    N = nn
    K = nn
    builder_timeout = 30
    runner_timeout = 70
    trial_time = 200
    size_name = str(M)+"*"+str(N)+"*"+str(K)
    tuner(dtype, M, N, K, size_name, builder_timeout, runner_timeout, trial_time)
    
    # No.2 8192*9216*9216
    dtype = "float32"
    M = 8192
    N = 9216
    K = 9216
    builder_timeout = 60
    runner_timeout = 120
    trial_time = 1000
    size_name = str(M)+"*"+str(N)+"*"+str(K)
    tuner(dtype, M, N, K, size_name, builder_timeout, runner_timeout, trial_time)
    
    # No.3 8192*9216*36864
    dtype = "float32"
    M = 8192
    N = 9216
    K = 36864
    builder_timeout = 60
    runner_timeout = 120
    trial_time = 1000
    size_name = str(M)+"*"+str(N)+"*"+str(K)
    tuner(dtype, M, N, K, size_name, builder_timeout, runner_timeout, trial_time)
    
    # No.4 8192*36864*9216
    dtype = "float32"
    M = 8192
    N = 36864
    K = 9216
    builder_timeout = 60
    runner_timeout = 90
    trial_time = 1000
    size_name = str(M)+"*"+str(N)+"*"+str(K)
    tuner(dtype, M, N, K, size_name, builder_timeout, runner_timeout, trial_time)
        
    # No.5 8192*14336*14336
    dtype = "float32"
    M = 8192
    N = 14336
    K = 14336
    builder_timeout = 60
    runner_timeout = 120
    trial_time = 1000
    size_name = str(M)+"*"+str(N)+"*"+str(K)
    tuner(dtype, M, N, K, size_name, builder_timeout, runner_timeout, trial_time)
        
    # No.6 16384*16384*16384
    dtype = "float32"
    nn = 16384
    M = nn
    N = nn
    K = nn
    builder_timeout = 60
    runner_timeout = 150
    trial_time = 400
    size_name = str(M)+"*"+str(N)+"*"+str(K)
    tuner(dtype, M, N, K, size_name, builder_timeout, runner_timeout, trial_time)
    
    # No.7 8192*28672*8192
    dtype = "float32"
    M = 8192
    N = 28672
    K = 8192
    builder_timeout = 60
    runner_timeout = 120
    trial_time = 1000
    size_name = str(M)+"*"+str(N)+"*"+str(K)
    # tuner(dtype, M, N, K, size_name, builder_timeout, runner_timeout, trial_time)

    # No.8 8192 8192 28672
    dtype = "float32"
    M = 8192
    N = 8192
    K = 28672
    builder_timeout = 60
    runner_timeout = 150
    trial_time = 1000
    size_name = str(M)+"*"+str(N)+"*"+str(K)
    tuner(dtype, M, N, K, size_name, builder_timeout, runner_timeout, trial_time)