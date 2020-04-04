-------------------------#模型0403xx------------------------------------
##1hz 5倍速
##地图1
##不收敛：结果
在400episodes左右，成功率达到最高的70%左右。 随后下降，最终在10%。与随机算法无差别

-------------------------#模型040416------------------------------------

##2hz 2.5倍速
##400个 episodes
##地图1
##收敛了！！！！！
average rate: 1.997
	min: 0.422s max: 0.577s std dev: 0.06186s window: 9
average rate: 2.013
	min: 0.364s max: 0.577s std dev: 0.06713s window: 15
average rate: 1.999
	min: 0.364s max: 0.577s std dev: 0.06378s window: 19
average rate: 2.014
	min: 0.364s max: 0.580s std dev: 0.07165s window: 25
average rate: 2.003
	min: 0.364s max: 0.590s std dev: 0.07008s window: 29
    /home/abby/anaconda3/envs/tur/bin/python /home/abby/DQNTUR/src/turt_q_learn/scripts/DoubleQ_GET_PAUSE_batch.py
Using hyperparams: {'Pause': False, 'Epsilon Decay': 0.992, 'Crash Penalty': -50, 'Action Rate': 2, 'Epsilon Initial': 1, 'Reward Direction': True, 'Action Dim': 5, 'Optimizer': <class 'tensorflow.python.keras.optimizer_v2.rmsprop.RMSprop'>, 'Memory Length': 1000000, 'State Dim': 23, 'First Activation': <function relu at 0x7fe8c3b1fc50>, 'Model Type': 2, 'Loss': <tensorflow.python.keras.losses.Huber object at 0x7fe8c0049b50>, 'Last Activation': <function linear at 0x7fe8c3b1fed0>, 'Scan Ratio': 18, 'Batch Size': 64, 'Double Q Network': True, 'Gamma': 0.99, 'Direction Scalar': 1, 'Max Scan Range': 1, 'Load Model': False, 'Hidden Activations': <function relu at 0x7fe8c3b1fc50>, 'Learning Rate': 0.0002, 'Initializer': <tensorflow.python.ops.init_ops.VarianceScaling object at 0x7fe8c0049b10>, 'Episodes': 400, 'Epsilon Min': 0.05, 'Goal Reward': 50, 'Scan Reward Scalar': 1, 'Reset Target': 500, 'Episode Length': 350}
stateDim: 23
2
WARNING:tensorflow:From /home/abby/anaconda3/envs/tur/lib/python2.7/site-packages/tensorflow/python/util/deprecation.py:507: calling __init__ (from tensorflow.python.ops.init_ops) with distribution=normal is deprecated and will be removed in a future version.
Instructions for updating:
`normal` is a deprecated alias for `truncated_normal`
WARNING:tensorflow:From /home/abby/anaconda3/envs/tur/lib/python2.7/site-packages/tensorflow/python/ops/init_ops.py:97: calling __init__ (from tensorflow.python.ops.init_ops) with dtype is deprecated and will be removed in a future version.
Instructions for updating:
Call initializer instance with the dtype argument instead of passing it to the constructor
WARNING:tensorflow:From /home/abby/anaconda3/envs/tur/lib/python2.7/site-packages/tensorflow/python/ops/init_ops.py:97: calling __init__ (from tensorflow.python.ops.init_ops) with dtype is deprecated and will be removed in a future version.
Instructions for updating:
Call initializer instance with the dtype argument instead of passing it to the constructor
2020-04-04 15:38:39.335878: I tensorflow/core/platform/cpu_feature_guard.cc:145] This TensorFlow binary is optimized with Intel(R) MKL-DNN to use the following CPU instructions in performance critical operations:  SSE4.1 SSE4.2 AVX AVX2 FMA
To enable them in non-MKL-DNN operations, rebuild TensorFlow with the appropriate compiler flags.
2020-04-04 15:38:39.358221: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2599990000 Hz
2020-04-04 15:38:39.358564: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55c1f68bb220 executing computations on platform Host. Devices:
2020-04-04 15:38:39.358576: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (0): <undefined>, <undefined>
OMP: Info #212: KMP_AFFINITY: decoding x2APIC ids.
OMP: Info #210: KMP_AFFINITY: Affinity capable, using global cpuid leaf 11 info
OMP: Info #154: KMP_AFFINITY: Initial OS proc set respected: 0-11
OMP: Info #156: KMP_AFFINITY: 12 available OS procs
OMP: Info #157: KMP_AFFINITY: Uniform topology
OMP: Info #179: KMP_AFFINITY: 1 packages x 6 cores/pkg x 2 threads/core (6 total cores)
OMP: Info #214: KMP_AFFINITY: OS proc to physical thread map:
OMP: Info #171: KMP_AFFINITY: OS proc 0 maps to package 0 core 0 thread 0 
OMP: Info #171: KMP_AFFINITY: OS proc 6 maps to package 0 core 0 thread 1 
OMP: Info #171: KMP_AFFINITY: OS proc 1 maps to package 0 core 1 thread 0 
OMP: Info #171: KMP_AFFINITY: OS proc 7 maps to package 0 core 1 thread 1 
OMP: Info #171: KMP_AFFINITY: OS proc 2 maps to package 0 core 2 thread 0 
OMP: Info #171: KMP_AFFINITY: OS proc 8 maps to package 0 core 2 thread 1 
OMP: Info #171: KMP_AFFINITY: OS proc 3 maps to package 0 core 3 thread 0 
OMP: Info #171: KMP_AFFINITY: OS proc 9 maps to package 0 core 3 thread 1 
OMP: Info #171: KMP_AFFINITY: OS proc 4 maps to package 0 core 4 thread 0 
OMP: Info #171: KMP_AFFINITY: OS proc 10 maps to package 0 core 4 thread 1 
OMP: Info #171: KMP_AFFINITY: OS proc 5 maps to package 0 core 5 thread 0 
OMP: Info #171: KMP_AFFINITY: OS proc 11 maps to package 0 core 5 thread 1 
OMP: Info #250: KMP_AFFINITY: pid 3382 tid 3382 thread 0 bound to OS proc set 0
2020-04-04 15:38:39.359105: I tensorflow/core/common_runtime/process_util.cc:115] Creating new thread pool with default inter op setting: 2. Tune using inter_op_parallelism_threads for best performance.
2020-04-04 15:38:39.380480: W tensorflow/compiler/jit/mark_for_compilation_pass.cc:1412] (One-time warning): Not using XLA:CPU for cluster because envvar TF_XLA_FLAGS=--tf_xla_cpu_global_jit was not set.  If you want XLA:CPU, either set that envvar, or use experimental_jit_scope to enable XLA:CPU.  To confirm that XLA is active, pass --vmodule=xla_compilation_cache=1 (as a proper command-line flag, not via TF_XLA_FLAGS) or set the envvar XLA_FLAGS=--xla_hlo_profile.
Episode 0:
	Start!
	New goal at 0.5, -0.5!
	Crashed!
	Total Step: 12
	Score: -55.7771550285
Episode 1:
	Start!
	New goal at -1.5, -1.5!
	Crashed!
	Total Step: 45
	Score: -64.9289924733
Episode 2:
	Start!
	New goal at -0.5, -1.5!
	Crashed!
	Total Step: 64
	Score: -63.1907107865
Episode 3:
	Start!
	New goal at 1.5, 1.5!
OMP: Info #250: KMP_AFFINITY: pid 3382 tid 3461 thread 1 bound to OS proc set 1
OMP: Info #250: KMP_AFFINITY: pid 3382 tid 4001 thread 2 bound to OS proc set 2
OMP: Info #250: KMP_AFFINITY: pid 3382 tid 4002 thread 3 bound to OS proc set 3
OMP: Info #250: KMP_AFFINITY: pid 3382 tid 4003 thread 4 bound to OS proc set 4
OMP: Info #250: KMP_AFFINITY: pid 3382 tid 4006 thread 6 bound to OS proc set 6
OMP: Info #250: KMP_AFFINITY: pid 3382 tid 4010 thread 7 bound to OS proc set 7
OMP: Info #250: KMP_AFFINITY: pid 3382 tid 4012 thread 9 bound to OS proc set 9
OMP: Info #250: KMP_AFFINITY: pid 3382 tid 4011 thread 8 bound to OS proc set 8
OMP: Info #250: KMP_AFFINITY: pid 3382 tid 4013 thread 10 bound to OS proc set 10
OMP: Info #250: KMP_AFFINITY: pid 3382 tid 4014 thread 11 bound to OS proc set 11
OMP: Info #250: KMP_AFFINITY: pid 3382 tid 4015 thread 12 bound to OS proc set 0
OMP: Info #250: KMP_AFFINITY: pid 3382 tid 4005 thread 5 bound to OS proc set 5
	Crashed!
	Total Step: 87
	Score: -33.9976800351
Episode 4:
	Start!
	New goal at -0.5, -0.5!
OMP: Info #250: KMP_AFFINITY: pid 3382 tid 3460 thread 13 bound to OS proc set 1
OMP: Info #250: KMP_AFFINITY: pid 3382 tid 4234 thread 14 bound to OS proc set 2
OMP: Info #250: KMP_AFFINITY: pid 3382 tid 4235 thread 15 bound to OS proc set 3
OMP: Info #250: KMP_AFFINITY: pid 3382 tid 4236 thread 16 bound to OS proc set 4
OMP: Info #250: KMP_AFFINITY: pid 3382 tid 4237 thread 17 bound to OS proc set 5
OMP: Info #250: KMP_AFFINITY: pid 3382 tid 4239 thread 19 bound to OS proc set 7
OMP: Info #250: KMP_AFFINITY: pid 3382 tid 4240 thread 20 bound to OS proc set 8
OMP: Info #250: KMP_AFFINITY: pid 3382 tid 4238 thread 18 bound to OS proc set 6
OMP: Info #250: KMP_AFFINITY: pid 3382 tid 4242 thread 22 bound to OS proc set 10
OMP: Info #250: KMP_AFFINITY: pid 3382 tid 4241 thread 21 bound to OS proc set 9
OMP: Info #250: KMP_AFFINITY: pid 3382 tid 4243 thread 23 bound to OS proc set 11
OMP: Info #250: KMP_AFFINITY: pid 3382 tid 4244 thread 24 bound to OS proc set 0
	Crashed!
	Total Step: 99
	Score: -60.0711139129
Episode 5:
	Start!
	New goal at 0.5, 1.5!
	Crashed!
	Total Step: 120
	Score: -31.8279436675
Episode 6:
	Start!
	New goal at 1.5, -0.5!
WARNING:tensorflow:From /home/abby/anaconda3/envs/tur/lib/python2.7/site-packages/tensorflow/python/ops/math_grad.py:1250: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
	Crashed!
	Total Step: 149
	Score: -47.040586193
Episode 7:
	Start!
	New goal at 0.5, 1.5!
 Reached!
	Total Step: 153
	Score: 52.9027167923
Episode 8:
	Start!
	New goal at 1.5, -1.5!
	Crashed!
	Total Step: 213
	Score: -63.0839287902
Episode 9:
	Start!
	New goal at 1.5, 1.5!
	Crashed!
	Total Step: 241
	Score: -33.9567814839
Episode 10:
	Start!
	New goal at -1.5, 0.5!
 Reached!
	Total Step: 251
	Score: 52.0576836773
Episode 11:
	Start!
	New goal at -0.5, 1.5!
	Crashed!
	Total Step: 264
	Score: -57.9693583748
Episode 12:
	Start!
	New goal at -1.5, -1.5!
	Crashed!
	Total Step: 293
	Score: -43.9953596577
Episode 13:
	Start!
	New goal at 1.5, -1.5!
	Crashed!
	Total Step: 318
	Score: -34.333394872
Episode 14:
	Start!
	New goal at -0.5, 1.5!
	Crashed!
	Total Step: 336
	Score: -36.80787151
Episode 15:
	Start!
	New goal at 0.5, -1.5!
	Crashed!
	Total Step: 362
	Score: -63.4062799074
Episode 16:
	Start!
	New goal at 0.5, 1.5!
	Crashed!
	Total Step: 374
	Score: -56.2268812869
Episode 17:
	Start!
	New goal at -1.5, -0.5!
 Reached!
	Total Step: 384
	Score: 52.2108000225
Episode 18:
	Start!
	New goal at 0.5, -1.5!
	Crashed!
	Total Step: 449
	Score: -53.7376158672
Episode 19:
	Start!
	New goal at 1.5, 1.5!
	Crashed!
	Total Step: 463
	Score: -52.7409036155
Episode 20:
	Start!
	New goal at 0.5, -0.5!
	Crashed!
	Total Step: 480
	Score: -62.933526472
Episode 21:
	Start!
	New goal at 1.5, 0.5!
 Reached!
	Total Step: 503
	Score: 52.4555851113
Episode 22:
	Start!
	New goal at -1.5, -0.5!
	Crashed!
	Total Step: 521
	Score: -60.9071130466
Episode 23:
	Start!
	New goal at 0.5, 0.5!
	Crashed!
	Total Step: 534
	Score: -61.5603616126
Episode 24:
	Start!
	New goal at 1.5, 1.5!
	Crashed!
	Total Step: 558
	Score: -34.9460068383
Episode 25:
	Start!
	New goal at 1.5, 0.5!
	Crashed!
	Total Step: 583
	Score: -45.2386398327
Episode 26:
	Start!
	New goal at -0.5, -0.5!
	Crashed!
	Total Step: 602
	Score: -47.7466639819
Episode 27:
	Start!
	New goal at 0.5, -1.5!
reset weights
	Crashed!
	Total Step: 690
	Score: -19.2441123942
Episode 28:
	Start!
	New goal at 0.5, 0.5!
 Reached!
	Total Step: 723
	Score: 63.1460332077
Episode 29:
	Start!
	New goal at -1.5, 1.5!
	Crashed!
	Total Step: 762
	Score: -50.7948082622
Episode 30:
	Start!
	New goal at 1.5, -1.5!
	Crashed!
	Total Step: 811
	Score: -59.6187059699
Episode 31:
	Start!
	New goal at 1.5, -0.5!
	Crashed!
	Total Step: 834
	Score: -35.7367383337
Episode 32:
	Start!
	New goal at -1.5, 1.5!
	Crashed!
	Total Step: 856
	Score: -43.9329625752
Episode 33:
	Start!
	New goal at 0.5, -1.5!
	Crashed!
	Total Step: 890
	Score: -29.8905949441
Episode 34:
	Start!
	New goal at 0.5, -0.5!
	Crashed!
	Total Step: 907
	Score: -46.8811505829
Episode 35:
	Start!
	New goal at -0.5, 0.5!
	Crashed!
	Total Step: 928
	Score: -47.015966346
Episode 36:
	Start!
	New goal at -1.5, 1.5!
	Crashed!
	Total Step: 944
	Score: -49.268734265
Episode 37:
	Start!
	New goal at 0.5, 0.5!
	Crashed!
	Total Step: 1015
	Score: -33.8656205475
Episode 38:
	Start!
	New goal at 1.5, -0.5!
 Reached!
	Total Step: 1056
	Score: 75.2972458708
Episode 39:
	Start!
	New goal at 0.5, 1.5!
	Crashed!
	Total Step: 1112
	Score: -29.3260356677
Episode 40:
	Start!
	New goal at 0.5, 0.5!
reset weights
	Crashed!
	Total Step: 1137
	Score: -46.4826918804
Episode 41:
	Start!
	New goal at -1.5, 0.5!
	Crashed!
	Total Step: 1170
	Score: -39.2150404615
Episode 42:
	Start!
	New goal at -1.5, -1.5!
	Crashed!
	Total Step: 1186
	Score: -56.4328500612
Episode 43:
	Start!
	New goal at -0.5, -1.5!
 Reached!
	Total Step: 1211
	Score: 51.8390876323
Episode 44:
	Start!
	New goal at 1.5, -1.5!
	Crashed!
	Total Step: 1235
	Score: -38.6009250188
Episode 45:
	Start!
	New goal at -1.5, 1.5!
	Crashed!
	Total Step: 1283
	Score: -37.1895659362
Episode 46:
	Start!
	New goal at 0.5, -1.5!
	Crashed!
	Total Step: 1297
	Score: -52.3760413157
Episode 47:
	Start!
	New goal at -0.5, -1.5!
	Crashed!
	Total Step: 1347
	Score: -36.0365449907
Episode 48:
	Start!
	New goal at 1.5, 1.5!
	Crashed!
	Total Step: 1408
	Score: -33.3782743719
Episode 49:
	Start!
	New goal at 0.5, 0.5!
 Reached!
	Total Step: 1425
	Score: 62.8032094177
Episode 50:
	Start!
	New goal at 1.5, -0.5!
	Crashed!
	Total Step: 1473
	Score: -15.8375484768
Episode 51:
	Start!
	New goal at -1.5, 1.5!
	Crashed!
	Total Step: 1491
	Score: -40.6146681354
Episode 52:
	Start!
	New goal at -0.5, 0.5!
	Crashed!
	Total Step: 1517
	Score: -49.2575299777
Episode 53:
	Start!
	New goal at 0.5, -0.5!
	Crashed!
	Total Step: 1549
	Score: -32.5463784488
Episode 54:
	Start!
	New goal at -0.5, 0.5!
	Crashed!
	Total Step: 1587
	Score: -54.6413912591
Episode 55:
	Start!
	New goal at -0.5, 0.5!
 Reached!
	Total Step: 1609
	Score: 62.5061590456
Episode 56:
	Start!
	New goal at -1.5, 0.5!
	Crashed!
	Total Step: 1632
	Score: -42.3518638326
Episode 57:
	Start!
	New goal at 0.5, 0.5!
reset weights
	Crashed!
	Total Step: 1669
	Score: -24.9401905339
Episode 58:
	Start!
	New goal at 0.5, -0.5!
	Crashed!
	Total Step: 1693
	Score: -30.0550110267
Episode 59:
	Start!
	New goal at 0.5, 1.5!
 Reached!
	Total Step: 1697
	Score: 52.9328448895
Episode 60:
	Start!
	New goal at -1.5, 0.5!
	Crashed!
	Total Step: 1716
	Score: -38.6134457802
Episode 61:
	Start!
	New goal at -1.5, 1.5!
	Crashed!
	Total Step: 1731
	Score: -52.2262917887
Episode 62:
	Start!
	New goal at 0.5, 0.5!
 Reached!
	Total Step: 1755
	Score: 62.7307561043
Episode 63:
	Start!
	New goal at -1.5, -1.5!
 Reached!
	Total Step: 1779
	Score: 69.0959275097
Episode 64:
	Start!
	New goal at 0.5, -1.5!
	Crashed!
	Total Step: 1833
	Score: -8.19294322343
Episode 65:
	Start!
	New goal at 1.5, 0.5!
	Crashed!
	Total Step: 1858
	Score: -30.5146607457
Episode 66:
	Start!
	New goal at -1.5, -0.5!
 Reached!
	Total Step: 1926
	Score: 93.6352663743
Episode 67:
	Start!
	New goal at -1.5, -1.5!
	Crashed!
	Total Step: 1949
	Score: -36.4142023637
Episode 68:
	Start!
	New goal at 0.5, -0.5!
	Crashed!
	Total Step: 1983
	Score: -34.5112988734
Episode 69:
	Start!
	New goal at 1.5, -1.5!
	Crashed!
	Total Step: 2025
	Score: -10.5576586628
Episode 70:
	Start!
	New goal at -1.5, 1.5!
	Crashed!
	Total Step: 2044
	Score: -61.6929495652
Episode 71:
	Start!
	New goal at -0.5, 0.5!
 Reached!
	Total Step: 2064
	Score: 61.8123534898
Episode 72:
	Start!
	New goal at -1.5, 1.5!
	Crashed!
	Total Step: 2095
	Score: -34.0277752108
Episode 73:
	Start!
	New goal at 0.5, 0.5!
 Reached!
	Total Step: 2117
	Score: 62.0190553116
Episode 74:
	Start!
	New goal at -1.5, 1.5!
reset weights
 Reached!
	Total Step: 2157
	Score: 68.7429798365
Episode 75:
	Start!
	New goal at -1.5, 1.5!
	Crashed!
	Total Step: 2175
	Score: -37.0789476511
Episode 76:
	Start!
	New goal at -0.5, -0.5!
 Reached!
	Total Step: 2224
	Score: 78.9745698473
Episode 77:
	Start!
	New goal at 0.5, 1.5!
	Crashed!
	Total Step: 2246
	Score: -34.8236778463
Episode 78:
	Start!
	New goal at -1.5, -0.5!
	Crashed!
	Total Step: 2283
	Score: -29.9773935282
Episode 79:
	Start!
	New goal at 0.5, 0.5!
	Crashed!
	Total Step: 2310
	Score: -32.540522832
Episode 80:
	Start!
	New goal at 0.5, 0.5!
	Crashed!
	Total Step: 2338
	Score: -30.3530151871
Episode 81:
	Start!
	New goal at 0.5, 0.5!
 Reached!
	Total Step: 2373
	Score: 79.6653000055
Episode 82:
	Start!
	New goal at -0.5, 0.5!
	Crashed!
	Total Step: 2402
	Score: -34.6044097801
Episode 83:
	Start!
	New goal at 1.5, 0.5!
	Crashed!
	Total Step: 2448
	Score: -13.6174073601
Episode 84:
	Start!
	New goal at -0.5, -1.5!
	Crashed!
	Total Step: 2466
	Score: -36.4561230111
Episode 85:
	Start!
	New goal at 1.5, 0.5!
 Reached!
	Total Step: 2473
	Score: 52.9371438611
Episode 86:
	Start!
	New goal at 1.5, -0.5!
	Crashed!
	Total Step: 2488
	Score: -37.2352768629
Episode 87:
	Start!
	New goal at 1.5, 1.5!
	Crashed!
	Total Step: 2542
	Score: -7.52974468662
Episode 88:
	Start!
	New goal at -1.5, 0.5!
 Reached!
	Total Step: 2551
	Score: 52.4140015759
Episode 89:
	Start!
	New goal at 1.5, -0.5!
	Crashed!
	Total Step: 2577
	Score: -33.8039556226
Episode 90:
	Start!
	New goal at 0.5, 0.5!
 Reached!
	Total Step: 2593
	Score: 62.8631271437
Episode 91:
	Start!
	New goal at -1.5, -0.5!
	Crashed!
	Total Step: 2621
	Score: -29.9971756027
Episode 92:
	Start!
	New goal at 1.5, 0.5!
reset weights
	Crashed!
	Total Step: 2638
	Score: -35.7814831405
Episode 93:
	Start!
	New goal at 1.5, 0.5!
	Crashed!
	Total Step: 2661
	Score: -30.7058649508
Episode 94:
	Start!
	New goal at 0.5, 1.5!
	Crashed!
	Total Step: 2678
	Score: -39.8758912536
Episode 95:
	Start!
	New goal at -0.5, 1.5!
 Reached!
	Total Step: 2714
	Score: 74.9960338634
Episode 96:
	Start!
	New goal at 0.5, -0.5!
	Crashed!
	Total Step: 2725
	Score: -58.6349133956
Episode 97:
	Start!
	New goal at 1.5, -0.5!
 Reached!
	Total Step: 2745
	Score: 52.6773747481
Episode 98:
	Start!
	New goal at 1.5, 1.5!
 Reached!
	Total Step: 2768
	Score: 70.5493070886
Episode 99:
	Start!
	New goal at -0.5, -1.5!
	Crashed!
	Total Step: 2788
	Score: -34.0798771312
Episode 100:
	Start!
	New goal at -1.5, -1.5!
	Crashed!
	Total Step: 2812
	Score: -39.1825386625
Episode 101:
	Start!
	New goal at 0.5, -1.5!
 Reached!
	Total Step: 2854
	Score: 86.2310193081
Episode 102:
	Start!
	New goal at 0.5, 0.5!
 Reached!
	Total Step: 2890
	Score: 79.3190299119
Episode 103:
	Start!
	New goal at 1.5, -1.5!
	Crashed!
	Total Step: 2937
	Score: -9.29252178513
Episode 104:
	Start!
	New goal at -0.5, -1.5!
	Crashed!
	Total Step: 2965
	Score: -27.901352464
Episode 105:
	Start!
	New goal at -0.5, -1.5!
	Crashed!
	Total Step: 2996
	Score: -41.9809323101
Episode 106:
	Start!
	New goal at -1.5, -0.5!
	Crashed!
	Total Step: 3025
	Score: -33.0342792211
Episode 107:
	Start!
	New goal at -1.5, 1.5!
	Crashed!
	Total Step: 3042
	Score: -38.9483750188
Episode 108:
	Start!
	New goal at 0.5, 1.5!
 Reached!
	Total Step: 3100
	Score: 93.6214964594
Episode 109:
	Start!
	New goal at -0.5, 0.5!
 Reached!
	Total Step: 3116
	Score: 61.0473990969
Episode 110:
	Start!
	New goal at -0.5, 0.5!
reset weights
 Reached!
	Total Step: 3138
	Score: 62.2225690817
Episode 111:
	Start!
	New goal at -0.5, 0.5!
 Reached!
	Total Step: 3175
	Score: 75.6020893755
Episode 112:
	Start!
	New goal at -0.5, -1.5!
 Reached!
	Total Step: 3242
	Score: 93.8771499557
Episode 113:
	Start!
	New goal at 0.5, -1.5!
	Crashed!
	Total Step: 3274
	Score: -31.0667320806
Episode 114:
	Start!
	New goal at 1.5, 1.5!
	Crashed!
	Total Step: 3298
	Score: -32.7525203868
Episode 115:
	Start!
	New goal at 1.5, 0.5!
	Crashed!
	Total Step: 3313
	Score: -50.7398519604
Episode 116:
	Start!
	New goal at -0.5, 1.5!
	Crashed!
	Total Step: 3338
	Score: -33.6253274733
Episode 117:
	Start!
	New goal at -1.5, -0.5!
 Reached!
	Total Step: 3397
	Score: 83.5798674185
Episode 118:
	Start!
	New goal at -0.5, 0.5!
 Reached!
	Total Step: 3414
	Score: 63.8044551532
Episode 119:
	Start!
	New goal at 0.5, 0.5!
 Reached!
	Total Step: 3458
	Score: 78.8533878694
Episode 120:
	Start!
	New goal at 0.5, 0.5!
 Reached!
	Total Step: 3474
	Score: 62.4250195147
Episode 121:
	Start!
	New goal at -0.5, 1.5!
 Reached!
	Total Step: 3524
	Score: 86.567638013
Episode 122:
	Start!
	New goal at 1.5, 1.5!
 Reached!
	Total Step: 3588
	Score: 98.8592881461
Episode 123:
	Start!
	New goal at 0.5, 1.5!
 Reached!
	Total Step: 3628
	Score: 85.2314536399
Episode 124:
	Start!
	New goal at -1.5, -0.5!
reset weights
	Crashed!
	Total Step: 3647
	Score: -39.5978202555
Episode 125:
	Start!
	New goal at -1.5, 1.5!
	Crashed!
	Total Step: 3699
	Score: -9.7825343623
Episode 126:
	Start!
	New goal at -1.5, -0.5!
 Reached!
	Total Step: 3703
	Score: 51.3363165502
Episode 127:
	Start!
	New goal at -1.5, -1.5!
 Reached!
	Total Step: 3726
	Score: 68.3510584548
Episode 128:
	Start!
	New goal at -0.5, 0.5!
	Crashed!
	Total Step: 3756
	Score: -39.2995048295
Episode 129:
	Start!
	New goal at 1.5, 0.5!
 Reached!
	Total Step: 3807
	Score: 94.9097845609
Episode 130:
	Start!
	New goal at -1.5, 1.5!
	Crashed!
	Total Step: 3852
	Score: -11.8819395414
Episode 131:
	Start!
	New goal at 0.5, -0.5!
	Crashed!
	Total Step: 3887
	Score: -26.3204740081
Episode 132:
	Start!
	New goal at 0.5, -0.5!
 Reached!
	Total Step: 3913
	Score: 61.2769060729
Episode 133:
	Start!
	New goal at -1.5, 0.5!
 Reached!
	Total Step: 3990
	Score: 93.2927108247
Episode 134:
	Start!
	New goal at -1.5, 0.5!
 Reached!
	Total Step: 4035
	Score: 83.4574504092
Episode 135:
	Start!
	New goal at 1.5, -1.5!
	Crashed!
	Total Step: 4088
	Score: -7.77384826886
Episode 136:
	Start!
	New goal at 1.5, -0.5!
	Crashed!
	Total Step: 4103
	Score: -37.9200132319
Episode 137:
	Start!
	New goal at -0.5, 0.5!
 Reached!
	Total Step: 4123
	Score: 62.8174698166
Episode 138:
	Start!
	New goal at -0.5, -1.5!
 Reached!
	Total Step: 4134
	Score: 51.3546855636
Episode 139:
	Start!
	New goal at 1.5, -1.5!
reset weights
 Reached!
	Total Step: 4163
	Score: 68.8170427205
Episode 140:
	Start!
	New goal at 1.5, -1.5!
	Crashed!
	Total Step: 4218
	Score: -6.67293421198
Episode 141:
	Start!
	New goal at 1.5, -1.5!
	Crashed!
	Total Step: 4238
	Score: -36.3797922761
Episode 142:
	Start!
	New goal at 0.5, -1.5!
	Crashed!
	Total Step: 4256
	Score: -40.4503570706
Episode 143:
	Start!
	New goal at -0.5, 1.5!
 Reached!
	Total Step: 4304
	Score: 84.6278455966
Episode 144:
	Start!
	New goal at 0.5, 1.5!
 Reached!
	Total Step: 4308
	Score: 52.9504994483
Episode 145:
	Start!
	New goal at 1.5, -0.5!
	Crashed!
	Total Step: 4326
	Score: -35.4479202889
Episode 146:
	Start!
	New goal at 0.5, 0.5!
	Crashed!
	Total Step: 4352
	Score: -29.1567038546
Episode 147:
	Start!
	New goal at 0.5, 0.5!
 Reached!
	Total Step: 4370
	Score: 62.9304864286
Episode 148:
	Start!
	New goal at -1.5, 1.5!
 Reached!
	Total Step: 4401
	Score: 67.1810734674
Episode 149:
	Start!
	New goal at -1.5, 0.5!
 Reached!
	Total Step: 4410
	Score: 52.3528704627
Episode 150:
	Start!
	New goal at -1.5, 1.5!
	Crashed!
	Total Step: 4431
	Score: -39.6601554943
Episode 151:
	Start!
	New goal at 0.5, -1.5!
 Reached!
	Total Step: 4436
	Score: 53.9894612676
Episode 152:
	Start!
	New goal at -1.5, -0.5!
 Reached!
	Total Step: 4441
	Score: 52.0428295174
Episode 153:
	Start!
	New goal at 1.5, 1.5!
 Reached!
	Total Step: 4507
	Score: 99.6855321543
Episode 154:
	Start!
	New goal at 0.5, -0.5!
 Reached!
	Total Step: 4526
	Score: 63.1370366189
Episode 155:
	Start!
	New goal at -1.5, -0.5!
 Reached!
	Total Step: 4538
	Score: 51.9735288646
Episode 156:
	Start!
	New goal at -1.5, -1.5!
	Crashed!
	Total Step: 4593
	Score: -11.2421692636
Episode 157:
	Start!
	New goal at 1.5, 1.5!
 Reached!
	Total Step: 4616
	Score: 69.6812805198
Episode 158:
	Start!
	New goal at -0.5, -1.5!
reset weights
 Reached!
	Total Step: 4665
	Score: 85.7345613734
Episode 159:
	Start!
	New goal at 0.5, -1.5!
	Crashed!
	Total Step: 4693
	Score: -27.8565525055
Episode 160:
	Start!
	New goal at -1.5, -0.5!
	Crashed!
	Total Step: 4703
	Score: -54.2489167562
Episode 161:
	Start!
	New goal at -0.5, -1.5!
	Crashed!
	Total Step: 4720
	Score: -36.5746763649
Episode 162:
	Start!
	New goal at -1.5, 1.5!
 Reached!
	Total Step: 4754
	Score: 67.5570528218
Episode 163:
	Start!
	New goal at 0.5, -0.5!
 Reached!
	Total Step: 4770
	Score: 63.0767985419
Episode 164:
	Start!
	New goal at -0.5, -0.5!
	Crashed!
	Total Step: 4805
	Score: -31.508818309
Episode 165:
	Start!
	New goal at -1.5, 0.5!
 Reached!
	Total Step: 4810
	Score: 51.672841059
Episode 166:
	Start!
	New goal at -1.5, -0.5!
 Reached!
	Total Step: 4863
	Score: 86.0181473045
Episode 167:
	Start!
	New goal at 1.5, -1.5!
 Reached!
	Total Step: 4886
	Score: 68.6929365312
Episode 168:
	Start!
	New goal at -1.5, 1.5!
 Reached!
	Total Step: 4913
	Score: 69.068080208
Episode 169:
	Start!
	New goal at -1.5, -0.5!
	Crashed!
	Total Step: 4942
	Score: -35.5974463259
Episode 170:
	Start!
	New goal at 0.5, -1.5!
 Reached!
	Total Step: 4946
	Score: 52.9840801822
Episode 171:
	Start!
	New goal at -0.5, -0.5!
 Reached!
	Total Step: 4993
	Score: 77.6702106387
Episode 172:
	Start!
	New goal at -1.5, -0.5!
	Crashed!
	Total Step: 5043
	Score: -15.553911412
Episode 173:
	Start!
	New goal at -1.5, 1.5!
	Crashed!
	Total Step: 5095
	Score: -10.3835212704
Episode 174:
	Start!
	New goal at 1.5, 1.5!
 Reached!
	Total Step: 5120
	Score: 69.4753308007
Episode 175:
	Start!
	New goal at 0.5, -0.5!
reset weights
 Reached!
	Total Step: 5154
	Score: 78.6833807571
Episode 176:
	Start!
	New goal at -1.5, -1.5!
	Crashed!
	Total Step: 5193
	Score: -32.5044173116
Episode 177:
	Start!
	New goal at -0.5, 1.5!
	Crashed!
	Total Step: 5211
	Score: -34.7037125817
Episode 178:
	Start!
	New goal at 1.5, 0.5!
 Reached!
	Total Step: 5245
	Score: 73.9993652847
Episode 179:
	Start!
	New goal at 1.5, -0.5!
	Crashed!
	Total Step: 5271
	Score: -36.6377968031
Episode 180:
	Start!
	New goal at -1.5, -0.5!
	Crashed!
	Total Step: 5299
	Score: -32.5673648508
Episode 181:
	Start!
	New goal at -1.5, -1.5!
	Crashed!
	Total Step: 5317
	Score: -37.4001302018
Episode 182:
	Start!
	New goal at 0.5, 0.5!
	Crashed!
	Total Step: 5338
	Score: -32.9913077698
Episode 183:
	Start!
	New goal at 0.5, -0.5!
 Reached!
	Total Step: 5381
	Score: 78.1097069945
Episode 184:
	Start!
	New goal at 0.5, 1.5!
	Crashed!
	Total Step: 5406
	Score: -32.4732026629
Episode 185:
	Start!
	New goal at 0.5, 0.5!
 Reached!
	Total Step: 5426
	Score: 61.3280354593
Episode 186:
	Start!
	New goal at -1.5, 0.5!
	Crashed!
	Total Step: 5449
	Score: -36.8737931856
Episode 187:
	Start!
	New goal at 1.5, -0.5!
	Crashed!
	Total Step: 5463
	Score: -38.2244720686
Episode 188:
	Start!
	New goal at 0.5, 0.5!
 Reached!
	Total Step: 5499
	Score: 79.3113310695
Episode 189:
	Start!
	New goal at -1.5, 1.5!
	Crashed!
	Total Step: 5517
	Score: -36.1654145722
Episode 190:
	Start!
	New goal at 1.5, 0.5!
 Reached!
	Total Step: 5561
	Score: 85.3001499103
Episode 191:
	Start!
	New goal at 1.5, -1.5!
	Crashed!
	Total Step: 5612
	Score: -8.97302510488
Episode 192:
	Start!
	New goal at -1.5, -0.5!
reset weights
 Reached!
	Total Step: 5655
	Score: 77.2304501496
Episode 193:
	Start!
	New goal at -1.5, 1.5!
	Crashed!
	Total Step: 5667
	Score: -44.3742224088
Episode 194:
	Start!
	New goal at 1.5, -0.5!
	Crashed!
	Total Step: 5680
	Score: -50.4982072355
Episode 195:
	Start!
	New goal at 1.5, -1.5!
 Reached!
	Total Step: 5707
	Score: 69.1655809256
Episode 196:
	Start!
	New goal at 1.5, -0.5!
	Crashed!
	Total Step: 5733
	Score: -28.9130259493
Episode 197:
	Start!
	New goal at -1.5, -1.5!
 Reached!
	Total Step: 5794
	Score: 98.5791241371
Episode 198:
	Start!
	New goal at 0.5, 1.5!
	Crashed!
	Total Step: 5813
	Score: -40.3858034947
Episode 199:
	Start!
	New goal at -0.5, 1.5!
 Reached!
	Total Step: 5824
	Score: 51.9108856699
Episode 200:
	Start!
	New goal at -1.5, -0.5!
	Crashed!
	Total Step: 5850
	Score: -35.2296501375
Episode 201:
	Start!
	New goal at 1.5, -0.5!
 Reached!
	Total Step: 5855
	Score: 51.8426908896
Episode 202:
	Start!
	New goal at 1.5, 1.5!
 Reached!
	Total Step: 5883
	Score: 68.9256337905
Episode 203:
	Start!
	New goal at -1.5, -0.5!
 Reached!
	Total Step: 5889
	Score: 52.4315692647
Episode 204:
	Start!
	New goal at 1.5, -1.5!
	Crashed!
	Total Step: 5930
	Score: -13.4083033283
Episode 205:
	Start!
	New goal at -1.5, -1.5!
	Crashed!
	Total Step: 5981
	Score: -11.6844804074
Episode 206:
	Start!
	New goal at -0.5, -1.5!
	Crashed!
	Total Step: 5998
	Score: -41.1484835222
Episode 207:
	Start!
	New goal at 1.5, 0.5!
 Reached!
	Total Step: 6004
	Score: 52.5803088835
Episode 208:
	Start!
	New goal at 1.5, 0.5!
 Reached!
	Total Step: 6010
	Score: 53.0772874474
Episode 209:
	Start!
	New goal at 0.5, -0.5!
 Reached!
	Total Step: 6031
	Score: 61.1809137495
Episode 210:
	Start!
	New goal at 0.5, -1.5!
 Reached!
	Total Step: 6095
	Score: 95.1244457736
Episode 211:
	Start!
	New goal at 1.5, -0.5!
	Crashed!
	Total Step: 6110
	Score: -37.7126708387
Episode 212:
	Start!
	New goal at -1.5, 0.5!
reset weights
	Crashed!
	Total Step: 6142
	Score: -31.711772062
Episode 213:
	Start!
	New goal at 0.5, 0.5!
 Reached!
	Total Step: 6166
	Score: 60.8028583047
Episode 214:
	Start!
	New goal at 1.5, 1.5!
 Reached!
	Total Step: 6195
	Score: 69.3592219476
Episode 215:
	Start!
	New goal at 1.5, 0.5!
	Crashed!
	Total Step: 6219
	Score: -28.289210731
Episode 216:
	Start!
	New goal at -0.5, 0.5!
	Crashed!
	Total Step: 6245
	Score: -29.9181719038
Episode 217:
	Start!
	New goal at -1.5, -0.5!
 Reached!
	Total Step: 6292
	Score: 84.3991073589
Episode 218:
	Start!
	New goal at 0.5, -0.5!
 Reached!
	Total Step: 6309
	Score: 63.0235539197
Episode 219:
	Start!
	New goal at -0.5, -1.5!
 Reached!
	Total Step: 6320
	Score: 52.7533614578
Episode 220:
	Start!
	New goal at 1.5, 1.5!
	Crashed!
	Total Step: 6342
	Score: -38.2279306422
Episode 221:
	Start!
	New goal at -0.5, 0.5!
 Reached!
	Total Step: 6358
	Score: 62.7406342304
Episode 222:
	Start!
	New goal at 0.5, 1.5!
	Crashed!
	Total Step: 6379
	Score: -36.7778304912
Episode 223:
	Start!
	New goal at -0.5, -0.5!
 Reached!
	Total Step: 6396
	Score: 63.1337897641
Episode 224:
	Start!
	New goal at 1.5, -0.5!
	Crashed!
	Total Step: 6425
	Score: -25.2306071306
Episode 225:
	Start!
	New goal at -1.5, -1.5!
	Crashed!
	Total Step: 6452
	Score: -36.3265873404
Episode 226:
	Start!
	New goal at -1.5, 1.5!
 Reached!
	Total Step: 6485
	Score: 68.0784263555
Episode 227:
	Start!
	New goal at -1.5, -0.5!
 Reached!
	Total Step: 6490
	Score: 52.0020093153
Episode 228:
	Start!
	New goal at -1.5, 1.5!
 Reached!
	Total Step: 6519
	Score: 68.8344594943
Episode 229:
	Start!
	New goal at -0.5, -0.5!
 Reached!
	Total Step: 6539
	Score: 63.0025962304
Episode 230:
	Start!
	New goal at 0.5, 0.5!
 Reached!
	Total Step: 6559
	Score: 62.3426969612
Episode 231:
	Start!
	New goal at -1.5, -1.5!
 Reached!
	Total Step: 6587
	Score: 67.9197760275
Episode 232:
	Start!
	New goal at -1.5, -0.5!
reset weights
	Crashed!
	Total Step: 6640
	Score: -13.5874685491
Episode 233:
	Start!
	New goal at -0.5, -1.5!
 Reached!
	Total Step: 6650
	Score: 51.5134175826
Episode 234:
	Start!
	New goal at 1.5, -1.5!
	Crashed!
	Total Step: 6701
	Score: -12.2887793492
Episode 235:
	Start!
	New goal at 0.5, -1.5!
 Reached!
	Total Step: 6748
	Score: 75.2553802627
Episode 236:
	Start!
	New goal at 1.5, 0.5!
 Reached!
	Total Step: 6754
	Score: 52.588732437
Episode 237:
	Start!
	New goal at -0.5, -1.5!
 Reached!
	Total Step: 6806
	Score: 84.8779313124
Episode 238:
	Start!
	New goal at 0.5, 1.5!
	Crashed!
	Total Step: 6821
	Score: -41.5759261521
Episode 239:
	Start!
	New goal at 1.5, 0.5!
 Reached!
	Total Step: 6827
	Score: 52.6641304155
Episode 240:
	Start!
	New goal at 1.5, -0.5!
 Reached!
	Total Step: 6883
	Score: 91.6834451224
Episode 241:
	Start!
	New goal at -1.5, 0.5!
	Crashed!
	Total Step: 6944
	Score: -12.4955737139
Episode 242:
	Start!
	New goal at -0.5, -1.5!
 Reached!
	Total Step: 6990
	Score: 82.9865736541
Episode 243:
	Start!
	New goal at -1.5, -1.5!
 Reached!
	Total Step: 7021
	Score: 67.4723228016
Episode 244:
	Start!
	New goal at -0.5, -0.5!
 Reached!
	Total Step: 7066
	Score: 78.5710377916
Episode 245:
	Start!
	New goal at -0.5, 1.5!
 Reached!
reset weights
	Total Step: 7135
	Score: 94.2097953934
Episode 246:
	Start!
	New goal at -1.5, 0.5!
	Crashed!
	Total Step: 7155
	Score: -38.264710197
Episode 247:
	Start!
	New goal at -1.5, -1.5!
 Reached!
	Total Step: 7192
	Score: 66.7939874852
Episode 248:
	Start!
	New goal at -1.5, -1.5!
	Crashed!
	Total Step: 7245
	Score: -12.8218144469
Episode 249:
	Start!
	New goal at -0.5, 0.5!
 Reached!
	Total Step: 7261
	Score: 62.2855154416
Episode 250:
	Start!
	New goal at -1.5, 0.5!
 Reached!
	Total Step: 7308
	Score: 85.335330168
Episode 251:
	Start!
	New goal at 0.5, 0.5!
	Crashed!
	Total Step: 7327
	Score: -33.4249020453
Episode 252:
	Start!
	New goal at -1.5, -1.5!
 Reached!
	Total Step: 7352
	Score: 67.7956813987
Episode 253:
	Start!
	New goal at -0.5, 1.5!
 Reached!
	Total Step: 7370
	Score: 52.7605534931
Episode 254:
	Start!
	New goal at 1.5, 0.5!
 Reached!
	Total Step: 7375
	Score: 52.2903039434
Episode 255:
	Start!
	New goal at 1.5, -1.5!
 Reached!
	Total Step: 7401
	Score: 67.6008502486
Episode 256:
	Start!
	New goal at -1.5, -0.5!
 Reached!
	Total Step: 7462
	Score: 92.8810996125
Episode 257:
	Start!
	New goal at -0.5, -0.5!
 Reached!
	Total Step: 7479
	Score: 62.8081196493
Episode 258:
	Start!
	New goal at 1.5, 1.5!
 Reached!
	Total Step: 7548
	Score: 101.891877087
Episode 259:
	Start!
	New goal at -0.5, -0.5!
 Reached!
	Total Step: 7588
	Score: 75.5835239651
Episode 260:
	Start!
	New goal at 0.5, 1.5!
 Reached!
	Total Step: 7628
	Score: 82.9061921996
Episode 261:
	Start!
	New goal at -0.5, 0.5!
reset weights
 Reached!
	Total Step: 7673
	Score: 75.7511685641
Episode 262:
	Start!
	New goal at 1.5, -1.5!
 Reached!
	Total Step: 7741
	Score: 94.8428124416
Episode 263:
	Start!
	New goal at -1.5, -0.5!
	Crashed!
	Total Step: 7776
	Score: -28.171154836
Episode 264:
	Start!
	New goal at -1.5, 1.5!
	Crashed!
	Total Step: 7800
	Score: -38.2942741647
Episode 265:
	Start!
	New goal at 1.5, 0.5!
 Reached!
	Total Step: 7840
	Score: 72.2073587938
Episode 266:
	Start!
	New goal at -1.5, 0.5!
	Crashed!
	Total Step: 7860
	Score: -39.3727292046
Episode 267:
	Start!
	New goal at -0.5, -1.5!
	Crashed!
	Total Step: 7884
	Score: -34.2789472475
Episode 268:
	Start!
	New goal at -1.5, 1.5!
 Reached!
	Total Step: 7913
	Score: 67.9043773957
Episode 269:
	Start!
	New goal at 1.5, -1.5!
 Reached!
	Total Step: 7937
	Score: 68.7384220419
Episode 270:
	Start!
	New goal at -1.5, -0.5!
 Reached!
	Total Step: 7943
	Score: 52.7397017052
Episode 271:
	Start!
	New goal at 1.5, 1.5!
 Reached!
	Total Step: 7971
	Score: 70.3551947658
Episode 272:
	Start!
	New goal at -0.5, -1.5!
 Reached!
	Total Step: 8018
	Score: 85.1830968272
Episode 273:
	Start!
	New goal at 0.5, -0.5!
 Reached!
	Total Step: 8056
	Score: 79.0766062801
Episode 274:
	Start!
	New goal at -0.5, -1.5!
 Reached!
	Total Step: 8114
	Score: 93.4715609114
Episode 275:
	Start!
	New goal at 0.5, -1.5!
 Reached!
	Total Step: 8118
	Score: 52.5837346509
Episode 276:
	Start!
	New goal at 1.5, -0.5!
reset weights
 Reached!
	Total Step: 8155
	Score: 82.2709415523
Episode 277:
	Start!
	New goal at -1.5, 0.5!
 Reached!
	Total Step: 8162
	Score: 51.4972647027
Episode 278:
	Start!
	New goal at 0.5, 1.5!
	Crashed!
	Total Step: 8190
	Score: -29.4412154478
Episode 279:
	Start!
	New goal at -1.5, 1.5!
	Crashed!
	Total Step: 8217
	Score: -36.6437574273
Episode 280:
	Start!
	New goal at 0.5, 1.5!
	Crashed!
	Total Step: 8243
	Score: -30.1234684966
Episode 281:
	Start!
	New goal at -1.5, -0.5!
 Reached!
	Total Step: 8304
	Score: 91.8035824169
Episode 282:
	Start!
	New goal at -0.5, -1.5!
 Reached!
	Total Step: 8354
	Score: 82.7558471046
Episode 283:
	Start!
	New goal at -0.5, 1.5!
 Reached!
	Total Step: 8403
	Score: 80.8266992642
Episode 284:
	Start!
	New goal at 0.5, 0.5!
	Crashed!
	Total Step: 8421
	Score: -33.381666148
Episode 285:
	Start!
	New goal at 1.5, -0.5!
	Crashed!
	Total Step: 8454
	Score: -30.9958844183
Episode 286:
	Start!
	New goal at -0.5, -0.5!
	Crashed!
	Total Step: 8482
	Score: -30.6175612372
Episode 287:
	Start!
	New goal at 0.5, 1.5!
 Reached!
	Total Step: 8486
	Score: 52.6995584328
Episode 288:
	Start!
	New goal at -0.5, -0.5!
 Reached!
	Total Step: 8523
	Score: 73.3180354793
Episode 289:
	Start!
	New goal at 0.5, 0.5!
 Reached!
	Total Step: 8560
	Score: 77.0162351841
Episode 290:
	Start!
	New goal at -1.5, -1.5!
 Reached!
	Total Step: 8634
	Score: 95.8185979265
Episode 291:
	Start!
	New goal at -0.5, 0.5!
reset weights
 Reached!
	Total Step: 8656
	Score: 62.9168445815
Episode 292:
	Start!
	New goal at 0.5, -1.5!
	Crashed!
	Total Step: 8676
	Score: -36.2095121868
Episode 293:
	Start!
	New goal at 0.5, -1.5!
 Reached!
	Total Step: 8728
	Score: 89.3985107991
Episode 294:
	Start!
	New goal at -0.5, 0.5!
	Crashed!
	Total Step: 8759
	Score: -31.3359352009
Episode 295:
	Start!
	New goal at -0.5, 1.5!
	Crashed!
	Total Step: 8773
	Score: -38.6428843892
Episode 296:
	Start!
	New goal at -0.5, -1.5!
 Reached!
	Total Step: 8838
	Score: 94.7593558372
Episode 297:
	Start!
	New goal at 0.5, -1.5!
 Reached!
	Total Step: 8875
	Score: 71.1021314605
Episode 298:
	Start!
	New goal at 1.5, -1.5!
	Crashed!
	Total Step: 8923
	Score: -8.6182053055
Episode 299:
	Start!
	New goal at 0.5, -0.5!
 Reached!
	Total Step: 8960
	Score: 77.5635324895
Episode 300:
	Start!
	New goal at 1.5, -1.5!
 Reached!
	Total Step: 8983
	Score: 70.1891575678
Episode 301:
	Start!
	New goal at -1.5, 0.5!
 Reached!
	Total Step: 9028
	Score: 72.702510343
Episode 302:
	Start!
	New goal at -1.5, 1.5!
	Crashed!
	Total Step: 9058
	Score: -31.8016441998
Episode 303:
	Start!
	New goal at -0.5, 0.5!
 Reached!
	Total Step: 9077
	Score: 62.2894299673
Episode 304:
	Start!
	New goal at -1.5, 0.5!
 Reached!
	Total Step: 9082
	Score: 51.7328373198
Episode 305:
	Start!
	New goal at 0.5, -1.5!
 Reached!
	Total Step: 9130
	Score: 86.812061851
Episode 306:
	Start!
	New goal at -1.5, 0.5!
reset weights
	Crashed!
	Total Step: 9155
	Score: -37.011314612
Episode 307:
	Start!
	New goal at 0.5, -1.5!
 Reached!
	Total Step: 9159
	Score: 52.7027917456
Episode 308:
	Start!
	New goal at 0.5, -1.5!
 Reached!
	Total Step: 9162
	Score: 51.9998691507
Episode 309:
	Start!
	New goal at 0.5, 1.5!
 Reached!
	Total Step: 9224
	Score: 92.2086244253
Episode 310:
	Start!
	New goal at -0.5, 1.5!
 Reached!
	Total Step: 9290
	Score: 93.8977705516
Episode 311:
	Start!
	New goal at 1.5, 0.5!
	Crashed!
	Total Step: 9310
	Score: -31.2598919391
Episode 312:
	Start!
	New goal at -0.5, -0.5!
 Reached!
	Total Step: 9348
	Score: 73.2147925803
Episode 313:
	Start!
	New goal at -1.5, -0.5!
	Crashed!
	Total Step: 9376
	Score: -35.1955244732
Episode 314:
	Start!
	New goal at -1.5, -0.5!
 Reached!
	Total Step: 9420
	Score: 74.7042547251
Episode 315:
	Start!
	New goal at 1.5, 0.5!
 Reached!
	Total Step: 9467
	Score: 85.2264995889
Episode 316:
	Start!
	New goal at -0.5, 1.5!
 Reached!
	Total Step: 9507
	Score: 74.5358345695
Episode 317:
	Start!
	New goal at -1.5, 1.5!
	Crashed!
	Total Step: 9531
	Score: -39.1558306251
Episode 318:
	Start!
	New goal at 0.5, 0.5!
 Reached!
	Total Step: 9552
	Score: 61.5099542837
Episode 319:
	Start!
	New goal at -0.5, 1.5!
	Crashed!
	Total Step: 9566
	Score: -38.9934953078
Episode 320:
	Start!
	New goal at 1.5, -0.5!
 Reached!
	Total Step: 9612
	Score: 72.1146832575
Episode 321:
	Start!
	New goal at -0.5, 0.5!
 Reached!
	Total Step: 9629
	Score: 63.4431154091
Episode 322:
	Start!
	New goal at 1.5, -1.5!
reset weights
 Reached!
	Total Step: 9655
	Score: 68.136936126
Episode 323:
	Start!
	New goal at 0.5, -1.5!
 Reached!
	Total Step: 9725
	Score: 74.9154563103
Episode 324:
	Start!
	New goal at -1.5, 1.5!
 Reached!
	Total Step: 9753
	Score: 65.8366660127
Episode 325:
	Start!
	New goal at 0.5, -1.5!
	Crashed!
	Total Step: 9768
	Score: -40.5431411824
Episode 326:
	Start!
	New goal at -1.5, -1.5!
	Crashed!
	Total Step: 9821
	Score: -17.8396145671
Episode 327:
	Start!
	New goal at 1.5, 1.5!
	Crashed!
	Total Step: 9840
	Score: -39.577492143
Episode 328:
	Start!
	New goal at -0.5, -1.5!
	Crashed!
	Total Step: 9864
	Score: -35.221786988
Episode 329:
	Start!
	New goal at -0.5, 1.5!
 Reached!
	Total Step: 9918
	Score: 89.8396083557
Episode 330:
	Start!
	New goal at 1.5, -0.5!
 Reached!
	Total Step: 9959
	Score: 84.4729900822
Episode 331:
	Start!
	New goal at 1.5, 0.5!
 Reached!
	Total Step: 10017
	Score: 90.1206281897
Episode 332:
	Start!
	New goal at 1.5, 1.5!
 Reached!
	Total Step: 10044
	Score: 67.6431920294
Episode 333:
	Start!
	New goal at -0.5, -0.5!
	Crashed!
	Total Step: 10076
	Score: -29.0465236268
Episode 334:
	Start!
	New goal at 1.5, -1.5!
	Crashed!
	Total Step: 10126
	Score: -8.54061225351
Episode 335:
	Start!
	New goal at -0.5, -1.5!
reset weights
 Reached!
	Total Step: 10174
	Score: 83.2597202534
Episode 336:
	Start!
	New goal at -0.5, -0.5!
 Reached!
	Total Step: 10217
	Score: 77.2517817608
Episode 337:
	Start!
	New goal at 1.5, 0.5!
	Crashed!
	Total Step: 10231
	Score: -37.8072489821
Episode 338:
	Start!
	New goal at 0.5, -1.5!
 Reached!
	Total Step: 10235
	Score: 52.9750547552
Episode 339:
	Start!
	New goal at -0.5, 1.5!
	Crashed!
	Total Step: 10250
	Score: -38.1440463015
Episode 340:
	Start!
	New goal at -0.5, 1.5!
 Reached!
	Total Step: 10262
	Score: 51.3931278838
Episode 341:
	Start!
	New goal at -1.5, 0.5!
 Reached!
	Total Step: 10305
	Score: 71.6028442797
Episode 342:
	Start!
	New goal at 1.5, -1.5!
 Reached!
	Total Step: 10332
	Score: 65.9042431702
Episode 343:
	Start!
	New goal at -0.5, -0.5!
 Reached!
	Total Step: 10372
	Score: 73.6295371988
Episode 344:
	Start!
	New goal at 0.5, -0.5!
 Reached!
	Total Step: 10390
	Score: 61.0337010802
Episode 345:
	Start!
	New goal at 1.5, 1.5!
	Crashed!
	Total Step: 10438
	Score: -10.7502273369
Episode 346:
	Start!
	New goal at -1.5, -0.5!
 Reached!
	Total Step: 10483
	Score: 83.274494293
Episode 347:
	Start!
	New goal at -1.5, -0.5!
 Reached!
	Total Step: 10519
	Score: 74.3026927865
Episode 348:
	Start!
	New goal at -1.5, -0.5!
	Crashed!
	Total Step: 10538
	Score: -39.0304621218
Episode 349:
	Start!
	New goal at -1.5, 1.5!
	Crashed!
	Total Step: 10594
	Score: -6.95222179268
Episode 350:
	Start!
	New goal at -1.5, -1.5!
reset weights
 Reached!
	Total Step: 10661
	Score: 99.4582509871
Episode 351:
	Start!
	New goal at 0.5, 0.5!
 Reached!
	Total Step: 10688
	Score: 61.0660075993
Episode 352:
	Start!
	New goal at -0.5, 1.5!
 Reached!
	Total Step: 10741
	Score: 85.1433974028
Episode 353:
	Start!
	New goal at -0.5, -1.5!
	Crashed!
	Total Step: 10768
	Score: -32.2339683363
Episode 354:
	Start!
	New goal at -1.5, 0.5!
 Reached!
	Total Step: 10815
	Score: 84.9560220934
Episode 355:
	Start!
	New goal at 0.5, 0.5!
 Reached!
	Total Step: 10857
	Score: 78.8007646909
Episode 356:
	Start!
	New goal at 1.5, 0.5!
 Reached!
	Total Step: 10903
	Score: 87.2919318239
Episode 357:
	Start!
	New goal at -1.5, 1.5!
 Reached!
	Total Step: 10976
	Score: 99.3796720104
Episode 358:
	Start!
	New goal at -1.5, -0.5!
 Reached!
	Total Step: 11019
	Score: 84.9081282272
Episode 359:
	Start!
	New goal at -1.5, 1.5!
	Crashed!
	Total Step: 11045
	Score: -31.3097483959
Episode 360:
	Start!
	New goal at -1.5, -1.5!
 Reached!
	Total Step: 11076
	Score: 68.1698682898
Episode 361:
	Start!
	New goal at -0.5, 1.5!
 Reached!
	Total Step: 11086
	Score: 51.3377035783
Episode 362:
	Start!
	New goal at 0.5, -0.5!
 Reached!
	Total Step: 11109
	Score: 61.4517486852
Episode 363:
	Start!
	New goal at 1.5, 1.5!
 Reached!
reset weights
	Total Step: 11135
	Score: 70.2989608138
Episode 364:
	Start!
	New goal at -1.5, 0.5!
 Reached!
	Total Step: 11200
	Score: 94.6213522395
Episode 365:
	Start!
	New goal at -0.5, 0.5!
 Reached!
	Total Step: 11245
	Score: 78.1694570988
Episode 366:
	Start!
	New goal at -1.5, -1.5!
 Reached!
	Total Step: 11271
	Score: 69.2934135404
Episode 367:
	Start!
	New goal at 1.5, 1.5!
 Reached!
	Total Step: 11295
	Score: 69.9267706283
Episode 368:
	Start!
	New goal at -1.5, -0.5!
 Reached!
	Total Step: 11360
	Score: 94.3131184094
Episode 369:
	Start!
	New goal at 1.5, -1.5!
 Reached!
	Total Step: 11383
	Score: 69.1059386665
Episode 370:
	Start!
	New goal at 0.5, 0.5!
 Reached!
	Total Step: 11422
	Score: 79.0786106197
Episode 371:
	Start!
	New goal at -0.5, -1.5!
 Reached!
	Total Step: 11491
	Score: 94.2182917474
Episode 372:
	Start!
	New goal at -0.5, 1.5!
 Reached!
	Total Step: 11529
	Score: 74.8011323231
Episode 373:
	Start!
	New goal at -1.5, -1.5!
	Crashed!
	Total Step: 11587
	Score: -8.23623460533
Episode 374:
	Start!
	New goal at 1.5, 0.5!
 Reached!
	Total Step: 11593
	Score: 52.7445548594
Episode 375:
	Start!
	New goal at -1.5, -1.5!
 Reached!
	Total Step: 11622
	Score: 69.049853068
Episode 376:
	Start!
	New goal at -1.5, -1.5!
reset weights
	Crashed!
	Total Step: 11683
	Score: -7.2969104051
Episode 377:
	Start!
	New goal at -1.5, 0.5!
 Reached!
	Total Step: 11750
	Score: 94.9524215408
Episode 378:
	Start!
	New goal at -1.5, -0.5!
	Crashed!
	Total Step: 11777
	Score: -39.9939223227
Episode 379:
	Start!
	New goal at 0.5, 0.5!
 Reached!
	Total Step: 11796
	Score: 63.0758974366
Episode 380:
	Start!
	New goal at 0.5, -1.5!
 Reached!
	Total Step: 11801
	Score: 53.7945248679
Episode 381:
	Start!
	New goal at 1.5, 0.5!
 Reached!
	Total Step: 11848
	Score: 86.9696447838
Episode 382:
	Start!
	New goal at 1.5, -0.5!
 Reached!
	Total Step: 11894
	Score: 86.4196835202
Episode 383:
	Start!
	New goal at 0.5, 0.5!
 Reached!
	Total Step: 11916
	Score: 61.7822634342
Episode 384:
	Start!
	New goal at 0.5, 0.5!
 Reached!
	Total Step: 11957
	Score: 78.6863315662
Episode 385:
	Start!
	New goal at -0.5, 0.5!
 Reached!
	Total Step: 12000
	Score: 78.525102697
Episode 386:
	Start!
	New goal at -0.5, -0.5!
 Reached!
	Total Step: 12017
	Score: 64.1563238819
Episode 387:
	Start!
	New goal at -1.5, -1.5!
 Reached!
	Total Step: 12084
	Score: 99.4806636461
Episode 388:
	Start!
	New goal at 1.5, -0.5!
 Reached!
	Total Step: 12090
	Score: 52.7958177587
Episode 389:
	Start!
	New goal at 1.5, -1.5!
 Reached!
	Total Step: 12115
	Score: 69.3916151961
Episode 390:
	Start!
	New goal at -0.5, 0.5!
reset weights
 Reached!
	Total Step: 12136
	Score: 62.4640544685
Episode 391:
	Start!
	New goal at -0.5, -0.5!
 Reached!
	Total Step: 12177
	Score: 77.5552029987
Episode 392:
	Start!
	New goal at -1.5, 0.5!
	Crashed!
	Total Step: 12205
	Score: -33.2602314675
Episode 393:
	Start!
	New goal at 1.5, 0.5!
 Reached!
	Total Step: 12254
	Score: 75.6103617041
Episode 394:
	Start!
	New goal at 1.5, 0.5!
 Reached!
	Total Step: 12260
	Score: 52.5821462526
Episode 395:
	Start!
	New goal at -0.5, 1.5!
 Reached!
	Total Step: 12323
	Score: 95.4716447277
Episode 396:
	Start!
	New goal at 1.5, 0.5!
 Reached!
	Total Step: 12369
	Score: 74.8839327204
Episode 397:
	Start!
	New goal at 1.5, 0.5!
 Reached!
	Total Step: 12375
	Score: 52.7069878487
Episode 398:
	Start!
	New goal at -1.5, 0.5!
 Reached!
	Total Step: 12447
	Score: 92.857379228
Episode 399:
	Start!
	New goal at -1.5, -0.5!
	Crashed!
	Total Step: 12509
	Score: -18.4748751525
('success_rate of last 50 eps', array([15., 15., 20., 20., 20., 20., 20., 20., 15., 20., 20., 15., 15.,
       15., 15., 15., 15., 15., 10., 15., 15., 15., 10., 10., 15., 15.,
       15., 15., 15., 10., 15., 15., 15., 15., 15., 15., 20., 20., 20.,
       15., 20., 20., 20., 25., 25., 25., 25., 30., 30., 30., 25., 25.,
       30., 30., 35., 40., 35., 40., 40., 40., 35., 35., 40., 35., 30.,
       30., 35., 30., 30., 35., 35., 40., 35., 35., 30., 25., 30., 25.,
       30., 35., 35., 35., 35., 40., 40., 40., 35., 35., 35., 35., 40.,
       40., 45., 50., 50., 50., 45., 45., 45., 45., 50., 55., 55., 55.,
       60., 60., 60., 65., 70., 65., 65., 60., 55., 55., 60., 65., 65.,
       65., 65., 65., 65., 60., 55., 50., 50., 55., 55., 50., 50., 55.,
       55., 55., 60., 60., 60., 60., 65., 65., 65., 65., 60., 60., 60.,
       65., 65., 60., 65., 70., 70., 70., 65., 70., 70., 65., 60., 60.,
       60., 60., 55., 55., 55., 55., 55., 50., 50., 50., 50., 45., 40.,
       40., 40., 40., 35., 40., 40., 35., 35., 35., 40., 35., 40., 40.,
       45., 50., 50., 50., 45., 45., 50., 50., 55., 55., 55., 50., 55.,
       60., 55., 55., 55., 60., 60., 60., 60., 55., 55., 55., 55., 60.,
       60., 60., 60., 60., 65., 65., 65., 60., 65., 70., 70., 65., 65.,
       70., 65., 70., 70., 75., 80., 75., 75., 70., 70., 70., 65., 70.,
       70., 75., 75., 75., 75., 80., 80., 80., 85., 85., 80., 75., 75.,
       75., 70., 75., 75., 75., 80., 80., 80., 80., 80., 80., 80., 75.,
       70., 65., 65., 65., 70., 70., 65., 65., 70., 70., 70., 70., 70.,
       65., 65., 60., 55., 55., 55., 55., 60., 65., 65., 60., 60., 65.,
       70., 70., 70., 70., 70., 70., 65., 70., 65., 70., 75., 75., 70.,
       75., 70., 70., 70., 75., 75., 75., 70., 70., 65., 60., 60., 60.,
       65., 65., 65., 60., 60., 60., 60., 60., 60., 60., 60., 60., 60.,
       60., 60., 65., 70., 70., 65., 65., 65., 65., 65., 70., 70., 70.,
       75., 75., 75., 75., 75., 75., 75., 75., 80., 80., 80., 85., 90.,
       90., 90., 90., 90., 90., 90., 85., 85., 80., 85., 85., 85., 85.,
       85., 85., 85., 85., 85., 85., 85., 85., 85., 80., 85., 85., 85.,
       90., 90., 95., 90.]))
Train Time: 3077.33930421

Process finished with exit code 0
