# from multiprocessing.pool import Pool as PoolParent
# from multiprocessing import Process, Pool
# from attack_tf_altMI import main_experiment

from attack_tf_altMI import main_experiment
#from test import test_func

import sys


if __name__ ==  '__main__': 


    preprocess_argstr = "{} --target_model={} --alt_mi=True --target_l2_ratio=1e-4 --target_privacy=post --target_dp={} --target_privacy=noisy_data --preprocess=dp-noise --target_epsilon={} --run={} --better_ai=True --better_ai_n=10"
        
    classic_argstr = "{} --target_model={} --alt_mi=True --target_l2_ratio=1e-4 --target_privacy=post --target_dp={} --target_epsilon={} --run={} --better_ai=True --better_ai_n=10"

    neuralnet_argstr = "{} --target_model={} --alt_mi=True --target_l2_ratio=1e-4 --target_privacy=grad_pert --target_dp={} --target_epsilon={} --target_delta=1e-100 --run={} --better_ai=True --better_ai_n=10"

    nopriv_argstr = "{} --target_model={} --target_privacy=no_privacy --target_dp={} --target_epsilon={} --alt_mi=True --target_l2_ratio=1e-4 --run={} --better_ai=True --better_ai_n=10"
    
    n_classes = str(sys.argv[1])
    target_eps = float(sys.argv[2])
    config = int(sys.argv[3])
    cuda_device = str(sys.argv[4])
    run_num = int(sys.argv[5])
    
    need_gpu = False
    
    
    if cuda_device == '-1':
        import random
        cuda_device = str(int(random.randint(0,4)))
    
    # BETTER AI TEST
#     print("ATERNATE CONFIG WARNING")
#     print("ATERNATE CONFIG WARNING")
#     print("ATERNATE CONFIG WARNING")
#     print("ATERNATE CONFIG WARNING")
#     classic_argstr = "{} --target_model={} --alt_mi=True --target_l2_ratio=1e-4 --target_privacy=post --target_dp={} --target_epsilon={} --run=5 --better_ai=True --better_ai_n={}"
    
    
    # Hardcode our existing configurations
    # Level 1
    if config == 0:
        dptype = 'dp'
        modeltype = 'rndf'
        argstr = preprocess_argstr.format(n_classes, modeltype, dptype, target_eps, run_num)
    elif config == 1:
        dptype = 'dp'
        modeltype = 'nn'
        argstr = preprocess_argstr.format(n_classes, modeltype, dptype, target_eps, run_num)
        need_gpu=True
    elif config == 2:
        dptype = 'dp'
        modeltype = 'nb'
        argstr = preprocess_argstr.format(n_classes, modeltype, dptype, target_eps, run_num)
    elif config == 3:
        dptype = 'dp'
        modeltype = 'lr'
        argstr = preprocess_argstr.format(n_classes, modeltype, dptype, target_eps, run_num)

    # Level 2
    elif config == 4:
        dptype = 'dp'
        modeltype = 'nn'
        argstr = neuralnet_argstr.format(n_classes, modeltype, dptype, target_eps, run_num)
        need_gpu=True

    elif config == 5:
        dptype = 'dp'
        modeltype = 'dp-rndf'
        argstr = classic_argstr.format(n_classes, modeltype, dptype, target_eps, run_num)

    # Level 3
    elif config == 6:
        dptype = 'dp'
        modeltype = 'ibmNB'
        argstr = classic_argstr.format(n_classes, modeltype, dptype, target_eps, run_num)
    elif config == 7:
        dptype = 'dp'
        modeltype = 'ibmLR'
        argstr = classic_argstr.format(n_classes, modeltype, dptype, target_eps, run_num)


    ## No privacy
    elif config == 20:
        dptype = 'dp'
        modeltype = 'rndf'
        argstr = nopriv_argstr.format(n_classes, modeltype, dptype, target_eps, run_num)
    elif config == 21:
        dptype = 'dp'
        modeltype = 'nn'
        argstr = nopriv_argstr.format(n_classes, modeltype, dptype, target_eps, run_num)
        need_gpu=True
    elif config == 22:
        dptype = 'dp'
        modeltype = 'nb'
        argstr = nopriv_argstr.format(n_classes, modeltype, dptype, target_eps, run_num)
    elif config == 23:
        dptype = 'dp'
        modeltype = 'lr'
        argstr = nopriv_argstr.format(n_classes, modeltype, dptype, target_eps, run_num)

    else:
        assert False, 'Invalid Config {}'.format(config)

    # SPECIFY GPU DEVICE
    need_gpu = False
    if need_gpu:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"]=cuda_device
        import tensorflow as tf

        conf = tf.ConfigProto()
        conf.gpu_options.allow_growth=True
        session = tf.Session(config=conf)
    
    
    argstrings = []
    argstrings.append(argstr)
    print(argstr)

    for arges in argstrings:
        main_experiment(arges)

    # PRE-PROCESSING
#     preprocess_argstr = "{} --target_model={} --alt_mi=True --target_l2_ratio=1e-4 --target_privacy=post --target_dp={} --target_privacy=noisy_data --preprocess=dp-noise --target_epsilon={} --run={}".format(n_classes, modeltype, dptype, target_eps, run_num)

#     # NO-PREPROCESSING
#     # Classic models
#     classic_argstr = "{} --target_model={} --alt_mi=True --target_l2_ratio=1e-4 --target_privacy=post --target_dp={} --target_epsilon={} --run={}".format(n_classes, modeltype, dptype, target_eps)
    
#     # Neural Network
#     neuralnet_argstr = "{} --target_model={} --alt_mi=True --target_l2_ratio=1e-4 --target_privacy=grad_pert --target_dp={} --target_epsilon={} --run={}".format(n_classes, modeltype, dptype, target_eps)
    
#     # NO PRIVACY
#     nopriv_argstr = "{} --target_model={} --target_dp={} --alt_mi=True --target_l2_ratio=1e-4".format(n_classes, modeltype, dptype)

