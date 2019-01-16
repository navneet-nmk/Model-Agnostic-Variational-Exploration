from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.zero_baseline import ZeroBaseline

from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite

from sandbox.rocky.tf.algos.vpg_maesn import VPG as vpgMaesn
from sandbox.rocky.tf.algos.vpg_lsBaseline import VPG as vpgLS

from rllab.envs.mujoco.wheeled_robot import WheeledEnv
from rllab.envs.mujoco.pusher import PusherEnv
from rllab.envs.mujoco.ant_env_rand_goal_ring import AntEnvRandGoalRing

import pickle
import argparse
from sandbox.rocky.tf.envs.base import TfEnv

import csv
import joblib
import numpy as np
import pickle
import tensorflow as tf

stub(globals())
mode = 'local_docker'
 
#parser = argparse.ArgumentParser()
#parser.add_argument('algo' , type=str , help = 'Maesn or LSBaseline', default='Maesn')
#parser.add_argument('--env', type=str,
                    #help='currently supported envs are Pusher, Wheeled and Ant', default='Pusher')
#parser.add_argument('--initial_params_file' , type=str)
#parser.add_argument('--learning_rate', type = float , default = 1)
#parser.add_argument('--latent_dim', type = int , default = 2)

#args = parser.parse_args()
#assert args.algo in ['Maesn' , 'LSBaseline']
#assert args.env in ['Ant', 'Pusher', 'Wheeled']
initial_params_file = '/root/code/rllab/metaTrainedPolicies/' + 'hyperParams.txt'

n_itr = 100
ldim = 2 ;  default_step = 1

aalgo = 'Maesn'
eenv = 'Pusher'

goals = np.array(range(30))
for counter, goal in enumerate(goals):

    ####################Env Selection#####################
    if eenv == 'Pusher':
        env = TfEnv( normalize(PusherEnv(sparse = True , train = False)))
        max_path_length = 100

    elif eenv == 'Wheeled':
        env = TfEnv( normalize(WheeledEnv(sparse = True , train = False)))
        max_path_length = 200

    elif eenv == 'Ant':
        env = TfEnv( normalize(AntEnvRandGoalRing(sparse = True , train = False)))
        max_path_length = 200

    else:
        raise AssertionError('Not Implemented')
    ########################################################

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    if aalgo == 'Maesn':
        algo = vpgMaesn(
            env=env,
            policy=None,
            load_policy=initial_params_file,
            baseline=baseline,
            batch_size=10000,  
            max_path_length=max_path_length,
            n_itr=n_itr,
            latent_dim=ldim,
            num_total_tasks=100,  #numTotalTasks while training
            noise_opt = True,
            default_step = default_step,
            reset_arg=np.asscalar(goal),
        )

    elif aalgo == 'LSBaseline':
        algo = vpgLS(
            env=env,
            policy=None,
            load_policy=initial_params_file,
            baseline=baseline,
            batch_size=10000, 
            max_path_length=max_path_length,
            n_itr=n_itr,
            latent_dim=ldim,
            num_total_tasks=100,  #numTotalTasks while training
            noise_opt = True,
            default_step = 1,
            reset_arg=np.asscalar(goal),       
    )
    else:
        raise AssertionError('Not Implemented')
  
    run_experiment_lite(
        algo.train(),
        # For this implementation, n_parallel has to be 1. Otherwise, the sparse and train attributes of the environment will be set to their default values
        # Running on ec2 will parallelize testing, and hence give results much faster
        n_parallel=1,
        snapshot_mode="all",
        seed=1,
        exp_prefix=aalgo+'_'+eenv+'_Test',
        exp_name=str(counter),
        docker_image='russellm888/rllab3',
        mode = mode,
        sync_s3_pkl = True,
      
    )

