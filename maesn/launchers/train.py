import sys
sys.path.append("/Users/navneetmkumar/Downloads/maesn_suite/maesn/")
print(sys.path)
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
import sandbox
from sandbox.rocky.tf.policies.maml_minimal_gauss_mlp_policy_adaStep import MAMLGaussianMLPPolicy as adaGaussPolicy
from sandbox.rocky.tf.policies.maml_minimal_gauss_mlp_policy import MAMLGaussianMLPPolicy as GaussPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.algos.maesn_trpo import MAESN_TRPO
import argparse
import tensorflow as tf
from launchers.hyperparam_sweep import VG
from rllab.envs.mujoco.wheeled_robot import WheeledEnv
from rllab.envs.mujoco.pusher import PusherEnv
from rllab.envs.mujoco.ant_env_rand_goal_ring import AntEnvRandGoalRing

mode = 'local'
#mode = 'ec2'
#parser = argparse.ArgumentParser()
#parser.add_argument('algo' , type=str , help = 'Maesn or LSBaseline', default='Maesn')
#parser.add_argument('--env', type=str,
#                    help='currently supported envs are Pusher, Wheeled and Ant', default='Pusher')
#args = parser.parse_args()
#assert args.algo in ['Maesn' , 'LSBaseline']
#assert args.env in ['Ant' , 'Pusher', 'Wheeled']

variants = VG().variants()
num_total_tasks = 100 ; num_grad_updates = 1 ; n_itr = 500

aalgo = 'Maesn'
eenv = 'Pusher'

for v in variants:

    
    #stub(globals())
    ####################Env Selection#####################
    if eenv == 'Pusher':
        env = TfEnv(normalize(PusherEnv()))
        max_path_length = 100

    elif eenv == 'Wheeled':
        env = TfEnv(normalize(WheeledEnv()))
        max_path_length = 200

    elif eenv == 'Ant':
        env = TfEnv(normalize(AntEnvRandGoalRing()))
        max_path_length = 200

    else:
        raise AssertionError('Not Implemented')
    ########################################################

    #####################Algo Selection####################
    if aalgo == 'Maesn':
        assert v['fast_learning_rate'] != 0 , 'Fast learning rate needs to be non 0 for Maesn'
        policy = adaGaussPolicy(
            name="policy",
            env_spec=env.spec,
            grad_step_size=v['fast_learning_rate'],
            hidden_nonlinearity=tf.nn.relu,
            hidden_sizes=(100,100),
            latent_dim=v['latent_dim'],
            num_total_tasks=num_total_tasks,
            init_std = v['init_std'],
        )
    elif aalgo == 'LSBaseline':
        assert v['fast_learning_rate'] == 0 ,'Fast learning rate needs to be 0 for LS Baseline'
        policy =  GaussPolicy(
            name="policy",
            env_spec=env.spec,
            grad_step_size=v['fast_learning_rate'],
            hidden_nonlinearity=tf.nn.relu,
            hidden_sizes=(100,100),
            latent_dim=v['latent_dim'],
            num_total_tasks=num_total_tasks,
            init_std = v['init_std']
        )
    else:
        raise AssertionError('Not Implemented')
    ########################################################

    baseline = LinearFeatureBaseline(env_spec=env.spec)
    print(env)

    algo = MAESN_TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=v['fast_batch_size'], # number of trajs for grad update
        max_path_length=max_path_length,
        meta_batch_size=v['meta_batch_size'],
        num_grad_updates=num_grad_updates,
        n_itr=n_itr,
        use_maml=True,
        step_size=v['meta_learning_rate'],
        plot=False,
        latent_dim=v['latent_dim'],
        num_total_tasks=num_total_tasks,
        kl_weighting=v['kl_weighting'],
        #plottingFolder = "Sparse_BP_kl0.05_ldim2",
        kl_scheme=None
    )

    algo.train()

    #run_experiment_lite(
    #    algo.train(),
    #    n_parallel=1,
    #    snapshot_mode="all",
    #    #python_command='python3',
    #    log_dir='/Users/navneetmkumar/Downloads/maesn_suite/maesn/tmp',
    #    seed=v['seed'],
    #    exp_prefix=aalgo+'_'+eenv,
    #    exp_name= v['exp_name'],
    #    plot=False,
    #    sync_s3_pkl=True,
    #    mode=mode,
    #)
