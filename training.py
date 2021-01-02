import tensorflow as tf
import numpy as np

from garage import wrap_experiment
from garage.envs import GymEnv, normalize
from garage.experiment.deterministic import set_seed
from garage.sampler import RaySampler, MultiprocessingSampler, Sampler, LocalSampler
from garage.tf.algos import PPO
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.policies import GaussianMLPPolicy
from garage.trainer import TFTrainer
import garage

from dowel import logger, tabular
import akro

from env.carenv import CarEnv



@wrap_experiment(snapshot_mode="gap", snapshot_gap=10,)
def ppo_car(ctxt=None, specs=None):
    mem_history = []
    assert specs is not None
   
    set_seed(1)
    tf.keras.backend.clear_session()
    with TFTrainer(snapshot_config=ctxt) as trainer:
        #env = normalize(GymEnv("LunarLanderContinuous-v2"))
        env = normalize(CarEnv(specs), normalize_obs=True)

        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(64, 64),
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=None,
        )
        

        baseline = GaussianMLPBaseline(
            env_spec=env.spec,
            hidden_sizes=(32, 32),
            use_trust_region=True,
        )

        sampler = RaySampler(agents=policy,
                             envs=env,
                             max_episode_length=500,
                             is_tf_worker=True)

        # NOTE: make sure when setting entropy_method to 'max', set
        # center_adv to False and turn off policy gradient. See
        # tf.algos.NPO for detailed documentation.
        algo = PPO(
            env_spec=env.spec,
            policy=policy,
            baseline=baseline,
            sampler=sampler,
            discount=0.99,
            gae_lambda=0.95,
            lr_clip_range=0.07,
            optimizer_args=dict(
                batch_size=128,
                max_optimization_epochs=10,
            ),
            stop_entropy_gradient=True,
            entropy_method='max',
            policy_ent_coeff=0.02,
            center_adv=False,
        )

        trainer.setup(algo, env)
        trainer.train(n_epochs=300, batch_size=2048, plot=False)
        trainer.save()