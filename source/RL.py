import random
import json
import random
from os.path import join
from pathlib import Path
from argparse import Namespace

import numpy as np
import torch
from torch import nn
import os
import gym

from sample_factory.algorithms.appo.actor_worker import transform_dict_observations
from sample_factory.algorithms.appo.learner import LearnerWorker
from sample_factory.algorithms.appo.model import create_actor_critic
from sample_factory.algorithms.appo.model_utils import get_hidden_size
from sample_factory.envs.create_env import create_env
from sample_factory.utils.utils import AttrDict
from sample_factory.envs.env_registry import global_env_registry
from sample_factory.algorithms.appo.model_utils import register_custom_encoder
from sample_factory.algorithms.appo.model_utils import get_obs_shape, EncoderBase, ResBlock, nonlinearity
from sample_factory.algorithms.utils.pytorch_utils import calc_num_elements
from sample_factory.algorithms.utils.algo_utils import EXTRA_EPISODIC_STATS_PROCESSING, EXTRA_PER_POLICY_SUMMARIES
from pydantic import Extra, BaseModel, validator
import multiprocessing

from pogema import GridConfig
from pogema.wrappers.multi_time_limit import MultiTimeLimit
from pogema.wrappers.metrics import MetricsWrapper


class AsyncPPO(BaseModel, extra=Extra.forbid):
    experiment_summaries_interval: int = 20
    adam_eps: float = 1e-6
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    gae_lambda: float = 0.95
    rollout: int = 32
    num_workers: int = multiprocessing.cpu_count()
    recurrence: int = 32
    use_rnn: bool = True
    rnn_type: str = 'gru'
    rnn_num_layers: int = 1
    ppo_clip_ratio: float = 0.1
    ppo_clip_value: float = 1.0
    batch_size: int = 1024
    num_batches_per_iteration: int = 1
    ppo_epochs: int = 1
    num_minibatches_to_accumulate: int = -1
    max_grad_norm: float = 4.0

    exploration_loss_coeff: float = 0.003
    value_loss_coeff: float = 0.5
    kl_loss_coeff: float = 0.0
    exploration_loss: str = 'entropy'
    num_envs_per_worker: int = 2
    worker_num_splits: int = 2
    num_policies: int = 1
    policy_workers_per_policy: int = 1
    max_policy_lag: int = 10000
    traj_buffers_excess_ratio: int = 2
    decorrelate_experience_max_seconds: int = 10
    decorrelate_envs_on_one_worker: bool = True

    with_vtrace: bool = True
    vtrace_rho: float = 1.0
    vtrace_c: float = 1.0
    set_workers_cpu_affinity: bool = True
    force_envs_single_thread: bool = True
    reset_timeout_seconds: int = 120
    default_niceness: int = 0
    train_in_background_thread: bool = True
    learner_main_loop_num_cores: int = 1
    actor_worker_gpus = []

    with_pbt: bool = False
    pbt_mix_policies_in_one_env: bool = True
    pbt_period_env_steps: int = 5e6
    pbt_start_mutation: int = 2e7
    pbt_replace_fraction: float = 0.3
    pbt_mutation_rate: float = 0.15
    pbt_replace_reward_gap: float = 0.1
    pbt_replace_reward_gap_absolute: float = 1e-6
    pbt_optimize_batch_size: bool = False
    pbt_target_objective: str = 'true_reward'

    use_cpc: bool = False
    cpc_forward_steps: int = 8
    cpc_time_subsample: int = 6
    cpc_forward_subsample: int = 2
    benchmark: bool = False
    sampler_only: bool = False


class ExperimentSettings(BaseModel, extra=Extra.forbid):
    save_every_sec: int = 120
    keep_checkpoints: int = 1
    save_milestones_sec: int = -1
    stats_avg: int = 100
    learning_rate: float = 1e-4
    train_for_env_steps: int = 1e10
    train_for_seconds: int = 1e10

    obs_subtract_mean: float = 0.0
    obs_scale: float = 1.0

    gamma: float = 0.99
    reward_scale: float = 1.0
    reward_clip: float = 10.0

    encoder_type: str = 'conv'
    encoder_custom: str = 'pogema_residual'
    encoder_subtype: str = 'convnet_simple'
    encoder_extra_fc_layers: int = 1

    pogema_encoder_num_filters: int = 64
    pogema_encoder_num_res_blocks: int = 3

    hidden_size: int = 512
    nonlinearity: str = 'relu'
    policy_initialization: str = 'orthogonal'
    policy_init_gain: float = 1.0
    actor_critic_share_weights: bool = True

    use_spectral_norm: bool = False
    adaptive_stddev: bool = True
    initial_stddev: float = 1.0


class GlobalSettings(BaseModel, extra=Extra.forbid):
    algo: str = 'APPO'
    env: str = None
    experiment: str = None
    experiments_root: str = None
    train_dir: str = 'train_dir/experiment'
    device: str = 'gpu'
    seed: int = None
    cli_args: dict = {}
    use_wandb: bool = True


class Evaluation(BaseModel, extra=Extra.forbid):
    fps: int = 0
    render_action_repeat: int = None
    no_render: bool = True
    policy_index: int = 0
    record_to: str = join(os.getcwd(), '..', 'recs')
    continuous_actions_sample: bool = True
    env_frameskip: int = None


class Environment(BaseModel, ):
    grid_config: GridConfig = GridConfig()
    name: str = 'Pogema-v0'
    # framestack: int = 1
    max_episode_steps: int = 256
    animation_monitor: bool = False
    animation_dir: str = './renders'
    evaluation: bool = False
    grid_memory_radius: int = None
    path_to_grid_configs: str = None
    auto_reset: bool = True


class Experiment(BaseModel):
    name: str = None
    environment: Environment = Environment()
    async_ppo: AsyncPPO = AsyncPPO()
    experiment_settings: ExperimentSettings = ExperimentSettings()
    global_settings: GlobalSettings = GlobalSettings()
    evaluation: Evaluation = Evaluation()


def validate_config(config):
    exp = Experiment(**config)
    flat_config = Namespace(**exp.async_ppo.dict(),
                            **exp.experiment_settings.dict(),
                            **exp.global_settings.dict(),
                            **exp.evaluation.dict(),
                            full_config=exp.dict()
                            )
    return exp, flat_config


class ResnetEncoder(EncoderBase):
    def __init__(self, cfg, obs_space, timing):
        super().__init__(cfg, timing)
        # noinspection Pydantic
        settings: ExperimentSettings = ExperimentSettings(**cfg.full_config['experiment_settings'])

        obs_shape = get_obs_shape(obs_space)
        input_ch = obs_shape.obs[0]

        resnet_conf = [[settings.pogema_encoder_num_filters, settings.pogema_encoder_num_res_blocks]]

        curr_input_channels = input_ch
        layers = []
        for i, (out_channels, res_blocks) in enumerate(resnet_conf):
            # noinspection PyTypeChecker
            layers.extend([
                nn.Conv2d(curr_input_channels, out_channels, kernel_size=3, stride=1, padding=1),  # padding SAME
            ])

            for j in range(res_blocks):
                layers.append(ResBlock(cfg, out_channels, out_channels, self.timing))

            curr_input_channels = out_channels

        layers.append(nonlinearity(cfg))

        self.conv_head = nn.Sequential(*layers)
        self.conv_head_out_size = calc_num_elements(self.conv_head, obs_shape.obs)

        self.init_fc_blocks(self.conv_head_out_size)

    def forward(self, x):
        if isinstance(x, dict):
            x = x['obs']
        x = self.conv_head(x)
        x = x.contiguous().view(-1, self.conv_head_out_size)
        x = self.forward_fc_blocks(x)
        return x


def make_pogema(env_cfg):
    env = gym.make(env_cfg.name, config=env_cfg.grid_config)

    if env_cfg.max_episode_steps:
        env = MultiTimeLimit(env, max_episode_steps=env_cfg.max_episode_steps)

    env = MetricsWrapper(env)

    return env

def override_default_params_func(env, parser):
    parser.set_defaults(
        encoder_custom='pogema_residual',
        hidden_size=128,
    )


def create_pogema_env(full_env_name, cfg=None, env_config=None):
    environment_config: Environment = Environment(**cfg.full_config['environment'])
    if env_config is None or env_config.get("remove_seed", True):
        environment_config.grid_config.seed = None
    return make_pogema(environment_config)


def pogema_extra_episodic_stats_processing(policy_id, stat_key, stat_value, cfg):
    pass


def pogema_extra_summaries(policy_id, policy_avg_stats, env_steps, summary_writer, cfg):
    for key in policy_avg_stats:
        for metric in ['ISR', 'CSR']:
            if metric in key:
                avg = np.mean(policy_avg_stats[key])
                summary_writer.add_scalar(key, avg, env_steps)


def register_custom_components():
    global_env_registry().register_env(
        env_name_prefix='Pogema-v0',
        make_env_func=create_pogema_env,
        override_default_params_func=override_default_params_func,
    )

    register_custom_encoder('pogema_residual', ResnetEncoder)

    EXTRA_EPISODIC_STATS_PROCESSING.append(pogema_extra_episodic_stats_processing)
    EXTRA_PER_POLICY_SUMMARIES.append(pogema_extra_summaries)


class APPOHolder:
    def __init__(self, path, device='cuda'):
        register_custom_components()

        self.path = path
        self.env = None
        config_path = join(path, 'cfg.json')
        with open(config_path, "r") as f:
            config = json.load(f)
        exp, flat_config = validate_config(config['full_config'])
        cfg = flat_config

        env = create_env(cfg.env, cfg=cfg, env_config={})

        actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
        env.close()

        # force cpu workers for parallel evaluation
        # cfg.device = 'cpu'
        if device == 'cpu' or not torch.cuda.is_available():
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
        self.device = device

        self.estimator = None

        # actor_critic.share_memory()
        actor_critic.model_to_device(device)
        policy_id = cfg.policy_index
        checkpoints = join(path, f'checkpoint_p{policy_id}')
        checkpoints = LearnerWorker.get_checkpoints(checkpoints)
        checkpoint_dict = LearnerWorker.load_checkpoint(checkpoints, device)
        actor_critic.load_state_dict(checkpoint_dict['model'])

        self.ppo = actor_critic
        self.device = device
        self.cfg = cfg

        self.rnn_states = None

    def after_reset(self, env):
        self.env = env

    def get_name(self):
        return Path(self.path).name

    def act(self, observations):
        if self.rnn_states is None or len(self.rnn_states) != len(observations):
            self.rnn_states = torch.zeros([len(observations), get_hidden_size(self.cfg)], dtype=torch.float32,
                                          device=self.device)

        with torch.no_grad():
            obs_torch = AttrDict(transform_dict_observations(observations))
            for key, x in obs_torch.items():
                obs_torch[key] = torch.from_numpy(x).to(self.device).float()
            policy_outputs = self.ppo(obs_torch, self.rnn_states, with_action_distribution=True)
            self.rnn_states = policy_outputs.rnn_states
            actions = policy_outputs.actions

        return actions.cpu().numpy()

    def after_step(self, dones):
        for agent_i, done_flag in enumerate(dones):
            if done_flag:
                self.rnn_states[agent_i] = torch.zeros([get_hidden_size(self.cfg)], dtype=torch.float32,
                                                       device=self.device)
        if all(dones):
            self.rnn_states = None


