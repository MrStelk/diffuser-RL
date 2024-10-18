import pdb

import diffuser.sampling as sampling
import diffuser.utils as utils


#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'walker2d-medium-replay-v2'
    config: str = 'config.locomotion'

args = Parser().parse_args('plan') # All args in /config/locomotion.py under 'plan' key

"""
args:
'plan': {
    'guide': 'sampling.ValueGuide',
    'policy': 'sampling.GuidedPolicy',
    'max_episode_length': 1000,
    'batch_size': 64,
    'preprocess_fns': [],
    'device': 'cuda',
    'seed': None,

    ## sample_kwargs
    'n_guide_steps': 2,
    'scale': 0.1,
    't_stopgrad': 2,
    'scale_grad_by_std': True,

    ## serialization
    'loadbase': None,
    'logbase': logbase,
    'prefix': 'plans/',
    'exp_name': watch(args_to_watch),
    'vis_freq': 100,
    'max_render': 8,

    ## diffusion model
    'horizon': 32,
    'n_diffusion_steps': 20,

    ## value function
    'discount': 0.997,

    ## loading
    'diffusion_loadpath': 'f:diffusion/defaults_H{horizon}_T{n_diffusion_steps}',
    'value_loadpath': 'f:values/defaults_H{horizon}_T{n_diffusion_steps}_d{discount}',

    'diffusion_epoch': 'latest',
    'value_epoch': 'latest',

    'verbose': True,
    'suffix': '0',
}

"""

#-----------------------------------------------------------------------------#
#---------------------------------- loading ----------------------------------#
#-----------------------------------------------------------------------------#

## load diffusion model and value function from disk
diffusion_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.diffusion_loadpath,
    epoch=args.diffusion_epoch, seed=args.seed,
)

# Loads value model. Value class (defined in sampling/guides.py) is a wrapper around torch.nn.Module
# This will be the classifier to guide the diffusion model.
value_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.value_loadpath,
    epoch=args.value_epoch, seed=args.seed,
)

value2_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.value2_loadpath,
    epoch=args.value2_epoch, seed=args.seed,
)
"""
load_diffusion calls load_config function (both defined in /diffusers/utils/serialization.py).
load_config is give the paths to configuration files (.pkl extension). It loads them and returns the configurations.
From the configurations, classes are instantiated (/diffuser/utils/config.py __call__ method).
Trainer class is defined in /diffuser/utils/training.py. 
After instantiaition, Trainer.load() method is called which actually loads the diffusion model weights.
All the instantiations are retuned in  named tuple with (dataset renderer model diffusion ema trainer epoch) keys.
.ema containes the actual model class like an nn.module class for models.
"""

## ensure that the diffusion model and value function are compatible with each other
utils.check_compatibility(diffusion_experiment, value_experiment)
utils.check_compatibility(diffusion_experiment, value2_experiment)
"""
check_compatability returns True if experiment_1 and experiment_2 have
the same normalizers and number of diffusion steps.
"""

diffusion = diffusion_experiment.ema # model
dataset = diffusion_experiment.dataset # dataset
renderer = diffusion_experiment.renderer # renderer

## initialize value guide
value_function = value_experiment.ema # Classifier model weights.
value2_function = value2_experiment.ema

# Config class of the guide. configs in the load_diffusion are of model_config.
# Here a guide config is being done. sampling.ValueGuide class and model is registered in member variables.
guide_config = utils.Config(args.guide, model=value_function, verbose=False)
guide = guide_config()


guide2_config = utils.Config(args.guide2, model=value2_function, verbose=False)
guide2 = guide2_config()

# logger
logger_config = utils.Config(
    utils.Logger,
    renderer=renderer,
    logpath=args.savepath,
    vis_freq=args.vis_freq,
    max_render=args.max_render,
)

"""
policies are wrappers around an unconditional diffusion model and a value guide
Policy class (defined in sampling/policies.py) has the __call__ method which runs the 
denoinsing process. args.policy is sampling.GuidedPolicy defined in sampling/policies.py
This config class is being instantiated here.
"""
# policy_config = utils.Config(
#     args.policy,
#     guide=guide,
#     scale=args.scale,
#     diffusion_model=diffusion,
#     normalizer=dataset.normalizer,
#     preprocess_fns=args.preprocess_fns,
#     ## sampling kwargs
#     sample_fn=sampling.n_step_guided_p_sample,
#     n_guide_steps=args.n_guide_steps,
#     t_stopgrad=args.t_stopgrad,
#     scale_grad_by_std=args.scale_grad_by_std,
#     verbose=False,
# )

policy_config = utils.Config(
    args.policy,
    guide=guide,
    guide2=guide2,
    scale=args.scale,
    diffusion_model=diffusion,
    normalizer=dataset.normalizer,
    preprocess_fns=args.preprocess_fns,
    ## sampling kwargs
    sample_fn=sampling.n_step_guided_p_sample2,
    n_guide_steps=args.n_guide_steps,
    t_stopgrad=args.t_stopgrad,
    scale_grad_by_std=args.scale_grad_by_std,
    verbose=False,
)


logger = logger_config()
policy = policy_config() # Actual policy class defined in sampling/policies.py


#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

env = dataset.env
observation = env.reset() # Initial observation

## observations for rendering
rollout = [observation.copy()]

total_reward = 0
for t in range(args.max_episode_length):

    if t % 10 == 0: print(args.savepath, flush=True)

    ## save state for rendering only. Basically current observation.
    state = env.state_vector().copy()

    ## format current observation for conditioning
    conditions = {0: observation}
    action, samples = policy(conditions, batch_size=args.batch_size, verbose=args.verbose)
    
    # print("samles: ", samples[0].shape)
    ## execute action in environment
    next_observation, reward, terminal, _ = env.step(action)

    ## print reward and score
    total_reward += reward
    score = env.get_normalized_score(total_reward)
    print(
        f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
        f'values: {samples.values} | scale: {args.scale}',
        flush=True,
    )

    ## update rollout observations
    rollout.append(next_observation.copy())

    ## render every `args.vis_freq` steps
    # logger.log(t, samples, state, rollout)
    # print("rollout : ", len(rollout), len(rollout[0]), rollout)
    # print("samples: ", len(samples[1][0]), samples[1][0])
    # print("state: ", len(state), state)

    if terminal:
        break

    observation = next_observation # Update observation.

## write results to json file at `args.savepath`
# Stores the args. diffusion_experiment and value_experiment are being passed just to store the epoch
# Full 'rollout' is not being stored.
logger.finish(t, score, total_reward, terminal, diffusion_experiment, value_experiment)
