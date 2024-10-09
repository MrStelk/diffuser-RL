import diffuser.utils as utils

"""
'diffusion': {
    ## model
    'model': 'models.TemporalUnet',
    'diffusion': 'models.GaussianDiffusion',
    'horizon': 32,
    'n_diffusion_steps': 20,
    'action_weight': 10,
    'loss_weights': None,
    'loss_discount': 1,
    'predict_epsilon': False,
    'dim_mults': (1, 2, 4, 8),
    'attention': False,
    'renderer': 'utils.MuJoCoRenderer',

    ## dataset
    'loader': 'datasets.SequenceDataset',
    'normalizer': 'GaussianNormalizer',
    'preprocess_fns': [],
    'clip_denoised': False,
    'use_padding': True,
    'max_path_length': 1000,

    ## serialization
    'logbase': logbase,
    'prefix': 'diffusion/defaults',
    'exp_name': watch(args_to_watch),

    ## training
    'n_steps_per_epoch': 10000,
    'loss_type': 'l2',
    'n_train_steps': 1e6,
    'batch_size': 32,
    'learning_rate': 2e-4,
    'gradient_accumulate_every': 2,
    'ema_decay': 0.995,
    'save_freq': 20000,
    'sample_freq': 20000,
    'n_saves': 5,
    'save_parallel': False,
    'n_reference': 8,
    'bucket': None,
    'device': 'cuda',
    'seed': None,
}
"""
#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'hopper-medium-expert-v2'
    config: str = 'config.locomotion'

args = Parser().parse_args('diffusion') # all args in config/locomotion.py under 'diffusion' key


#-----------------------------------------------------------------------------#
#---------------------------------- dataset ----------------------------------#
#-----------------------------------------------------------------------------#

# instantiate a config class for datasets.SequenceDataset class
dataset_config = utils.Config(
    args.loader,
    savepath=(args.savepath, 'dataset_config.pkl'),
    env=args.dataset,
    horizon=args.horizon,
    normalizer=args.normalizer,
    preprocess_fns=args.preprocess_fns,
    use_padding=args.use_padding,
    max_path_length=args.max_path_length,
)

# instantiate a config class for utils.MuJoCoRenderer class
render_config = utils.Config(
    args.renderer,
    savepath=(args.savepath, 'render_config.pkl'),
    env=args.dataset,
)

dataset = dataset_config() # Creates a new instance of dataset i.e datasets.SequenceDataset class in diffuser/datasets/sequence.py
renderer = render_config() # Creates a new instance of dataset i.e utils.MuJoCoRenderer class in diffuser/utils/rendering.py

# Dimensions of env stored in SequenceDataset class member variables.
observation_dim = dataset.observation_dim
action_dim = dataset.action_dim


#-----------------------------------------------------------------------------#
#------------------------------ model & trainer ------------------------------#
#-----------------------------------------------------------------------------#

# instantiate a config class for models.TemporalUnet  class
model_config = utils.Config(
    args.model,
    savepath=(args.savepath, 'model_config.pkl'),
    horizon=args.horizon,
    transition_dim=observation_dim + action_dim,
    cond_dim=observation_dim,
    dim_mults=args.dim_mults,
    attention=args.attention,
    device=args.device,
)

# instantiate a config class for models.GaussianDiffusion class
diffusion_config = utils.Config(
    args.diffusion,
    savepath=(args.savepath, 'diffusion_config.pkl'),
    horizon=args.horizon,
    observation_dim=observation_dim,
    action_dim=action_dim,
    n_timesteps=args.n_diffusion_steps,
    loss_type=args.loss_type,
    clip_denoised=args.clip_denoised,
    predict_epsilon=args.predict_epsilon,
    ## loss weighting
    action_weight=args.action_weight,
    loss_weights=args.loss_weights,
    loss_discount=args.loss_discount,
    device=args.device,
)

# instantiate a config class for utils.Trainer class
trainer_config = utils.Config(
    utils.Trainer,
    savepath=(args.savepath, 'trainer_config.pkl'),
    train_batch_size=args.batch_size,
    train_lr=args.learning_rate,
    gradient_accumulate_every=args.gradient_accumulate_every,
    ema_decay=args.ema_decay,
    sample_freq=args.sample_freq,
    save_freq=args.save_freq,
    label_freq=int(args.n_train_steps // args.n_saves),
    save_parallel=args.save_parallel,
    results_folder=args.savepath,
    bucket=args.bucket,
    n_reference=args.n_reference,
)

#-----------------------------------------------------------------------------#
#-------------------------------- instantiate --------------------------------#
#-----------------------------------------------------------------------------#

model = model_config() # Creates an instance of the models.TemporalUnet class in diffuser/models/temporal.py

diffusion = diffusion_config(model) # Creates an instance of the models.GaussianDiffusion class in diffuser/models/diffusion.py

trainer = trainer_config(diffusion, dataset, renderer) # Creates an instance of the utils.Trainer class in diffuser/utils/training.py


#-----------------------------------------------------------------------------#
#------------------------ test forward & backward pass -----------------------#
#-----------------------------------------------------------------------------#

utils.report_parameters(model) # Prints out some stats of parameter for the TempralUnet model

print('Testing forward...', end=' ', flush=True)
# SequenceDataset.__getitem__ returns a namedtuple Batch(trajectories, conditions) where trajectories are concatination of actions and observations,
# conditions = {0:observations[0]} representing the initial state of trajectories. utils.batchify converts numpy batch to tesnsors for proper input to model.
batch = utils.batchify(dataset[0])

# diffusion.loss() computes loss by first adding noise to trajectories, replace observation dimension of noisy trajectory with that from conditions dict.
# Then the noise in the image is predicted and loss in computed. call returns loss, info containing 'a0_loss'(not sure what).
# GaussianDiffusion.loss -> GaussianDiffusion.p_losses -> apply_conditioning, GaussianDiffusion.loss_fn
loss, _ = diffusion.loss(*batch)
loss.backward() # Backward pass
print('✓')


#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)

for i in range(n_epochs):
    print(f'Epoch {i} / {n_epochs} | {args.savepath}')
    trainer.train(n_train_steps=args.n_steps_per_epoch) # Trainer class train method trains the model for n_steps_per_epoch epochs.

