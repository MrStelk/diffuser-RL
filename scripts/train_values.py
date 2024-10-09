import diffuser.utils as utils
import pdb

"""
'values': {
    'model': 'models.ValueFunction',
    'diffusion': 'models.ValueDiffusion',
    'horizon': 32,
    'n_diffusion_steps': 20,
    'dim_mults': (1, 2, 4, 8),
    'renderer': 'utils.MuJoCoRenderer',

    ## value-specific kwargs
    'discount': 0.99,
    'termination_penalty': -100,
    'normed': False,

    ## dataset
    'loader': 'datasets.ValueDataset',
    'normalizer': 'GaussianNormalizer',
    'preprocess_fns': [],
    'use_padding': True,
    'max_path_length': 1000,

    ## serialization
    'logbase': logbase,
    'prefix': 'values/defaults',
    'exp_name': watch(args_to_watch),

    ## training
    'n_steps_per_epoch': 10000,
    'loss_type': 'value_l2',
    'n_train_steps': 200e3,
    'batch_size': 32,
    'learning_rate': 2e-4,
    'gradient_accumulate_every': 2,
    'ema_decay': 0.995,
    'save_freq': 1000,
    'sample_freq': 0,
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
    dataset: str = 'walker2d-medium-replay-v2'
    config: str = 'config.locomotion'

args = Parser().parse_args('values') # all args stored in config/locomotion.py under 'values' key.


#-----------------------------------------------------------------------------#
#---------------------------------- dataset ----------------------------------#
#-----------------------------------------------------------------------------#

# instantiate a config class for datasets.ValueDataset class
dataset_config = utils.Config(
    args.loader,
    savepath=(args.savepath, 'dataset_config.pkl'),
    env=args.dataset,
    horizon=args.horizon,
    normalizer=args.normalizer,
    preprocess_fns=args.preprocess_fns,
    use_padding=args.use_padding,
    max_path_length=args.max_path_length,
    ## value-specific kwargs
    discount=args.discount,
    termination_penalty=args.termination_penalty,
    normed=args.normed,
)

# instantiate a config class for utils.MuJoCoRenderer class
render_config = utils.Config(
    args.renderer,
    savepath=(args.savepath, 'render_config.pkl'),
    env=args.dataset,
)

dataset = dataset_config() # Creates a new instance of dataset i.e datasets.ValueDataset class in diffuser/datasets/sequence.py
renderer = render_config() # Creates a new instance of dataset i.e utils.MuJoCoRenderer class in diffuser/utils/rendering.py

"""
ValueDataset is a subclass of SequenceDataset with addtional information on value. For all trajectories, the cummulative reward is 
calculated. 
value = (discounts * rewards).sum()
This value is added to the batch.
"""

# Dimensions of env stored in SequenceDataset class member variables.
observation_dim = dataset.observation_dim
action_dim = dataset.action_dim

#-----------------------------------------------------------------------------#
#------------------------------ model & trainer ------------------------------#
#-----------------------------------------------------------------------------#

# instantiate a config class for models.ValueFunction class
model_config = utils.Config(
    args.model,
    savepath=(args.savepath, 'model_config.pkl'),
    horizon=args.horizon,
    transition_dim=observation_dim + action_dim,
    cond_dim=observation_dim,
    dim_mults=args.dim_mults,
    device=args.device,
)

# instantiate a config class for models.ValueDiffusion class
diffusion_config = utils.Config(
    args.diffusion,
    savepath=(args.savepath, 'diffusion_config.pkl'),
    horizon=args.horizon,
    observation_dim=observation_dim,
    action_dim=action_dim,
    n_timesteps=args.n_diffusion_steps,
    loss_type=args.loss_type,
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

model = model_config() # Creates an instance of the models.ValueFunction class in diffuser/models/temporal.py

# Creates an instance of the models.ValueDiffusion (subclass of GaussianDiffusion) class in diffuser/models/diffusion.py
diffusion = diffusion_config(model) 

trainer = trainer_config(diffusion, dataset, renderer) # Creates an instance of the utils.Trainer class in diffuser/utils/training.py

#-----------------------------------------------------------------------------#
#------------------------ test forward & backward pass -----------------------#
#-----------------------------------------------------------------------------#

print('Testing forward...', end=' ', flush=True)

"""
ValueDataset.__getitem__ returns a namedtuple ValueBatch(trajectories, conditions, value) where trajectories are concat of actions and observations,
conditions = {0:observations[0]} representing the initial state of trajectories. Values are cummulative reward of trajectories.
utils.batchify converts numpy batch to tesnsors for proper input to model.
"""
batch = utils.batchify(dataset[0])

"""
diffusion.loss() computes loss by first adding noise to trajectories, replace observation dimension of 
noisy trajectory with that from conditions dict.
Then the noisy trajectory is passed through the model and model predicts cummulative reward. call returns 
loss, info containing 'a0_loss'(not sure what).
ValueDiffusion.loss -> ValueDiffusion.p_losses -> apply_conditioning, ValueDiffusion.loss_fn
"""
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
