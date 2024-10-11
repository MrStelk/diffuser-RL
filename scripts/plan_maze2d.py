import json
import numpy as np
from os.path import join
import pdb

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils


class Parser(utils.Parser):
    dataset: str = 'maze2d-umaze-v1'
    config: str = 'config.maze2d'

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('plan') # All args in the /config folder under "plan" key.

# logger = utils.Logger(args)

env = datasets.load_environment(args.dataset)

#---------------------------------- loading ----------------------------------#

diffusion_experiment = utils.load_diffusion(args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch)
"""
load_diffusion() calls load_config() function. Both defined in /diffuser/utils/serialization.py file.
load_config() loads in the dataset, renderer, diffusion, trainer classes from the respective .pkl files.
classes are then instantiated.
Trainer class is defined in /diffuser/utils/training.py. 
After instantiaition, Trainer.load() method is called which actually loads the diffusion model weights.
All the instantiations are retuned in  named tuple with (dataset renderer model diffusion ema trainer epoch) keys.
.ema containes the actual model class like an nn.module class for models.
""" 

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

policy = Policy(diffusion, dataset.normalizer)
"""
Policy is a class defined in /diffuser/guides/policy.py. When instantiated, it simply registers the model and
a function(normalize in this case) into its member variables. 
__call__ to Policy class as in the main loop calls the diffusion model and retunes the model output in a 
named tuple with (actions, observations) as keys.
"""

#---------------------------------- main loop ----------------------------------#

observation = env.reset() # Initial setting. 

if args.conditional:
    print('Resetting target')
    env.set_target()

# env.unwrapped._target = (4.0,6.0)

## set conditioning xy position to be the goal
target = env._target
cond = {
    diffusion.horizon - 1: np.array([*target, 0, 0]), # Goal to reach.
}
## observations for rendering
rollout = [observation.copy()] # list to contain all observations.

total_reward = 0

"""
state_tmp = env.sim.get_state()
print("state: ", state_tmp)
qpos = state_tmp.qpos.copy()
qvel = state_tmp.qvel.copy()
qpos[0] = 1.0
qpos[1] = 1.0
env.set_state(qpos, qvel)
"""

env.max_episode_steps=2

for t in range(env.max_episode_steps):

    state = env.state_vector().copy() # current state(observation)
    ## can replan if desired, but the open-loop plans are good enough for maze2d
    ## that we really only need to plan once
    if t == 0:
        cond[0] = observation

        # print("conditions: ", cond)
        action, samples = policy(cond, batch_size=args.batch_size)
        actions = samples.actions[0]
        sequence = samples.observations[0]
        # print("samples : ", samples)
    # pdb.set_trace()

    
    # print("predicted state : ", samples.observations[0][t])

    # ####
    if t < len(sequence) - 1:
        next_waypoint = sequence[t+1]
    else:
        next_waypoint = sequence[-1].copy()
        next_waypoint[2:] = 0
        # pdb.set_trace()

    ## can use actions or define a simple controller based on state predictions
    action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])
    """
    computes the action from the observations. why is this being done when actions are also generated?
    """
    # pdb.set_trace()
    ####

    # else:
    #     actions = actions[1:]
    #     if len(actions) > 1:
    #         action = actions[0]
    #     else:
    #         # action = np.zeros(2)
    #         action = -state[2:]
    #         pdb.set_trace()



    next_observation, reward, terminal, _ = env.step(action) # takes action in environment.
    total_reward += reward
    score = env.get_normalized_score(total_reward)
    print(
        f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
        f'{action}'
    )

    if 'maze2d' in args.dataset:
        xy = next_observation[:2]
        goal = env.unwrapped._target
        print(
            f'maze | pos: {xy} | goal: {goal}'
        )

    ## update rollout observations
    rollout.append(next_observation.copy()) # Adds observation.

    # logger.log(score=score, step=t)

    if t % args.vis_freq == 0 or terminal:
        fullpath = join(args.savepath, f'{t}.png')

        if t == 0: renderer.composite(fullpath, samples.observations, ncol=1)


        # renderer.render_plan(join(args.savepath, f'{t}_plan.mp4'), samples.actions, samples.observations, state)

        ## save rollout thus far
        renderer.composite(join(args.savepath, 'rollout.png'), np.array(rollout)[None], ncol=1)

        # renderer.render_rollout(join(args.savepath, f'rollout.mp4'), rollout, fps=80)

        # logger.video(rollout=join(args.savepath, f'rollout.mp4'), plan=join(args.savepath, f'{t}_plan.mp4'), step=t)

    if terminal:
        break

    observation = next_observation # Update observation.

# logger.finish(t, env.max_episode_steps, score=score, value=0)

## save result as a json file
json_path = join(args.savepath, 'rollout.json')
json_data = {'score': score, 'step': t, 'return': total_reward, 'term': terminal,
    'epoch_diffusion': diffusion_experiment.epoch}
json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)
