import os
import time
from tqdm import tqdm
import torch
import math
import numpy as np
import wandb
from torch.utils.data import DataLoader
from torch.nn import DataParallel

from torch.distributions import Categorical
from nets.attention_model import set_decode_type
from utils.log_utils import log_values
from utils import move_to


def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def validate(model, dataset, opts):
    # Validate
    print('Validating...')
    cost = rollout(model, dataset, opts)
    avg_cost = cost.mean()
    print('Validation overall avg_cost: {} +- {}'.format(
        avg_cost, torch.std(cost) / math.sqrt(len(cost))))

    return avg_cost


def rollout(model, dataset, opts):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _, __ = model(move_to(bat, opts.device))
        return cost.data.cpu()

    return torch.cat([
        eval_model_bat(bat)
        for bat
        in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar)
    ], 0)


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def train_epoch(model, optimizer, baseline, lr_scheduler, epoch, val_dataset, problem, tb_logger, opts, runs, old_log_probabilities = None):
    print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()

    if not opts.no_tensorboard:
        tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)

    # Generate new training data for each epoch
    training_dataset = baseline.wrap_dataset(problem.make_dataset(
        size=opts.graph_size, num_samples=opts.epoch_size, distribution=opts.data_distribution))
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=1)

    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")
    collect_old_log_probabilities = []
    # collect_old_log_probabilities = move_to(collect_old_log_probabilities, opts.device)
    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):

        old_log_prob = train_batch(
            model,
            optimizer,
            baseline,
            epoch,
            batch_id,
            step,
            batch,
            tb_logger,
            opts,
            runs,
            old_log_probabilities

        )
        collect_old_log_probabilities.append(old_log_prob)
        step += 1
    print(collect_old_log_probabilities[-1], len(collect_old_log_probabilities))
    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
        print('Saving model and state...')
        torch.save(
            {
                'model': get_inner_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'baseline': baseline.state_dict()
            },
            os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )

    avg_reward = validate(model, val_dataset, opts)

    if not opts.no_tensorboard:
        tb_logger.log_value('val_avg_reward', avg_reward, step)

    baseline.epoch_callback(model, epoch)

    # lr_scheduler should be called at end of epoch
    lr_scheduler.step()
    last_iter_log_prob = move_to(collect_old_log_probabilities[-1], opts.device)
    return last_iter_log_prob


def train_batch(
        model,
        optimizer,
        baseline,
        epoch,
        batch_id,
        step,
        batch,
        tb_logger,
        opts,
        runs,
        old_log_prob = None
):
    x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None
    
    # Evaluate model, get costs and log probabilities
    cost, log_likelihood, log_probabilities = model(x)
    shapeOfLog_probabilities = log_probabilities.shape
    
    if old_log_prob == None:
      old_log_prob = torch.zeros(size = shapeOfLog_probabilities)
    with torch.no_grad():
      entropy = Categorical(probs = torch.exp(log_probabilities)).entropy().mean()
      entropy = move_to(entropy, opts.device)

    # print(old_log_prob.shape)
    # print(log_probabilities.shape)
    
    diff = old_log_prob.shape[1] - log_probabilities.shape[1]
    if diff > 0:
        log_probabilites_arr = log_probabilities.cpu().detach().numpy()
        padded_log_probabilities_arr = np.pad(log_probabilites_arr, ((0, 0), (0, diff)), 'constant', constant_values=(-999,))
        log_probabilities = torch.tensor(padded_log_probabilities_arr)
    elif diff < 0:
        old_log_probabilites_arr = old_log_prob.cpu().detach().numpy()
        # padded_old_log_probabilities_arr = np.pad(old_log_probabilites_arr, ((0, 0), (0, -diff)), -999)
        padded_old_log_probabilities_arr = np.pad(old_log_probabilites_arr, ((0, 0), (0, -diff)), 'constant', constant_values=(-999,))
        old_log_prob = torch.tensor(padded_old_log_probabilities_arr)

    old_log_prob = move_to(old_log_prob, opts.device)
    log_probabilities = move_to(log_probabilities, opts.device)

    # Evaluate baseline, get baseline loss if any (only for critic)
    bl_val, bl_loss = baseline.eval(x['loc'], cost) if bl_val is None else (bl_val, 0)
    bl_val_1 = torch.reshape(bl_val, (-1,))
    
    # Calculate loss
    reinforce_loss = ((cost - bl_val) * log_likelihood).mean()
    # loss = reinforce_loss + bl_loss
    advantage = cost - bl_val_1
    advantage = torch.reshape(advantage, (advantage.shape[0], 1))
    clip_param = 0.2
    # log_probabilities = torch.tensor(log_probabilities).cuda()
    # old_log_prob = torch.tensor(old_log_prob).cuda()
    ratio = (log_probabilities - old_log_prob).exp() #calc ratio for ppo

    #print('Shape of Ratio: ', ratio.shape)
    #print('Shape of Advantage: ', advantage.shape)
    #print('Baseline loss: ', bl_loss)
    surr1 = torch.mul(advantage, ratio)     
    surr2 = torch.mul(torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param), advantage) # torch.mul(advantage, torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)) advantage * torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)
    actor_loss  = - torch.min(surr1, surr2).mean() 
    critic_loss = bl_loss
    # print('Actor loss: ', actor_loss)
    # print('Critic loss: ', critic_loss)
    # print('Entropy: ', entropy)
    loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

    wandb.log({"LR_model": runs.config.lr_model_val, "LR_critic": runs.config.lr_critic_val, "loss": loss})

    # Perform backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()
    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()

    # Logging
    if step % int(opts.log_step) == 0:
        log_values(cost, grad_norms, epoch, batch_id, step,
                   log_likelihood, reinforce_loss, bl_loss, tb_logger, opts)
    return log_probabilities
