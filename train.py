## File for training and evaluation of model
import os
import time
from tqdm import tqdm
import torch
import math
import numpy as np

from torch.utils.data import DataLoader
from torch.nn import DataParallel

from nets.attention_model import set_decode_type
from utils.log_utils import log_values
from utils import move_to

# Load model based on the architecture of machine
def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model

# Validating the model based on validation dataset and reporting average distance
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
            cost, _ = model(move_to(bat, opts.device))
        return cost.data.cpu()

    return torch.cat([
        eval_model_bat(bat)
        for bat
        in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar)
    ], 0)


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
        Clips the norms for all param groups to max_norm and returns gradient norms before clipping
        :param optimizer: Adams Optimizer with variable learning rate
        :param max_norm: The maximum value of gradient, given by opts.max_grad_norm
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


def train_epoch(model, optimizer, baseline, lr_scheduler, epoch, val_dataset, problem, tb_logger, opts, old_log_likelihood = None):
    """
        Training of model for given number of epochs
        :param model: Attention model selected in run.py
        :param optimizer: Adams Optimizer with variable learning rate
        :param baseline: Baseline or Critic chosen to perform experiment
        :param lr_scheduler: Change the learning rate after each epoch
        :param epoch: Number of training epochs for each experiment
        :param val_dataset: Validation dataset used to validate the model after every epoch
        :param problem: Problem to solve (CVRP)
        :param tb_logger: Logger to save important value after every few steps 
        :param opts: Configuration established in the run command or by default values
        :param old_log_likelihood: Policy for previous iterations                                                                           
        :return: Likelihood values(Policy) of the current epoch 
    """
    print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))

    # Calculate number of times we train a model on problem instance in each epoch [Parameter Used: opts.epoch_size, opts.batch_size]
    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()

    # Connecting model to tensorboard
    if not opts.no_tensorboard:
        tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)

    # Generate new training data for each epoch, each trains on a different problem instance for 'step' times
    training_dataset = baseline.wrap_dataset(problem.make_dataset(
        size=opts.graph_size, num_samples=opts.epoch_size, distribution=opts.data_distribution))
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=1)

    # Put model in train mode!
    model.train()

    # Training happens in sampling mode to allow the model to explore different paths
    """
        Different Sampling modes
        :sample: Use multinomial distribution to sample node
        :greedy: Use max probability to sample node
    """  
    set_decode_type(model, "sampling")
    collect_old_log_likelihoods = []

    # Run a particular dataset 'step' times in each epochs
    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):
        # Run each step in an epoch
        old_log_like = train_batch(
            model,
            optimizer,
            baseline,
            epoch,
            batch_id,
            step,
            batch,
            tb_logger,
            opts,
            old_log_likelihood
        )
        collect_old_log_likelihoods.append(old_log_like)
        step += 1
    # print(collect_old_log_likelihoods[-1], len(collect_old_log_likelihoods))
    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    # Save model after every epoch, allows to load model again if it fails at any given epoch
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

    # Validate the model after completion of epoch on validation datasets
    avg_reward = validate(model, val_dataset, opts)

    if not opts.no_tensorboard:
        tb_logger.log_value('val_avg_reward', avg_reward, step)

    # Update the baseline with the newly trained model (only for baseline = rollout)
    baseline.epoch_callback(model, epoch)

    # lr_scheduler should be called at end of epoch
    lr_scheduler.step()

    # Select the final policy on a given dataset
    last_iter_log_prob = move_to(collect_old_log_likelihoods[-1], opts.device)
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
        old_log_likelihood = None
):
    """
        Training of model for each step in epoch
        :param model: Attention model selected in run.py
        :param optimizer: Adams Optimizer with variable learning rate
        :param baseline: Baseline or Critic chosen to perform experiment
        :param epoch: Number of training epochs for each experiment
        :param step: Number of times we train a model on problem instance in each epoch  
        :param batch_id: Batch id of each batch
        :param batch: Problem instances in a given batch (default 512) 
        :param tb_logger: Logger to save important value after every few steps
        :param opts: Configuration established in the run command or by default values
        :param old_log_likelihood: Policy for previous iterations                                                                           
        :return: Likelihood values(Policy) of the current step 
    """

    # Seperate the inputvalue of batch size and any inital value of cost 
    x, bl_val = baseline.unwrap_batch(batch)

    # Move the variable to device based on whether GPU is available or not [Parameter used: opts.device] 
    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None
    
    # Evaluate model, get costs and log likelihood
    cost, log_likelihood = model(x)

    # For the first epoch we do not have any old policy, therefore we use the average value of current likelihood as subsitute 
    shapeOfLikelihood = log_likelihood.shape
    if old_log_likelihood == None:
        # Getting the mean value of likelihood for a given batch
        mean_likelihood = log_likelihood.mean().item()
        old_log_likelihood = np.full(shapeOfLikelihood[0], mean_likelihood)
        old_log_likelihood = torch.as_tensor(old_log_likelihood)

    old_log_likelihood = move_to(old_log_likelihood, opts.device)
    log_likelihood = move_to(log_likelihood, opts.device)

    # Evaluate baseline/critic, get baseline/critic loss if any
    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)

    # Reshape critic output to for faster operations
    bl_val_1 = torch.reshape(bl_val, (-1,))
    
    # Calculate Advantage function, this enable algorithm to compare action loss with critic or baseline loss   
    advantage = cost - bl_val

    # Clipping values for PPO loss (default 0.2, selected in paper)
    clip_param = 0.2

    # Define the PPO loss function
    # Note: Dividing likelihoods can lead to unexpected behavior, therefore we subtract log likelihoods and then exponentiate them 
    ratio = log_likelihood - old_log_likelihood
    ratio = torch.exp(ratio)

    surr1 = ratio
    # Clipping the gradients
    surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)

    # Selecting the minimum values between original and clipped gradients
    actor_loss = torch.min(surr1, surr2)
    
    # Combining all the values
    # Note: We add an extra 0.5xbl_loss to allow the LSTM model to update its parameters using backpropogation
    loss = 0.5 * bl_loss + (-(bl_val_1 - cost) * actor_loss).mean() #Added negative sign

    # Perform backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()

    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()

    # Logging values after every few steps [parameter used: opts.log_step] 
    if step % int(opts.log_step) == 0:
        log_values(cost, grad_norms, epoch, batch_id, step,
                   log_likelihood, loss, bl_loss, tb_logger, opts)

    # Retunr Likelihood values after each step
    return log_likelihood