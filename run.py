#!/usr/bin/env python

## File to setup the environment for training and evaluation of any problem ## 
# Refer to options.py for definition of values for describing the problem and environment

import os
import json
import pprint as pp

import torch
import torch.optim as optim
from tensorboard_logger import Logger as TbLogger

from nets.critic_network import CriticNetwork
from options import get_options
from train import train_epoch, validate, get_inner_model
from reinforce_baselines import NoBaseline, ExponentialBaseline, CriticBaseline, RolloutBaseline, WarmupBaseline
from nets.attention_model import AttentionModel
from nets.pointer_network import PointerNetwork, CriticNetworkLSTM
from utils import torch_load_cpu, load_problem, move_to


def run(opts):

    # Pretty print the run args
    pp.pprint(vars(opts))

    # Set the random seed [Parameter Used: opts.seed]
    torch.manual_seed(opts.seed)

    # Optionally configure tensorboard
    tb_logger = None
    if not opts.no_tensorboard:
        tb_logger = TbLogger(os.path.join(opts.log_dir, "{}_{}".format(opts.problem, opts.graph_size), opts.run_name))

    os.makedirs(opts.save_dir)
    # Save arguments so exact configuration can always be found
    with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
        json.dump(vars(opts), f, indent=True)

    # Set the device
    opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")

    # Loads the given problem and its methods [Paramter Used: opts.problem]
    # Refer to problems/vrp for CVRP problem setup
    problem = load_problem(opts.problem)

    # Load data from the option load_path
    load_data = {}
    assert opts.load_path is None or opts.resume is None, "Only one of load path and resume can be given"
    load_path = opts.load_path if opts.load_path is not None else opts.resume
    if load_path is not None:
        print('  [*] Loading data from {}'.format(load_path))
        load_data = torch_load_cpu(load_path)

    # Initialize model with the required parameters [Parameter Used: opts.model]
    # We choose to work with attention model
    # Note: Pointer network is not created for CVRP 
    model_class = {
        'attention': AttentionModel,
        'pointer': PointerNetwork
    }.get(opts.model, None)

    assert model_class is not None, "Unknown model: {}".format(model_class)

    # Use the parameters to create model instance [Parameters Used: opts.embedding_dim, opts.hidden_dim, opts.n_encode_layers]
    # Default values for embedding dimension(opts.embedding_dim) and hidden dimensions(opts.hidden_dim) are 128
    # Default value for number of encoding layers(opts.n_encode_layers) = 3 
    model = model_class(
        opts.embedding_dim,
        opts.hidden_dim,
        problem,
        n_encode_layers=opts.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping,
        checkpoint_encoder=opts.checkpoint_encoder,
        shrink_size=opts.shrink_size
    ).to(opts.device)

    # Parallelize model if multiple GPU or GPU cores are available [parameter used opts.use_cuda]
    if opts.use_cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # In case of saved parameters, overwrite model parameters by parameters to load
    model_ = get_inner_model(model)
    model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})

    # Initialize baseline [Parameter Used: opts.baseline]
    """
        For the experiment we use the following values of opts.baseline
        :rollout: Performs PPO loss with Greedy Baseline
        :critic_lstm: Performs PPO loss with Critic LSTM
    """ 
    # Note: Critic network is not defined for CVRP problems
    if opts.baseline == 'exponential':
        baseline = ExponentialBaseline(opts.exp_beta)
    elif opts.baseline == 'critic' or opts.baseline == 'critic_lstm':
        # assert problem.NAME == 'tsp', "Critic only supported for TSP"
        baseline = CriticBaseline(
            (
                CriticNetworkLSTM(
                    2,
                    opts.embedding_dim,
                    opts.hidden_dim,
                    opts.n_encode_layers,
                    opts.tanh_clipping
                )
                if opts.baseline == 'critic_lstm'
                else
                CriticNetwork(
                    2,
                    opts.embedding_dim,
                    opts.hidden_dim,
                    opts.n_encode_layers,
                    opts.normalization
                )
            ).to(opts.device)
        )
    elif opts.baseline == 'rollout':
        baseline = RolloutBaseline(model, problem, opts)
    else:
        assert opts.baseline is None, "Unknown baseline: {}".format(opts.baseline)
        baseline = NoBaseline()

    if opts.bl_warmup_epochs > 0:
        baseline = WarmupBaseline(baseline, opts.bl_warmup_epochs, warmup_exp_beta=opts.exp_beta)

    # Load baseline from data, make sure script is called with same type of baseline
    if 'baseline' in load_data:
        baseline.load_state_dict(load_data['baseline'])

    # Initialize Adam optimizer with given learning rate [Parameter Used: opts.lr_model]
    # In case of Actor-Critic architecture, initialize Critic learnin rate [Parameter Used: opts.lr_critic]
    optimizer = optim.Adam(
        [{'params': model.parameters(), 'lr': opts.lr_model}]
        + (
            [{'params': baseline.get_learnable_parameters(), 'lr': opts.lr_critic}]
            if len(baseline.get_learnable_parameters()) > 0
            else []
        )
    )

    # Load optimizer state
    if 'optimizer' in load_data:
        optimizer.load_state_dict(load_data['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                # if isinstance(v, torch.Tensor):
                if torch.is_tensor(v):
                    state[k] = v.to(opts.device)

    # Initialize learning rate scheduler or learning rate decay [Parameter Used: opts.lr_decay]
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)

    # Create validation dataset for a specific problem instance
    """
        Creating dataset for a configurations
        :param size: Number of nodes in the problem instance (We use 20 for our experiments)
        :param num_samples: Number of problem instances with given number of nodes (default: 20)
        :param filename: Filename for the pickle file generated
        :param distribution: The distribution for sampling coordinates of nodes (default: random uniform)
    """ 
    val_dataset = problem.make_dataset(
        size=opts.graph_size, num_samples=opts.val_size, filename=opts.val_dataset, distribution=opts.data_distribution)
    
    # Resuming model training process in case of failure
    if opts.resume:
        epoch_resume = int(os.path.splitext(os.path.split(opts.resume)[-1])[0].split("-")[1])

        torch.set_rng_state(load_data['rng_state'])
        if opts.use_cuda:
            torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
        # Set the random states
        # Dumping of state was done before epoch callback, so do that now (model is loaded)
        baseline.epoch_callback(model, epoch_resume)
        print("Resuming after {}".format(epoch_resume))
        opts.epoch_start = epoch_resume + 1

    # Deciding to use model in train or inference mode [Parameter Used: opts.eval_only]
    if opts.eval_only:
        # For using any model to validate
        validate(model, val_dataset, opts)
    else:
        # Training process starts

        # Initialize old log probabilities as None (Old log probilities = Policy in previous iterations)
        old_log_probabilities = None
        
        # Run the training loop for given numner of epochs [Parameter Used: opts.epoch_start, opts.n_epochs]
        # For this experiment we use opts.n_epochs = 10, opts.epoch_start = 0
        for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):

            # Call the training each epoch function
            old_log_probabilities = train_epoch(
                model,
                optimizer,
                baseline,
                lr_scheduler,
                epoch,
                val_dataset,
                problem,
                tb_logger,
                opts,
                old_log_probabilities
            )
            
            # Saving the old log probabilites 
            # Note: It is important to detach the old_log_probabilites otherwise training iteration will fail after 1st epoch
            old_log_probabilities = move_to(old_log_probabilities.detach(), opts.device)


if __name__ == "__main__":
    run(get_options())
