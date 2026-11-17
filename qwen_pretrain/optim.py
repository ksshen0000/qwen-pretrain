"""AdamW parameter groups and checkpoint optimizer-state loading."""


def parameter_groups(model, weight_decay):
    """Decay matrices (including embeddings); exempt norms, biases and other vectors."""
    decay, no_decay = [], []
    for parameter in model.parameters():
        if parameter.requires_grad:
            (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    return [{'params': decay, 'weight_decay': weight_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def build_optimizer(model, config, *, fused=True):
    import torch
    return torch.optim.AdamW(parameter_groups(model, config.weight_decay), lr=config.lr,
                             betas=(0.9, 0.95), eps=1e-8, fused=fused)


def load_optimizer_state(optimizer, model, saved):
    """Load AdamW moments, remapping older single-group checkpoints by parameter order.

    Saved state is keyed by each parameter's position in the saved groups. The
    current groups' hyperparameters (weight decay, fused, ...) always win;
    only the learning rate comes from the checkpoint, so the schedule's first
    step starts from the saved value. Moments are cast to the parameter dtype
    by torch, so BF16 states load into FP32 parameters losslessly.
    """
    parameters = [item for item in model.parameters() if item.requires_grad]
    saved_order = [index for group in saved['param_groups'] for index in group['params']]
    if len(saved_order) != len(parameters):
        raise ValueError(f'Optimizer checkpoint has {len(saved_order)} parameters, '
                         f'model has {len(parameters)}')
    # Older checkpoints used one group over model.parameters(), in model order.
    if len(saved['param_groups']) == 1:
        by_parameter = {id(parameter): saved_order[position]
                        for position, parameter in enumerate(parameters)}
    elif len(saved['param_groups']) == len(optimizer.param_groups):
        by_parameter = {}
        for group, saved_group in zip(optimizer.param_groups, saved['param_groups']):
            if len(group['params']) != len(saved_group['params']):
                raise ValueError('Optimizer parameter groups differ from the checkpoint')
            by_parameter.update((id(parameter), index)
                                for parameter, index in zip(group['params'], saved_group['params']))
    else:
        raise ValueError(f'Cannot map {len(saved["param_groups"])} saved optimizer groups '
                         f'onto {len(optimizer.param_groups)}')
    lrs = {group['lr'] for group in saved['param_groups']}
    if len(lrs) != 1:
        raise ValueError('Expected a single learning rate across saved optimizer groups')
    lr = lrs.pop()
    groups, state, position = [], {}, 0
    for group in optimizer.param_groups:
        ids = []
        for parameter in group['params']:
            old = by_parameter[id(parameter)]
            if old in saved['state']:
                state[position] = saved['state'][old]
            ids.append(position)
            position += 1
        groups.append({**{key: value for key, value in group.items() if key != 'params'},
                       'lr': lr, 'params': ids})
    optimizer.load_state_dict({'state': state, 'param_groups': groups})
