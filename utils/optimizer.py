from torch import optim

def get_optimizer(config, model):
    params = [p for p in model.parameters()]
    if config.optimizer == 'Adam':
        return optim.Adam(params, lr=config.lr, weight_decay = config.weight_decay)
    