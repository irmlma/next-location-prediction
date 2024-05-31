import os

import pickle as pickle
import datetime
import json

from mobpredict.train import train_net, single_test, get_performance_dict
from mobpredict.networks import TransEncoder, RNNs


def get_trained_nets(config, model, train_loader, val_loader, device, log_dir):
    best_model, perf = train_net(config, model, train_loader, val_loader, device, log_dir=log_dir)
    perf["type"] = "vali"
    return best_model, perf


def get_test_result(config, best_model, test_loader, device):
    return_dict = single_test(config, best_model, test_loader, device)
    performance = get_performance_dict(return_dict)
    performance["type"] = "test"

    return performance


def get_models(config, device):
    if config.networkName == "mhsa":
        model = TransEncoder(config=config).to(device)
    elif config.networkName == "rnn":
        model = RNNs(config=config).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Total number of trainable parameters: ", total_params)

    return model


def init_save_path(config):
    """define the path to save, and save the configuration file."""
    log_dir = os.path.join(config.run_save_root, config.run_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "conf.json"), "w") as fp:
        json.dump(config, fp, indent=4, sort_keys=True)

    return log_dir
