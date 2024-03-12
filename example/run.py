import argparse
import numpy as np
import random

import torch
import os

import pandas as pd
import geopandas as gpd
import datetime

import yaml

from shapely import wkt

from easydict import EasyDict as edict

from mobpredict.utils import prepare_nn_dataset, get_dataloaders
from mobpredict.train import init_save_path, get_models, get_test_result, get_trained_nets


def load_config(path):
    """
    Loads config file
    Args:
        path (str): path to the config file
    Returns:
        config (dict): dictionary of the configuration parameters, merge sub_dicts
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    config = dict()
    for _, value in cfg.items():
        for k, v in value.items():
            config[k] = v

    return config


def setup_seed(seed):
    """
    fix random seed for deterministic training
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def single_run(train_loader, val_loader, test_loader, config, device, log_dir):
    result_ls = []

    # get models
    model = get_models(config, device)

    # train, returns validation performances
    best_model, perf = get_trained_nets(config, model, train_loader, val_loader, device, log_dir)
    result_ls.append(perf)

    # test, return test performances
    perf = get_test_result(config, best_model, test_loader, device)

    result_ls.append(perf)

    return result_ls


def load_data(sp):
    sp["geometry"] = sp["geometry"].apply(wkt.loads)
    sp = gpd.GeoDataFrame(sp, geometry="geometry", crs="EPSG:4326")

    sp.index.name = "id"
    sp.reset_index(inplace=True)

    def _transfer_time_to_absolute(df, start_time):
        duration_arr = df["duration"].to_list()[:-1]
        duration_arr.insert(0, 0)
        timedelta_arr = np.array([datetime.timedelta(hours=i) for i in np.cumsum(duration_arr)])

        df["started_at"] = timedelta_arr + start_time
        df["finished_at"] = df["started_at"] + pd.to_timedelta(df["duration"], unit="hours")

        return df

    # transfer duration to absolut time format for each user
    sp = sp.groupby("user_id", as_index=False).apply(
        _transfer_time_to_absolute, start_time=datetime.datetime(2023, 1, 1, hour=8)
    )
    sp.reset_index(drop=True, inplace=True)

    def _get_time_info(df):
        min_day = pd.to_datetime(df["started_at"].min().date())

        df["start_day"] = (df["started_at"] - min_day).dt.days
        df["start_min"] = df["started_at"].dt.hour * 60 + df["started_at"].dt.minute
        df["weekday"] = df["started_at"].dt.weekday
        df["duration"] = (df["duration"] * 60).round()
        return df

    sp = sp.groupby("user_id", group_keys=False).apply(_get_time_info)
    return sp


if __name__ == "__main__":
    setup_seed(0)

    # load configs
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        type=str,
        nargs="?",
        help="Path to the config file.",
        default="./example/config/config.yml",
    )
    args = parser.parse_args()

    # initialization
    config = load_config(args.config)
    config = edict(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = init_save_path(config)

    if config.training:  # for training
        # read and preprocess
        sp = pd.read_csv(os.path.join(config.data_save_root, f"{config.train_dataset}.csv"), index_col="index")
        sp = load_data(sp)

        # get data for nn, initialize the location and user number
        max_locations, max_users = prepare_nn_dataset(sp, config.data_save_root)
        config["total_loc_num"] = int(max_locations + 1)
        config["total_user_num"] = int(max_users + 1)

        # get dataloaders
        train_loader, val_loader, test_loader = get_dataloaders(config)

        # train, validate and test
        # res_single contains the performance of validation and test of the current run
        res_single = single_run(train_loader, val_loader, test_loader, config, device, log_dir)

        # save results
        result_df = pd.DataFrame(res_single)
        type = "default"
        filename = os.path.join(log_dir, f"{config.train_dataset}_{config.networkName}_{type}.csv")
        result_df.to_csv(filename, index=False)
    else:  # for inference
        pass
