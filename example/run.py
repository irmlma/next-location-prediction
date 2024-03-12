import argparse
import numpy as np
import random

import torch
import os
import glob

import pandas as pd
import geopandas as gpd
import datetime

import yaml

from shapely import wkt

from easydict import EasyDict as edict

from mobpredict.utils import (
    prepare_nn_dataset_train,
    prepare_nn_dataset_inference,
    get_train_vali_loaders,
    get_test_loaders,
)
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


def train_run(config, device, log_dir):
    result_ls = []

    # get train and validation loaders
    train_path = os.path.join(config.data_save_root, "temp", config.train_dataset + "_trained_train.pk")
    vali_path = os.path.join(config.data_save_root, "temp", config.train_dataset + "_trained_validation.pk")
    train_loader, val_loader = get_train_vali_loaders(config, train_path=train_path, vali_path=vali_path)

    # get models
    model = get_models(config, device)

    # train, returns validation performances
    best_model, perf = get_trained_nets(config, model, train_loader, val_loader, device, log_dir)
    result_ls.append(perf)

    # get test loader
    test_path = os.path.join(config.data_save_root, "temp", config.train_dataset + "_trained_test.pk")
    test_loader = get_test_loaders(config, test_path=test_path)

    # test, return test performances
    perf = get_test_result(config, best_model, test_loader, device)

    result_ls.append(perf)

    return result_ls


def inference_run(config, device, log_dir, all_files):
    # get default train, vali and test dataset
    default_files = [files for files in all_files if "default" in files]
    default_train = [files for files in default_files if "train" in files][0]
    default_validation = [files for files in default_files if "validation" in files][0]

    result_ls = []

    # get data
    train_loader, val_loader = get_dataloaders(config, default_train, default_validation)

    # get model
    model = get_models(config, device)

    # train
    model, perf = get_trained_nets(config, model, train_loader, val_loader, device, log_dir)
    # model.load_state_dict(torch.load(r"D:\Code\HAI-IRMA\WP1\npp\outputs\gc_transformer_1679872715/checkpoint.pt"))

    # get the performance
    # perf["dataset"] = default_validation.split("\\")[-1].split(".")[0]
    # result_ls.append(perf)

    for dataset_dir in all_files:
        if dataset_dir == default_train or dataset_dir == default_validation:
            continue

        filename = dataset_dir.split("\\")[-1].split(".")[0]

        if "validation" in filename:
            continue

        test_loader = get_test_loader(config, dataset_dir)

        # test
        perf, user = get_test_result(config, model, test_loader, device)
        user.to_csv(os.path.join(log_dir, "user_detail.csv"))

        perf["dataset"] = dataset_dir.split("\\")[-1].split(".")[0]
        print(perf)
        result_ls.append(perf)

    return result_ls


def load_data(sp):
    def _transfer_time_to_absolute(df, start_time):
        duration_arr = df["duration"].to_list()[:-1]
        duration_arr.insert(0, 0)
        timedelta_arr = np.array([datetime.timedelta(hours=i) for i in np.cumsum(duration_arr)])

        df["started_at"] = timedelta_arr + start_time
        df["finished_at"] = df["started_at"] + pd.to_timedelta(df["duration"], unit="hours")

        min_day = pd.to_datetime(df["started_at"].min().date())

        df["start_day"] = (df["started_at"] - min_day).dt.days
        df["start_min"] = df["started_at"].dt.hour * 60 + df["started_at"].dt.minute
        df["weekday"] = df["started_at"].dt.weekday
        df["duration"] = (df["duration"] * 60).round()

        return df

    # sp["geometry"] = sp["geometry"].apply(wkt.loads)
    # sp = gpd.GeoDataFrame(sp, geometry="geometry", crs="EPSG:4326")

    sp.index.name = "id"
    sp.reset_index(inplace=True)

    # transfer duration to absolut time format for each user
    sp = sp.groupby("user_id", as_index=False, group_keys=False).apply(
        _transfer_time_to_absolute, start_time=datetime.datetime(2023, 1, 1, hour=8)
    )
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
        max_locations, max_users = prepare_nn_dataset_train(
            sp,
            train_name=config.train_dataset,
            save_root=config.data_save_root,
        )
        config["total_loc_num"] = int(max_locations + 1)
        config["total_user_num"] = int(max_users + 1)

        # res_single contains the performance of validation and test of the current run
        res_single = train_run(config, device, log_dir)

        # save results
        result_df = pd.DataFrame(res_single)
        type = "default"
        filename = os.path.join(log_dir, f"{config.train_dataset}_{config.networkName}_{type}.csv")
        result_df.to_csv(filename, index=False)
    else:  # for inference
        all_files = glob.glob(os.path.join(config.data_save_root, config.inference_data_dir, "*.csv"))

        inf_sps_ls = []
        filename_ls = []
        for file in all_files:
            sp = pd.read_csv(file, index_col="index")
            sp = load_data(sp)
            inf_sps_ls.append(sp)
            filename_ls.append(file.split(os.sep)[-1].split(".")[0])

        train_sp = pd.read_csv(os.path.join(config.data_save_root, f"{config.train_dataset}.csv"), index_col="index")
        train_sp = load_data(train_sp)

        # get data, initialize the location and user number
        max_locations, max_users = prepare_nn_dataset_inference(
            inf_sps_ls,
            train_sp,
            save_root=config.data_save_root,
            inference_names=filename_ls,
            train_name=config.train_dataset,
        )
        config["total_loc_num"] = int(max_locations + 1)
        config["total_user_num"] = int(max_users + 1)
