# Next location prediction

[![arXiv](https://img.shields.io/badge/arXiv-2311.11749-b31b1b.svg)](https://arxiv.org/abs/2311.11749)

## Install

Install the package in edit mode using:
```
pip install -e .
```

## Neural network implementation

`mobpredict/networks/` contains network implementation for multi-head self-attentional (MHSA) model and LSTM models.

## Training

Run 
```
python example/run.py
```
with `training: True` in `example/config/config.yml` file. The code will train a neural network for next location prediction with a dataset generated from [mobility-simulation](https://github.com/irmlma/mobility-simulation). The `train_dataset` shall be avilable as a `.csv` file stored in `data_save_root`. The other hyper parameters are defined in the config yml file. A folder with specified folder name (`run_name`) containing the trained nn parameters will be created in `run_save_root`. 


## Inference

Run 
```
python example/run.py
```
with `training: False` in `example/config/config.yml` file. The code will take an already trained neural network for next location prediction, stored in `run_save_root` with dir name `pretrain_dir`, for inference for all datasets stored in `data_save_root` under the dir `inference_data_dir`. The datasets shall be in the format generated with [mobility-simulation](https://github.com/irmlma/mobility-simulation). A folder containing the evaluation results will be created in `run_save_root`. 

We provide an already trained model with the default config parameters on the provided dtepr dataset. 
## Known issues:
None

## TODO:
None

## Citation
If you find this code useful for your work or use it in your project, please consider citing:

```shell
@misc{hong_revealing_2023,
    title={Revealing behavioral impact on mobility prediction networks through causal interventions},
    author={Hong, Ye and Xin, Yanan and Dirmeier, Simon and Perez-Cruz, Fernando and Raubal, Martin},
    publisher={arXiv},
    year={2023},
    url = {https://arxiv.org/abs/2311.11749},
    doi = {10.48550/arXiv.2311.11749},
}
```

## Contact
If you have any questions, open an issue or let me know: 
- Ye Hong {hongy@ethz.ch}