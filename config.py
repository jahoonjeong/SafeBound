import torch


class BaseConfig:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = (42, 43, 44, 45, 46)
    epochs = 50
    lr = 1e-3
    dual_lr_ratio = 0.1
    weight_decay = 1e-4
    dropout = 0.1
    hidden_dim = 32
    num_layers = 3
    validation_ratio = 0.2
    risk_omega = 10.0
    gamma = 1.0


class CMAPSSConfig(BaseConfig):
    name = "CMAPSS"
    data_dir = "./CMAPSSData"
    subsets = ("FD001", "FD002", "FD003", "FD004")
    window_size = 60
    batch_size = 64
    kappa = 50.0
    epsilon = 0.10
    tau = 0.05
    alpha = 50.0
    beta = 1.0


class NCMAPSSConfig(BaseConfig):
    name = "NCMAPSS"
    data_dir = "./N-CMAPSS"
    subsets = ("DS01", "DS02", "DS03")
    window_size = 50
    batch_size = 64
    kappa = 50.0
    epsilon = 0.10
    tau = 0.05
    alpha = 100.0
    beta = 1.0
    sample_stride = 10


class BatteryConfig(BaseConfig):
    name = "Battery"
    data_dir = "./NASA_Battery"
    battery_ids = ("B0005", "B0006", "B0007", "B0018")
    window_size = 256
    batch_size = 64
    kappa = 30.0
    epsilon = 0.05
    tau = 0.02
    alpha = 100.0
    beta = 1.0
