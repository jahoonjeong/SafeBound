import os

import h5py
import numpy as np
import pandas as pd
import scipy.io
import torch
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


def _split_ids(ids, ratio, seed):
    ids = np.asarray(ids)
    if len(ids) < 2:
        raise ValueError("At least two units are required for a train/validation split.")
    rng = np.random.default_rng(seed)
    ids = rng.permutation(ids)
    n_validation = max(1, min(len(ids) - 1, int(round(len(ids) * ratio))))
    return ids[n_validation:], ids[:n_validation]


def _split_indices(size, ratio, seed):
    if size < 2:
        raise ValueError("At least two samples are required for a train/validation split.")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(size)
    n_validation = max(1, min(size - 1, int(round(size * ratio))))
    return indices[n_validation:], indices[:n_validation]


def _rate_bounds(target):
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    positive = target[target > 0.0]
    if positive.size == 0:
        raise ValueError("Training targets must contain positive values.")
    return 1.0 / float(np.max(positive)), 1.0 / float(np.min(positive))


class CMAPSSLoader:
    def __init__(self, config, seed):
        self.data_dir = config.data_dir
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.validation_ratio = config.validation_ratio
        self.seed = seed
        self.sensor_columns = [
            "s2", "s3", "s4", "s7", "s8", "s9", "s11",
            "s12", "s13", "s14", "s15", "s17", "s20", "s21",
        ]
        self.condition_columns = ["os1", "os2", "os3"]
        self.columns = ["unit", "cycle", "os1", "os2", "os3"] + [
            f"s{i}" for i in range(1, 22)
        ]

    def _read(self, subset, train):
        prefix = "train" if train else "test"
        return pd.read_csv(
            os.path.join(self.data_dir, f"{prefix}_{subset}.txt"),
            sep=r"\s+",
            header=None,
            names=self.columns,
        )

    def _windows(self, frame):
        sensors, conditions, targets = [], [], []
        for unit in frame["unit"].unique():
            unit_frame = frame[frame["unit"] == unit]
            if len(unit_frame) < self.window_size:
                continue
            x = unit_frame[self.sensor_columns].to_numpy()
            c = unit_frame[self.condition_columns].to_numpy()
            y = unit_frame["RUL"].to_numpy()
            count = len(unit_frame) - self.window_size + 1
            index = np.arange(count)[:, None] + np.arange(self.window_size)[None, :]
            sensors.append(x[index])
            conditions.append(c[index])
            targets.append(y[self.window_size - 1:])
        if not sensors:
            raise ValueError("No valid C-MAPSS windows were created.")
        return np.concatenate(sensors), np.concatenate(conditions), np.concatenate(targets)

    def get_loaders(self, subset):
        frame = self._read(subset, train=True)
        eol = frame.groupby("unit")["cycle"].max()
        frame = frame.merge(eol.rename("eol"), on="unit")
        frame["RUL"] = frame["eol"] - frame["cycle"]
        frame = frame.drop(columns="eol")

        train_units, validation_units = _split_ids(
            frame["unit"].unique(), self.validation_ratio, self.seed
        )
        train_frame = frame[frame["unit"].isin(train_units)].copy()
        validation_frame = frame[frame["unit"].isin(validation_units)].copy()

        sensor_scaler = StandardScaler().fit(train_frame[self.sensor_columns])
        condition_scaler = StandardScaler().fit(train_frame[self.condition_columns])
        for split in (train_frame, validation_frame):
            split[self.sensor_columns] = sensor_scaler.transform(split[self.sensor_columns])
            split[self.condition_columns] = condition_scaler.transform(split[self.condition_columns])

        train_x, train_c, train_y = self._windows(train_frame)
        validation_x, validation_c, validation_y = self._windows(validation_frame)

        test_frame = self._read(subset, train=False)
        test_frame[self.sensor_columns] = sensor_scaler.transform(test_frame[self.sensor_columns])
        test_frame[self.condition_columns] = condition_scaler.transform(test_frame[self.condition_columns])
        test_truth = pd.read_csv(
            os.path.join(self.data_dir, f"RUL_{subset}.txt"),
            sep=r"\s+",
            header=None,
            names=["RUL"],
        )["RUL"].to_numpy()

        test_x, test_c, test_y = [], [], []
        for unit in test_frame["unit"].unique():
            unit_frame = test_frame[test_frame["unit"] == unit]
            if len(unit_frame) < self.window_size:
                continue
            test_x.append(unit_frame[self.sensor_columns].to_numpy()[-self.window_size:])
            test_c.append(unit_frame[self.condition_columns].to_numpy()[-self.window_size:])
            test_y.append(test_truth[int(unit) - 1])

        train = TensorDataset(
            torch.tensor(train_x, dtype=torch.float32),
            torch.tensor(train_c, dtype=torch.float32),
            torch.tensor(train_y, dtype=torch.float32),
        )
        validation = TensorDataset(
            torch.tensor(validation_x, dtype=torch.float32),
            torch.tensor(validation_c, dtype=torch.float32),
            torch.tensor(validation_y, dtype=torch.float32),
        )
        test = TensorDataset(
            torch.tensor(np.asarray(test_x), dtype=torch.float32),
            torch.tensor(np.asarray(test_c), dtype=torch.float32),
            torch.tensor(np.asarray(test_y), dtype=torch.float32),
        )

        return (
            DataLoader(train, batch_size=self.batch_size, shuffle=True),
            DataLoader(validation, batch_size=256, shuffle=False),
            DataLoader(test, batch_size=256, shuffle=False),
            _rate_bounds(train_y),
        )


class NCMAPSSLoader:
    def __init__(self, config, seed):
        self.data_dir = config.data_dir
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.stride = config.sample_stride
        self.validation_ratio = config.validation_ratio
        self.seed = seed

    def _windows(self, sensors, conditions, target, units):
        x_list, c_list, y_list = [], [], []
        for unit in np.unique(units):
            mask = units == unit
            x = sensors[mask]
            c = conditions[mask]
            y = target[mask]
            if len(x) < self.window_size:
                continue
            count = (len(x) - self.window_size) // self.stride + 1
            starts = np.arange(0, count * self.stride, self.stride)
            index = starts[:, None] + np.arange(self.window_size)[None, :]
            x_list.append(x[index])
            c_list.append(c[index])
            y_list.append(y[starts + self.window_size - 1])
        if not x_list:
            raise ValueError("No valid N-CMAPSS windows were created.")
        return np.concatenate(x_list), np.concatenate(c_list), np.concatenate(y_list)

    def get_loaders(self, subset):
        files = sorted(
            name for name in os.listdir(self.data_dir)
            if subset in name and name.endswith(".h5")
        )
        if not files:
            raise FileNotFoundError(f"H5 file for {subset} not found.")

        with h5py.File(os.path.join(self.data_dir, files[0]), "r") as handle:
            w_dev = np.asarray(handle["W_dev"])
            x_dev = np.asarray(handle["X_s_dev"])
            y_dev = np.asarray(handle["Y_dev"]).reshape(-1)
            a_dev = np.asarray(handle["A_dev"])

            if all(key in handle for key in ("W_test", "X_s_test", "Y_test", "A_test")):
                w_test = np.asarray(handle["W_test"])
                x_test = np.asarray(handle["X_s_test"])
                y_test = np.asarray(handle["Y_test"]).reshape(-1)
                a_test = np.asarray(handle["A_test"])
            else:
                dev_units = np.unique(a_dev[:, 0])
                keep_units, test_units = _split_ids(dev_units, 0.2, self.seed + 1000)
                test_mask = np.isin(a_dev[:, 0], test_units)
                keep_mask = np.isin(a_dev[:, 0], keep_units)
                w_test, x_test, y_test, a_test = (
                    w_dev[test_mask], x_dev[test_mask], y_dev[test_mask], a_dev[test_mask]
                )
                w_dev, x_dev, y_dev, a_dev = (
                    w_dev[keep_mask], x_dev[keep_mask], y_dev[keep_mask], a_dev[keep_mask]
                )

        train_units, validation_units = _split_ids(
            np.unique(a_dev[:, 0]), self.validation_ratio, self.seed
        )
        train_mask = np.isin(a_dev[:, 0], train_units)
        validation_mask = np.isin(a_dev[:, 0], validation_units)

        sensor_scaler = StandardScaler().fit(x_dev[train_mask])
        condition_scaler = StandardScaler().fit(w_dev[train_mask])

        train_x, train_c, train_y = self._windows(
            sensor_scaler.transform(x_dev[train_mask]),
            condition_scaler.transform(w_dev[train_mask]),
            y_dev[train_mask],
            a_dev[train_mask, 0],
        )
        validation_x, validation_c, validation_y = self._windows(
            sensor_scaler.transform(x_dev[validation_mask]),
            condition_scaler.transform(w_dev[validation_mask]),
            y_dev[validation_mask],
            a_dev[validation_mask, 0],
        )
        test_x, test_c, test_y = self._windows(
            sensor_scaler.transform(x_test),
            condition_scaler.transform(w_test),
            y_test,
            a_test[:, 0],
        )

        train = TensorDataset(
            torch.tensor(train_x, dtype=torch.float32),
            torch.tensor(train_c, dtype=torch.float32),
            torch.tensor(train_y, dtype=torch.float32),
        )
        validation = TensorDataset(
            torch.tensor(validation_x, dtype=torch.float32),
            torch.tensor(validation_c, dtype=torch.float32),
            torch.tensor(validation_y, dtype=torch.float32),
        )
        test = TensorDataset(
            torch.tensor(test_x, dtype=torch.float32),
            torch.tensor(test_c, dtype=torch.float32),
            torch.tensor(test_y, dtype=torch.float32),
        )

        return (
            DataLoader(train, batch_size=self.batch_size, shuffle=True),
            DataLoader(validation, batch_size=256, shuffle=False),
            DataLoader(test, batch_size=256, shuffle=False),
            _rate_bounds(train_y),
        )


class BatteryLoader:
    def __init__(self, config, seed):
        self.data_dir = config.data_dir
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.battery_ids = config.battery_ids
        self.validation_ratio = config.validation_ratio
        self.seed = seed

    def _resample(self, data):
        if len(data) == 0:
            return np.zeros((self.window_size, data.shape[1]))
        if len(data) == 1:
            return np.repeat(data, self.window_size, axis=0)
        old_axis = np.linspace(0.0, 1.0, len(data))
        new_axis = np.linspace(0.0, 1.0, self.window_size)
        return np.stack(
            [interp1d(old_axis, data[:, i], kind="linear")(new_axis) for i in range(data.shape[1])],
            axis=1,
        )

    def load_battery(self, battery_id):
        path = os.path.join(self.data_dir, f"{battery_id}.mat")
        if not os.path.exists(path):
            return None, None, None
        try:
            mat = scipy.io.loadmat(path)
            cycles = mat[battery_id][0, 0]["cycle"][0]
        except (OSError, KeyError, TypeError, ValueError, IndexError):
            return None, None, None

        sensors, targets, conditions = [], [], []
        for cycle in cycles:
            if cycle["type"][0] != "discharge":
                continue
            data = cycle["data"][0, 0]
            if "Capacity" not in data.dtype.names:
                continue
            voltage = np.asarray(data["Voltage_measured"][0]).reshape(-1)
            current = np.asarray(data["Current_measured"][0]).reshape(-1)
            temperature = np.asarray(data["Temperature_measured"][0]).reshape(-1)
            if len(voltage) == 0:
                continue
            features = np.stack((voltage, current, temperature), axis=1)
            sensors.append(self._resample(features))
            conditions.append(np.mean(features, axis=0))
            targets.append(float(np.asarray(data["Capacity"][0, 0]).squeeze()))

        if not sensors:
            return None, None, None
        return np.asarray(sensors), np.asarray(targets), np.asarray(conditions)

    def load_loo(self, test_id):
        data = {}
        for battery_id in self.battery_ids:
            sensors, targets, conditions = self.load_battery(battery_id)
            if sensors is not None:
                data[battery_id] = (sensors, targets, conditions)

        if test_id not in data:
            raise ValueError(f"Test battery {test_id} was not loaded.")

        train_sensors, train_targets, train_conditions = [], [], []
        for battery_id in self.battery_ids:
            if battery_id == test_id or battery_id not in data:
                continue
            sensors, targets, conditions = data[battery_id]
            train_sensors.append(sensors)
            train_targets.append(targets)
            train_conditions.append(conditions)

        train_x = np.concatenate(train_sensors)
        train_y = np.concatenate(train_targets).reshape(-1)
        train_c = np.concatenate(train_conditions)
        test_x, test_y, test_c = data[test_id]
        test_y = test_y.reshape(-1)

        train_index, validation_index = _split_indices(
            len(train_y), self.validation_ratio, self.seed
        )

        sensor_mean = np.mean(train_x[train_index], axis=(0, 1))
        sensor_std = np.std(train_x[train_index], axis=(0, 1)) + 1e-6
        condition_mean = np.mean(train_c[train_index], axis=0)
        condition_std = np.std(train_c[train_index], axis=0) + 1e-6

        train_x = (train_x - sensor_mean) / sensor_std
        test_x = (test_x - sensor_mean) / sensor_std
        train_c = (train_c - condition_mean) / condition_std
        test_c = (test_c - condition_mean) / condition_std

        train = TensorDataset(
            torch.tensor(train_x[train_index], dtype=torch.float32),
            torch.tensor(train_c[train_index], dtype=torch.float32),
            torch.tensor(train_y[train_index], dtype=torch.float32),
        )
        validation = TensorDataset(
            torch.tensor(train_x[validation_index], dtype=torch.float32),
            torch.tensor(train_c[validation_index], dtype=torch.float32),
            torch.tensor(train_y[validation_index], dtype=torch.float32),
        )
        test = TensorDataset(
            torch.tensor(test_x, dtype=torch.float32),
            torch.tensor(test_c, dtype=torch.float32),
            torch.tensor(test_y, dtype=torch.float32),
        )

        return (
            DataLoader(train, batch_size=self.batch_size, shuffle=True),
            DataLoader(validation, batch_size=self.batch_size, shuffle=False),
            DataLoader(test, batch_size=self.batch_size, shuffle=False),
            _rate_bounds(train_y[train_index]),
        )
