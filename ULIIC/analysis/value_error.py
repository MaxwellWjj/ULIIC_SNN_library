"""
Author: Jiajun Wu, HUST, China
Main Library: Pytorch
Description: It is a library with spiking neural network. We would like to implement this NN in hardware.
File Information: This file includes value analysis.
Log: 2020/1/20 Build firstly
Reference: Bindsnet library https://bindsnet-docs.readthedocs.io/

"""

import torch
import numpy as np


def spike_timing_error(
    spikes: torch.Tensor,
    spikes_b: torch.Tensor,
    time: int = 500,
):
    """
    Calculate spikes error for any group(s) of neurons.

    :param spikes: Mapping from layer names to spiking data. Spike data has shape ``[time, n_1, ..., n_k]``, where
                   ``[n_1, ..., n_k]`` is the shape of the recorded layer.
    :param spikes_b: Mapping from layer names to spiking data. Spike data has shape ``[time, n_1, ..., n_k]``, where
                   ``[n_1, ..., n_k]`` is the shape of the recorded layer.
    :param time: Total time for a encoding.

    """

    if spikes.shape != spikes_b.shape:
        print("spikes shapes don't match!")
        print(spikes.shape, spikes_b.shape)
        return 0.0

    tensor_shape = spikes.shape
    spike_time = np.zeros(tensor_shape[2], dtype=int)
    spike_b_time = np.zeros(spikes_b.shape[2], dtype=int)

    spike_trace = []
    spike_b_trace = []
    for n in range(tensor_shape[2]):
        spike_trace.append([])
        spike_b_trace.append([])

    error_sum = np.zeros(tensor_shape[2], dtype=float)
    ster_result = np.zeros(tensor_shape[2], dtype=float)

    for n in range(tensor_shape[2]):
        for t in range(tensor_shape[0]):
            if spikes[t][0][n] == 1:
                spike_time[n] += 1
                spike_trace[n].append(t)
            if spikes_b[t][0][n] == 1:
                spike_b_time[n] += 1
                spike_b_trace[n].append(t)
        for k in range(min(spike_time[n], spike_b_time[n])):
            error_sum[n] += spike_b_trace[n][k] - spike_trace[n][k]
        ster_result[n] = error_sum[n] / min(spike_time[n], spike_b_time[n])

    return ster_result / time


def nrmsd_error(
    voltages: torch.Tensor,
    voltages_b: torch.Tensor,
    time: int = 500,
):
    """
    Calculate spikes error for any group(s) of neurons.

    :param voltages: Mapping from layer names to spiking data. Spike data has shape ``[time, n_1, ..., n_k]``, where
                   ``[n_1, ..., n_k]`` is the shape of the recorded layer.
    :param voltages_b: Mapping from layer names to spiking data. Spike data has shape ``[time, n_1, ..., n_k]``, where
                   ``[n_1, ..., n_k]`` is the shape of the recorded layer.
    :param time: Total time for a encoding.

    """
    if voltages.shape != voltages_b.shape:
        print("spikes shapes don't match!")
        print(voltages.shape, voltages_b.shape)
        return 0.0

    error_sum = np.zeros(voltages.shape[2], dtype=float)
    nrmsd_result = np.zeros(voltages.shape[2], dtype=float)

    for n in range(voltages.shape[2]):
        for t in range(voltages.shape[0]):
            minv = 0.0
            maxv = 0.0
            error_sum[n] += (voltages[t][0][n]-voltages_b[t][0][n]) ** 2
            if voltages[t][0][n] > maxv:
                maxv = voltages[t][0][n]
            if voltages[t][0][n] < minv:
                minv = voltages[t][0][n]
        nrmsd_result[n] = (error_sum[n] / time) ** 0.5

    return nrmsd_result


def weights_error(
    weights: torch.Tensor,
    weights_b: torch.Tensor,
):
    """
    Calculate weights error for any group(s) of neurons connections.

    :param weights: Mapping from layer names to spiking data. Spike data has shape ``[time, n_1, ..., n_k]``, where
                   ``[n_1, ..., n_k]`` is the shape of the recorded layer.
    :param weights_b: Mapping from layer names to spiking data. Spike data has shape ``[time, n_1, ..., n_k]``, where
                   ``[n_1, ..., n_k]`` is the shape of the recorded layer.

    """

    if weights.shape != weights_b.shape:
        print("weights shapes don't match!")
        print(weights.shape, weights_b.shape)
        return 0.0

    error_sum = 0.0

    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            error_sum += (weights[i][j] - weights_b[i][j]) ** 2

    weights_error_result = (error_sum / (weights.shape[0] * weights.shape[1])) ** 0.5
    return weights_error_result
