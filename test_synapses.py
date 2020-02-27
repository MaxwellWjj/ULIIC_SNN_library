"""
Author: Jiajun Wu, HUST, China
Main Library: Pytorch
Description: It is a library with spiking neural network. We would like to implement this NN in hardware.
File Information: This file includes a test for synapses weights updating.
Log: 2020/1/20 Build firstly
Reference: Bindsnet library https://bindsnet-docs.readthedocs.io/

"""

import torch
import matplotlib.pyplot as plt
import argparse
import numpy as np

from ULIIC.architectures.models import TwoLayerNetwork
from ULIIC.network.monitors import Monitor
from ULIIC.analysis.plotting import plot_weights, plot_voltages
from ULIIC.auxiliary.snn_utils import get_square_weights, get_square_assignments
from ULIIC.analysis.value_error import nrmsd_error, spike_timing_error, weights_error


# Simulation time and epoch.
time = 100
epoch = 10

# Create the network.
n_neurons = 100
network_a = TwoLayerNetwork(
    n_input=784,
    n_neurons=n_neurons,
    neederror=False,
)

network_b = TwoLayerNetwork(
    n_input=784,
    n_neurons=n_neurons,
    neederror=True,
)

n_sqrt = int(np.ceil(np.sqrt(n_neurons)))

# Create the monitor.
out_monitor_a = Monitor(
    network_a.layers["Y"],
    ["v"],
    time=time
)

out_monitor_b = Monitor(
    network_b.layers["Y"],
    ["v"],
    time=time
)

network_a.add_monitor(monitor=out_monitor_a, name="monitor A")
network_b.add_monitor(monitor=out_monitor_b, name="monitor B")

# Create input spike data, where each spike is distributed according to Bernoulli(0.1).
input_data = torch.bernoulli(0.1 * torch.ones(time, network_a.layers["X"].n)).byte()
inputs = {"X": input_data}
inputs_b = {"X": input_data}

voltage_axes, voltage_ims = None, None

# weights_im = None

# Training
for i in range(epoch):

    print("epoch-", i)
    # plt.ioff()
    network_a.run(inputs=inputs, time=time)
    weights = network_a.connections[("X", "Y")].w

    network_b.run(inputs=inputs_b, time=time)
    weights_b = network_b.connections[("X", "Y")].w

    voltage = {"voltage_a": out_monitor_a.get("v"), "voltage_b": out_monitor_b.get("v"),
               "difference": out_monitor_a.get("v") - out_monitor_b.get("v")}

    print("weights error = ", weights_error(weights, weights_b))
    # square_weights = get_square_weights(
    #     weights.view(784, 100), n_sqrt, 28
    # )
    # weights_im = plot_weights(square_weights, im=weights_im)
    # voltage_ims, voltage_axes = plot_voltages(
    #     voltage, plot_type="line"
    # )  # , ims=voltage_ims, axes=voltage_axes,

    # plt.pause(1e-8)
    # plt.show()

    network_a.reset_()  # Reset state variables.
