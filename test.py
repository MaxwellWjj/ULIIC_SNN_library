"""
Author: Jiajun Wu, HUST, China
Main Library: Pytorch
Description: It is a library with spiking neural network. We would like to implement this NN in hardware.
File Information: This file includes a test for monitors.
Log: 2020/1/20 Build firstly
Reference: Bindsnet library https://bindsnet-docs.readthedocs.io/

"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from ULIIC.network.networks import Network
from ULIIC.network.neurons import Input, SRM0Neurons, LIFNeurons
from ULIIC.network.synapese import Connection

from ULIIC.network.monitors import Monitor
from ULIIC.analysis.value_error import spike_timing_error
from ULIIC.analysis.value_error import nrmsd_error
from ULIIC.analysis.plotting import plot_spikes, plot_voltages

# Simulation time and params.
time = 250
ineurons = 100
oneurons = 1

# Create the network.
network = Network()
network_with_cordic = Network()  # with cordic

# Create and add input, output layers.
source_layer = Input(n=ineurons)

source_layer_with_cordic = Input(n=ineurons)

target_layer = LIFNeurons(n=oneurons, neederror=False)

target_layer_with_cordic = LIFNeurons(n=oneurons, neederror=True)  # with cordic

network.add_layer(
    layer=source_layer, name="accuracy input"
)
network.add_layer(
    layer=target_layer, name="accuracy output"
)

network_with_cordic.add_layer(
    layer=source_layer_with_cordic, name="cordic input"
)

network_with_cordic.add_layer(
    layer=target_layer_with_cordic, name="cordic output"
)

# Create connection between input and output layers.
w_test = 0.05 + 0.1 * torch.randn(source_layer.n, target_layer.n)

forward_connection = Connection(
    source=source_layer,
    target=target_layer,
    w=w_test,  # Normal(0.05, 0.01) weights.
)

forward_connection_with_cordic = Connection(
    source=source_layer_with_cordic,
    target=target_layer_with_cordic,
    w=w_test,  # Normal(0.05, 0.01) weights.
)

network.add_connection(
    connection=forward_connection, source="accuracy input", target="accuracy output"
)

network_with_cordic.add_connection(
    connection=forward_connection_with_cordic, source="cordic input", target="cordic output"
)

# Create and add input and output layer monitors.
source_monitor = Monitor(
    obj=source_layer,
    state_vars=("s",),  # Record spikes and voltages.
    time=time,  # Length of simulation (if known ahead of time).
)
target_monitor = Monitor(
    obj=target_layer,
    state_vars=("s", "v"),  # Record spikes and voltages.
    time=time,  # Length of simulation (if known ahead of time).
)

source_monitor_with_cordic = Monitor(
    obj=source_layer_with_cordic,
    state_vars=("s",),
    time=time,
)

target_monitor_with_cordic = Monitor(
    obj=target_layer_with_cordic,
    state_vars=("s", "v"),
    time=time,
)

network.add_monitor(monitor=source_monitor, name="accuracy monitor input")
network.add_monitor(monitor=target_monitor, name="accuracy monitor output")

network_with_cordic.add_monitor(monitor=source_monitor_with_cordic, name="cordic monitor input")
network_with_cordic.add_monitor(monitor=target_monitor_with_cordic, name="cordic monitor output")

# Create input spike data, where each spike is distributed according to Bernoulli(0.1).
input_data = torch.bernoulli(0.1 * torch.ones(time, source_layer.n)).byte()
inputs = {"accuracy input": input_data}

inputs_with_cordic = {"cordic input": input_data}

# Simulate network on input data.
network.run(inputs=inputs, time=time)

network_with_cordic.run(inputs=inputs_with_cordic, time=time)

# Retrieve and plot simulation spike, voltage data from monitors.
spikes = {
    "input with accuracy": source_monitor.get("s"), "accuracy": target_monitor.get("s"),
    "cordic": target_monitor_with_cordic.get("s")
}
voltages = {"accuracy": target_monitor.get("v"), "cordic": target_monitor_with_cordic.get("v"),
            "differences": target_monitor.get("v") - target_monitor_with_cordic.get("v")}

ster = np.mean(spike_timing_error(target_monitor.get("s"), target_monitor_with_cordic.get("s"), time))
print("The Spike Timing Error(STER) is")
print(ster)

nrmsd = np.mean(nrmsd_error(target_monitor.get("v"), target_monitor_with_cordic.get("v"), time))
print("The NRMSD Error is")
print(nrmsd)

plt.ioff()
plot_spikes(spikes)
plot_voltages(voltages, plot_type="line")
plt.show()

