"""
Author: Jiajun Wu, HUST, China
Main Library: Pytorch
Description: It is a library with spiking neural network. We would like to implement this NN in hardware.
File Information: This file includes monitors, which are used to watch the performance of NN.
Log: 2020/1/20 Build firstly
Reference: Bindsnet library https://bindsnet-docs.readthedocs.io/

"""

import os
import torch
import numpy as np

from abc import ABC, abstractmethod
from typing import Union, Optional, Iterable, Dict

from ULIIC.network.neurons import Neurons
from ULIIC.network.synapese import AbstractConnection


class AbstractMonitor(ABC):
    # language=rst
    """
    Abstract base class for state variable monitors.
    """


class Monitor(AbstractMonitor):
    # language=rst
    """
    Records state variables of interest.
    """

    def __init__(
        self,
        obj: Union[Neurons, AbstractConnection],
        state_vars: Iterable[str],
        time: Optional[int] = None,
        batch_size: int = 1,
    ):
        # language=rst
        """
        Constructs a ``Monitor`` object.

        :param obj: An object to record state variables from during network simulation.
        :param state_vars: Iterable of strings indicating names of state variables to record.
        :param time: If not ``None``, pre-allocate memory for state variable recording.
        """
        super().__init__()

        self.obj = obj
        self.state_vars = state_vars
        self.time = time
        self.batch_size = batch_size

        # Deal with time later, the same underlying list is used
        self.recording = {v: [] for v in self.state_vars}

    def get(self, var: str) -> torch.Tensor:
        # language=rst
        """
        Return recording to user.

        :param var: State variable recording to return.
        :return: Tensor of shape ``[time, n_1, ..., n_k]``, where ``[n_1, ..., n_k]`` is the shape of the recorded
                 state variable.
        """
        return torch.cat(self.recording[var], 0)

    def record(self) -> None:
        # language=rst
        """
        Appends the current value of the recorded state variables to the recording.
        """
        for v in self.state_vars:
            data = getattr(self.obj, v).unsqueeze(0)
            self.recording[v].append(data.detach().clone())

        # remove the oldest element (first in the list)
        if self.time is not None:
            for v in self.state_vars:
                if len(self.recording[v]) > self.time:
                    self.recording[v].pop(0)

    def reset_(self) -> None:
        # language=rst
        """
        Resets recordings to empty ``torch.Tensor``s.
        """
        self.recording = {v: [] for v in self.state_vars}


class NetworkMonitor(AbstractMonitor):
    # language=rst
    """
    Record state variables of all layers and connections.
    """

    def __init__(
        self,
        network: "Network",
        layers: Optional[Iterable[str]] = None,
        connections: Optional[Iterable[str]] = None,
        state_vars: Optional[Iterable[str]] = None,
        time: Optional[int] = None,
    ):
        # language=rst
        """
        Constructs a ``NetworkMonitor`` object.

        :param network: Network to record state variables from.
        :param layers: Layers to record state variables from.
        :param connections: Connections to record state variables from.
        :param state_vars: List of strings indicating names of state variables to record.
        :param time: If not ``None``, pre-allocate memory for state variable recording.
        """
        super().__init__()

        self.network = network
        self.layers = layers if layers is not None else list(self.network.layers.keys())
        self.connections = (
            connections
            if connections is not None
            else list(self.network.connections.keys())
        )
        self.state_vars = state_vars if state_vars is not None else ("v", "s", "w")
        self.time = time

        if self.time is not None:
            self.i = 0

        # Initialize empty recording.
        self.recording = {k: {} for k in self.layers + self.connections}

        # If no simulation time is specified, specify 0-dimensional recordings.
        if self.time is None:
            for v in self.state_vars:
                for l in self.layers:
                    if hasattr(self.network.layers[l], v):
                        self.recording[l][v] = torch.Tensor()

                for c in self.connections:
                    if hasattr(self.network.connections[c], v):
                        self.recording[c][v] = torch.Tensor()

        # If simulation time is specified, pre-allocate recordings in memory for speed.
        else:
            for v in self.state_vars:
                for l in self.layers:
                    if hasattr(self.network.layers[l], v):
                        self.recording[l][v] = torch.zeros(
                            self.time, *getattr(self.network.layers[l], v).size()
                        )

                for c in self.connections:
                    if hasattr(self.network.connections[c], v):
                        self.recording[c][v] = torch.zeros(
                            self.time, *getattr(self.network.connections[c], v).size()
                        )

    def get(self) -> Dict[str, Dict[str, Union[Neurons, AbstractConnection]]]:
        # language=rst
        """
        Return entire recording to user.

        :return: Dictionary of dictionary of all layers' and connections' recorded state variables.
        """
        return self.recording

    def record(self) -> None:
        # language=rst
        """
        Appends the current value of the recorded state variables to the recording.
        """
        if self.time is None:
            for v in self.state_vars:
                for l in self.layers:
                    if hasattr(self.network.layers[l], v):
                        data = getattr(self.network.layers[l], v).unsqueeze(0).float()
                        self.recording[l][v] = torch.cat(
                            (self.recording[l][v], data), 0
                        )

                for c in self.connections:
                    if hasattr(self.network.connections[c], v):
                        data = getattr(self.network.connections[c], v).unsqueeze(0)
                        self.recording[c][v] = torch.cat(
                            (self.recording[c][v], data), 0
                        )

        else:
            for v in self.state_vars:
                for l in self.layers:
                    if hasattr(self.network.layers[l], v):
                        data = getattr(self.network.layers[l], v).float().unsqueeze(0)
                        self.recording[l][v] = torch.cat(
                            (self.recording[l][v][1:].type(data.type()), data), 0
                        )

                for c in self.connections:
                    if hasattr(self.network.connections[c], v):
                        data = getattr(self.network.connections[c], v).unsqueeze(0)
                        self.recording[c][v] = torch.cat(
                            (self.recording[c][v][1:].type(data.type()), data), 0
                        )

            self.i += 1

    def save(self, path: str, fmt: str = "npz") -> None:
        # language=rst
        """
        Write the recording dictionary out to file.

        :param path: The directory to which to write the monitor's recording.
        :param fmt: Type of file to write to disk. One of ``"pickle"`` or ``"npz"``.
        """
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        if fmt == "npz":
            # Build a list of arrays to write to disk.
            arrays = {}
            for o in self.recording:
                if type(o) == tuple:
                    arrays.update(
                        {
                            "_".join(["-".join(o), v]): self.recording[o][v]
                            for v in self.recording[o]
                        }
                    )
                elif type(o) == str:
                    arrays.update(
                        {
                            "_".join([o, v]): self.recording[o][v]
                            for v in self.recording[o]
                        }
                    )

            np.savez_compressed(path, **arrays)

        elif fmt == "pickle":
            with open(path, "wb") as f:
                torch.save(self.recording, f)

    def reset_(self) -> None:
        # language=rst
        """
        Resets recordings to empty ``torch.Tensors``.
        """
        # Reset to empty recordings
        self.recording = {k: {} for k in self.layers + self.connections}

        if self.time is not None:
            self.i = 0

        # If no simulation time is specified, specify 0-dimensional recordings.
        if self.time is None:
            for v in self.state_vars:
                for l in self.layers:
                    if hasattr(self.network.layers[l], v):
                        self.recording[l][v] = torch.Tensor()

                for c in self.connections:
                    if hasattr(self.network.connections[c], v):
                        self.recording[c][v] = torch.Tensor()

        # If simulation time is specified, pre-allocate recordings in memory for speed.
        else:
            for v in self.state_vars:
                for l in self.layers:
                    if hasattr(self.network.layers[l], v):
                        self.recording[l][v] = torch.zeros(
                            self.time, *getattr(self.network.layers[l], v).size()
                        )

                for c in self.connections:
                    if hasattr(self.network.connections[c], v):
                        self.recording[c][v] = torch.zeros(
                            self.time, *getattr(self.network.layers[c], v).size()
                        )
