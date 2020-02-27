"""
Author: Jiajun Wu, HUST, China
Main Library: Pytorch
Description: It is a library with spiking neural network. We would like to implement this NN in hardware.
File Information: This file includes models in some papers.
Log: 2020/1/20 Build firstly
Reference: Bindsnet library https://bindsnet-docs.readthedocs.io/

"""

from typing import Optional, Union, Tuple, List, Sequence, Iterable

import numpy as np
import torch
from scipy.spatial.distance import euclidean
from torch.nn.modules.utils import _pair

from ULIIC.learning.learning_rules import PostPre, ExpWeightSTDP, MSTDPET
from ULIIC.network.networks import Network
from ULIIC.network.neurons import Input, RealInput, LIFNeurons, DiehlAndCookNeurons
from ULIIC.network.synapese import Connection, LocalConnection


class TwoLayerNetwork(Network):
    # language=rst
    """
    Implements an ``Input`` instance connected to a ``LIFNeurons`` instance with a fully-connected ``Connection``.
    """

    def __init__(
        self,
        n_input: int,
        n_neurons: int = 100,
        dt: float = 1.0,
        wmin: float = 0.0,
        wmax: float = 1.0,
        nu: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
        reduction: Optional[callable] = None,
        norm: float = 78.4,
        neederror: bool = True,
    ) -> None:
        # language=rst
        """
        Constructor for class ``TwoLayerNetwork``.

        :param n_input: Number of input neurons. Matches the 1D size of the input data.
        :param n_neurons: Number of neurons in the ``LIFNeurons`` population.
        :param dt: Simulation time step.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events, respectively.
        :param reduction: Method for reducing parameter updates along the minibatch dimension.
        :param wmin: Minimum allowed weight on ``Input`` to ``LIFNeurons`` synapses.
        :param wmax: Maximum allowed weight on ``Input`` to ``LIFNeurons`` synapses.
        :param norm: ``Input`` to ``LIFNeurons`` layer connection weights normalization constant.
        :param neederror: Whether need error.

        """
        super().__init__(dt=dt)

        self.n_input = n_input
        self.n_neurons = n_neurons
        self.dt = dt
        self.neederror = neederror

        self.add_layer(Input(n=self.n_input, traces=True, tc_trace=20.0), name="X")
        self.add_layer(
            LIFNeurons(
                n=self.n_neurons,
                neederror=self.neederror,
                traces=True,
                rest=-65.0,
                reset=-65.0,
                thresh=-52.0,
                refrac=5,
                tc_decay=100.0,
                tc_trace=20.0,
            ),
            name="Y",
        )

        w = 0.5 * torch.ones(self.n_input, self.n_neurons)  # w = 0.3 * torch.rand(self.n_input, self.n_neurons)

        self.add_connection(
            Connection(
                source=self.layers["X"],
                target=self.layers["Y"],
                neederror=self.neederror,
                w=w,
                update_rule=ExpWeightSTDP,
                nu=nu,
                reduction=reduction,
                wmin=wmin,
                wmax=wmax,
                norm=norm,
            ),
            source="X",
            target="Y",
        )


class DiehlAndCook2015(Network):
    # language=rst
    """
    Implements the spiking neural network architecture from `(Diehl & Cook 2015)
    <https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full>`_.
    """

    def __init__(
        self,
        n_input: int,
        n_neurons: int = 100,
        exc: float = 22.5,
        inh: float = 17.5,
        dt: float = 1.0,
        nu: Optional[Union[float, Sequence[float]]] = (1e-2, 1e-2),
        reduction: Optional[callable] = None,
        wmin: float = 0.0,
        wmax: float = 1.0,
        norm: float = 78.4,
        theta_plus: float = 0.05,
        tc_theta_decay: float = 1e7,
        input_shape: Optional[Iterable[int]] = None,
        neederror: bool = True,
    ) -> None:
        # language=rst
        """
        Constructor for class ``DiehlAndCook2015``.

        :param n_input: Number of input neurons. Matches the 1D size of the input data.
        :param n_neurons: Number of excitatory, inhibitory neurons.
        :param exc: Strength of synapse weights from excitatory to inhibitory layer.
        :param inh: Strength of synapse weights from inhibitory to excitatory layer.
        :param dt: Simulation time step.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events, respectively.
        :param reduction: Method for reducing parameter updates along the minibatch dimension.
        :param wmin: Minimum allowed weight on input to excitatory synapses.
        :param wmax: Maximum allowed weight on input to excitatory synapses.
        :param norm: Input to excitatory layer connection weights normalization constant.
        :param theta_plus: On-spike increment of ``DiehlAndCookNeurons`` membrane threshold potential.
        :param tc_theta_decay: Time constant of ``DiehlAndCookNeurons`` threshold potential decay.
        :param input_shape: The dimensionality of the input layer.
        """
        super().__init__(dt=dt)

        self.n_input = n_input
        self.input_shape = input_shape
        self.n_neurons = n_neurons
        self.exc = exc
        self.inh = inh
        self.dt = dt
        self.neederror = neederror

        # Layers
        input_layer = Input(
            n=self.n_input, shape=self.input_shape, traces=True, tc_trace=20.0
        )
        exc_layer = DiehlAndCookNeurons(
            n=self.n_neurons,
            neederror=False,
            traces=True,
            traces_additive=True,
            rest=0.0,  # -65.0
            reset=5.0,  # -60.0
            thresh=13.0,  # -52.0
            refrac=5,
            tc_decay=100.0,
            tc_trace=20.0,
            theta_plus=theta_plus,
            tc_theta_decay=tc_theta_decay,
        )
        inh_layer = LIFNeurons(
            n=self.n_neurons,
            neederror=self.neederror,
            traces=True,
            traces_additive=True,
            rest=0.0,  # -60.0
            reset=15.0,  # -45.0
            thresh=20.0,  # -40.0
            tc_decay=10.0,
            refrac=2,
            tc_trace=20.0,
        )

        # Connections
        w = 0.3 * torch.rand(self.n_input, self.n_neurons)
        input_exc_conn = Connection(
            source=input_layer,
            target=exc_layer,
            neederror=self.neederror,
            w=w,
            update_rule=ExpWeightSTDP,
            nu=nu,
            reduction=reduction,
            wmin=wmin,
            wmax=wmax,
            norm=norm,
        )
        w = self.exc * torch.diag(torch.ones(self.n_neurons))
        exc_inh_conn = Connection(
            source=exc_layer, target=inh_layer, w=w, wmin=0, wmax=self.exc
        )
        w = -self.inh * (
            torch.ones(self.n_neurons, self.n_neurons)
            - torch.diag(torch.ones(self.n_neurons))
        )
        inh_exc_conn = Connection(
            source=inh_layer, target=exc_layer, w=w, wmin=-self.inh, wmax=0
        )

        # Add to network
        self.add_layer(input_layer, name="X")
        self.add_layer(exc_layer, name="Ae")
        self.add_layer(inh_layer, name="Ai")
        self.add_connection(input_exc_conn, source="X", target="Ae")
        self.add_connection(exc_inh_conn, source="Ae", target="Ai")
        self.add_connection(inh_exc_conn, source="Ai", target="Ae")


class DiehlAndCook2015v2(Network):
    # language=rst
    """
    Slightly modifies the spiking neural network architecture from `(Diehl & Cook 2015)
    <https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full>`_ by removing the inhibitory layer and
    replacing it with a recurrent inhibitory connection in the output layer (what used to be the excitatory layer).
    """

    def __init__(
        self,
        n_input: int,
        n_neurons: int = 100,
        inh: float = 17.5,
        dt: float = 1.0,
        nu: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
        reduction: Optional[callable] = None,
        wmin: Optional[float] = 0.0,
        wmax: Optional[float] = 1.0,
        norm: float = 78.4,
        theta_plus: float = 0.05,
        tc_theta_decay: float = 1e7,
        input_shape: Optional[Iterable[int]] = None,
    ) -> None:
        # language=rst
        """
        Constructor for class ``DiehlAndCook2015v2``.

        :param n_input: Number of input neurons. Matches the 1D size of the input data.
        :param n_neurons: Number of excitatory, inhibitory neurons.
        :param inh: Strength of synapse weights from inhibitory to excitatory layer.
        :param dt: Simulation time step.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events, respectively.
        :param reduction: Method for reducing parameter updates along the minibatch dimension.
        :param wmin: Minimum allowed weight on input to excitatory synapses.
        :param wmax: Maximum allowed weight on input to excitatory synapses.
        :param norm: Input to excitatory layer connection weights normalization constant.
        :param theta_plus: On-spike increment of ``DiehlAndCookNeurons`` membrane threshold potential.
        :param tc_theta_decay: Time constant of ``DiehlAndCookNeurons`` threshold potential decay.
        :param input_shape: The dimensionality of the input layer.
        """
        super().__init__(dt=dt)

        self.n_input = n_input
        self.input_shape = input_shape
        self.n_neurons = n_neurons
        self.inh = inh
        self.dt = dt

        input_layer = Input(
            n=self.n_input, shape=self.input_shape, traces=True, tc_trace=20.0
        )
        self.add_layer(input_layer, name="X")

        output_layer = DiehlAndCookNeurons(
            n=self.n_neurons,
            traces=True,
            rest=-65.0,
            reset=-60.0,
            thresh=-52.0,
            refrac=5,
            tc_decay=100.0,
            tc_trace=20.0,
            theta_plus=theta_plus,
            tc_theta_decay=tc_theta_decay,
        )
        self.add_layer(output_layer, name="Y")

        w = 0.3 * torch.rand(self.n_input, self.n_neurons)
        input_connection = Connection(
            source=self.layers["X"],
            target=self.layers["Y"],
            w=w,
            update_rule=PostPre,
            nu=nu,
            reduction=reduction,
            wmin=wmin,
            wmax=wmax,
            norm=norm,
        )
        self.add_connection(input_connection, source="X", target="Y")

        w = -self.inh * (
            torch.ones(self.n_neurons, self.n_neurons)
            - torch.diag(torch.ones(self.n_neurons))
        )
        recurrent_connection = Connection(
            source=self.layers["Y"],
            target=self.layers["Y"],
            w=w,
            wmin=-self.inh,
            wmax=0,
        )
        self.add_connection(recurrent_connection, source="Y", target="Y")


class IncreasingInhibitionNetwork(Network):
    # language=rst
    """
    Implements the inhibitory layer structure of the spiking neural network architecture from `(Hazan et al. 2018)
    <https://arxiv.org/abs/1807.09374>`_
    """

    def __init__(
        self,
        n_input: int,
        n_neurons: int = 100,
        start_inhib: float = 1.0,
        max_inhib: float = 100.0,
        dt: float = 1.0,
        nu: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
        reduction: Optional[callable] = None,
        wmin: float = 0.0,
        wmax: float = 1.0,
        norm: float = 78.4,
        theta_plus: float = 0.05,
        tc_theta_decay: float = 1e7,
    ) -> None:
        # language=rst
        """
        Constructor for class ``IncreasingInhibitionNetwork``.

        :param n_input: Number of input neurons. Matches the 1D size of the input data.
        :param n_neurons: Number of excitatory, inhibitory neurons.
        :param inh: Strength of synapse weights from inhibitory to excitatory layer.
        :param dt: Simulation time step.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events, respectively.
        :param reduction: Method for reducing parameter updates along the minibatch dimension.
        :param wmin: Minimum allowed weight on input to excitatory synapses.
        :param wmax: Maximum allowed weight on input to excitatory synapses.
        :param norm: Input to excitatory layer connection weights normalization constant.
        :param theta_plus: On-spike increment of ``DiehlAndCookNeurons`` membrane threshold potential.
        :param tc_theta_decay: Time constant of ``DiehlAndCookNeurons`` threshold potential decay.
        """
        super().__init__(dt=dt)

        self.n_input = n_input
        self.n_neurons = n_neurons
        self.n_sqrt = int(np.sqrt(n_neurons))
        self.start_inhib = start_inhib
        self.max_inhib = max_inhib
        self.dt = dt

        input_layer = Input(n=self.n_input, traces=True, tc_trace=20.0)
        self.add_layer(input_layer, name="X")

        output_layer = DiehlAndCookNeurons(
            n=self.n_neurons,
            traces=True,
            rest=-65.0,
            reset=-60.0,
            thresh=-52.0,
            refrac=5,
            tc_decay=100.0,
            tc_trace=20.0,
            theta_plus=theta_plus,
            tc_theta_decay=tc_theta_decay,
        )
        self.add_layer(output_layer, name="Y")

        w = 0.3 * torch.rand(self.n_input, self.n_neurons)
        input_output_conn = Connection(
            source=self.layers["X"],
            target=self.layers["Y"],
            w=w,
            update_rule=PostPre,
            nu=nu,
            reduction=reduction,
            wmin=wmin,
            wmax=wmax,
            norm=norm,
        )
        self.add_connection(input_output_conn, source="X", target="Y")

        w = torch.zeros(self.n_neurons, self.n_neurons)
        for i in range(self.n_neurons):
            for j in range(self.n_neurons):
                if i != j:
                    x1, y1 = i // self.n_sqrt, i % self.n_sqrt
                    x2, y2 = j // self.n_sqrt, j % self.n_sqrt

                    inhib = self.start_inhib * np.sqrt(euclidean([x1, y1], [x2, y2]))
                    w[i, j] = -min(self.max_inhib, inhib)

        recurrent_output_conn = Connection(
            source=self.layers["Y"],
            target=self.layers["Y"],
            w=w,
            wmin=-self.max_inhib,
            wmax=0,
        )
        self.add_connection(recurrent_output_conn, source="Y", target="Y")


class LocallyConnectedNetwork(Network):
    # language=rst
    """
    Defines a two-layer network in which the input layer is "locally connected" to the output layer, and the output
    layer is recurrently inhibited connected such that neurons with the same input receptive field inhibit each other.
    """

    def __init__(
        self,
        n_input: int,
        input_shape: List[int],
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]],
        n_filters: int,
        inh: float = 25.0,
        dt: float = 1.0,
        nu: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
        reduction: Optional[callable] = None,
        theta_plus: float = 0.05,
        tc_theta_decay: float = 1e7,
        wmin: float = 0.0,
        wmax: float = 1.0,
        norm: Optional[float] = 0.2,
        real=False,
    ) -> None:
        # language=rst
        """
        Constructor for class ``LocallyConnectedNetwork``. Uses ``DiehlAndCookNeurons`` to avoid multiple spikes per
        timestep in the output layer population.

        :param n_input: Number of input neurons. Matches the 1D size of the input data.
        :param input_shape: Two-dimensional shape of input population.
        :param kernel_size: Size of input windows. Integer or two-tuple of integers.
        :param stride: Length of horizontal, vertical stride across input space. Integer or two-tuple of integers.
        :param n_filters: Number of locally connected filters per input region. Integer or two-tuple of integers.
        :param inh: Strength of synapse weights from output layer back onto itself.
        :param dt: Simulation time step.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events, respectively.
        :param reduction: Method for reducing parameter updates along the minibatch dimension.
        :param wmin: Minimum allowed weight on ``Input`` to ``DiehlAndCookNeurons`` synapses.
        :param wmax: Maximum allowed weight on ``Input`` to ``DiehlAndCookNeurons`` synapses.
        :param theta_plus: On-spike increment of ``DiehlAndCookNeurons`` membrane threshold potential.
        :param tc_theta_decay: Time constant of ``DiehlAndCookNeurons`` threshold potential decay.
        :param norm: ``Input`` to ``DiehlAndCookNeurons`` layer connection weights normalization constant.
        :param real: Whether to use real-valued (non-spiking) input (implemented as a "clamp").
        """
        super().__init__(dt=dt)

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)

        self.n_input = n_input
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_filters = n_filters
        self.inh = inh
        self.dt = dt
        self.theta_plus = theta_plus
        self.tc_theta_decay = tc_theta_decay
        self.wmin = wmin
        self.wmax = wmax
        self.norm = norm

        if kernel_size == input_shape:
            conv_size = [1, 1]
        else:
            conv_size = (
                int((input_shape[0] - kernel_size[0]) / stride[0]) + 1,
                int((input_shape[1] - kernel_size[1]) / stride[1]) + 1,
            )

        if real:
            input_layer = RealInput(n=self.n_input, traces=True, tc_trace=20.0)
        else:
            input_layer = Input(n=self.n_input, traces=True, tc_trace=20.0)

        output_layer = DiehlAndCookNeurons(
            n=self.n_filters * conv_size[0] * conv_size[1],
            traces=True,
            rest=-65.0,
            reset=-60.0,
            thresh=-52.0,
            refrac=5,
            tc_decay=100.0,
            tc_trace=20.0,
            theta_plus=theta_plus,
            tc_theta_decay=tc_theta_decay,
        )
        input_output_conn = LocalConnection(
            input_layer,
            output_layer,
            kernel_size=kernel_size,
            stride=stride,
            n_filters=n_filters,
            nu=nu,
            reduction=reduction,
            update_rule=PostPre,
            wmin=wmin,
            wmax=wmax,
            norm=norm,
            input_shape=input_shape,
        )

        w = torch.zeros(n_filters, *conv_size, n_filters, *conv_size)
        for fltr1 in range(n_filters):
            for fltr2 in range(n_filters):
                if fltr1 != fltr2:
                    for i in range(conv_size[0]):
                        for j in range(conv_size[1]):
                            w[fltr1, i, j, fltr2, i, j] = -inh

        w = w.view(
            n_filters * conv_size[0] * conv_size[1],
            n_filters * conv_size[0] * conv_size[1],
        )
        recurrent_conn = Connection(output_layer, output_layer, w=w)

        self.add_layer(input_layer, name="X")
        self.add_layer(output_layer, name="Y")
        self.add_connection(input_output_conn, source="X", target="Y")
        self.add_connection(recurrent_conn, source="Y", target="Y")