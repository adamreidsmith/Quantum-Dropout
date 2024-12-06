'''
Implementation of a quantum dropout layer. Updated for qiskit 1.0.
'''

from collections import defaultdict
import random
from hashlib import sha256
from math import prod
from pathlib import Path
from typing import Optional
from warnings import warn

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit import ParameterVector
from qiskit_ibm_runtime import IBMBackend

import torch
from torch import nn

import networkx as nx
import dill


SEED = 1234567890


class QuantumDropout(nn.Module):
    '''Quantum dropout layer.'''

    def __init__(
        self,
        p: float,
        input_shape: tuple[int, ...],
        img_shape: tuple[int, ...],
        observables: list[str],
        qubits_to_measure: list[int] = [1, 3, 7, 8, 12, 14, 18, 19, 23, 25],
        shots: int = 100_000,
        backend: Optional[AerSimulator | IBMBackend] = None,
        seed: Optional[int] = None,
        max_threads: Optional[int] = None,
        max_experiments: int = 100,
    ) -> None:
        '''
        Parameters
        ----------
        p : float
            Threshold for setting neurons to zero.
        input_shape : tuple[int]
            The shape of the input tensor to the dropout layer including batch size.
        img_shape : tuple[int]
            The shape of the images input into the neural network including batch size.
        observables : list[str]
            A list of observables. Each element should be a string composed of characters 'I',
            'X', 'Y', and 'Z' of the same length as `qubits_to_measure` representing Pauli observables.
            The order is such that the ith operator in an observable corresponds to the ith qubit in
            `qubits_to_measure`.
        qubits_to_measure : list[int]
            The qubits which will be measured.
        shots : int, optional
            Number of shots per run of the quantum circuit. Default: 100_000
        backend_name : AerSimulator | IBMBackend, optional
            Name of the qiskit backend to use. If None, use a local AerSimulator. Default: None
        seed : int, optional
            Seed value for the qiskit simulator. If the backend is not a simulator, this is ignored.
            Default: None
        max_threads : int, optional
            Maximum number of CPU cores for parallelization. Only applies when the backend is a local
            simulator. If None, all available cores are used. Default: None
        max_experiments : int, optional
            Maximum number of experiments to send to the backend in one job. Default: 100
        '''

        super().__init__()

        self.p = p

        # Define the backend
        self.backend = backend
        if self.backend is None:
            self.backend = AerSimulator(method='automatic')
        self.is_local_backend = isinstance(self.backend, AerSimulator)
        if self.is_local_backend:
            if max_threads is not None:
                if max_threads > 0:
                    self.backend.set_options(max_parallel_threads=max_threads)
                else:
                    self.backend.set_options(
                        max_parallel_threads=0, max_parallel_experiments=0, statevector_parallel_threshold=6
                    )
            if seed is not None:
                self.backend.set_options(seed_simulator=seed)

        # Maximum number of circuits to send to the backend at once
        self.max_experiments = max_experiments
        if not self.is_local_backend:
            self.max_experiments = min(self.max_experiments, self.backend.configuration().max_experiments)

        # Ensure the number of shots is viable for the given backend
        if shots > (max_shots := self.backend.configuration().max_shots):
            raise ValueError(f'Max allowed shots is {max_shots} for the given backend')
        self.shots = shots

        # Define the coupling map for the given backend
        self.define_coupling_map()

        self.qubits_to_measure = qubits_to_measure
        self.n_measure = len(self.qubits_to_measure)

        # Check that the observables are valid
        assert all(
            map(self.is_valid_observable, observables)
        ), f'Invalid list of observables for {self.n_measure} measured qubits'
        self.observables = observables
        self.n_observables = len(self.observables)

        # self.batch_size = input_shape[0]
        self.n_inputs = prod(input_shape[1:])  # Number of neurons in the dropout layer
        self.n_pixels = prod(img_shape[1:])  # Number of pixels in the input image

        self.params = ParameterVector('inputs', self.n_pixels)
        self.build_quantum_circuits()

        # Cache to store distributions for reuse.
        # Maps a hash (int) to a distribution if n_observabels == 1, or a
        # list of distributions, one for each observable, if n_observables > 1.
        self._dist_cache_directory = Path('./caches/dist_caches/')

        self.p0 = defaultdict(list)  # Dict mapping class labels to list of proportion of neurons zeroed

    def define_coupling_map(self) -> None:
        '''Define the coupling map and the number of qubits associated to the backend.
        If the backend is a simulator, get the coupling map from FakeKolkata.
        '''

        backend = self.backend
        if backend.configuration().simulator:
            # If the backend is a simulator, get the coupling map from FakeKolkata
            from qiskit_ibm_runtime.fake_provider import FakeKolkataV2

            backend = FakeKolkataV2()

        # Get the coupling map
        self.coupling_map = backend.configuration().coupling_map

        # Reduce the coupling map to an undirected graph.
        # This is viable since all edges in the coupling maps for the IBM devices of interest are
        # bidirectional with the same CNOT error in both directions.
        undirected_coupling_map = []
        for edge in self.coupling_map:
            if edge[::-1] not in undirected_coupling_map:
                undirected_coupling_map.append(edge)
        self.coupling_map = undirected_coupling_map

        # Create an undirected graph of the coupling map with CNOT errors as edge weights
        coupling_map_graph = nx.Graph()
        for edge in self.coupling_map:
            coupling_map_graph.add_edge(*edge, weight=backend.properties().gate_error('cx', edge))

        # Compute the minimum spanning tree
        self.coupling_map = nx.minimum_spanning_tree(coupling_map_graph).edges

        # Reorder the coupling map so that the control qubit is the one with the lower index
        self.coupling_map = [(i, j) if i < j else (j, i) for i, j in self.coupling_map]

        self.n_qubits = backend.configuration().n_qubits

    def is_valid_observable(self, observable: str) -> bool:
        return len(observable) == self.n_measure and all(obs in 'IXYZ' for obs in observable)

    def build_quantum_circuits(self) -> None:
        '''Builds a list of quantum circuits with the following architecture:

        |0> -- √X -- Rz(θ_0) ------ √X -- Rz(θ_1) -------- √X -- ... -- √X -- Rz(θ_{m-1}) ---- √X --
        |0> -- √X -- Rz(θ_m) ------ √X -- Rz(θ_{m+1}) ---- √X -- ... -- √X -- Rz(θ_{2*m-1}) -- √X --
        ...
        |0> -- √X -- Rz(θ_{n-m}) -- √X -- Rz(θ_{n-m+1}) -- √X -- ... -- √X -- Rz(θ_{n-1}) ---- √X --

        where n is the number of inputs to the quantum circuit and m == n // q where q is the
        number of qubits in the circuit. CX gates are then applied according to the topology
        of the quantum computer. Finally, gates are applied corresponding to each observable.
        '''

        quantum_circuit = QuantumCircuit(self.n_qubits, self.n_measure)

        # Apply alternating √X and Rz gates to each qubit with parameters as input
        depth, remaining = divmod(self.n_pixels, self.n_qubits)
        param_index = 0
        for q in range(self.n_qubits):
            for _ in range(depth + (q < remaining)):
                quantum_circuit.sx(q)
                quantum_circuit.rz(self.params[param_index], q)
                param_index += 1
        for q in range(self.n_qubits):
            quantum_circuit.sx(q)

        # Apply CX gates between adjacent qubits
        for q1, q2 in self.coupling_map:
            quantum_circuit.cx(q1, q2)

        # Make a copy of the quantum circuit for each observable
        self.quantum_circuits = [quantum_circuit.copy() for _ in range(self.n_observables)]

        # Apply transformations on the qubits to be measured corresponding to the observables
        for circuit_index, observable in enumerate(self.observables):
            for op_index, obs in enumerate(observable):
                if obs == 'X':
                    self.quantum_circuits[circuit_index].h(self.qubits_to_measure[op_index])
                elif obs == 'Y':
                    self.quantum_circuits[circuit_index].sdg(self.qubits_to_measure[op_index])
                    self.quantum_circuits[circuit_index].h(self.qubits_to_measure[op_index])
                # No need to apply gates for 'Z' or 'I' Pauli operators

        # Measure the qubits specified by `qubits_to_measure`
        for quantum_circuit in self.quantum_circuits:
            for i, qubit_index in enumerate(self.qubits_to_measure):
                quantum_circuit.measure(qubit_index, i)

        self.quantum_circuits = transpile(self.quantum_circuits, self.backend)

    def _dist_cache(self, img_hash: int, dist: Optional[dict[int, float]] = None) -> dict[int, float] | None:
        path = self._dist_cache_directory / f'{img_hash}.pkl'
        if dist is not None:
            if path.exists():
                raise FileExistsError(f'Distribution {img_hash} is already cached')
            with open(path, 'wb') as f:
                dill.dump(dist, f)
            return
        if not path.exists():
            return
        with open(path, 'rb') as f:
            dist = dill.load(f)
        return dist

    def forward(self, t: torch.Tensor, x: torch.Tensor | None = None, y: torch.Tensor | None = None) -> torch.Tensor:
        '''Run a forward pass of the quantum dropout layer.

        Parameters
        ----------
        t : torch.Tensor
            The input to the dropout layer. len(observables) copies of this tensor will be returned
            with some neurons killed.
        x : torch.Tensor
            The batch of original images input to the NN that produce the tensor t
        y : torch.Tensor
            Class labels. If present, used to compute the proportion of elements set to zero for each class

        Returns
        -------
        torch.Tensor
            The output of the dropout layer. A tensor of shape (n_observables, *t.shape)
        '''

        if not self.training or self.p == 1:
            return t

        assert t.shape[0] == x.shape[0], 'Batch sizes of input and img do not match'

        if self.n_observables == 1:
            all_dists = []
            bound_circuits = []
            # Loop over each input image in the batch
            for flat_img in x.cpu().flatten(start_dim=1).detach().numpy():
                img_hash = sha256(flat_img.tobytes()).hexdigest()
                dist = self._dist_cache(img_hash)
                if dist is not None:
                    # Extract the normalized distributions from the cache
                    all_dists.append(dist)
                    continue
                # Bind the parameters to each quantum circuit
                bound_circuit_for_input = self.quantum_circuits[0].assign_parameters(flat_img)
                all_dists.append(img_hash)
                bound_circuits.append(bound_circuit_for_input)

            # Run all quantum circuits on the backend (there are batch_size * self.n_observables of them at most)
            normalized_dists = []
            for i in range(0, len(bound_circuits), self.max_experiments):
                counts = (
                    self.backend.run(bound_circuits[i : i + self.max_experiments], shots=self.shots)
                    .result()
                    .get_counts()
                )
                if not isinstance(counts, list):
                    counts = [counts]

                # Scale the distributions such that the max value is 1
                dist_maxs = (max(dist.values()) for dist in counts)
                normalized_dists.extend(
                    [{int(k, 2): v / mx for k, v in dist.items()} for dist, mx in zip(counts, dist_maxs)]
                )

            # Replace the instances of hashes in all_dists with the computed distributions
            for i in range(len(all_dists) - 1, -1, -1):
                if not isinstance(all_dists[i], dict):
                    img_hash = all_dists[i]
                    all_dists[i] = normalized_dists.pop()
                    self._dist_cache(img_hash, all_dists[i])
            assert len(normalized_dists) == 0, 'Dists remain!!'

        else:
            all_dists = []
            # Loop over each input image in the batch
            for flat_img in x.cpu().flatten(start_dim=1).numpy():
                dist = self._dist_cache(img_hash)
                if dist is not None:
                    # Extract the normalized distributions from the cache
                    all_dists.extend(dist)
                    continue

                # Bind the parameters to each quantum circuit
                param_dict = {param: val for param, val in zip(self.params, flat_img)}
                bound_circuits = [qc.assign_parameters(param_dict) for qc in self.quantum_circuits]

                # Run all quantum circuits on the backend (there are self.n_observables of them)
                normalized_dists = []
                for i in range(0, len(bound_circuits), self.max_experiments):
                    counts = (
                        self.backend.run(bound_circuits[i : i + self.max_experiments], shots=self.shots)
                        .result()
                        .get_counts()
                    )
                    if not isinstance(counts, list):
                        counts = [counts]

                    # Scale the distributions such that the max value is 1
                    dist_maxs = (max(dist.values()) for dist in counts)
                    normalized_dists.extend(
                        [{int(k, 2): v / mx for k, v in dist.items()} for dist, mx in zip(counts, dist_maxs)]
                    )

                # Add the normalized distributions to the cache
                self._dist_cache(img_hash, normalized_dists)
                all_dists.extend(normalized_dists)

        mask = torch.ones((self.n_observables, t.shape[0], self.n_inputs), dtype=torch.bool, device=t.device)
        for batch_index in range(t.shape[0]):
            # Get the distributions associated to the image specified by `batch_index`
            dists = all_dists[batch_index * self.n_observables : (batch_index + 1) * self.n_observables]

            # Select a random sample (with replacement) of measurement outcomes for each observable
            random_measurement_outcomes = (
                random.choices(range(2**self.n_measure), k=self.n_inputs) for _ in range(self.n_observables)
            )

            # Get the probabilities associated to each random measurement outcome
            random_probs = torch.Tensor(
                [[dist.get(b, 0) for b in rmo] for dist, rmo in zip(dists, random_measurement_outcomes)]
            )

            # Zero out the corresponding values in the mask
            for i in range(self.n_observables):
                mask[i, batch_index, random_probs[i] > self.p] = False

        # Repeat the input tensor `self.n_observables` times in a new dimension
        if self.n_observables > 1:
            t = t.unsqueeze(0).repeat(self.n_observables, *([1] * len(t.shape)))

        # Reshape the mask to match the input tensor
        if (shape := t.shape) != mask.shape:
            mask = mask.reshape(shape)

        # Scale t such that the expected value of t is unchanged
        r = mask.numel() / mask.sum().item()

        if y is not None:
            for batch_index in range(t.shape[0]):
                mask_element = mask[batch_index]
                p0 = (~mask_element).sum().item() / mask_element.numel()
                self.p0[y[batch_index].item()].append(p0)

        return (t * mask) * r


if __name__ == '__main__':
    # Test the layer
    torch.manual_seed(SEED)
    random.seed(SEED)

    bs = 2
    t = torch.rand((bs, 2048))
    img = torch.rand((bs, 3, 64, 64))
    qd = QuantumDropout(
        0.3,
        t.shape,
        img.shape,
        qubits_to_measure=[1, 7, 8, 12, 14, 18, 19, 25],
        observables=['Z' * 8],
        backend=AerSimulator(method='matrix_product_state'),
        seed=SEED,
        max_threads=0,
        shots=10_000,
    )
    result: torch.Tensor = qd(t, img)
    print(result.shape)
    print(f'Proportion of elements set to zero: {(result == 0).sum().item() / (prod(t.shape) * qd.n_observables)}')
