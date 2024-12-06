'''
This file analyzes the Jensen-Shannon divergences and Chebyshev distances of distributions 
of measurement outcomes produced by the quantum dropout layer on the EuroSAT data.
'''

import math
import os
import pickle
import shutil
import random
import time
from itertools import product
from functools import partial
from multiprocessing import Pool

import torch

# from qiskit import execute
from qiskit_aer import AerSimulator
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

from dropout import QuantumDropout
from lulc_classification_pytorch import load_data as load_eurosat_data


SEED = 1234567898
N_CLASSES = 10  # Number of classes to use. Should be an integer in [2, 10]
SHOTS = 100_000
SAMPLES_PER_CLASS = 1000  # Number of images from each class to use
OBSERVABLES = ['Z' * 8]
QUBITS_TO_MEASURE = [1, 7, 8, 12, 14, 18, 19, 25]
# BACKEND = 'aer_simulator'

# Maximum number of CPU cores to use for simulation and divergence computation
# Only applies to simulation when the backend is a local simulator
MAX_THREADS = 8

CLASS_PAIRS = [(c1, c2) for c1, c2 in product(range(N_CLASSES), range(N_CLASSES)) if c1 <= c2]


def load_data(
    shuffle: bool = False, seed: int = None, return_labels: bool = False
) -> tuple[dict[int, list[torch.Tensor]]]:
    '''Load the EuroSAT dataset and separate images by class. Keep only SAMPLES_PER_CLASS images per class.'''

    data = load_eurosat_data(seed=0 if seed is None else seed, return_labels=return_labels)
    if return_labels:
        train_loader, test_loader, class_labels = data
        class_labels = {v: k for k, v in class_labels.items()}
    else:
        train_loader, test_loader = data

    train_data, test_data = {i: [] for i in range(N_CLASSES)}, {i: [] for i in range(N_CLASSES)}

    # Split the data by class
    for j, dataloader in enumerate((train_loader, test_loader)):
        dataset = (train_data, test_data)[j]
        for imgs, labels in dataloader:
            for i in range(len(imgs)):
                label = labels[i].item()
                if label < N_CLASSES and len(dataset[label]) < SAMPLES_PER_CLASS:
                    dataset[label].append(imgs[i])

    if shuffle:
        if seed is not None:
            random.seed(seed)

        for key in train_data.keys():
            random.shuffle(train_data[key])
            random.shuffle(test_data[key])

    if return_labels:
        return train_data, test_data, class_labels
    return train_data, test_data


def space_words(s: str):
    out = s[0]
    for c in s[1:]:
        if c.isupper():
            out += ' '
        out += c
    return out


def run_quantum_circuit_for_dataset_batched_by_class(
    dataset: dict[int, list[torch.Tensor]],
    qd: QuantumDropout,
    save_distributions: bool = True,
    distributions_root: str = None,
) -> dict[str, dict[int, list[dict[int, float]]]] | None:
    '''Run the samples provided in `dataset` through the quantum circuit.
    If `save_distributions` is True, the distributions will be saved as pickle files,
    one per class, with path `./distributions/{observable}/class_{class_id}_dists.pkl`.

    Otherwise, the distributions will be returned in a dictionary keyed by observables,
    with values of dictionaries with the class ID as keys and list of distributions as
    values.  This sometimes consumes too much memory when the dataset is large.
    '''

    if save_distributions and distributions_root is None:
        raise ValueError('Must supply root when save_distributions=True')

    distributions = {obs: {} for obs in qd.observables}

    for obs_id, circuit in enumerate(qd.quantum_circuits):
        for class_id, samples in tqdm(
            dataset.items(), desc=f'Processing data through qc with observable {qd.observables[obs_id]}'
        ):
            # We can only send `qd.max_experiments` circuits to the backend at once, so we compute the distributions
            # over several runs
            class_dists = []
            for i in range(0, len(samples), qd.max_experiments):
                # Bind parameters to the quantum circuits
                bound_circuits_for_class = []
                for sample in samples[i : i + qd.max_experiments]:
                    param_dict = {param: val for param, val in zip(qd.params, sample.flatten().tolist())}
                    bound_circuits_for_class.append(circuit.assign_parameters(param_dict))

                print(f'Running {len(bound_circuits_for_class)} circuits')
                counts = qd.backend.run(bound_circuits_for_class, shots=SHOTS).result().get_counts()

                # Normalize the counts to obtain distributions
                class_dists.extend([{key: value / SHOTS for key, value in dist.items()} for dist in counts])

            if save_distributions:
                dirnames = (distributions_root, qd.observables[obs_id])
                mkdir_recursive(*dirnames)
                path = os.path.join(*dirnames, f'class_{class_id}_dists.pkl')
                with open(path, 'wb') as f:
                    pickle.dump(class_dists, f)
            else:
                distributions[qd.observables[obs_id]][class_id] = class_dists

    if not save_distributions:
        return distributions


def chebyshev(p: dict, q: dict) -> float:
    '''Chebyshev distance between counts dictionaries p and q'''

    keys = p.keys() | q.keys()
    max_key = max(keys, key=lambda k: abs(p.get(k, 0) - q.get(k, 0)))
    return abs(p.get(max_key, 0) - q.get(max_key, 0))


def kl_div(p: dict, q: dict) -> float:
    '''Kullback-Liebler divergence between counts dictionaries p and q.
    Here it is assumed that all dict values are non-zero, as is the case for
    counts dictionaries returned by Qiskit.
    '''

    # The keys of p must be a subset of the keys of q, otherwise there is a key k
    # such that p(k) is non-zero and q(k) is zero, and p(k)/q(k) is undefined.
    p_keys, q_keys = p.keys(), q.keys()
    if p_keys - q_keys:
        raise ValueError('q has zero values for which p is non-zero')

    # Only consider keys for which p(k) is non-zero
    return sum(p[k] * math.log(p[k] / q[k]) for k in q_keys & p_keys)


def js_div(p: dict, q: dict) -> float:
    '''Jensen-Shannon divergence between counts dictionaries p and q'''

    m = {k: (p.get(k, 0) + q.get(k, 0)) / 2 for k in p.keys() | q.keys()}
    return 0.5 * (kl_div(p, m) + kl_div(q, m))


def load_distributions_for_class(root: str, observable: str, class_id: str) -> list[dict[int, float]]:
    '''Load and return the distributions for a given class saved at the path
    `./{root}/{observable}/class_{class_id}_dists.pkl`.
    '''

    with open(os.path.join(root, observable, f'class_{class_id}_dists.pkl'), 'rb') as f:
        distributions = pickle.load(f)
    return distributions


def compute_divergences_single_instance(
    id_tuple: tuple[str, int, int], distributions_root: str, divergences_root: str
) -> None:
    '''Compute JS divergences and Chebyshev distances for a single pair of classes.
    This function is used to support parallel computation of divergences.
    '''

    obs, class_1_id, class_2_id = id_tuple

    # Initialize arrays to store the divergences
    divergences = [torch.empty((SAMPLES_PER_CLASS, SAMPLES_PER_CLASS)) for _ in range(2)]

    # Load the lists of distributions
    class_1_dists = load_distributions_for_class(distributions_root, obs, class_1_id)
    class_2_dists = load_distributions_for_class(distributions_root, obs, class_2_id)

    # Loop over each pair of distributions
    for sample_1_id, sample_2_id in product(range(SAMPLES_PER_CLASS), range(SAMPLES_PER_CLASS)):
        if class_1_id == class_2_id and sample_1_id > sample_2_id:
            # The divergences are symmetric, so there is no need to compute twice
            divergences[0][sample_2_id, sample_1_id] = divergences[0][sample_1_id, sample_2_id]
            divergences[1][sample_2_id, sample_1_id] = divergences[1][sample_1_id, sample_2_id]
        else:
            # Compute the JS divergences and Chebyshev distances
            divergences[0][sample_2_id, sample_1_id] = js_div(class_1_dists[sample_1_id], class_2_dists[sample_2_id])
            divergences[1][sample_2_id, sample_1_id] = chebyshev(
                class_1_dists[sample_1_id], class_2_dists[sample_2_id]
            )

    # Save the divergences in `./{divergences_root}/{observable}/class_{class_1_id}_vs_class_{class_2_id}.pkl`
    dirnames = (divergences_root, obs)
    mkdir_recursive(*dirnames)
    with open(os.path.join(*dirnames, f'class_{class_1_id}_vs_class_{class_2_id}.pkl'), 'wb') as f:
        pickle.dump(divergences, f)


def compute_divergences_from_saved(
    distributions_root: str, divergences_root: str, use_multiprocessing=True, processes=6
) -> None:
    '''Compute Jensen-Shannon divergences and Chebyshev distances between all pairs of classes from
    saved data with the option of using multiprocessing to speed up the computation.
    Assumes the distributions are saved as `./{distributions_root}/{observable}/class_{class_id}_dists.pkl`.

    The divergences are saved as pickle files with the path
    `./{divergences_root}/{observable}/class_{class_1_id}_vs_class_{class_2_id}.pkl`.
    '''

    # Get the list of observables from the directory names
    is_valid_obs = lambda obs: all(o in 'IXYZ' for o in obs)
    observables = [obs for obs in os.listdir(distributions_root) if is_valid_obs(obs)]

    if use_multiprocessing:
        # Use multiprocessing to speed up the divergence calculation
        distribution_ids = [(obs, c1, c2) for obs in observables for c1, c2 in CLASS_PAIRS]
        compute_divergences_single_instance_partial = partial(
            compute_divergences_single_instance,
            distributions_root=distributions_root,
            divergences_root=divergences_root,
        )
        print(f'Computing divergences from saved distributions using {processes} processes')
        start = time.time()
        with Pool(processes=processes) as pool:
            pool.map(compute_divergences_single_instance_partial, distribution_ids)
        print(f'Completed in {time.time() - start:.2f} seconds')
    else:
        for class_1_id, class_2_id in tqdm(CLASS_PAIRS, desc='Computing divergences from saved distributions'):
            for obs in observables:
                compute_divergences_single_instance(
                    (obs, class_1_id, class_2_id), distributions_root, divergences_root
                )


def load_divergences_for_classes(root: str, observable: str, class_1_id: int, class_2_id: int) -> list[torch.Tensor]:
    '''Load and return the arrays of divergences for a given pair of classes saved at the path
    `./{root}/{observable}/class_{class_1_id}_vs_class_{class_2_id}.pkl`.  Returns a list of divergences
    tensors of length 2.
    '''

    with open(os.path.join(root, observable, f'class_{class_1_id}_vs_class_{class_2_id}.pkl'), 'rb') as f:
        divergences = pickle.load(f)
    return divergences


def normalize_saved_divergences(divergences_root: str) -> None:
    '''Compute the global maxima and minima of the JS and Chebyshev divergences and normalize the arrays.
    This assumes the arrays of divergences are saved at the path
    `./{divergences_root}/{observable}/class_{class_1_id}_vs_class_{class_2_id}.pkl`.
    Note that this overwrites the arrays divergences with their normalized equivalents.
    '''

    # Get the list of observables from the directory names
    is_valid_obs = lambda obs: all(o in 'IXYZ' for o in obs)
    observables = [obs for obs in os.listdir(divergences_root) if is_valid_obs(obs)]

    for obs in tqdm(observables, desc='Normalizing saved divergences'):
        # Initialize variables to store the global minima and maxima
        js_max, chebyshev_max, js_min, chebyshev_min = 0, 0, float('inf'), float('inf')

        # Loop over each pair of classes
        for class_1_id, class_2_id in CLASS_PAIRS:
            divergences = load_divergences_for_classes(divergences_root, obs, class_1_id, class_2_id)

            # Update the global minima and maxima
            js_max = max(js_max, divergences[0].max())
            js_min = min(js_min, divergences[0].min())
            chebyshev_max = max(chebyshev_max, divergences[1].max())
            chebyshev_min = min(chebyshev_min, divergences[1].min())

        # Once we have the global minima and maxima for the given observable, loop over pairs of classes again
        for class_1_id, class_2_id in CLASS_PAIRS:
            divergences = load_divergences_for_classes(divergences_root, obs, class_1_id, class_2_id)

            # Normalize each array of divergences identically
            divergences[0] = (divergences[0] - js_min) / (js_max - js_min)
            divergences[1] = (divergences[1] - chebyshev_min) / (chebyshev_max - chebyshev_min)

            # Overwrite the original arrays of divergences with the normalized ones
            path = os.path.join(divergences_root, obs, f'class_{class_1_id}_vs_class_{class_2_id}.pkl')
            with open(path, 'wb') as f:
                pickle.dump(divergences, f)


def mkdir_recursive(*dirnames: str) -> None:
    '''Create a nested set of directories.  If a directory already exists, it is left as is.'''

    path = ''
    for dirname in dirnames:
        path = os.path.join(path, dirname)
        if not os.path.exists(path):
            os.mkdir(path)


def rm_dirs(*dirnames: str) -> None:
    '''Recursively remove a directory and all subdirectories'''

    for dir in dirnames:
        shutil.rmtree(dir)


def plot_heatmaps_from_saved(divergences_root: str, plots_root: str) -> None:
    '''Plot heatmaps of the JS divergences and Chebyshev distances.
    This assumes arrays of divergences are saved at the path
    `./{divergences_root}/{observable}/class_{class_1_id}_vs_class_{class_2_id}.pkl`.
    Plots are saved at the path
    `./{plots_root}/{observable}/{class_1_id}/class_{class_1_id}_vs_class_{class_2_id}.png`.
    '''

    is_valid_obs = lambda obs: all(o in 'IXYZ' for o in obs)
    observables = [obs for obs in os.listdir(divergences_root) if is_valid_obs(obs)]

    titles = ['JS Divergence', 'Chebyshev Distance']

    for obs in observables:
        for class_1_id, class_2_id in tqdm(CLASS_PAIRS, desc=f'Plotting heatmaps for observable {obs}'):
            fig = plt.figure(figsize=(9, 5))
            divergences = load_divergences_for_classes(divergences_root, obs, class_1_id, class_2_id)
            for i in range(2):
                ax = plt.subplot(1, 2, i + 1)
                ax.imshow(divergences[i], interpolation='nearest', cmap='hot_r', origin='lower', vmin=0, vmax=1)
                ax.set_title(titles[i])

            # Resize the subplots
            fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9)

            # Add the colorbar
            cb_ax = fig.add_axes([0.93, 0.19, 0.02, 0.62])
            fig.colorbar(ScalarMappable(cmap='hot_r'), cax=cb_ax)

            # Add axes labels
            fig.text(0.5, 0.07, s=f'Class {class_1_id}', ha='center', va='center', fontsize=11)
            fig.text(0.03, 0.5, s=f'Class {class_2_id}', ha='center', va='center', fontsize=11, rotation='vertical')

            plt.suptitle(f'Normalized divergences for observable {obs}', fontsize=12)

            dirnames = (plots_root, obs, str(class_1_id))
            mkdir_recursive(*dirnames)
            plt.savefig(os.path.join(*dirnames, f'class_{class_1_id}_vs_class_{class_2_id}.png'), dpi=300)

            plt.close()


def plot_dropout_probability_from_saved_distributions(distributions_root: str, class_labels: dict[int, str]):
    '''Assumes the distributions are saved as `./{distributions_root}/{observable}/class_{class_id}_dists.pkl`.'''
    from pathlib import Path
    import matplotlib.colors as mcolors
    import numpy as np

    plt.rcParams.update(
        {
            'text.usetex': True,
            'font.family': 'serif',
            'font.serif': ['Computer Modern Roman'],
            'text.latex.preamble': r'\usepackage{amsmath,amssymb}',
        }
    )

    N_MEASURE = 8

    # Get the list of observables from the directory names
    is_valid_obs = lambda obs: all(o in 'IXYZ' for o in obs)
    observables = [obs for obs in os.listdir(distributions_root) if is_valid_obs(obs)]

    # Neurons are killed if a uniformly randomly chosen bit string b satisfied P(b) >= p
    p = torch.arange(0, 1 + 1e-12, 0.0002)
    # p = torch.arange(0, 1 + 1e-12, 0.002)

    for obs in observables:
        fig, axes = plt.subplots(3, 4, figsize=(13, 7))
        axes = axes.flatten()
        mean_neuron_kill_curves = []
        std_neuron_kill_curves = []
        median_neuron_kill_curves = []
        q1_neuron_kill_curves = []
        q3_neuron_kill_curves = []

        cls_paths = list((Path(distributions_root) / obs).glob('class_*_dists.pkl'))
        classes = sorted(int(path.name.split('_')[1]) for path in cls_paths)
        for c, cls in enumerate(classes):
            dists = load_distributions_for_class(distributions_root, obs, cls)

            # Scale the distributions such that the max value is 1
            dist_maxs = [max(dist.values()) for dist in dists]
            dists = [{k: v / mx for k, v in dist.items()} for dist, mx in zip(dists, dist_maxs)]

            # Dist curves show the probabiliity of killing a neuron vs p
            neuron_kill_curves: list[torch.Tensor] = []
            for dist in tqdm(dists, desc=f'Computing neuron kill curve for observable {obs}, class {cls}'):
                neuron_kill_curve = torch.zeros(len(p))
                dist = torch.from_numpy(
                    np.fromiter((dist.get(f'{n:08b}', 0) for n in range(2**N_MEASURE)), dtype=np.float64)
                )  # Convert the distribution to a torch tensor
                for i, p_i in enumerate(p):
                    n_killed = (dist >= p_i).sum().item()  # Count the number of bitstring that satisfy P(b) > p_i
                    if n_killed == 0:
                        break
                    neuron_kill_curve[i] = n_killed / 2**N_MEASURE

                neuron_kill_curves.append(neuron_kill_curve)

            neuron_kill_curves: torch.Tensor = torch.vstack(neuron_kill_curves)
            mean_neuron_kill_curve = neuron_kill_curves.mean(dim=0)
            mean_neuron_kill_curves.append(mean_neuron_kill_curve)
            std_neuron_kill_curve = neuron_kill_curves.std(dim=0)
            std_neuron_kill_curves.append(std_neuron_kill_curve)

            median_neuron_kill_curve = neuron_kill_curves.median(dim=0)[0]
            median_neuron_kill_curves.append(median_neuron_kill_curve)
            q1_neuron_kill_curve = neuron_kill_curves.quantile(0.25, dim=0)
            q1_neuron_kill_curves.append(q1_neuron_kill_curve)
            q3_neuron_kill_curve = neuron_kill_curves.quantile(0.75, dim=0)
            q3_neuron_kill_curves.append(q3_neuron_kill_curve)

            ax = axes[c] if c < 8 else axes[c + 1]
            for neuron_kill_curve in neuron_kill_curves:
                ax.plot(p, neuron_kill_curve, c='grey', alpha=0.07, zorder=0, linewidth=0.5)
            ax.fill_between(
                p,
                mean_neuron_kill_curve + std_neuron_kill_curve,
                mean_neuron_kill_curve - std_neuron_kill_curve,
                # q1_neuron_kill_curve,
                # q3_neuron_kill_curve,
                facecolor='b',
                linewidth=0,
                alpha=0.2,
                zorder=1,
            )
            ax.plot(p, mean_neuron_kill_curve, 'b-', zorder=2)
            # ax.plot(p, median_neuron_kill_curve, 'b-', zorder=2, linewidth=1)
            ax.text(
                0.97,
                0.95,
                space_words(class_labels[c]),
                transform=ax.transAxes,
                ha='right',
                va='top',
                fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgrey', boxstyle='round,pad=0.3', linewidth=1),
            )
            ax.set_xlim(p.min(), 1)
            if c % 4 != 0:  # Hide y-axis labels for non-leftmost plots
                ax.tick_params(labelleft=False)
            if c not in [4, 7, 8, 9]:  # Hide x-axis labels for non-bottommost plots
                ax.tick_params(labelbottom=False)
        axes[8].set_visible(False)
        axes[11].set_visible(False)
        xlabel = r'$\sigma$'
        fig.text(0.525, 0.02, xlabel, ha='center', fontsize=14)
        # ylabel = r'$\mathbb{P}_{b\sim F(0,2^k-1)}(P(b) > p)$'
        ylabel = '$p_d$'
        fig.text(0.02, 0.525, ylabel, va='center', rotation='vertical', fontsize=14)
        plt.tight_layout(rect=[0.03, 0.03, 1, 1], w_pad=-1, h_pad=0)
        # plt.show()
        plt.savefig('thesis_plots/mean_neuron_kill_curves.png', dpi=400)
        # plt.savefig('thesis_plots/median_neuron_kill_curves.png', dpi=400)

        _, ax = plt.subplots(1, 1, figsize=(6, 5))
        colors = list(mcolors.TABLEAU_COLORS.keys())
        for i, (cv, std) in enumerate(zip(mean_neuron_kill_curves, std_neuron_kill_curves)):
        # for i, (cv, q1, q3) in enumerate(zip(median_neuron_kill_curves, q1_neuron_kill_curves, q3_neuron_kill_curves)):
            ax.plot(p, cv, c=colors[i], linewidth=1, label=space_words(class_labels[i]))
            # ax.fill_between(p, curve + std, curve - std, facecolor=colors[i], linewidth=0, alpha=0.35)
            # ax.fill_between(p, q1, q3, facecolor=colors[i], linewidth=0, alpha=0.35)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        plt.tight_layout()
        # plt.show()
        plt.savefig('thesis_plots/overlayed_mean_neuron_kill_curves.png', dpi=400)
        # plt.savefig('thesis_plots/overlayed_median_neuron_kill_curves.png', dpi=400)


def plot_js_heatmaps(divergences_root: str, class_labels: dict[int, str]):
    from matplotlib.axes import Axes

    # Get the list of observables from the directory names
    is_valid_obs = lambda obs: all(o in 'IXYZ' for o in obs)
    observables = [obs for obs in os.listdir(divergences_root) if is_valid_obs(obs)]

    colormap = 'hot_r'

    for obs in observables:
        for class_1_id in range(N_CLASSES):
            fig, axs = plt.subplots(2, 5, figsize=(15, 6))
            axs: list[Axes] = axs.flatten()
            for class_2_id in range(N_CLASSES):
                ax = axs[class_2_id]
                try:
                    divs = load_divergences_for_classes(divergences_root, obs, class_1_id, class_2_id)[0]
                except FileNotFoundError:
                    divs = load_divergences_for_classes(divergences_root, obs, class_2_id, class_1_id)[0].T
                ax.imshow(divs, origin='lower', cmap=colormap, interpolation=None, vmin=0, vmax=1)
                ticks = range(0, 1001, 200)
                ax.set_yticks(ticks, labels=[] if class_2_id % 5 != 0 else ticks)
                ax.set_xticks(ticks, labels=[] if class_2_id < 5 else ticks)
                ax.set_xlabel(space_words(class_labels[class_1_id]))
                ax.set_ylabel(space_words(class_labels[class_2_id]))

            # Add the colorbar
            cb_ax = fig.add_axes([0.94, 0.1115, 0.01, 0.837])
            fig.colorbar(ScalarMappable(cmap=colormap), cax=cb_ax)

            fig.text(0.98, 0.5, s='Normalized JS divergence', rotation='vertical', verticalalignment='center')

            plt.tight_layout(rect=[0, 0, 0.94, 1], h_pad=0, w_pad=0.5)
            # plt.show()
            plt.savefig(f'thesis_plots/js_divs/js_divs_{class_labels[class_1_id].lower()}.png', dpi=400)


def main():
    '''Computes divergences between all pairs of images in the dataset and saves them as heatmaps'''
    train_data, _, class_labels = load_data(shuffle=True, seed=SEED, return_labels=True)

    qd = QuantumDropout(
        p=0,
        input_shape=(0, 0),
        img_shape=(0, *train_data[0][0].shape),
        observables=OBSERVABLES,
        qubits_to_measure=QUBITS_TO_MEASURE,
        shots=SHOTS,
        backend=AerSimulator(method='matrix_product_state'),
        seed=SEED,
        max_threads=MAX_THREADS,
        max_experiments=100,
    )

    distributions_root = 'dists'
    divergences_root = 'divs'
    heatmaps_root = 'heatmaps'

    # Run the quantum circuit to compute the distributions for every input image
    run_quantum_circuit_for_dataset_batched_by_class(
        train_data, qd=qd, save_distributions=True, distributions_root=distributions_root
    )

    # Compute the divergences from the saved distributions
    compute_divergences_from_saved(distributions_root, divergences_root, processes=MAX_THREADS)

    # Normalize the divergences
    normalize_saved_divergences(divergences_root)

    # Compute and save the heatmaps
    plot_heatmaps_from_saved(divergences_root, heatmaps_root)

    # # Remove the directories containing the raw distributions and divergences
    # rm_dirs(distributions_root, divergences_root)

    plot_dropout_probability_from_saved_distributions(distributions_root, class_labels)
    plot_js_heatmaps(divergences_root, class_labels)


if __name__ == '__main__':
    main()
