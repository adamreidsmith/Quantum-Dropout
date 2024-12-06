'''
This file implements a land cover classification model on the EuroSAT data using PyTorch.
'''

import os
import statistics as stats
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam

from torchvision.models import resnet50, ResNet50_Weights
from torchvision.datasets import ImageFolder
from torchvision import transforms

import dill
import torchinfo
from tqdm import tqdm

from qiskit_aer import AerSimulator

from dropout import QuantumDropout


DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

SEED = 1234567890

N_CLASSES = 10
TRAIN_SPLIT = 0.8
BATCH_SIZE = 100
EPOCHS_HEAD = 16
EPOCHS_BACKBONE = 20
LR_HEAD = 3e-4
LR_BACKBONE = LR_HEAD / 100

POOL_OUTPUT_SIZE = (1, 1)
DROPOUT_P = 0.471
HIDDEN_NEURONS = 512

QD_ARGS = dict(
    p=0.1,  # ~DROPOUT_P = 0.5
    img_shape=(BATCH_SIZE, 3, 64, 64),
    observables=['Z' * 8],
    qubits_to_measure=[1, 7, 8, 12, 14, 18, 19, 25],
    shots=10_000,
    backend=AerSimulator(method='matrix_product_state'),
    seed=SEED,
    max_threads=0,
    max_experiments=100,
)

QUANTUM = True
EXCLUDE_DROPOUT = False


def load_data(
    encode_targets: bool = False,
    seed: int = None,
    return_labels: bool = False,
    resize_for_resnet: bool = False,
    return_unresized: bool = False,
) -> tuple[DataLoader, DataLoader]:
    '''Load the EuroSAT dataset. Returns a tuple of torch.utils.data.DataLoader objects.
    If return_labels is True, also return a dictionary mapping class names to integers.
    '''

    root = './2750'

    # Download the data, if necessary
    if not os.path.exists(root):
        import gdown
        import zipfile

        gdown.download(
            url='https://drive.google.com/u/0/uc?id=15EnOOMXVRlFjAa7OOQgXYlCPE3y3V2Xm&export=download',
            output='EuroSAT.zip',
            quiet=False,
        )
        with zipfile.ZipFile('EuroSAT.zip', 'r') as zip_ref:
            zip_ref.extractall('.')

    target_transform = None
    if encode_targets:
        # One-hot encode the targets
        target_transform = lambda i: torch.eye(N_CLASSES)[i]

    transform = transforms.ToTensor()  # Convert images to tensors and normalize them
    if resize_for_resnet:
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),  # ResNet50 expects tensors of size [bs, 3, 224, 224]
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    # Load the EuroSAT data
    data = ImageFolder(
        root=root,
        transform=transform,
        target_transform=target_transform,
        is_valid_file=lambda path: path.endswith('.jpg'),
    )
    # Load the EuroSAT data
    data_unresized = ImageFolder(
        root=root,
        transform=transforms.ToTensor(),
        target_transform=target_transform,
        is_valid_file=lambda path: path.endswith('.jpg'),
    )

    # Fix the generator for reproducibility
    generator = torch.Generator()
    if seed is not None:
        generator = generator.manual_seed(seed)

    # Split into training and testing sets
    train_data, test_data = random_split(data, [TRAIN_SPLIT, 1 - TRAIN_SPLIT], generator=generator)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, generator=generator)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, generator=generator)

    # Fix the generator for reproducibility
    generator = torch.Generator()
    if seed is not None:
        generator = generator.manual_seed(seed)

    # Split into training and testing sets
    train_data_unresized, test_data_unresized = random_split(
        data_unresized, [TRAIN_SPLIT, 1 - TRAIN_SPLIT], generator=generator
    )
    train_loader_unresized = DataLoader(train_data_unresized, batch_size=BATCH_SIZE, generator=generator)
    test_loader_unresized = DataLoader(test_data_unresized, batch_size=BATCH_SIZE, generator=generator)

    ret = (train_loader, test_loader)
    if return_unresized:
        ret = (*ret, train_loader_unresized, test_loader_unresized)
    if return_labels:
        ret = (*ret, data.class_to_idx)
    return ret


class LULCClassifier(nn.Module):
    '''The main classifier, consisting of a ResNet50 backbone and a custom head.'''

    def __init__(self, quantum: bool = False):
        super().__init__()

        self.quantum = quantum

        # Use ResNet50 as the backbone
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # This removes the last two layers (pooling and linear) of the ResNet50 model
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # Disable training of the ResNet50 backbone
        self.backbone.requires_grad_(False)

        # Reduce dimension from [BS, 2048, 7, 7] to [BS, 2048, *POOL_OUTPUT_SIZE]
        self.pool = nn.AdaptiveAvgPool2d(output_size=POOL_OUTPUT_SIZE)
        self.flatten = nn.Flatten()  # Flatten to [BS, 2048 * prod(POOL_OUTPUT_SIZE)]
        if self.quantum:
            self.dropout1 = QuantumDropout(
                input_shape=(BATCH_SIZE, 2048 * POOL_OUTPUT_SIZE[0] * POOL_OUTPUT_SIZE[1]), **QD_ARGS
            )
        else:
            self.dropout1 = nn.Dropout(DROPOUT_P)
        self.linear1 = nn.Linear(2048 * POOL_OUTPUT_SIZE[0] * POOL_OUTPUT_SIZE[1], HIDDEN_NEURONS)
        self.relu = nn.ReLU()
        if self.quantum:
            self.dropout2 = QuantumDropout(input_shape=(BATCH_SIZE, HIDDEN_NEURONS), **QD_ARGS)
        else:
            self.dropout2 = nn.Dropout(DROPOUT_P)
        self.linear2 = nn.Linear(HIDDEN_NEURONS, N_CLASSES)

    def forward(self, x: torch.Tensor, x_u: torch.Tensor | None = None, y: torch.Tensor | None = None) -> torch.Tensor:
        x = self.backbone(x)
        x = self.flatten(self.pool(x))
        if not EXCLUDE_DROPOUT:
            if self.quantum:
                x = self.dropout1(x, x_u, y)
            else:
                x = self.dropout1(x)
        x = self.relu(self.linear1(x))
        if not EXCLUDE_DROPOUT:
            if self.quantum:
                x = self.dropout2(x, x_u, y)
            else:
                x = self.dropout2(x)
        return self.linear2(x)  # No need to apply softmax as it is applied by the loss function


def confusion_matrix(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    '''Generate a confusion matrix from 1D tensors of predicted classes and target classes.
    Returns a confusion matrix of shape (N_CLASSES, N_CLASSES) where entry (i, j) is the
    count of examples with true class i and predicted class j.
    '''

    cm = torch.zeros((N_CLASSES, N_CLASSES), dtype=torch.int64)
    for t, p in zip(targets, predictions):
        cm[t.long(), p.long()] += 1
    return cm


def acc_from_cm(cm: torch.Tensor) -> float:
    '''Compute the accuracy from a confusion matrix.'''

    return cm.trace().item() / cm.sum().item()


def mcc_from_cm(cm: torch.Tensor) -> float:
    '''Compute the Matthew's correlation coefficient from a confusion matrix.'''

    t_k = cm.sum(dim=1)  # The number of times class k truly occurred
    p_k = cm.sum(dim=0)  # The number of times class k was predicted
    c = cm.trace()  # The total number of samples correctly predicted
    s = cm.sum()  # The total number of samples

    numer = (c * s - t_k @ p_k).item()
    denom = (torch.sqrt(s**2 - p_k @ p_k) * torch.sqrt(s**2 - t_k @ t_k)).item()
    if denom == 0:
        return -1
    return numer / denom


def train(
    model: LULCClassifier,
    dataloader: DataLoader,
    dataloader_unresized: DataLoader,
    loss_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    epoch: int,
    epochs: int,
    eval_in_eval_mode: bool = False,
) -> tuple[list[float], torch.Tensor]:
    train_loss = []
    train_cm = None

    data_iter = zip(dataloader, dataloader_unresized) if QUANTUM else dataloader

    model.train()
    for data in tqdm(data_iter, desc=f'Training epoch {epoch + 1}/{epochs}', total=len(dataloader)):
        if QUANTUM:
            (x, y), (x_u, _) = data
        else:
            x, y = data
            x_u = None
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        # Zero gradients and compute the prediction
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x, x_u, y)

        # Loss computation and backpropagation
        loss = loss_func(prediction, y)
        loss.backward()

        # Parameter optimization
        optimizer.step()

        if not eval_in_eval_mode:
            # Track loss and accuracy metrics
            train_loss.append(loss.item())
            predictions = torch.argmax(nn.functional.softmax(prediction, dim=1), dim=1)
            if train_cm is None:
                train_cm = confusion_matrix(predictions, y)
            else:
                train_cm += confusion_matrix(predictions, y)

    if eval_in_eval_mode:
        data_iter = zip(dataloader, dataloader_unresized) if QUANTUM else dataloader
        model.eval()
        with torch.no_grad():
            for data in tqdm(
                data_iter, desc=f'Evaluating train set epoch {epoch + 1}/{epochs}', total=len(dataloader)
            ):
                if QUANTUM:
                    (x, y), (x_u, _) = data
                else:
                    x, y = data
                    x_u = None
                x = x.to(DEVICE)
                y = y.to(DEVICE)

                prediction = model(x, x_u, y)
                # Track loss and accuracy metrics
                train_loss.append(loss_func(prediction, y).item())
                predictions = torch.argmax(nn.functional.softmax(prediction, dim=1), dim=1)
                if train_cm is None:
                    train_cm = confusion_matrix(predictions, y)
                else:
                    train_cm += confusion_matrix(predictions, y)

    return train_loss, train_cm


@torch.no_grad()
def test(
    model: LULCClassifier,
    dataloader: DataLoader,
    dataloader_unresized: DataLoader,
    loss_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    epoch: int,
    epochs: int,
) -> tuple[list[float], torch.Tensor]:
    test_loss = []
    test_cm = None

    data_iter = zip(dataloader, dataloader_unresized) if QUANTUM else dataloader

    model.eval()
    for data in tqdm(data_iter, desc=f'Evaluating test set epoch {epoch + 1}/{epochs}', total=len(dataloader)):
        if QUANTUM:
            (x, y), (x_u, _) = data
        else:
            x, y = data
            x_u = None
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        # Obtain predictions and track loss and accuracy metrics
        prediction = model(x, x_u, y)

        # Track loss and cm metrics
        test_loss.append(loss_func(prediction, y).item())
        predictions = torch.argmax(nn.functional.softmax(prediction, dim=1), dim=1)
        if test_cm is None:
            test_cm = confusion_matrix(predictions, y)
        else:
            test_cm += confusion_matrix(predictions, y)

    return test_loss, test_cm


def load_trained_classifier(weights_file: str = 'lulc_classifier_weights.pt', quantum: bool = False) -> LULCClassifier:
    model = LULCClassifier(quantum=quantum)
    model.load_state_dict(torch.load(weights_file))
    return model


def main(save: bool = True) -> None:
    torch.manual_seed(SEED)

    train_loader, test_loader, train_loader_unresized, test_loader_unresized = load_data(
        seed=SEED, resize_for_resnet=True, return_unresized=True
    )
    # train_loader = list(train_loader)
    # test_loader = list(test_loader)

    model = LULCClassifier(quantum=QUANTUM).to(DEVICE)

    torchinfo.summary(model, input_size=(BATCH_SIZE, 3, 224, 224), device=DEVICE)

    ### Train the head of the network #############################################################
    loss_func = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), LR_HEAD)

    head_mean_train_loss, head_mean_test_loss = [], []  # list[float]
    head_train_cms, head_test_cms = [], []  # list[torch.Tensor]

    for epoch in range(EPOCHS_HEAD):
        # Train the model
        train_loss, train_cm = train(
            model,
            train_loader,
            train_loader_unresized,
            loss_func,
            optimizer,
            epoch,
            EPOCHS_HEAD,
            eval_in_eval_mode=True,
        )
        head_mean_train_loss.append(stats.mean(train_loss))
        head_train_cms.append(train_cm)

        # Test the model
        test_loss, test_cm = test(model, test_loader, test_loader_unresized, loss_func, epoch, EPOCHS_HEAD)
        head_mean_test_loss.append(stats.mean(test_loss))
        head_test_cms.append(test_cm)

        print(
            f'Epoch {epoch + 1}/{EPOCHS_HEAD}  |  '
            f'train loss {head_mean_train_loss[-1]:.4f}  |  '
            f'train acc {acc_from_cm(head_train_cms[-1]):.2%}  |  '
            f'test loss {head_mean_test_loss[-1]:.4f}  |  '
            f'test acc {acc_from_cm(head_test_cms[-1]):.2%}'
        )

    ### Fine-tune the backbone ####################################################################
    # Enable training of the ResNet50 backbone
    model.backbone.requires_grad_(True)

    torchinfo.summary(model, device=DEVICE)

    # Train with a reduced learning rate
    optimizer = Adam(model.parameters(), LR_BACKBONE)

    bb_mean_train_loss, bb_mean_test_loss = [], []  # list[float]
    bb_train_cms, bb_test_cms = [], []  # list[torch.Tensor]

    for epoch in range(EPOCHS_BACKBONE):
        # Train the model
        train_loss, train_cm = train(
            model,
            train_loader,
            train_loader_unresized,
            loss_func,
            optimizer,
            epoch,
            EPOCHS_BACKBONE,
            eval_in_eval_mode=True,
        )
        bb_mean_train_loss.append(stats.mean(train_loss))
        bb_train_cms.append(train_cm)

        # Test the model
        test_loss, test_cm = test(model, test_loader, test_loader_unresized, loss_func, epoch, EPOCHS_BACKBONE)
        bb_mean_test_loss.append(stats.mean(test_loss))
        bb_test_cms.append(test_cm)

        print(
            f'Epoch {epoch + 1}/{EPOCHS_BACKBONE}  |  '
            f'train loss {bb_mean_train_loss[-1]:.4f}  |  '
            f'train acc {acc_from_cm(bb_train_cms[-1]):.2%}  |  '
            f'test loss {bb_mean_test_loss[-1]:.4f}  |  '
            f'test acc {acc_from_cm(bb_test_cms[-1]):.2%}'
        )

    full_mean_train_loss = head_mean_train_loss + bb_mean_train_loss
    full_mean_test_loss = head_mean_test_loss + bb_mean_test_loss
    full_train_cms = head_train_cms + bb_train_cms
    full_test_cms = head_test_cms + bb_test_cms

    if save:
        pth = 'lulc_classifier_weights.pt'
        if EXCLUDE_DROPOUT:
            pth = 'bare_' + pth
        elif QUANTUM:
            pth = 'q_' + pth
        torch.save(model.state_dict(), pth)
        config = dict(
            seed=SEED,
            train_split=TRAIN_SPLIT,
            batch_size=BATCH_SIZE,
            epochs_head=EPOCHS_HEAD,
            epochs_backbone=EPOCHS_BACKBONE,
            lr_head=LR_HEAD,
            lr_backbone=LR_BACKBONE,
            pool_output_size=POOL_OUTPUT_SIZE,
            hidden_neurons=HIDDEN_NEURONS,
        )
        if QUANTUM and not EXCLUDE_DROPOUT:
            config |= dict(qd_args=QD_ARGS)
        else:
            config |= dict(dropout_p=DROPOUT_P)
        pth = 'lulc_classifier_results.pkl'
        if EXCLUDE_DROPOUT:
            pth = 'bare_' + pth
        elif QUANTUM:
            pth = 'q_' + pth
        to_save = (full_mean_train_loss, full_train_cms, full_mean_test_loss, full_test_cms, config)
        if QUANTUM and not EXCLUDE_DROPOUT:
            to_save = (*to_save, model.dropout1.p0, model.dropout2.p0)
        with open(pth, 'wb') as f:
            dill.dump(to_save, f)


def run_analysis():
    from pathlib import Path
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    import numpy as np
    import seaborn as sns

    plt.rcParams.update(
        {
            'text.usetex': True,
            'font.family': 'serif',
            'font.serif': ['Computer Modern Roman'],
        }
    )

    tl, tcm, vl, vcm = [], [], [], []
    for prefix in ('bare_', '', 'q_'):
        results_path = Path(f'{prefix}lulc_classifier_results.pkl')
        if prefix == 'q_':
            with open(results_path, 'rb') as f:
                train_loss, train_cm, test_loss, test_cm, config, d1p0, d2p0 = dill.load(f)
        else:
            with open(results_path, 'rb') as f:
                train_loss, train_cm, test_loss, test_cm, config = dill.load(f)
        tl.append(train_loss)
        tcm.append(train_cm)
        vl.append(test_loss)
        vcm.append(test_cm)

        print(f'CONFIG: {config}')

    tacc = [[acc_from_cm(cm) for cm in tcm[i]] for i in range(3)]
    vacc = [[acc_from_cm(cm) for cm in vcm[i]] for i in range(3)]
    tmcc = [[mcc_from_cm(cm) for cm in tcm[i]] for i in range(3)]
    vmcc = [[mcc_from_cm(cm) for cm in vcm[i]] for i in range(3)]

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    # _, axs = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['darkblue', 'firebrick', 'darkgreen']
    epochs = range(1, len(tl[0]) + 1)
    for q, label in enumerate(('No dropout', 'Classical dropout', 'Quantum dropout')):
        # Plot loss curves
        axs[0].plot(epochs, tl[q], c=colors[q], linewidth=1, label=f'{label} - train')
        axs[0].plot(epochs, vl[q], c=colors[q], linewidth=1, linestyle='--', label=f'{label} - test')
        # Plot MCC curves
        axs[1].plot(epochs, tacc[q], c=colors[q], linewidth=1, label=f'{label} - train')
        axs[1].plot(epochs, vacc[q], c=colors[q], linewidth=1, linestyle='--', label=f'{label} - test')
        # # Plot accuracy curves
        # axs[2].plot(epochs, tmcc[q], c=colors[q], linewidth=1, label=f'{label} - train')
        # axs[2].plot(epochs, vmcc[q], c=colors[q], linewidth=1, linestyle='--', label=f'{label} - test')
    for ax in axs:
        ax.axvline(EPOCHS_HEAD, c='k', zorder=0, linestyle='--', linewidth=1, alpha=0.5, ymin=-0.06, clip_on=False)
        ax.set_xticks(range(0, 36, 5), labels=[*range(0, EPOCHS_HEAD, 5), *range(5, EPOCHS_BACKBONE + 1, 5)])
        ax.legend()
        ax.set_xlabel(r'Head Epochs\hspace{9em}Backbone Epochs', x=0.1275, ha='left')
    axs[0].set_ylabel('Loss')
    axs[1].set_ylabel('Accuracy')
    # axs[2].set_ylabel('MCC')
    plt.tight_layout()
    # plt.show()
    plt.savefig('plots/training_loss_acc.png', dpi=400)

    class_labels = [
        'Annual Crop',
        'Forest',
        'Herbaceous\nVegetation',
        'Highway',
        'Industrial',
        'Pasture',
        'Permanent\nCrop',
        'Residential',
        'River',
        'Sea Lake',
    ]

    # fig, axs = plt.subplots(1, 2, figsize=(12, 7))
    # for i in range(len(class_labels)):
    #     sns.violinplot(y=[class_labels[i]] * len(d1p0[i]), x=d1p0[i], orient='h', split=True, ax=axs[0])
    #     sns.violinplot(y=[class_labels[i]] * len(d2p0[i]), x=d2p0[i], orient='h', split=True, ax=axs[1])
    # axs[1].set_yticks([])
    fig, axs = plt.subplots(1, 1, figsize=(7, 7))
    for i in range(len(class_labels)):
        sns.violinplot(y=[class_labels[i]] * len(d1p0[i]), x=d1p0[i], orient='h', split=True, ax=axs, bw_adjust=0.8)
    fig.text(0.535, 0.02, r'$p_d$', ha='center', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    # plt.show()
    plt.savefig('plots/true_dropout_probability_distributions.png', dpi=400)

    for i, cm in enumerate((tcm, vcm)):
        fig, axs = plt.subplots(1, 3, figsize=(13, 6.5))
        for q, label in enumerate(('No Dropout', 'Classical Dropout', 'Quantum Dropout')):
            lcm = cm[q][-1].detach().numpy().astype(np.float64)
            for row in range(lcm.shape[0]):
                lcm[row] /= lcm[row].sum()

            cm_disp = ConfusionMatrixDisplay(lcm, display_labels=class_labels)
            # cm_disp.plot(values_format='.2%', colorbar=False, xticks_rotation='vertical', ax=axs[q], cmap='Blues_r')
            cm_disp.plot(values_format='.2%', colorbar=False, xticks_rotation='vertical', ax=axs[q], cmap='Blues')
            axs[q].set_title(label, fontsize=13)
            axs[q].set_xlabel('')
            if q == 0:
                axs[q].set_ylabel('True Class', fontsize=13)
            else:
                axs[q].set_ylabel('')
            if q > 0:
                axs[q].set_yticks([])
        fig.text(0.542, 0.02, 'Predicted Class', ha='center', fontsize=13)
        plt.tight_layout()
        # plt.show()
        plt.savefig(f'plots/confusion_matrix_{"test" if i else "train"}.png', dpi=400)


if __name__ == '__main__':
    main()
    # run_analysis()
