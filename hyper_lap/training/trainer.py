from jaxtyping import Array, Float
from typing import Callable

import shutil
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import jax.tree as jt
import numpy as np
import wandb
from matplotlib import pyplot as plt
from optax import GradientTransformation, OptState
from torch.utils.data import DataLoader
from tqdm import tqdm

from hyper_lap.datasets import MultiDataLoader
from hyper_lap.datasets.base import Dataset
from hyper_lap.hyper import HyperNet, ResHyperNet
from hyper_lap.models import Unet
from hyper_lap.training.utils import to_PIL

from .metrics import calc_metrics


class Trainer[Net: Unet | HyperNet | ResHyperNet]:
    type TrainingStep = Callable[
        [Net, dict[str, Array], GradientTransformation, OptState],
        tuple[Float[Array, ""], Net, OptState],
    ]

    epoch: int

    opt: GradientTransformation
    opt_state: OptState

    train_loader: MultiDataLoader
    val_loader: MultiDataLoader

    training_step: TrainingStep

    @staticmethod
    def make_unet(net: Net, batch: dict[str, Array]) -> Unet:
        if isinstance(net, Unet):
            return net
        elif isinstance(net, (HyperNet, ResHyperNet)):
            return eqx.filter_jit(net)(batch["image"][0], batch["label"][0])
        else:
            raise ValueError(f"net has unexpected type {type(net)}")

    def __init__(
        self,
        train_loader: MultiDataLoader,
        val_loader: MultiDataLoader,
        opt: GradientTransformation,
        opt_state: OptState,
        training_step: TrainingStep,
        *,
        epoch: int = 0,
    ):
        super().__init__()

        # epoch gets incremented at start of train, so set to one less of start value
        self.epoch = epoch - 1

        self.opt = opt
        self.opt_state = opt_state

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.training_step = training_step

    def train(self, net: Net) -> Net:
        self.epoch += 1

        tqdm.write("Training:\n")

        losses = []

        for batch_tensor in tqdm(self.train_loader, leave=False):
            batch: dict[str, Array] = jt.map(jnp.asarray, batch_tensor)

            loss, net, self.opt_state = self.training_step(net, batch, self.opt, self.opt_state)

            losses.append(loss.item())

        loss_mean = jnp.mean(jnp.array(losses)).item()
        loss_std = jnp.std(jnp.array(losses)).item()

        tqdm.write(f"Loss: {loss_mean:.3}")

        if wandb.run is not None:
            wandb.run.log(
                {
                    "epoch": self.epoch,
                    "loss/train/mean": loss_mean,
                    "loss/train/std": loss_std,
                }
            )

        return net

    def validate(self, net: Net):
        tqdm.write("Validation:\n")

        all_metrics: dict[str, dict[str, Array]] = {}

        for dataloader in self.val_loader.dataloaders:
            dataset_name: str = dataloader.dataset.name  # type: ignore

            batch = jt.map(jnp.asarray, next(iter(dataloader)))

            unet = self.make_unet(net, batch)

            metrics = calc_metrics(unet, batch)

            tqdm.write(f"Dataset: {dataset_name}:")
            tqdm.write(f"    Dice score: {metrics['dice']:.3f}")
            tqdm.write(f"    IoU score : {metrics['iou']:.3f}")
            tqdm.write(f"    Hausdorff : {metrics['hausdorff']:.3f}")
            tqdm.write("")

            all_metrics[dataset_name] = metrics

        if wandb.run is not None:
            wandb.run.log(
                {
                    "epoch": self.epoch,
                }
                | {
                    f"{metric}/{dataset_name}": val.item()
                    for dataset_name, metrics in all_metrics.items()
                    for metric, val in metrics.items()
                }
            )

        tqdm.write("")

    def make_plots(self, net: Net, test_loader: DataLoader, *, image_folder: Path):
        if image_folder.exists():
            shutil.rmtree(image_folder)

        image_folder.mkdir(parents=True)

        for dataset, dataloader in zip(self.val_loader.datasets, self.val_loader.dataloaders):
            print(f"Dataset {dataset.name}")
            print()

            batch = jt.map(jnp.asarray, next(iter(dataloader)))

            unet = self.make_unet(net, batch)

            image = jnp.asarray(batch["image"][1])
            label = jnp.asarray(batch["label"][1])

            logits = eqx.filter_jit(unet)(image)
            pred = jnp.argmax(logits, axis=0)

            fig, axs = plt.subplots(ncols=3)

            axs[0].imshow(image.mean(axis=0), cmap="gray")
            axs[1].imshow(label, cmap="gray")
            axs[2].imshow(pred, cmap="gray")

            fig.savefig(image_folder / f"{dataset.name}.pdf")

            metrics = calc_metrics(unet, batch)

            print(f"Dataset {dataset.name}:")
            print(f"    Dice score: {metrics['dice']:.3f}")
            print(f"    IoU score : {metrics['iou']:.3f}")
            print(f"    Hausdorff : {metrics['hausdorff']:.3f}")

            print()
            print()

            if wandb.run is not None:
                class_labels = {0: "background", 1: "foreground"}

                image = wandb.Image(
                    to_PIL(image),
                    caption="Input image",
                    masks={
                        "ground_truth": {
                            "mask_data": np.asarray(label),
                            "class_labels": class_labels,
                        },
                        "prediction": {"mask_data": np.asarray(pred), "class_labels": class_labels},
                    },
                )

                wandb.run.log({f"images/train/{dataset.name}": image})

        assert isinstance(test_loader.dataset, Dataset)

        print()
        print()
        print(f"Test: {test_loader.dataset.name}")
        print()

        batch = jt.map(jnp.asarray, next(iter(test_loader)))

        unet = self.make_unet(net, batch)

        metrics = calc_metrics(unet, batch)

        print(f"Dice score: {metrics['dice']:.3f}")
        print(f"IoU score : {metrics['iou']:.3f}")
        print(f"Hausdorff : {metrics['hausdorff']:.3f}")

        image = jnp.asarray(batch["image"][1])
        label = jnp.asarray(batch["label"][1])

        logits = eqx.filter_jit(unet)(image)
        pred = jnp.argmax(logits, axis=0)

        fig, axs = plt.subplots(ncols=3)

        axs[0].imshow(image.mean(axis=0), cmap="gray")
        axs[1].imshow(label, cmap="gray")
        axs[2].imshow(pred, cmap="gray")

        fig.savefig(image_folder / f"{test_loader.dataset.name}_test.pdf")

        if wandb.run is not None:
            class_labels = {0: "background", 1: "foreground"}

            image = wandb.Image(
                to_PIL(image),
                caption="Input image",
                masks={
                    "ground_truth": {"mask_data": np.asarray(label), "class_labels": class_labels},
                    "prediction": {"mask_data": np.asarray(pred), "class_labels": class_labels},
                },
            )

            wandb.run.log({f"images/test/{test_loader.dataset.name}": image})
