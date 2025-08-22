from jaxtyping import Array, Float, Integer
from typing import Any, Callable, overload

import shutil
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree as jt
import numpy as np
import optax
import wandb
from chex import assert_axis_dimension, assert_equal_shape_suffix, assert_rank
from matplotlib import pyplot as plt
from optax import GradientTransformation, OptState, global_norm
from torch.utils.data import DataLoader
from tqdm import tqdm
from umap import UMAP

from hyper_lap.datasets import MultiDataLoader
from hyper_lap.datasets.base import Dataset
from hyper_lap.embedder import InputEmbedder
from hyper_lap.hyper import AttnHyperNet, HyperNet
from hyper_lap.models import AttentionUnet, FilmUnet, Unet, VitSegmentator
from hyper_lap.training.loss import loss_fn
from hyper_lap.training.utils import make_lr_schedule, to_PIL

from .metrics import calc_metrics


class Trainer[Net: Unet | HyperNet | AttnHyperNet | FilmUnet | AttentionUnet | VitSegmentator]:
    epoch: int

    lr_schedule: optax.Schedule

    opt: GradientTransformation
    opt_state: OptState

    train_loader: MultiDataLoader
    val_loader: MultiDataLoader

    _get_step: Callable[[], int]

    @staticmethod
    @eqx.filter_jit
    def net_forward(
        net: Net,
        embedder: InputEmbedder | None,
        images: Float[Array, "b c h w"],
        labels: Integer[Array, "b h w"],
        dataset_idx: Integer[Array, ""],
    ) -> Array:
        if embedder is not None:
            input_emb = embedder(images[0], labels[0], dataset_idx)
        else:
            input_emb = None

        return jax.vmap(net, in_axes=(0, None))(images, input_emb)

    @staticmethod
    @eqx.filter_jit
    def training_step(
        net: Net,
        embedder: InputEmbedder | None,
        batch: dict[str, Array],
        opt: optax.GradientTransformation,
        opt_state: OptState,
    ) -> tuple[Net, InputEmbedder | None, OptState, dict[str, Any]]:
        images = batch["image"]
        labels = batch["label"]
        dataset_idx = batch["dataset_idx"]

        net_embedder = (net, embedder)

        def grad_fn(net_embedder: tuple[Net, InputEmbedder | None]) -> Array:
            (net, embedder) = net_embedder

            logits = Trainer.net_forward(net, embedder, images, labels, dataset_idx)

            loss = jax.vmap(loss_fn)(logits, labels).mean()

            return loss

        loss, grads = eqx.filter_value_and_grad(grad_fn)(net_embedder)

        updates, opt_state = opt.update(grads, opt_state, net_embedder)  # type: ignore

        (net, embedder) = eqx.apply_updates(net_embedder, updates)

        aux = {
            "loss": loss,
            "grad_norm": global_norm(grads),  # type: ignore,
            "update_norm": global_norm(updates),  # type: ignore
        }

        return net, embedder, opt_state, aux

    def __init__(
        self,
        net: Net,
        embedder: InputEmbedder | None,
        train_loader: MultiDataLoader,
        val_loader: MultiDataLoader,
        *,
        first_epoch: int = 0,
        optim_config: dict[str, Any],
    ):
        super().__init__()

        # epoch gets incremented at start of train, so set to one less of start value
        self.epoch = first_epoch - 1

        lr_schedule = make_lr_schedule(
            len(train_loader),
            lr=optim_config["lr"],
            epochs=optim_config["epochs"],
            scheduler=optim_config["scheduler"],
        )

        self.lr_schedule = lr_schedule

        if optim_config["grad_clip"] is not None:
            self.opt = optax.chain(
                optax.clip_by_global_norm(optim_config["grad_clip"]), optax.adamw(self.lr_schedule)
            )

            self._get_step = lambda: self.opt_state[1][2].count  # type: ignore
        else:
            self.opt = optax.adamw(self.lr_schedule)

            self._get_step = lambda: self.opt_state[2].count  # type: ignore

        self.opt_state = self.opt.init(eqx.filter((net, embedder), eqx.is_array_like))

        self.train_loader = train_loader
        self.val_loader = val_loader

    @property
    def learning_rate(self) -> float:
        if isinstance(self.lr_schedule, float):
            return self.lr_schedule

        return float(self.lr_schedule(self._get_step()))  # type: ignore

    @overload
    def train(
        self, net: Net, embedder: InputEmbedder
    ) -> tuple[Net, InputEmbedder, dict[str, list[Any]]]: ...

    @overload
    def train(self, net: Net, embedder: None) -> tuple[Net, None, dict[str, list[Any]]]: ...

    def train(
        self, net: Net, embedder: InputEmbedder | None
    ) -> tuple[Net, InputEmbedder | None, dict[str, list[Any]]]:
        self.epoch += 1

        tqdm.write(f"Epoch {self.epoch: 3}: Training\n")

        auxs: list[dict[str, Any]] = []

        for batch_tensor in tqdm(self.train_loader, leave=False):
            batch: dict[str, Array] = jt.map(jnp.asarray, batch_tensor)

            net, embedder, self.opt_state, aux = eqx.filter_jit(self.training_step)(
                net, embedder, batch, self.opt, self.opt_state
            )

            auxs.append(aux)

        for aux in auxs:
            assert aux.keys() == auxs[0].keys()

        def try_unbox(x):
            """
            if x is a shapeless Array (scalar), unbox it, else do nothing
            """
            if eqx.is_array(x) and x.shape == ():
                return x.item()

            return x

        aux = {key: [try_unbox(aux[key]) for aux in auxs] for key in auxs[0].keys()}

        return net, embedder, aux

    def validate(self, net: Net, embedder: InputEmbedder | None):
        tqdm.write(f"Epoch {self.epoch: 3}: Validation\n")

        all_metrics: dict[str, dict[str, Array]] = {}

        losses = []

        for i, dataloader in enumerate(self.val_loader.dataloaders):
            dataset_name: str = dataloader.dataset.name  # type: ignore

            batch = jt.map(jnp.asarray, next(iter(dataloader)))

            images = batch["image"]
            labels = batch["label"]
            dataset_idx = batch["dataset_idx"]

            logits = Trainer.net_forward(net, embedder, images, labels, dataset_idx)

            metrics = calc_metrics(logits, labels)

            tqdm.write(f"Dataset: {dataset_name}:")
            tqdm.write(f"    Dice score: {metrics['dice']:.3f}")
            tqdm.write(f"    IoU score : {metrics['iou']:.3f}")
            tqdm.write(f"    Hausdorff : {metrics['hausdorff']:.3f}")
            tqdm.write("")

            all_metrics[dataset_name] = metrics

            loss = jax.jit(jax.vmap(loss_fn))(logits, labels).mean()

            losses.append(loss.item())

        if wandb.run is not None:
            wandb.run.log(
                {
                    "epoch": self.epoch,
                    "loss/validation/mean": np.mean(losses),
                }
                | {
                    f"{metric}/{dataset_name}": metrics[metric]
                    for dataset_name, metrics in all_metrics.items()
                    for metric in ["dice", "iou", "hausdorff"]
                }
            )
        else:
            tqdm.write(f"Validation loss: {np.mean(losses):.3}")

        tqdm.write("")

    def make_plots(
        self,
        net: Net,
        embedder: InputEmbedder | None,
        test_loader: DataLoader | None,
        *,
        image_folder: Path,
    ):
        if image_folder.exists():
            shutil.rmtree(image_folder)

        image_folder.mkdir(parents=True)

        def make_images(batch: dict[str, Array], *, dataset_name: str, test: bool = False):
            images = batch["image"]
            labels = batch["label"]
            dataset_idx = jnp.array(i)

            logits = self.net_forward(net, embedder, images, labels, dataset_idx)

            metrics = calc_metrics(logits, labels)

            image = images[1]
            label = labels[1]

            pred = jnp.argmax(logits[1], axis=0)

            fig, axs = plt.subplots(ncols=3)

            axs[0].imshow(image.mean(axis=0), cmap="gray")
            axs[1].imshow(label, cmap="gray")
            axs[2].imshow(pred, cmap="gray")

            fig.savefig(image_folder / f"{dataset_name}{'_test' if test else ''}.pdf")
            if not test:
                print(f"Dataset {dataset_name}:")
            else:
                print(f"Test Dataset {dataset_name}:")

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
                        "prediction": {
                            "mask_data": np.asarray(pred),
                            "class_labels": class_labels,
                        },
                    },
                )

                if not test:
                    wandb.run.log({f"images/train/{dataset_name}": image})
                else:
                    wandb.run.log({f"images/test/{dataset_name}": image})

        for i, (dataset, dataloader) in enumerate(
            zip(self.val_loader.datasets, self.val_loader.dataloaders)
        ):
            batch = jt.map(jnp.asarray, next(iter(dataloader)))

            make_images(batch, dataset_name=dataset.name)

        if test_loader is None:
            print("no test loader")

            return

        assert isinstance(test_loader.dataset, Dataset)

        print(f"{test_loader.dataset.name=}")

        batch = jt.map(jnp.asarray, next(iter(test_loader)))

        make_images(batch, dataset_name=f"{test_loader.dataset.name}", test=True)

    @staticmethod
    def make_umap(embedder: InputEmbedder, datasets: list[Dataset], image_folder: Path):
        if not image_folder.exists():
            image_folder.mkdir(parents=True)

        embedder_jit = jax.jit(jax.vmap(embedder, in_axes=(0, 0, None)))

        for dataset in datasets:
            assert isinstance(dataset, Dataset)

        multi_dataloader = MultiDataLoader(
            *datasets,
            num_samples=100,
            dataloader_args=dict(batch_size=100),
        )

        samples = {
            dataset.name: jt.map(jnp.asarray, next(iter(dataloader)))
            for dataset, dataloader in zip(multi_dataloader.datasets, multi_dataloader.dataloaders)
        }

        embs = {
            name: embedder_jit(X["image"], X["label"], jnp.array(i))
            for i, (name, X) in enumerate(samples.items())
        }

        umap = UMAP()
        umap.fit(jnp.concat([embs for embs in embs.values()]))

        projs: dict[str, Array] = {name: umap.transform(embs) for name, embs in embs.items()}  # type: ignore

        fig, ax = plt.subplots()

        for name, proj in projs.items():
            ax.scatter(proj[:, 0], proj[:, 1], 4.0, label=name)

        pos = ax.get_position()
        ax.set_position((pos.x0, pos.y0, pos.width * 0.75, pos.height))

        fig.legend(loc="outside center right")

        fig.savefig(image_folder / "umap.pdf")

        if wandb.run is not None:
            image = wandb.Image(fig, mode="RGBA", caption="UMAP")

            wandb.run.log({"images/umap": image})
