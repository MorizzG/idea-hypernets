from jaxtyping import Array, Float, Integer
from typing import Any, Callable, overload

import itertools
import shutil
from functools import partial
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree as jt
import numpy as np
import optax
import wandb
from grain import MapDataset, ReadOptions
from matplotlib import pyplot as plt
from optax import GradientTransformation, OptState, global_norm
from tqdm import tqdm
from umap import UMAP

from hyper_lap.embedder import InputEmbedder
from hyper_lap.training.loss import loss_fn
from hyper_lap.training.utils import make_lr_schedule, to_PIL

from .metrics import calc_metrics


def transpose(elems: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """
    Transpose a list of dicts into a dict of lists.

    Also unboxes scalar Arrays
    """
    assert isinstance(elems, list)
    assert all(isinstance(x, dict) for x in elems)
    assert all(x.keys() == elems[0].keys() for x in elems)

    if len(elems) == 0:
        return {}

    def try_unbox(x):
        """
        if x is a shapeless Array (scalar), unbox it, else do nothing
        """
        if eqx.is_array(x) and x.shape == ():
            return x.item()

        return x

    keys = elems[0].keys()

    return {key: [try_unbox(elem[key]) for elem in elems] for key in keys}


class Trainer[Net: Callable[[Array, Array | None], Array]]:
    # NUM_VALIDATION_BATCHES: ClassVar[int] = 20

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
        batches: list[dict[str, Array]],
        opt: optax.GradientTransformation,
        opt_state: OptState,
    ) -> tuple[Net, InputEmbedder | None, OptState, dict[str, Any]]:
        net_embedder = (net, embedder)

        @eqx.filter_value_and_grad
        def grad_fn(
            net_embedder: tuple[Net, InputEmbedder | None],
            images: Array,
            labels: Array,
            dataset_idx: Array,
        ) -> Array:
            (net, embedder) = net_embedder

            logits = Trainer.net_forward(net, embedder, images, labels, dataset_idx)

            loss = jax.vmap(loss_fn)(logits, labels).mean()

            return loss

        images = batches[0]["image"]
        labels = batches[0]["label"]
        dataset_idx = batches[0]["dataset_idx"]

        loss, grads = grad_fn(net_embedder, images, labels, dataset_idx)

        for batch in batches[1:]:
            images = batch["image"]
            labels = batch["label"]
            dataset_idx = batch["dataset_idx"]

            next_loss, next_grads = grad_fn(net_embedder, images, labels, dataset_idx)

            loss += next_loss
            grads = jt.map(jnp.add, grads, next_grads)

        n_batches = len(batches)

        loss /= n_batches
        grads = jt.map(lambda x: x / n_batches, grads)

        updates, opt_state = opt.update(grads, opt_state, net_embedder)  # type: ignore

        (net, embedder) = eqx.apply_updates(net_embedder, updates)

        aux = {
            "loss": loss,
            "grad_norm": global_norm(grads),  # type: ignore
            "update_norm": global_norm(updates),  # type: ignore
        }

        return net, embedder, opt_state, aux

    num_workers: int

    epoch: int

    lr_schedule: optax.Schedule

    opt: GradientTransformation
    opt_state: OptState

    grad_accu: int

    trainsets: list[MapDataset]
    valsets: list[MapDataset]

    num_validation_batches: int

    _get_step: Callable[[], int]

    def __init__(
        self,
        net: Net,
        embedder: InputEmbedder | None,
        trainsets: list[MapDataset],
        valsets: list[MapDataset],
        *,
        first_epoch: int = 0,
        optim_config: dict[str, Any],
        grad_accu: int,
        num_workers: int,
        num_validation_batches: int,
    ):
        super().__init__()

        self.batches_per_epoch = 100 * len(trainsets)
        self.num_workers = num_workers

        if grad_accu < 1:
            raise ValueError(f"grad_accu must be at least 1, but is {grad_accu}")

        # epoch gets incremented at start of train, so set to one less of start value
        self.epoch = first_epoch - 1

        self.grad_accu = grad_accu

        self.lr_schedule = make_lr_schedule(
            self.batches_per_epoch // grad_accu,
            lr=optim_config["lr"],
            epochs=optim_config["epochs"],
            scheduler=optim_config["scheduler"],
        )

        match optim_config["optimizer"]:
            case "sgd":
                optim = partial(optax.sgd, momentum=0.9, nesterov=True)
            case "adamw":
                optim = optax.adamw
            case "radam":
                optim = optax.radam
            case optim:
                raise ValueError(f"invalid optimizer {optim}")

        if optim_config["grad_clip"] is not None:
            self.opt = optax.chain(
                optax.clip_by_global_norm(optim_config["grad_clip"]), optim(self.lr_schedule)
            )

            self._get_step = lambda: self.opt_state[1][-1].count  # type: ignore
        else:
            self.opt = optim(self.lr_schedule)

            self._get_step = lambda: self.opt_state[-1].count  # type: ignore

        self.opt_state = self.opt.init(eqx.filter((net, embedder), eqx.is_array_like))

        self.trainsets = trainsets
        self.valsets = valsets

        self.num_validation_batches = num_validation_batches

    @property
    def learning_rate(self) -> float:
        if isinstance(self.lr_schedule, float):
            return self.lr_schedule

        return float(self.lr_schedule(self._get_step()))  # type: ignore

    @overload
    def train(self, net: Net, embedder: InputEmbedder) -> tuple[Net, InputEmbedder]: ...

    @overload
    def train(self, net: Net, embedder: None) -> tuple[Net, None]: ...

    def train(self, net: Net, embedder: InputEmbedder | None) -> tuple[Net, InputEmbedder | None]:
        self.epoch += 1

        tqdm.write(f"Epoch {self.epoch: 3}: Training\n")

        auxs: list[dict[str, Any]] = []

        mixed_trainset = (
            MapDataset.mix(self.trainsets)
            .seed(self.epoch)
            .shuffle()[: self.batches_per_epoch]
            .to_iter_dataset(
                read_options=ReadOptions(
                    num_threads=self.num_workers, prefetch_buffer_size=2 * self.num_workers
                )
            )
        )

        for batches_np in itertools.batched(
            tqdm(mixed_trainset, total=self.batches_per_epoch, leave=False), self.grad_accu
        ):
            for batch in batches_np:
                batch.pop("name")

            batches: list[dict[str, Array]] = jt.map(jnp.asarray, list(batches_np))

            net, embedder, self.opt_state, aux = self.training_step(
                net, embedder, batches, self.opt, self.opt_state
            )

            auxs.append(aux)

        aux = transpose(auxs)

        if wandb.run is not None:
            data: dict[str, Any] = {"epoch": self.epoch}

            for key, value in aux.items():
                if key == "loss":
                    data["loss/train/mean"] = np.mean(aux["loss"])
                    data["loss/train/std"] = np.std(aux["loss"])
                else:
                    data[key] = np.mean(value)

            wandb.run.log(data)
        else:
            tqdm.write(f"Loss:        {np.mean(aux['loss']):.3}")
            tqdm.write(f"Grad Norm:   {np.mean(aux['grad_norm']):.3}")
            tqdm.write(f"Update Norm: {np.mean(aux['update_norm']):.3}")
            tqdm.write("")

        return net, embedder

    def validate(self, net: Net, embedder: InputEmbedder | None):
        tqdm.write(f"Epoch {self.epoch: 3}: Validation\n")

        @jax.jit
        def loss_jit(logits, labels):
            return jax.vmap(loss_fn)(logits, labels)

        all_losses = []

        for valset in self.valsets:
            dataset_name: str = valset[0]["name"]  # type: ignore

            metrices = []

            data_loader = (
                valset.seed(self.epoch)
                .shuffle()[: self.num_validation_batches]
                .to_iter_dataset(
                    ReadOptions(num_threads=self.num_workers, prefetch_buffer_size=self.num_workers)
                )
            )

            for batch in data_loader:
                images = jnp.asarray(batch["image"])
                labels = jnp.asarray(batch["label"])
                dataset_idx = jnp.asarray(batch["dataset_idx"])

                logits = self.net_forward(net, embedder, images, labels, dataset_idx)

                metrics = calc_metrics(logits, labels)

                loss = loss_jit(logits, labels)

                metrices.append(metrics)
                all_losses += list(loss)

            metrics = transpose(metrices)

            if wandb.run is not None:
                wandb.run.log(
                    {
                        f"{metric}/{dataset_name}": np.mean(value)
                        for metric, value in metrics.items()
                    },
                    commit=False,
                )
            else:
                tqdm.write(f"Dataset: {dataset_name}:")
                tqdm.write(f"    Dice score: {np.mean(metrics['dice']):.3f}")
                tqdm.write(f"    IoU score : {np.mean(metrics['iou']):.3f}")
                tqdm.write(f"    Hausdorff : {np.mean(metrics['hausdorff']):.3f}")
                tqdm.write("")

        if wandb.run is not None:
            wandb.run.log(
                {
                    "epoch": self.epoch,
                    "loss/validation/mean": np.mean(all_losses),
                    "loss/validation/std": np.std(all_losses),
                }
            )
        else:
            tqdm.write(f"Validation loss: {np.mean(all_losses):.3}")
            tqdm.write("")

    def make_plots(
        self,
        net: Net,
        embedder: InputEmbedder | None,
        testset: MapDataset | None,
        *,
        image_folder: Path,
    ):
        if image_folder.exists():
            shutil.rmtree(image_folder)

        image_folder.mkdir(parents=True)

        def make_images(batch: dict[str, Array], *, dataset_name: str, test: bool = False):
            images = batch["image"]
            labels = batch["label"]
            dataset_idx = batch["dataset_idx"]

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

        for valset in self.valsets:
            batch = jt.map(lambda x: jnp.asarray(x) if eqx.is_array(x) else x, valset[0])

            make_images(batch, dataset_name=batch["name"])

        if testset is None:
            return

        assert isinstance(testset, MapDataset)

        batch = jt.map(lambda x: jnp.asarray(x) if eqx.is_array(x) else x, testset[0])

        make_images(batch, dataset_name=batch["name"], test=True)

    @staticmethod
    def make_umap(embedder: InputEmbedder, datasets: list[MapDataset], image_folder: Path):
        if not image_folder.exists():
            image_folder.mkdir(parents=True)

        embedder_jit = jax.jit(jax.vmap(embedder, in_axes=(0, 0, None)))

        assert all(isinstance(dataset, MapDataset) for dataset in datasets)

        batches: list[dict[str, Any]] = [dataset[0] for dataset in datasets]  # type: ignore

        samples = {
            batch["name"]: jt.map(lambda x: jnp.asarray(x) if eqx.is_array(x) else x, batch)
            for batch in batches
        }

        embs = {
            name: embedder_jit(X["image"], X["label"], X["dataset_idx"])
            for name, X in samples.items()
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
