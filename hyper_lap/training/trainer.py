from jaxtyping import Array
from typing import Any, Callable, ClassVar, Literal, overload

import math
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
from tqdm import tqdm, trange
from umap import UMAP

from hyper_lap.embedder import InputEmbedder
from hyper_lap.serialisation import save_with_config_safetensors
from hyper_lap.training.loss import ce_loss_fn, focal_loss_fn, hybrid_loss_fn
from hyper_lap.training.metrics import calc_metrics
from hyper_lap.training.utils import make_lr_schedule, timer, to_PIL


def unwrap[T](x: T | None) -> T:
    assert x is not None

    return x


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
    NUM_VALIDATION_BATCHES: ClassVar[int] = 20

    @staticmethod
    @eqx.filter_jit
    def net_forward(
        net: Net,
        embedder: InputEmbedder | None,
        batch: dict[str, Array],
    ) -> Array:
        if embedder is not None:
            example_image = batch["example_image"]
            example_label = batch["example_label"]
            dataset_idx = batch["dataset_idx"]

            input_emb = embedder(example_image, example_label, dataset_idx)
        else:
            input_emb = None

        images = batch["image"]

        return jax.vmap(net, in_axes=(0, None))(images, input_emb)

    @staticmethod
    @eqx.filter_jit
    def training_step(
        net: Net,
        embedder: InputEmbedder | None,
        batch: dict[str, Array],
        opt: optax.GradientTransformation,
        opt_state: OptState,
        loss_fn: Callable[[Array, Array], Array],
    ) -> tuple[Net, InputEmbedder | None, OptState, dict[str, Array]]:
        net_embedder = (net, embedder)

        @eqx.filter_value_and_grad
        def grad_fn(
            net_embedder: tuple[Net, InputEmbedder | None],
            batch: dict[str, Array],
        ) -> Array:
            (net, embedder) = net_embedder

            logits = Trainer.net_forward(net, embedder, batch)

            labels = batch["label"]

            loss = jax.vmap(loss_fn)(logits, labels).mean()

            return loss

        loss, grads = grad_fn(net_embedder, batch)

        updates, opt_state = opt.update(grads, opt_state, net_embedder)  # pyright: ignore

        (net, embedder) = eqx.apply_updates(net_embedder, updates)

        aux: dict[str, Array] = {
            "loss": loss,
            "grad_norm": global_norm(grads),
            "update_norm": global_norm(updates),
        }  # pyright: ignore

        return net, embedder, opt_state, aux

    model_name: str

    num_workers: int
    total_batches_per_epoch: int

    epoch: int

    loss_fn: Callable[[Array, Array], Array]

    lr_schedule: optax.Schedule

    opt: GradientTransformation
    opt_state: OptState

    trainsets: list[MapDataset]
    valsets: list[MapDataset]
    oodsets: list[MapDataset]

    _get_step: Callable[[], int]

    def __init__(
        self,
        net: Net,
        embedder: InputEmbedder | None,
        trainsets: list[MapDataset],
        valsets: list[MapDataset],
        oodsets: list[MapDataset],
        *,
        loss_fn: Literal["CE", "focal", "hybrid"],
        model_name: str,
        first_epoch: int = 1,
        optim_config: dict[str, Any],
        num_workers: int,
        batches_per_epoch: int,
    ):
        super().__init__()

        self.model_name = model_name

        self.total_batches_per_epoch = batches_per_epoch * len(trainsets)
        self.num_workers = num_workers

        # epoch gets incremented at start of train, so set to one less of start value
        self.epoch = first_epoch - 1

        match loss_fn:
            case "CE":
                self.loss_fn = ce_loss_fn
            case "focal":
                self.loss_fn = focal_loss_fn
            case "hybrid":
                self.loss_fn = hybrid_loss_fn

        self.lr_schedule = make_lr_schedule(
            self.total_batches_per_epoch,
            lr=optim_config["lr"],
            epochs=optim_config["epochs"] or 1,
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

            self._get_step = lambda: self.opt_state[1][-1].count  # pyright: ignore
        else:
            self.opt = optim(self.lr_schedule)

            self._get_step = lambda: self.opt_state[-1].count  # pyright: ignore

        self.opt_state = self.opt.init(eqx.filter((net, embedder), eqx.is_array_like))

        self.trainsets = trainsets
        self.valsets = valsets
        self.oodsets = oodsets

    @property
    def learning_rate(self) -> float:
        if isinstance(self.lr_schedule, float):
            return self.lr_schedule

        return float(self.lr_schedule(self._get_step()))

    @overload
    def train(
        self, net: Net, embedder: InputEmbedder
    ) -> tuple[Net, InputEmbedder, dict[str, float]]: ...

    @overload
    def train(self, net: Net, embedder: None) -> tuple[Net, None, dict[str, float]]: ...

    def train(
        self, net: Net, embedder: InputEmbedder | None
    ) -> tuple[Net, InputEmbedder | None, dict[str, float]]:
        auxs: list[dict[str, Array]] = []

        mixed_trainset = (
            MapDataset.mix(self.trainsets)
            .seed(self.epoch)
            .shuffle()[: self.total_batches_per_epoch]
            .to_iter_dataset(
                ReadOptions(num_threads=self.num_workers, prefetch_buffer_size=2 * self.num_workers)
            )
        )

        for batch_np in tqdm(mixed_trainset, total=self.total_batches_per_epoch):
            batch_np.pop("name")

            batch: dict[str, Array] = jt.map(jnp.asarray, batch_np)

            net, embedder, self.opt_state, aux = self.training_step(
                net, embedder, batch, self.opt, self.opt_state, self.loss_fn
            )

            auxs.append(aux)

        aux = transpose(auxs)

        loss_mean = float(np.mean(aux["loss"]))
        loss_std = float(np.std(aux["loss"]))

        grad_norm = float(np.mean(aux["grad_norm"]))
        update_norm = float(np.mean(aux["update_norm"]))

        tqdm.write(f"Loss:        {loss_mean:.3}")
        tqdm.write(f"Grad Norm:   {grad_norm:.3}")
        tqdm.write(f"Update Norm: {update_norm:.3}")
        tqdm.write("")

        metrics = {
            "loss/train/mean": loss_mean,
            "loss/train/std": loss_std,
            "grad_norm": grad_norm,
            "update_norm": update_norm,
        }

        return net, embedder, metrics

    def validate(
        self,
        net: Net,
        embedder: InputEmbedder | None,
        *,
        seed: int,
        num_batches: int | None = None,
    ) -> dict[str, float]:
        if num_batches is None:
            num_batches = self.NUM_VALIDATION_BATCHES

        @partial(jax.jit, static_argnums=(2,))
        def loss_jit(logits, labels, loss_fn):
            return jax.vmap(loss_fn)(logits, labels)

        val_losses = []
        metrics = {}

        valsets: dict[str, MapDataset] = {
            unwrap(valset[0])["name"]: valset.seed(seed).shuffle()[:num_batches]
            for valset in self.valsets
        }

        oodsets: dict[str, MapDataset] = {
            unwrap(valset[0])["name"]: valset.seed(seed).shuffle()[:num_batches]
            for valset in self.oodsets
        }

        it = iter(
            MapDataset.concatenate(list(valsets.values()) + list(oodsets.values())).to_iter_dataset(
                ReadOptions(num_threads=self.num_workers, prefetch_buffer_size=self.num_workers)
            )
        )

        def validate_dataset(dataset_name):
            losses = []
            first_losses = []
            dataset_metrices = []

            for _ in trange(num_batches):
                batch = next(it)

                assert batch.pop("name") == dataset_name

                labels = jnp.asarray(batch["label"])

                logits = self.net_forward(net, embedder, batch)

                dataset_metrics = calc_metrics(logits, labels)

                loss = loss_jit(logits, labels, self.loss_fn)

                losses += [x.item() for x in loss]
                first_losses.append(loss[0].item())

                dataset_metrices.append(dataset_metrics)

            dataset_metrics = transpose(dataset_metrices)

            return losses, first_losses, dataset_metrics

        for dataset_name in tqdm(valsets.keys()):
            losses, first_losses, dataset_metrics = validate_dataset(dataset_name)

            val_losses += losses

            metrics |= {
                f"{metric}/{dataset_name}": float(np.mean(value))
                for metric, value in dataset_metrics.items()
            }

            tqdm.write(f"Dataset: {dataset_name}:")
            tqdm.write(f"    Loss      : {np.mean(losses):.3}")
            tqdm.write(f"    First Loss: {np.mean(first_losses):.3}")
            tqdm.write(f"    Dice score: {np.mean(dataset_metrics['dice']):.3f}")
            tqdm.write(f"    IoU score : {np.mean(dataset_metrics['iou']):.3f}")
            tqdm.write(f"    Hausdorff : {np.mean(dataset_metrics['hausdorff']):.3f}")
            tqdm.write("")

        tqdm.write(f"Validation loss: {np.mean(val_losses):.3}")
        tqdm.write("")

        metrics["loss/validation/mean"] = float(np.mean(val_losses))
        metrics["loss/validation/std"] = float(np.std(val_losses))

        if self.oodsets:
            ood_losses = []

            for dataset_name in tqdm(oodsets.keys()):
                losses, first_losses, dataset_metrics = validate_dataset(dataset_name)

                ood_losses += losses

                metrics |= {
                    f"{metric}/{dataset_name}": float(np.mean(value))
                    for metric, value in dataset_metrics.items()
                }

                tqdm.write(f"Dataset: {dataset_name}:")
                tqdm.write(f"    Loss      : {np.mean(losses):.3}")
                tqdm.write(f"    First Loss: {np.mean(first_losses):.3}")
                tqdm.write(f"    Dice score: {np.mean(dataset_metrics['dice']):.3f}")
                tqdm.write(f"    IoU score : {np.mean(dataset_metrics['iou']):.3f}")
                tqdm.write(f"    Hausdorff : {np.mean(dataset_metrics['hausdorff']):.3f}")
                tqdm.write("")

            tqdm.write(f"OOD loss: {np.mean(ood_losses):.3}")
            tqdm.write("")

            metrics["loss/ood/mean"] = float(np.mean(ood_losses))
            metrics["loss/ood/std"] = float(np.std(ood_losses))

        return metrics

    @overload
    def run(
        self, net: Net, embedder: InputEmbedder, num_epochs: int, config: Any
    ) -> tuple[Net, InputEmbedder]: ...

    @overload
    def run(self, net: Net, embedder: None, num_epochs: int, config: Any) -> tuple[Net, None]: ...

    def run(
        self,
        net: Net,
        embedder: InputEmbedder | None,
        num_epochs: int,
        config: Any,
    ) -> tuple[Net, InputEmbedder | None]:
        best_epoch = self.epoch
        best_val_loss = np.inf
        no_improvement_counter = 0

        best_model = jax.device_get((net, embedder))

        val_interval = 3

        for i in trange(1, num_epochs + 1):
            self.epoch += 1

            metrics = {
                "epoch": self.epoch,
                "learning_rate": self.learning_rate,
            }

            tqdm.write(f"Epoch {self.epoch: 3}: Training\n")

            with timer("train", use_tqdm=True):
                net, embedder, train_metrics = self.train(net, embedder)

            metrics |= train_metrics

            # validate every val_interval epochs and after the first and final epoch
            if i % val_interval == 0 or i == 1 or i == num_epochs:
                tqdm.write(f"Epoch {self.epoch: 3}: Validation\n")

                with timer("validate", use_tqdm=True):
                    val_metrics = self.validate(net, embedder, seed=self.epoch)

                metrics |= val_metrics

                if val_metrics["loss/validation/mean"] < best_val_loss:
                    best_epoch = self.epoch
                    best_val_loss = val_metrics["loss/validation/mean"]

                    best_model = jax.device_get((net, embedder))

                    no_improvement_counter = 0
                else:
                    no_improvement_counter += 1

                    # no improvement for 3 validations and 20% of total epochs: early stop
                    if (
                        no_improvement_counter >= 3
                        and 5 * val_interval * no_improvement_counter >= num_epochs
                    ):
                        tqdm.write(
                            "Stopping early after no validation improvement for"
                            f" {val_interval * no_improvement_counter} epochs"
                        )

                        if wandb.run is not None:
                            wandb.run.log(metrics)

                        break

            if wandb.run is not None:
                wandb.run.log(metrics)

        best_net, best_embedder = jt.map(
            lambda x: jax.device_put(x) if eqx.is_array(x) else x, best_model
        )

        tqdm.write("Best model validation:\n")

        with timer("best_validation", use_tqdm=True):
            best_metrics = self.validate(best_net, best_embedder, seed=0, num_batches=100)

        if wandb.run is not None:
            for key in wandb.run.summary.keys():
                del wandb.run.summary[key]

            wandb.run.summary["epoch"] = best_epoch
            wandb.run.summary.update(best_metrics)

        model_path = Path(f"./models/{self.model_name}.safetensors")

        model_path.parent.mkdir(exist_ok=True)

        save_with_config_safetensors(model_path, config, best_model)

        if wandb.run is not None:
            model_artifact = wandb.Artifact(self.model_name, type="model")

            model_artifact.add_file(str(model_path.with_suffix(".json")), overwrite=True)
            model_artifact.add_file(str(model_path.with_suffix(".safetensors")), overwrite=True)

            wandb.run.log_artifact(model_artifact)

        return best_net, best_embedder

    def make_plots(
        self,
        net: Net,
        embedder: InputEmbedder | None,
        *,
        image_folder: Path,
    ):
        print("Making plots")

        if image_folder.exists():
            shutil.rmtree(image_folder)

        image_folder.mkdir(parents=True)

        def make_images(batch: dict[str, Any], ood: bool = False):
            dataset_name = batch.pop("name")

            batch["image"] = batch["image"][:1]
            batch["label"] = batch["label"][:1]

            logits = self.net_forward(net, embedder, batch)

            image = batch["image"][0]
            label = batch["label"][0]

            pred = jnp.argmax(logits[0], axis=0)

            metrics = calc_metrics(logits, batch["label"])

            tqdm.write(f"Dataset: {dataset_name}:")
            tqdm.write(f"    Dice score: {metrics['dice'].item():.3f}")
            tqdm.write(f"    IoU score : {metrics['iou'].item():.3f}")
            tqdm.write(f"    Hausdorff : {metrics['hausdorff'].item():.3f}")
            tqdm.write("")

            fig, axs = plt.subplots(ncols=3)

            axs[0].imshow(image.mean(axis=0), cmap="gray")
            axs[1].imshow(label, cmap="gray")
            axs[2].imshow(pred, cmap="gray")

            fig.savefig(image_folder / f"{dataset_name}{'_ood' if ood else ''}.pdf")

            if wandb.run is not None:
                class_labels = {
                    # 0: "background",
                    1: "foreground",
                }

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

                if not ood:
                    wandb.run.log({f"images/val/{dataset_name}": image})
                    del wandb.run.summary[f"images/val/{dataset_name}"]
                else:
                    wandb.run.log({f"images/ood/{dataset_name}": image})
                    del wandb.run.summary[f"images/ood/{dataset_name}"]

        for valset in self.valsets:
            batch = jt.map(lambda x: jnp.asarray(x) if eqx.is_array(x) else x, valset[0])

            make_images(batch)

        for oodset in self.oodsets:
            batch = jt.map(lambda x: jnp.asarray(x) if eqx.is_array(x) else x, oodset[0])

            make_images(batch, ood=True)

    def make_umap(self, embedder: InputEmbedder, image_folder: Path):
        print("Making UMAP")

        if not image_folder.exists():
            image_folder.mkdir(parents=True)

        with timer("UMAP fitting"):
            embedder_jit = eqx.filter_jit(eqx.filter_vmap(embedder, in_axes=(0, 0, None)))

            batches: list[dict[str, Any]] = [
                unwrap(dataset[0]) for dataset in self.valsets + self.oodsets
            ]

            samples = {
                batch["name"]: jt.map(lambda x: jnp.asarray(x) if eqx.is_array(x) else x, batch)
                for batch in batches
            }

            embs = {
                name: jnp.concat(
                    [
                        embedder_jit(
                            X["image"][32 * i : 32 * (i + 1)],
                            X["label"][32 * i : 32 * (i + 1)],
                            X["dataset_idx"],
                        )
                        for i in range(math.ceil(X["image"].shape[0] / 32))
                    ],
                    axis=0,
                )
                for name, X in samples.items()
            }

            umap = UMAP()
            umap.fit(jnp.concat([embs for embs in embs.values()]))

            projs: dict[str, Array] = {name: umap.transform(embs) for name, embs in embs.items()}  # pyright: ignore

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
