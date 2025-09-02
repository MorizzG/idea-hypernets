from jaxtyping import Array, Float, Integer
from typing import Any, Callable, ClassVar, overload

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
from tqdm import tqdm, trange
from umap import UMAP

from hyper_lap.embedder import InputEmbedder
from hyper_lap.serialisation import save_with_config_safetensors
from hyper_lap.training.loss import loss_fn
from hyper_lap.training.utils import make_lr_schedule, timer, to_PIL

from .metrics import calc_metrics


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
    NUM_VALIDATION_BATCHES: ClassVar[int] = 10

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

    model_name: str

    num_workers: int

    epoch: int

    lr_schedule: optax.Schedule

    opt: GradientTransformation
    opt_state: OptState

    grad_accu: int

    trainsets: list[MapDataset]
    valsets: list[MapDataset]

    _get_step: Callable[[], int]

    def __init__(
        self,
        net: Net,
        embedder: InputEmbedder | None,
        trainsets: list[MapDataset],
        valsets: list[MapDataset],
        *,
        model_name: str,
        first_epoch: int = 0,
        optim_config: dict[str, Any],
        grad_accu: int,
        num_workers: int,
    ):
        super().__init__()

        self.model_name = model_name

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

    @property
    def learning_rate(self) -> float:
        if isinstance(self.lr_schedule, float):
            return self.lr_schedule

        return float(self.lr_schedule(self._get_step()))  # type: ignore

    @overload
    def train(
        self, net: Net, embedder: InputEmbedder
    ) -> tuple[Net, InputEmbedder, dict[str, float]]: ...

    @overload
    def train(self, net: Net, embedder: None) -> tuple[Net, None, dict[str, float]]: ...

    def train(
        self, net: Net, embedder: InputEmbedder | None
    ) -> tuple[Net, InputEmbedder | None, dict[str, float]]:
        auxs: list[dict[str, Any]] = []

        mixed_trainset = (
            MapDataset.mix(self.trainsets)
            .seed(self.epoch)
            .shuffle()[: self.batches_per_epoch]
            .to_iter_dataset(
                read_options=ReadOptions(num_threads=self.num_workers, prefetch_buffer_size=200)
            )
        )

        for batches_np in itertools.batched(
            tqdm(mixed_trainset, total=self.batches_per_epoch), self.grad_accu
        ):
            for batch in batches_np:
                batch.pop("name")

            batches: list[dict[str, Array]] = jt.map(jnp.asarray, list(batches_np))

            net, embedder, self.opt_state, aux = self.training_step(
                net, embedder, batches, self.opt, self.opt_state
            )

            auxs.append(aux)

        aux = transpose(auxs)

        tqdm.write(f"Loss:        {np.mean(aux['loss']):.3}")
        tqdm.write(f"Grad Norm:   {np.mean(aux['grad_norm']):.3}")
        tqdm.write(f"Update Norm: {np.mean(aux['update_norm']):.3}")
        tqdm.write("")

        metrics = {
            "loss/train/mean": float(np.mean(aux["loss"])),
            "loss/train/std": float(np.std(aux["loss"])),
            "grad_norm": np.mean(aux["grad_norm"]),
            "update_norm": np.mean(aux["update_norm"]),
        }

        return net, embedder, metrics

    def validate(
        self,
        net: Net,
        embedder: InputEmbedder | None,
        *,
        num_batches: int | None = None,
    ) -> dict[str, float]:
        if num_batches is None:
            num_batches = self.NUM_VALIDATION_BATCHES

        @jax.jit
        def loss_jit(logits, labels):
            return jax.vmap(loss_fn)(logits, labels)

        all_losses = []
        metrics = {}

        valsets: dict[str, MapDataset] = {
            unwrap(valset[0])["name"]: valset.seed(self.epoch).shuffle()[:num_batches]
            for valset in self.valsets
        }

        it = iter(
            MapDataset.concatenate(list(valsets.values())).to_iter_dataset(
                ReadOptions(
                    num_threads=self.num_workers, prefetch_buffer_size=len(valsets) * num_batches
                )
            )
        )

        for dataset_name in valsets.keys():
            dataset_metrices = []

            for _ in trange(num_batches):
                batch = next(it)

                assert batch["name"] == dataset_name

                images = jnp.asarray(batch["image"])
                labels = jnp.asarray(batch["label"])
                dataset_idx = jnp.asarray(batch["dataset_idx"])

                logits = self.net_forward(net, embedder, images, labels, dataset_idx)

                dataset_metrics = calc_metrics(logits, labels)

                loss = loss_jit(logits, labels)

                dataset_metrices.append(dataset_metrics)
                all_losses += list(loss)

            dataset_metrics = transpose(dataset_metrices)

            metrics |= {
                f"{metric}/{dataset_name}": float(np.mean(value))
                for metric, value in dataset_metrics.items()
            }

            tqdm.write(f"Dataset: {dataset_name}:")
            tqdm.write(f"    Dice score: {np.mean(dataset_metrics['dice']):.3f}")
            tqdm.write(f"    IoU score : {np.mean(dataset_metrics['iou']):.3f}")
            tqdm.write(f"    Hausdorff : {np.mean(dataset_metrics['hausdorff']):.3f}")
            tqdm.write("")

        tqdm.write(f"Validation loss: {np.mean(all_losses):.3}")
        tqdm.write("")

        metrics["loss/validation/mean"] = float(np.mean(all_losses))
        metrics["loss/validation/std"] = float(np.std(all_losses))

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

        for _ in trange(num_epochs):
            self.epoch += 1

            learning_rate = self.learning_rate

            tqdm.write(f"Epoch {self.epoch: 3}: Training\n")

            with timer("train", use_tqdm=True):
                net, embedder, train_metrics = self.train(net, embedder)

            tqdm.write(f"Epoch {self.epoch: 3}: Validation\n")

            with timer("validate", use_tqdm=True):
                val_metrics = self.validate(net, embedder)

            metrics = {
                "epoch": self.epoch,
                "learning_rate": learning_rate,
            }

            metrics |= train_metrics
            metrics |= val_metrics

            if wandb.run is not None:
                wandb.run.log(metrics)

            if val_metrics["loss/validation/mean"] < best_val_loss:
                best_epoch = self.epoch
                best_val_loss = val_metrics["loss/validation/mean"]
                no_improvement_counter = 0

                best_model = jax.device_get((net, embedder))
            else:
                no_improvement_counter += 1

                if no_improvement_counter == 20:
                    tqdm.write(
                        "Stopping early after no validation improvement for"
                        f" {no_improvement_counter} epochs"
                    )

                    break

        net, embedder = jt.map(lambda x: jax.device_put(x) if eqx.is_array(x) else x, best_model)

        tqdm.write("Best model validation:\n")

        with timer("best_validation", use_tqdm=True):
            best_metrics = self.validate(net, embedder, num_batches=100)

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

        return net, embedder

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

            image = images[1]
            label = labels[1]

            pred = jnp.argmax(logits[1], axis=0)

            fig, axs = plt.subplots(ncols=3)

            axs[0].imshow(image.mean(axis=0), cmap="gray")
            axs[1].imshow(label, cmap="gray")
            axs[2].imshow(pred, cmap="gray")

            fig.savefig(image_folder / f"{dataset_name}{'_test' if test else ''}.pdf")

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
                    del wandb.run.summary[f"images/train/{dataset_name}"]
                else:
                    wandb.run.log({f"images/test/{dataset_name}": image})
                    del wandb.run.summary[f"images/test/{dataset_name}"]

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

        batches: list[dict[str, Any]] = [unwrap(dataset[0]) for dataset in datasets]

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
