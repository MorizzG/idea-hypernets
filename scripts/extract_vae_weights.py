from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from safetensors.torch import save_file
from torch import Tensor


def get_fixed_state_dict(vae: AutoencoderKL):
    state_dict = vae.state_dict()

    fixed_state_dict = {}

    for key, value in state_dict.items():
        # equinox wants bias as (c, 1, 1)
        if key.endswith(".bias") and "conv" in key and "conv_norm_out" not in key:
            value = value[:, None, None]

        # replace downsamplers by single conv
        if "downsamplers" in key:
            key = key.replace("downsamplers.0.conv", "downsample")

        # replace upsamplers by single conv
        if "upsamplers" in key:
            key = key.replace("upsamplers.0.conv", "upsample")

        # replace to_out by single linear in attentions
        if "to_out" in key:
            key = key.replace("to_out.0", "to_out")

        fixed_state_dict[key] = value

    return fixed_state_dict


def save_and_get(vae: AutoencoderKL) -> dict[str, Tensor]:
    state_dict = get_fixed_state_dict(vae)

    save_file(state_dict, "models/vae.safetensors")

    return state_dict


if __name__ == "__main__":
    vae: AutoencoderKL = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")

    state_dict = save_and_get(vae)
