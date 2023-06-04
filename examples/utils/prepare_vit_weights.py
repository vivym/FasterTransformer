import argparse
from pathlib import Path

import numpy as np
from safetensors import safe_open


weight_name_suffix_mapping = {
    ".layer_norm1.bias": "_ln0_bias",
    ".layer_norm1.weight": "_ln0_scale",
    ".layer_norm2.bias": "_ln2_bias",
    ".layer_norm2.weight": "_ln2_scale",
    ".mlp.fc1.bias": "_ffn_inter_bias",
    ".mlp.fc1.weight": "_ffn_inter_kern",
    ".mlp.fc2.bias": "_ffn_o_bias",
    ".mlp.fc2.weight": "_ffn_o_kern",
    ".self_attn.projection.bias": "_att_o_bias",
    ".self_attn.projection.weight": "_att_o_kern",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_path", required=True, type=str)
    parser.add_argument("--out_path", required=True, type=str)
    args = parser.parse_args()

    out_path = Path(args.out_path)
    arr_type = np.float16

    with safe_open(args.hf_path, framework="np") as f:
        for key in f.keys():
            if key == "embeddings.class_embedding":
                file_name = "cls_token.npy"
            elif key.startswith("embeddings.patch_embedding."):
                suffix = key.split(".")[-1]
                if suffix == "weight":
                    suffix = "kernel"
                file_name = f"conv_{suffix}.npy"
            elif key == "embeddings.position_embedding":
                file_name = "pos_embed.npy"
            elif key == "post_layernorm.bias":
                file_name = "enc_ln_bias.npy"
            elif key == "post_layernorm.weight":
                file_name = "enc_ln_scale.npy"
            elif key.startswith("encoder.layers."):
                layer_id, *suffix = key[len("encoder.layers."):].split(".")
                suffix = "." + ".".join(suffix)
                file_name = f"l{layer_id}"

                if "self_attn.qkv" in key:
                    qkv = f.get_tensor(key).astype(arr_type)
                    q, k, v = np.split(qkv, 3, axis=0)
                    if qkv.ndim == 2:
                        q, k, v = q.T, k.T, v.T
                    q = np.ascontiguousarray(q)
                    k = np.ascontiguousarray(k)
                    v = np.ascontiguousarray(v)
                    suffix = key.split(".")[-1]
                    if suffix == "weight":
                        suffix = "kern"

                    for name, value in zip(["q", "k", "v"], [q, k, v]):
                        file_path = out_path / (file_name + f"_{name}_{suffix}.npy")
                        file_path.parent.mkdir(parents=True, exist_ok=True)
                        np.save(file_path, value)

                    continue
                else:
                    file_name += f"{weight_name_suffix_mapping[suffix]}.npy"
            else:
                raise ValueError(f"Unknown key: {key}")

            value = f.get_tensor(key).astype(arr_type)
            if value.ndim == 2:
                value = value.T
            value = np.ascontiguousarray(value)

            file_path = out_path / file_name
            file_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(file_path, value)


if __name__ == "__main__":
    main()
