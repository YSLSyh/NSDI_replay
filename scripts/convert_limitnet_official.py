"""把官方可信 pickle 权重转换为无需官方依赖即可安全读取的 LimitNet state_dict。"""

from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
import pickle
import types

import torch
from torch import nn

from replay.baselines.limitnet import LimitNetCore


class _CheckpointModule(nn.Module):
    """只承接旧 pickle 的属性，不运行官方模型或分类器依赖。"""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__()


class _CheckpointDict(dict):
    pass


class _CheckpointUnpickler(pickle.Unpickler):
    """把旧文件中的外部类映射为仅用于读取参数的空壳。"""

    def find_class(self, module: str, name: str) -> object:
        if module == "models.model" and name in {
            "LimitNet", "Encoder", "Decoder", "SalDecoder"
        }:
            return _CheckpointModule
        if module.startswith(("torchmetrics.", "efficientnet_pytorch.")):
            if name in {"jit_distributed_available", "dim_zero_sum"}:
                return _unused_checkpoint_function
            return _CheckpointModule
        if module == "lightning.fabric.utilities.data" and name == "AttributeDict":
            return _CheckpointDict
        if module == "torchvision.transforms.transforms" and name == "Normalize":
            return _CheckpointModule
        return super().find_class(module, name)


def _unused_checkpoint_function(*args: object, **kwargs: object) -> None:
    return None


_CHECKPOINT_PICKLE = types.ModuleType("limitnet_checkpoint_pickle")
_CHECKPOINT_PICKLE.Unpickler = _CheckpointUnpickler
_CHECKPOINT_PICKLE.Pickler = pickle.Pickler
_CHECKPOINT_PICKLE.load = pickle.load
_CHECKPOINT_PICKLE.loads = pickle.loads
_CHECKPOINT_PICKLE.dump = pickle.dump
_CHECKPOINT_PICKLE.dumps = pickle.dumps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True,
                        help="仅可使用从官方项目获取且已核对来源的 pickle 权重")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha256", help="可选：预期 SHA-256，转换前核对")
    parser.add_argument("--dataset", choices=("imagenet", "cifar100"), required=True)
    args = parser.parse_args()
    source = args.input.resolve()
    repo = args.official_repo.resolve()
    if not (repo / "models" / "model.py").is_file():
        raise FileNotFoundError(f"不是官方 LimitNet 仓库：{repo}")
    digest = sha256(source.read_bytes()).hexdigest()
    if args.expected_sha256 and digest.lower() != args.expected_sha256.lower():
        raise ValueError("官方权重 SHA-256 不匹配")
    # 官方发布的是 torch.save 整个 Python 对象。仅转换核实哈希的官方文件；
    # 空壳类型使读取编码器参数时不必安装其无关的训练依赖。
    value = torch.load(source, map_location="cpu", weights_only=False,
                       pickle_module=_CHECKPOINT_PICKLE)
    state = value.get("state_dict", value) if isinstance(value, dict) else value.state_dict()
    core_state = {key: tensor.detach().cpu() for key, tensor in state.items()
                  if key.startswith(("encoder.", "sal_decoder.", "decoder."))}
    core = LimitNetCore()
    core.load_state_dict(core_state, strict=True)
    metadata = {
        "source": f"official_limitnet_{args.dataset}_pretrained",
        "official_repo": "https://github.com/ds-kiel/LimitNet",
        "official_checkpoint_sha256": digest,
        "conversion": "full_pickle_to_pure_state_dict",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": core_state, "metadata": metadata}, args.output)
    print(f"converted={args.output} sha256={digest} source={metadata['source']}")


if __name__ == "__main__":
    main()
