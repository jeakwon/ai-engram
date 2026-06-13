# Examples

Jupyter notebooks reproducing the paper's main results across modalities.

> **Note:** these notebooks accompany the paper and may use an **earlier API**. For the
> current `engram` package (0.6.0+) — `Statistics`, `EngramResult`, and pluggable
> `scale=` editing — follow the
> [Quickstart](https://jeakwon.github.io/ai-engram/quickstart/) and
> [Guide](https://jeakwon.github.io/ai-engram/guide/). The maintained, current-API TOFU
> reproduction lives in [`tests/`](../tests) (`test_tofu_unlearn.py`,
> `test_tofu_evaluate.py`, gated by `ENGRAM_RUN_TOFU` / `ENGRAM_RUN_TOFU_EVALUATE`).

| notebook | what |
|---|---|
| `llm_tofu.ipynb` | TOFU forget10 unlearning on Llama-3.2-1B (the package's primary target) |
| `mlp_mnist.ipynb` | MLP on MNIST |
| `resnet18_cifar10.ipynb` / `resnet18_cifar100.ipynb` | ResNet-18 on CIFAR-10 / CIFAR-100 |
| `vit_imagenet1k.ipynb` | ViT on ImageNet-1k |
| `wae_celeba.ipynb` | WAE on CelebA |

The current `engram` package covers `nn.Linear` and GPT-2 `Conv1D` (and fused-MoE via
`engram.moe`); the vision/`Conv2d` notebooks above use the broader research code from the
paper, not the minimal published package.
