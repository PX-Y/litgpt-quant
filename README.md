# Llama-3.2-1B Prox-QAT Training (LitGPT)

本项目基于 LitGPT（https://github.com/Lightning-AI/litgpt） 框架，面向 **Llama-3.2-1B** 模型，在 **Alpaca** 数据集上进行的量化训练实验代码。

---

## Project Layout

- **量化库**：`qat_prox2/`
- **主训练脚本**：`full_sens_quant2.py`  
  用于替代 LitGPT 原始的 full finetune 脚本（以便接入 dist_loss、gamma/dual 控制器、sensitivity 等模块）。

---

## Quantization Library (`qat_prox2/`)


### Parameter Filter
1. iter_named_quant_params`
* 遍历传入的 `model` 中的所有参数。
* 拦截敏感权重 (`exclude_substrings`)：`bias`、`norm`, `ln_`、`wte`, `wpe`，`lm_head`。

2. `QuantParamSelector.allow`
根据量化需求挑选特定参数。
例如目标是量化：`"mlp"`/ `"attn"`/ `"all_linear"`...定位目标参数。


### dist_loss 
计算模型当前权重与目标量化网格之间的距离损失。



### Controller
1. GammaController
量化项前系数 类似lambda调控量化项大小（单一的lambda修改起来比较麻烦）



2.DualController 
 控制拉格朗日乘子 lambda。

前期的`f-beta`不稳定，导致`lambda` 波动较大。考虑PI控制算法（用积分表示f-beta,保持lambd稳定）

### quant_ops.py 
1. `quantize_to_grid`：找格点离它最近的量化值

2. `selective_hard_quantize_model_inplace` 只把那些“距离网格点已经非常近（误差 ≤ atol）”的权重吸附成量化值，离得远的放过。

3. `hard_quantize_model_inplace` 不管误差多大，把所有选中的参数统统强行吸附到最近的量化网格点上。

4. `restore_model_from_backup` 吸附前的全精度备份


### quant_stats.py
统计当前权重的量化进度
返回：
1. `hit_rate`计算有多少比例的权重，其当前真实值与理想量化网格点的误差已经小于设定的容忍度 (`|w - q(w)| <= atol`)。
2. `sat_rate` 计算有多少比例的权重的绝对值太大超过了量化网格边界 


### sensitivity
 Hessian 矩阵大小将是 $N \times N$。太大了，用一阶梯度的平方，来近似代替二阶 Hessian 矩阵的对角线元素。

计算：
在每次反向传播（Backward）后，针对某一层参数 $w$，获取其当前 batch 的一阶梯度 $\nabla w$，并计算其平方的均值：
$$g^2 = \frac{1}{M} \sum_{i=1}^{M} (\nabla w_i)^2$$
由于单次算出的 $g^2$ 包含了巨大的随机噪声。因此引入了一个字典 (`self.state`) 来存储历史数据，并通过 $\beta=0.95$ 进行平滑融合：
$$S_{t} = \beta \cdot S_{t-1} + (1 - \beta) \cdot g^2_t$$

### config
   参数统计列表

---

## Usage

### 0) Prerequisites

1. 安装 LitGPT
2. 准备 Llama-3.2-1B 的 checkpoint：
   `checkpoints/meta-llama/Llama-3.2-1B/`



### 1) Minimal Training Script Example

下面示例展示如何从项目内调用 `setup(...)` 进行训练：

```python
from pathlib import Path

from litgpt.finetune.full_sens_quant2 import setup
from litgpt.args import TrainArgs, EvalArgs
from litgpt.data import Alpaca


def main():
    HERE = Path(__file__).resolve().parent          # .../litgpt_llama32_1b/litgpt
    PROJ = HERE.parent                              # .../litgpt_llama32_1b
    checkpoint_dir = (PROJ / "checkpoints/meta-llama/Llama-3.2-1B").resolve()

    setup(
        checkpoint_dir=checkpoint_dir,
        out_dir=Path("out/smoke/full_sens_quant"),
        devices=1,
        data=Alpaca(num_workers=0),

        train=TrainArgs(
            max_steps=3000,
            epochs=30,
            global_batch_size=128,
            micro_batch_size=4,
            lr_warmup_steps=100,
            log_interval=320,
            save_interval=None,
            max_seq_length=256,
        ),

        eval=EvalArgs(
            interval=100,
            max_iters=100,
            initial_validation=True,
            final_validation=True,
            max_new_tokens=50,
        ),
    )


if __name__ == "__main__":
    main()
