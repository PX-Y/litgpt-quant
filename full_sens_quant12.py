# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
from litgpt.qat_prox2.config import QATConfig
from litgpt.qat_prox2 import (
    GammaController, DualController, SensitivityEMA,
    QuantParamSelector, iter_named_quant_params, quantize_to_grid,
    compute_dist_loss, compute_quantization_rate_fast,
)
import dataclasses
import math

import os
import time
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Literal, Optional, Tuple, Union
from litgpt.qat_prox2.quant_ops import hard_quantize_model_inplace, restore_model_from_backup


import math
import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy
from torch.utils.data import ConcatDataset, DataLoader
from torchmetrics import RunningMean

from litgpt.args import EvalArgs, LogArgs, TrainArgs
from litgpt.data import Alpaca, DataModule
from litgpt.generate.base import generate
from litgpt.model import GPT, Block, Config
from litgpt.parser_config import save_hyperparameters
from litgpt.prompts import save_prompt_style
from litgpt.tokenizer import Tokenizer
from litgpt.types import LoggerChoice
from litgpt.utils import (
    CycleIterator,
    auto_download_checkpoint,
    check_nvlink_connectivity,
    check_valid_checkpoint_dir,
    choose_logger,
    chunked_cross_entropy,
    copy_config_files,
    create_finetuning_performance_report,
    find_resume_path,
    get_default_supported_precision,
    init_out_dir,
    instantiate_torch_optimizer,
    load_checkpoint,
    num_parameters,
    parse_devices,
    select_sft_generate_example,
)


def setup(
    checkpoint_dir: Path,
    out_dir: Path = Path("out/finetune/full"),
    precision: Optional[str] = None,
    devices: Union[int, str] = 1,
    num_nodes: int = 1,
    resume: Union[bool, Literal["auto"], Path] = False,
    data: Optional[DataModule] = None,
    train: TrainArgs = TrainArgs(
        save_interval=1000,
        log_interval=1,
        global_batch_size=16,
        micro_batch_size=1,
        lr_warmup_steps=100,
        epochs=5,
        max_seq_length=None,
    ),
    eval: EvalArgs = EvalArgs(interval=600, max_new_tokens=100, max_iters=100),
    log: LogArgs = LogArgs(),
    optimizer: Union[str, Dict] = "AdamW",
    logger_name: LoggerChoice = "csv",
    seed: int = 1337,
    access_token: Optional[str] = None,
) -> None:
    """Finetune a model.

    Arguments:
        checkpoint_dir: The path to the base model's checkpoint directory to load for finetuning.
        out_dir: Directory in which to save checkpoints and logs. If running in a Lightning Studio Job, look for it in
            /teamspace/jobs/<job-name>/share.
        precision: The precision to use for finetuning. Possible choices: "bf16-true", "bf16-mixed", "32-true".
        devices: How many devices/GPUs to use
        num_nodes: How many nodes the code is being run on.
        resume: Path to a checkpoint directory to resume from in case training was interrupted, or ``True`` to resume
            from the latest checkpoint in ``out_dir``. An error will be raised if no checkpoint is found. Passing
            ``'auto'`` will resume from the latest checkpoint but not error if no checkpoint exists.
        data: Data-related arguments. If not provided, the default is ``litgpt.data.Alpaca``.
        train: Training-related arguments. See ``litgpt.args.TrainArgs`` for details.
        eval: Evaluation-related arguments. See ``litgpt.args.EvalArgs`` for details.
        optimizer: An optimizer name (such as "AdamW") or config.
        logger_name: The name of the logger to send metrics to.
        seed: The random seed to use for reproducibility.
        access_token: Optional API token to access models with restrictions.
    """
    checkpoint_dir = auto_download_checkpoint(model_name=checkpoint_dir, access_token=access_token)
    pprint(locals())
    data = Alpaca() if data is None else data
    devices = parse_devices(devices)
    out_dir = init_out_dir(out_dir)

    check_valid_checkpoint_dir(checkpoint_dir)
    config = Config.from_file(checkpoint_dir / "model_config.yaml")

    precision = precision or get_default_supported_precision(training=True)
    logger = choose_logger(
        logger_name,
        out_dir,
        name=f"finetune-{config.name}",
        resume=bool(resume),
        log_interval=train.log_interval,
        log_args=dataclasses.asdict(log),
    )

    if devices * num_nodes > 1:
        strategy = FSDPStrategy(
            auto_wrap_policy={Block},
            activation_checkpointing_policy={Block},
            state_dict_type="full",
            limit_all_gathers=True,
            cpu_offload=False,
        )
    else:
        strategy = "auto"

    fabric = L.Fabric(devices=devices, num_nodes=num_nodes, strategy=strategy, precision=precision, loggers=logger)

    if torch.cuda.is_available() and devices > 1:
        check_nvlink_connectivity(fabric)

    fabric.launch(main, devices, resume, seed, config, data, checkpoint_dir, out_dir, train, eval, optimizer, num_nodes)


@torch.no_grad()
def validate_quantized(
    fabric: L.Fabric,
    model: GPT,
    dataloader: DataLoader,
    eval: EvalArgs,
    qat: QATConfig,
    q_selector: QuantParamSelector,
    verbose: bool = True,
) -> torch.Tensor:
    if verbose:
        fabric.print("Validating quantized ...")

    model.eval()
    losses = torch.zeros(min(len(dataloader), eval.max_iters), device=fabric.device)

    for k, batch in enumerate(dataloader):
        if k >= eval.max_iters:
            break

        input_ids, targets = batch["input_ids"], batch["labels"]

        backup_q = None
        if qat.enabled:
            backup_q = hard_quantize_model_inplace(
                model,
                n_bits_w=qat.n_bits_w,
                step=qat.step_w,
                selector=q_selector,
                include_substrings=qat.include_substrings,
                exclude_substrings=qat.exclude_substrings,
            )

        logits = model(input_ids)
        losses[k] = chunked_cross_entropy(logits[..., :-1, :], targets[..., 1:], chunk_size=0)

        if backup_q is not None:
            restore_model_from_backup(model, backup_q)

    val_loss_q = losses.mean()
    model.train()
    return val_loss_q




def compute_selected_grad_and_adamw_delta_norm(
    model,
    optimizer,
    selector,
    include_substrings=None,
    exclude_substrings=None,
):
    """
    Returns:
        grad_l2:   L2 norm of grads that AdamW is about to use
        adam_dx_l2: L2 norm of the actual AdamW parameter update for the selected params
    """
    grad_sq = 0.0
    adam_dx_sq = 0.0

    group_map = {}
    for group in optimizer.param_groups:
        for p in group["params"]:
            group_map[id(p)] = group

    with torch.no_grad():
        for name, p in iter_named_quant_params(
            model,
            selector=selector,
            include_substrings=include_substrings,
            exclude_substrings=exclude_substrings,
        ):
            if p.grad is None:
                continue

            g = p.grad.detach().float()
            grad_sq += torch.sum(g * g).item()

            group = group_map[id(p)]
            lr = float(group["lr"])
            beta1, beta2 = group["betas"]
            eps = float(group["eps"])
            wd = float(group.get("weight_decay", 0.0))

            state = optimizer.state[p]

            exp_avg = state.get("exp_avg", None)
            exp_avg_sq = state.get("exp_avg_sq", None)

            if exp_avg is None:
                exp_avg = torch.zeros_like(g)
            else:
                exp_avg = exp_avg.detach().float()

            if exp_avg_sq is None:
                exp_avg_sq = torch.zeros_like(g)
            else:
                exp_avg_sq = exp_avg_sq.detach().float()

            step_prev = state.get("step", 0)
            if torch.is_tensor(step_prev):
                step_prev = step_prev.item()
            step_t = int(step_prev) + 1

            # AdamW state after this step
            exp_avg_next = exp_avg.mul(beta1).add(g, alpha=1.0 - beta1)
            exp_avg_sq_next = exp_avg_sq.mul(beta2).addcmul(g, g, value=1.0 - beta2)

            bias_correction1 = 1.0 - beta1 ** step_t
            bias_correction2 = 1.0 - beta2 ** step_t

            denom = exp_avg_sq_next.sqrt().div(math.sqrt(bias_correction2)).add_(eps)
            adam_term = exp_avg_next.div(denom).mul(lr / bias_correction1)

            # AdamW is decoupled weight decay:
            # p <- p - lr*wd*p - adam_term
            total_delta = adam_term
            if wd != 0.0:
                total_delta = total_delta + p.detach().float() * (lr * wd)

            adam_dx_sq += torch.sum(total_delta * total_delta).item()

    return math.sqrt(grad_sq), math.sqrt(adam_dx_sq)


def main(
    fabric: L.Fabric,
    devices: int,
    resume: Union[bool, Literal["auto"], Path],
    seed: int,
    config: Config,
    data: DataModule,
    checkpoint_dir: Path,
    out_dir: Path,
    train: TrainArgs,
    eval: EvalArgs,
    optimizer: Union[str, Dict],
    num_nodes: int = 1,
) -> None:
    validate_args(train, eval)

    tokenizer = Tokenizer(checkpoint_dir)
    train_dataloader, val_dataloader = get_dataloaders(fabric, data, tokenizer, train)
    steps_per_epoch = len(train_dataloader) // train.gradient_accumulation_iters(devices, num_nodes)
    lr_max_steps = min(train.epochs * steps_per_epoch, (train.max_steps or float("inf")))

    fabric.seed_everything(seed)  # same seed for every process to init model (FSDP)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    checkpoint_path = checkpoint_dir / "lit_model.pth"
    with fabric.init_module(empty_init=(fabric.world_size > 1)):
        model = GPT(config)

    fabric.print(f"Number of trainable parameters: {num_parameters(model, requires_grad=True):,}")

    model = fabric.setup(model)

    optimizer = instantiate_torch_optimizer(optimizer, model.parameters())
    optimizer = fabric.setup_optimizers(optimizer)
    scheduler = get_lr_scheduler(optimizer, warmup_steps=train.lr_warmup_steps, max_steps=lr_max_steps)
    state = {"model": model, "optimizer": optimizer, "scheduler": scheduler, "iter_num": 0, "step_count": 0}

    resume = find_resume_path(resume, out_dir)
    if resume:
        fabric.print(f"Resuming training from {resume}")
        fabric.load(resume, state)
    else:
        load_checkpoint(fabric, state["model"], checkpoint_path)

    train_time = time.perf_counter()
    token_counts = fit(
        fabric=fabric,
        state=state,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        devices=devices,
        num_nodes=num_nodes,
        resume=resume,
        checkpoint_dir=checkpoint_dir,
        out_dir=out_dir,
        train=train,
        eval=eval,
        data=data,
    )
    training_time = time.perf_counter() - train_time
    output = create_finetuning_performance_report(training_time, token_counts, fabric.device.type)
    fabric.print(output)

    # Final evaluation
    if eval.final_validation:
        qat = QATConfig()
        q_selector = QuantParamSelector(mode=qat.selector_mode)

        val_loss = validate_quantized(
            fabric,
            model,
            val_dataloader,
            dataclasses.replace(eval, max_iters=len(val_dataloader)),
            qat=qat,
            q_selector=q_selector,
        )
        metrics = {"final_val_loss_q": val_loss, "final_val_ppl_q": math.exp(val_loss)}
        fabric.log_dict(metrics, step=state["iter_num"])
        fabric.print(
            f"Final evaluation | val_loss_q: {val_loss.item():.3f} | val_ppl_q: {math.exp(val_loss):.3f}"
        )
    # Save the final checkpoint at the end of training
    #save_path = out_dir / "final" / "lit_model.pth"
    #save_path.parent.mkdir(parents=True, exist_ok=True)
    #fabric.save(save_path, {"model": state["model"]})
    #if fabric.global_rank == 0:
        # Copy checkpoint files from original checkpoint dir
    #    copy_config_files(checkpoint_dir, save_path.parent)
    #    save_hyperparameters(setup, save_path.parent)
    #    save_prompt_style(data.prompt_style, save_path.parent)


@torch.no_grad()
def _compute_quant_rels(
    model,
    *,
    qat: QATConfig,
    selector: QuantParamSelector,
    sens: Optional[SensitivityEMA],
) -> Tuple[List[torch.nn.Parameter], List[float]]:
    """Collect selected quantized parameters and their optional sensitivity weights."""
    params: List[torch.nn.Parameter] = []
    s_vals = []
    for name, p in iter_named_quant_params(
        model,
        selector=selector,
        include_substrings=qat.include_substrings,
        exclude_substrings=qat.exclude_substrings,
    ):
        if not p.requires_grad:
            continue
        params.append(p)
        s_vals.append(sens.get(name, p.device) if sens is not None else None)

    if not params:
        return [], []

    if sens is None or all(s is None for s in s_vals):
        return params, [1.0] * len(params)

    S = torch.stack([s.float() for s in s_vals if s is not None]) + 1e-12
    z = torch.log(S)
    mu = z.mean()
    sigma = z.std() + 1e-6

    ws = []
    for s in s_vals:
        if s is None:
            w = torch.tensor(3.0, device=mu.device)
        else:
            zz = torch.log(s.float() + 1e-12)
            score = (mu - zz) / sigma
            w = 1.0 + 4.0 * torch.sigmoid(score)
        ws.append(w)
    ws = torch.stack(ws).detach().float()
    rels = (ws / (ws.mean() + 1e-12)).tolist()
    return params, rels


@torch.no_grad()
def prepare_theory_matched_quant_update(
    model,
    optimizer,
    *,
    qat: QATConfig,
    selector: QuantParamSelector,
    gamma: float,
    sens: Optional[SensitivityEMA],
    dual_lambda: float,
) -> Optional[Dict[str, object]]:
    """
        x_{k+1} = x_k - [ prox(x_k) + lambda * Adam{grad at q(x_k)} ]
    """
    if not qat.enabled:
        return None

    params, rels = _compute_quant_rels(model, qat=qat, selector=selector, sens=sens)
    if not params:
        return None

    scale = float(qat.dist_scale * gamma) if gamma > 0.0 else 0.0
    effective_lambda = max(float(dual_lambda), 0.05)
    prox_multiplier = min(scale / effective_lambda, 20.0) if scale > 0.0 else 0.0
    base_prox_lr = 5e-3
    task_scale = max(float(dual_lambda), 0.0) if qat.use_lagrange else 1.0

    group_map = {}
    for group in optimizer.param_groups:
        for p in group["params"]:
            group_map[id(p)] = group

    prepared = []
    grad_sq = 0.0
    task_delta_sq = 0.0
    prox_delta_sq = 0.0
    alpha_vals = []

    for rel, p in zip(rels, params):
        p_old = p.data.detach().clone()
        q_old = quantize_to_grid(p_old, qat.n_bits_w, qat.step_w)

        raw_alpha = (base_prox_lr * prox_multiplier) * float(rel) if prox_multiplier > 0.0 else 0.0
        safe_alpha = min(raw_alpha, 0.1)
        alpha_vals.append(float(safe_alpha))
        prox_delta = (safe_alpha * (p_old - q_old)).float()
        prox_delta_sq += torch.sum(prox_delta * prox_delta).item()

        if p.grad is None:
            adam_term = torch.zeros_like(p_old, dtype=torch.float32)
            wd_term = torch.zeros_like(p_old, dtype=torch.float32)
        else:
            g = p.grad.detach().float()
            grad_sq += torch.sum(g * g).item()

            group = group_map[id(p)]
            lr = float(group["lr"])
            beta1, beta2 = group["betas"]
            eps = float(group["eps"])
            wd = float(group.get("weight_decay", 0.0))

            state = optimizer.state[p]
            exp_avg = state.get("exp_avg", None)
            exp_avg_sq = state.get("exp_avg_sq", None)

            if exp_avg is None:
                exp_avg = torch.zeros_like(g)
            else:
                exp_avg = exp_avg.detach().float()

            if exp_avg_sq is None:
                exp_avg_sq = torch.zeros_like(g)
            else:
                exp_avg_sq = exp_avg_sq.detach().float()

            step_prev = state.get("step", 0)
            if torch.is_tensor(step_prev):
                step_prev = step_prev.item()
            step_t = int(step_prev) + 1

            exp_avg_next = exp_avg.mul(beta1).add(g, alpha=1.0 - beta1)
            exp_avg_sq_next = exp_avg_sq.mul(beta2).addcmul(g, g, value=1.0 - beta2)

            bias_correction1 = 1.0 - beta1 ** step_t
            bias_correction2 = 1.0 - beta2 ** step_t
            denom = exp_avg_sq_next.sqrt().div(math.sqrt(bias_correction2)).add_(eps)
            adam_term = exp_avg_next.div(denom).mul(lr / bias_correction1)

            wd_term = torch.zeros_like(adam_term)
            if wd != 0.0:
                wd_term = p_old.detach().float() * (lr * wd)

        task_delta = task_scale * adam_term + wd_term
        task_delta_sq += torch.sum(task_delta * task_delta).item()

        desired = p_old.detach().float() - task_delta - prox_delta
        prepared.append((p, desired.to(dtype=p.dtype), p_old, q_old))

    debug = {
        "grad_l2": math.sqrt(grad_sq),
        "task_dx_l2": math.sqrt(task_delta_sq),
        "prox/delta_l2": math.sqrt(prox_delta_sq),
        "prox/scale": scale,
        "prox/eff_lambda": effective_lambda,
        "prox/prox_multiplier": prox_multiplier,
        "prox/alpha_mean": float(sum(alpha_vals) / len(alpha_vals)) if alpha_vals else 0.0,
        "prox/alpha_max": float(max(alpha_vals)) if alpha_vals else 0.0,
        "prox/rel_min": float(min(rels)) if rels else 0.0,
        "prox/rel_mean": float(sum(rels) / len(rels)) if rels else 0.0,
        "prox/rel_max": float(max(rels)) if rels else 0.0,
        "task_scale": task_scale,
    }
    return {"prepared": prepared, "debug": debug}


@torch.no_grad()
def apply_prepared_quant_update(pkg: Optional[Dict[str, object]]) -> Optional[Dict[str, float]]:
    if pkg is None:
        return None
    for p, desired, _, _ in pkg["prepared"]:
        p.data.copy_(desired)
    return pkg["debug"]


def fit(
    fabric: L.Fabric,
    state: Dict,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    devices: int,
    resume: Union[bool, Literal["auto"], Path],
    checkpoint_dir: Path,
    out_dir: Path,
    train: TrainArgs,
    eval: EvalArgs,
    data: DataModule,
    num_nodes: int = 1,
) -> None:
    model = state["model"]
    optimizer = state["optimizer"]
    scheduler = state["scheduler"]
    tokenizer = Tokenizer(checkpoint_dir)
    longest_seq_length, longest_seq_ix = get_longest_seq_length(
        ConcatDataset([train_dataloader.dataset, val_dataloader.dataset])
    )
    model.max_seq_length = min(longest_seq_length, train.max_seq_length or float("inf"))
    fabric.print(
        f"The longest sequence length in the train data is {longest_seq_length}, the model's maximum sequence length is"
        f" {model.max_seq_length} and context length is {model.config.block_size}"
    )

    token_counts = {
        "raw_tokens": torch.tensor(0, device=fabric.device, dtype=torch.long),
        "raw_tokens_plus_prompt_template": torch.tensor(0, device=fabric.device, dtype=torch.long),
        "raw_tokens_plus_prompt_template_and_padding": torch.tensor(0, device=fabric.device, dtype=torch.long),
    }

    # ---------------- Lagrange config ----------------
    qat = QATConfig()

    # cache accum iters for readability 
    accum_iters = train.gradient_accumulation_iters(devices, num_nodes)

    # selector & state
    q_selector = QuantParamSelector(mode=qat.selector_mode)
    sens = SensitivityEMA(momentum=qat.sens_ema) if (qat.enabled and qat.sens_enable) else None




    if eval.initial_validation:
        val_loss = validate_quantized(
            fabric,
            model,
            val_dataloader,
            dataclasses.replace(eval, max_iters=len(val_dataloader)),
            qat=qat,
            q_selector=q_selector,
        )
        val_loss = f"{val_loss:.3f}"
    else:
        fabric.print("Verifying settings ...")
        validate_quantized(
            fabric,
            model,
            val_dataloader,
            dataclasses.replace(eval, max_iters=2),
            qat=qat,
            q_selector=q_selector,
            verbose=False,
        )
        val_loss = "n/a"



    initial_iter = state["iter_num"]
    max_steps = train.max_steps or float("inf")
    train_iterator = CycleIterator(train_dataloader)

    # resume data loader state by fast-forwarding through all seen batches
    if resume:
        resume_t0 = time.perf_counter()
        for resume_iter in range(initial_iter):
            next(train_iterator)
            if resume_iter % 1000 == 0:
                fabric.print(f"Resuming dataset: {resume_iter} / {initial_iter}")
        fabric.barrier()
        fabric.print(
            f"Resuming data loader finished. Took {time.perf_counter() - resume_t0:.1f} seconds to reach iteration"
            f" {initial_iter}."
        )

    running_loss = RunningMean(window=train.gradient_accumulation_iters(devices, num_nodes), sync_on_compute=False).to(
        fabric.device
    )

    running_loss_fp = RunningMean(
        window=train.gradient_accumulation_iters(devices, num_nodes),
        sync_on_compute=False
    ).to(fabric.device)

    gamma_ctl = GammaController(
        q_target=qat.q_target,
        gamma_init=qat.gamma_init,
        gamma_lr=qat.gamma_lr,
        gamma_max=qat.gamma_max,
        start_step=qat.gamma_start_step,
        ema_momentum=qat.qrate_ema_momentum,
    )
    dual_ctl = DualController(
        beta=qat.beta,
        dual_lr=qat.dual_lr,
        lambda_init=qat.lambda_init,
        lambda_max=qat.lambda_max,
        use_pi=qat.lambda_use_pi,
        kp=qat.lambda_kp,
        ki=qat.lambda_ki,
        i_clamp=qat.lambda_i_clamp,
    )

    gamma = gamma_ctl.gamma
    dual_lambda = dual_ctl.lam

    # per-step accumulators
    f_sum = 0.0

    qrate_last = None
    sat_last = None
    qtotal_last = None

    fabric.barrier()

    while state["step_count"] < max_steps:
        state["iter_num"] += 1
        iter_t0 = time.perf_counter()
        batch = next(train_iterator)
        if train_iterator.epoch >= train.epochs:
            break
        input_ids, targets = batch["input_ids"], batch["labels"]

        is_accumulating = state["iter_num"] % train.gradient_accumulation_iters(devices, num_nodes) != 0
        

        with fabric.no_backward_sync(model, enabled=is_accumulating):
            dist_loss = None
            dist_dbg = None
            if qat.enabled:
                # dist term only once per optimizer step
                dist_loss = None
                dist_dbg = None
                if qat.enabled and (not is_accumulating) and (gamma > 0.0):
                    with torch.no_grad():
                        dist_loss, dist_dbg = compute_dist_loss(
                            model,
                            n_bits_w=qat.n_bits_w,
                            step=qat.step_w,
                            selector=q_selector,
                            include_substrings=qat.include_substrings,
                            exclude_substrings=qat.exclude_substrings,
                            sens=sens,
                            return_debug=True,
                       )

            backup_q = None
            if qat.enabled:
                backup_q = hard_quantize_model_inplace(
                    model,
                    n_bits_w=qat.n_bits_w,
                    step=qat.step_w,
                    selector=q_selector,
                    include_substrings=qat.include_substrings,
                    exclude_substrings=qat.exclude_substrings,
                )

            logits = model(input_ids)
            f_loss_quantized = chunked_cross_entropy(logits[..., :-1, :], targets[..., 1:])  # f(q(x))

            # Adam now sees grad of f(q(x))
            fabric.backward(f_loss_quantized / accum_iters)

            if backup_q is not None:
                restore_model_from_backup(model, backup_q)

            with torch.no_grad():
                logits_unquant = model(input_ids)
                f_loss_unquant = chunked_cross_entropy(logits_unquant[..., :-1, :], targets[..., 1:])  # f(x)

            # logging accumulators
            running_loss.update(f_loss_quantized.detach())      # train_loss_q
            running_loss_fp.update(f_loss_unquant.detach())     # loss_fq
            f_sum += float(f_loss_unquant.item())


        if not is_accumulating:
        # update 
            if qat.enabled and qat.sens_enable and sens is not None:
                sens.update_from_grads(
                    model,
                    selector=q_selector,
                    include_substrings=qat.include_substrings,
                    exclude_substrings=qat.exclude_substrings,
                )

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            next_step = state["step_count"] + 1
            track_step = 85 <= next_step <= 110

            quant_step_pkg = None
            if qat.enabled:
                quant_step_pkg = prepare_theory_matched_quant_update(
                    model,
                    optimizer,
                    qat=qat,
                    selector=q_selector,
                    gamma=gamma,
                    sens=sens,
                    dual_lambda=dual_lambda,
                )

            optimizer.step()

            prox_dbg = apply_prepared_quant_update(quant_step_pkg)

            if track_step and prox_dbg is not None:
                fabric.print(
                    f"[stepdbg] step={next_step} "
                    f"grad_l2={prox_dbg['grad_l2']:.4e} "
                    f"task_dx_l2={prox_dbg['task_dx_l2']:.4e} "
                    f"prox_dx_l2={prox_dbg['prox/delta_l2']:.4e} "
                    f"gamma={gamma:.6f} "
                    f"lambda={dual_lambda:.6f} "
                    f"task_scale={prox_dbg['task_scale']:.6f}"
                )

            optimizer.zero_grad()
            scheduler.step()
            state["step_count"] += 1

            # dual update 
            if qat.enabled and qat.use_lagrange:
                with torch.no_grad():
                    logits_unquant_post = model(input_ids)
                    f_loss_unquant_post = chunked_cross_entropy(
                        logits_unquant_post[..., :-1, :],
                        targets[..., 1:]
                    )
                dual_lambda = dual_ctl.step(float(f_loss_unquant_post.item()))

            # debug print every optimizer step
            #if prox_dbg is not None:
                #fabric.print(
                #    f"[dbg] step={state['step_count']} "
                #    f"gamma={gamma:.6f} "
                #    f"lambda={dual_lambda:.6f} "
                #    f"prox_mult={prox_dbg['prox/prox_multiplier']:.6f} "
                #    f"alpha_mean={prox_dbg['prox/alpha_mean']:.6f} "
                #    f"alpha_max={prox_dbg['prox/alpha_max']:.6f}"
                #)

            f_sum = 0.0

            # gamma update
            if qat.enabled and (state["step_count"] % qat.qrate_every == 0):
                qrate_last, sat_last, qtotal_last = compute_quantization_rate_fast(
                    model,
                    n_bits_w=qat.n_bits_w,
                    step=qat.step_w,
                    atol=qat.atol,
                    selector=q_selector,
                    include_substrings=qat.include_substrings,
                    exclude_substrings=qat.exclude_substrings,
                )
                gamma = gamma_ctl.step(qrate_last, step_count=state["step_count"])
                fabric.print(
                    f"[gamma] step={state['step_count']} qrate={qrate_last:.4f} -> gamma={gamma:.6f} "
                    f"(target={qat.q_target}) sat={sat_last*100:.2f}% N={qtotal_last:,}"
                )

        token_counts["raw_tokens"] += batch["token_counts"]["raw"].sum().item()
        token_counts["raw_tokens_plus_prompt_template"] += (
            batch["token_counts"]["raw_plus_prompt_template"].sum().item()
        )
        token_counts["raw_tokens_plus_prompt_template_and_padding"] += input_ids.numel()

        if state["iter_num"] % train.log_interval == 0:
            train_loss_q = running_loss.compute().item()      # f(q(x))
            loss_fq = running_loss_fp.compute().item()        # f(x)

            t1 = time.perf_counter()
            metrics = {
                "train_loss_q": train_loss_q,
                "iter": state["iter_num"],
                "loss_fq": loss_fq,
                "step": state["step_count"],
                "epoch": train_iterator.epoch,
                "iter_time": t1 - iter_t0,
                "tokens": token_counts["raw_tokens_plus_prompt_template"],
                "total_tokens": token_counts["raw_tokens_plus_prompt_template"] * fabric.world_size,
                "learning_rate": scheduler.get_last_lr()[0],
            }

            metrics.update({
                "qat/gamma": gamma,
                "qat/dual_lambda": dual_lambda,
            })
            if qrate_last is not None:
                metrics.update({
                    "qat/qrate": qrate_last,
                    "qat/sat_rate": sat_last,
                })

            if isinstance(val_loss, torch.Tensor):
                val_loss = f"{val_loss:.3f}"
            fabric.print(
                f"Epoch {metrics['epoch'] + 1} | iter {metrics['iter']} step {metrics['step']} |"
                f" train_loss_q: {metrics['train_loss_q']:.3f},"
                f" loss_fq: {metrics['loss_fq']:.3f},"
                f" val_fqx: {val_loss} |"
                f" lambda: {dual_lambda:.4f} |"
                f" iter time: {metrics['iter_time'] * 1000:.2f} ms"
                f"{' (step)' if not is_accumulating else ''}"
            )
            fabric.log_dict(metrics, step=state["iter_num"])

        if not is_accumulating and state["step_count"] % eval.interval == 0:
            t0 = time.perf_counter()


            val_loss = validate_quantized(
                fabric,
                model,
                val_dataloader,
                eval,
                qat=qat,
                q_selector=q_selector,
                verbose=False,
            )

            t1 = time.perf_counter() - t0

            val_loss_tensor = val_loss.detach().clone().to(fabric.device)
            val_time_tensor = torch.tensor(t1, device=fabric.device, dtype=torch.float32)

            fabric.all_reduce(val_loss_tensor, reduce_op="mean")
            fabric.all_reduce(val_time_tensor, reduce_op="mean")

            fabric.print(
                f"iter {state['iter_num']}: val_fqx {val_loss_tensor.item():.4f}, "
                f"val time: {val_time_tensor.item() * 1000:.2f} ms"
            )
            metrics = {
                "val_loss_q": val_loss_tensor,
                "val_ppl_q": math.exp(val_loss_tensor),
            }
            fabric.log_dict(metrics, step=state["iter_num"])
            fabric.barrier()

            val_loss = val_loss_tensor

    # -------- every 100 steps: full train/val on f(q(x)) --------
            if state["step_count"] % 100 == 0:
                t0_full = time.perf_counter()

                full_train_loss_q = validate_quantized(
                    fabric,
                    model,
                    train_dataloader,
                    dataclasses.replace(eval, max_iters=len(train_dataloader)),
                    qat=qat,
                    q_selector=q_selector,
                    verbose=False,
                )
                full_val_loss_q = validate_quantized(
                    fabric,
                    model,
                    val_dataloader,
                    dataclasses.replace(eval, max_iters=len(val_dataloader)),
                    qat=qat,
                    q_selector=q_selector,
                    verbose=False,
                )

                t1_full = time.perf_counter() - t0_full

                full_train_loss_q_tensor = full_train_loss_q.detach().clone().to(fabric.device)
                full_val_loss_q_tensor = full_val_loss_q.detach().clone().to(fabric.device)
                full_eval_time_tensor = torch.tensor(t1_full, device=fabric.device, dtype=torch.float32)


                fabric.all_reduce(full_train_loss_q_tensor, reduce_op="mean")
                fabric.all_reduce(full_val_loss_q_tensor, reduce_op="mean")
                fabric.all_reduce(full_eval_time_tensor, reduce_op="mean")

                fabric.print(
                    f"iter {state['iter_num']}: "
                    f"full_train_loss_q {full_train_loss_q_tensor.item():.4f}, "
                    f"full_val_loss_q {full_val_loss_q_tensor.item():.4f}, "
                    f"eval time: {full_eval_time_tensor.item() * 1000:.2f} ms"
                )

                full_metrics = {
                    "full_train_loss_q": full_train_loss_q_tensor,
                    "full_train_ppl_q": math.exp(full_train_loss_q_tensor),
                    "full_val_loss_q": full_val_loss_q_tensor,
                    "full_val_ppl_q": math.exp(full_val_loss_q_tensor),
                }
                fabric.log_dict(full_metrics, step=state["iter_num"])
                fabric.barrier()

                val_loss = full_val_loss_q_tensor


        if train.save_interval is not None and not is_accumulating and state["step_count"] % train.save_interval == 0:
            checkpoint_file = out_dir / f"step-{state['step_count']:06d}" / "lit_model.pth"
            checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
            fabric.print(f"Saving checkpoint to {str(checkpoint_file.parent)!r}")
            fabric.save(checkpoint_file, state)
            if fabric.global_rank == 0:
                copy_config_files(checkpoint_dir, checkpoint_file.parent)
                save_hyperparameters(setup, checkpoint_file.parent)
                save_prompt_style(data.prompt_style, checkpoint_file.parent)

    total_token_counts = {}
    for key in token_counts:
        total = fabric.all_reduce(token_counts[key], reduce_op="sum")
        total_token_counts[key] = total.item()

    return total_token_counts


# FSDP has issues with `inference_mode`
@torch.no_grad()
def validate(
    fabric: L.Fabric, model: GPT, val_dataloader: DataLoader, eval: EvalArgs, verbose: bool = True
) -> torch.Tensor:
    if verbose:
        fabric.print("Validating ...")
    model.eval()
    losses = torch.zeros(min(len(val_dataloader), eval.max_iters))
    for k, batch in enumerate(val_dataloader):
        if k >= eval.max_iters:
            break
        input_ids, targets = batch["input_ids"], batch["labels"]
        logits = model(input_ids)
        losses[k] = chunked_cross_entropy(logits[..., :-1, :], targets[..., 1:], chunk_size=0)

    val_loss = losses.mean()
    model.train()
    return val_loss


@torch.no_grad()
def generate_example(fabric: L.Fabric, model: GPT, tokenizer: Tokenizer, eval: EvalArgs, data: DataModule):
    instruction = select_sft_generate_example(eval, data)
    fabric.print(instruction)
    prompt = data.prompt_style.apply(instruction)
    encoded = tokenizer.encode(prompt, device=fabric.device)
    model.eval()

    with fabric.init_tensor():
        # do not set `max_seq_length=max_returned_token` because memory is not a concern here
        model.set_kv_cache(batch_size=1)

    max_returned_tokens = len(encoded) + eval.max_new_tokens

    if max_returned_tokens < model.max_seq_length:
        with fabric.init_tensor():
            # do not set `max_seq_length=max_returned_token` because memory is not a concern here
            model.set_kv_cache(batch_size=1)
        output = generate(
            model, encoded, max_returned_tokens=max_returned_tokens, temperature=0.8, eos_id=tokenizer.eos_id
        )
        model.clear_kv_cache()
        model.train()
        output = tokenizer.decode(output)
        fabric.print(f"{output}\n")
    else:
        print(
            f"Length of encoded instruction ({len(encoded)}) and eval.max_new_tokens ({eval.max_new_tokens}) "
            f"exceeds model.max_seq_length ({model.max_seq_length}) used for training. Skipping example generation for efficiency. "
            f"The model's supported context size (post-training) is {model.config.block_size}."
        )


def get_lr_scheduler(optimizer, warmup_steps: int, max_steps: int):
    # linear warmup followed by cosine annealing
    scheduler1 = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: step / warmup_steps)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(max_steps - warmup_steps))
    return torch.optim.lr_scheduler.SequentialLR(optimizer, [scheduler1, scheduler2], milestones=[warmup_steps])


def get_dataloaders(
    fabric: L.Fabric, data: DataModule, tokenizer: Tokenizer, train: TrainArgs
) -> Tuple[DataLoader, DataLoader]:
    data.connect(tokenizer=tokenizer, batch_size=train.micro_batch_size, max_seq_length=train.max_seq_length)
    with fabric.rank_zero_first():
        data.prepare_data()
    data.setup()
    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()
    train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)
    return train_dataloader, val_dataloader


def get_longest_seq_length(data: List[Dict]) -> Tuple[int, int]:
    # find out the minimum max_seq_length required during fine-tuning (saves memory!)
    lengths = [len(d["input_ids"]) for d in data]
    longest_seq_length = max(lengths)
    longest_seq_ix = lengths.index(longest_seq_length)
    return longest_seq_length, longest_seq_ix


def validate_args(train: TrainArgs, eval: EvalArgs) -> None:
    issues = []
    unsupported = [(train, ["max_tokens", "max_norm", "tie_embeddings", "lr_warmup_fraction"])]
    for args, names in unsupported:
        for name in names:
            if getattr(args, name) is not None:
                issues.append(f"{__file__} doesn't support the {name!r} argument. This is set in {args}")
    required = [(train, ["epochs"]), (eval, ["max_new_tokens"])]
    for args, names in required:
        for name in names:
            if getattr(args, name) is None:
                issues.append(f"{__file__} requires the {name!r} argument. This is set in {args}")
    if not train.epochs and not train.max_steps:
        issues.append(f"{__file__} requires either epochs or max_steps to be set. This is set in {train}")
    if issues:
        raise ValueError("\n".join(issues))
