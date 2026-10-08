#!/usr/bin/env python3
"""Profile GROOT N1.7 FP8 on RTX with per-stage and per-layer CUDA timings.

This is a diagnostic profiler, not a replacement for the CUDA-graph latency
benchmark. It reports graph-replay end-to-end timings, then runs the same
kernel pipeline eagerly with one transformer layer per Python call so CUDA
events can attribute time to individual layers. The latter adds launch/Python
overhead and must not be summed as the production latency.

The standard aux fixture contains ``pixel_features`` and fused
``llm_input_embeds``. Thus patch embedding, raw-image preprocessing, and the
final ViT merger are outside this measured path; the 24 ViT blocks are still
run to produce the three DeepStack taps.
"""
from __future__ import annotations

import argparse
import csv
import functools
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.mean(values),
        "p50_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def _linear_flops(rows: int, *weights: torch.Tensor) -> int:
    # 2 operations per multiply-accumulate; bias/activation are omitted.
    return 2 * rows * sum(int(w.numel()) for w in weights)


def _make_flop_functions(fe, aux: dict, horizon: int):
    grid = aux["grid_thw"].reshape(-1, 3)
    nviews = int(grid.shape[0])
    sv = int(aux["pixel_features"].reshape(-1, 1024).shape[0])
    se = int(aux["llm_input_embeds"].reshape(-1, 2048).shape[0])
    sper = sv // nviews
    mask = aux["visual_pos_masks"].reshape(-1).bool()
    ntext = int((~mask).sum())
    nimage = int(mask.sum())

    def vit(d):
        s, dim, ff = int(d["S"]), int(d["D"]), int(d["ff_inner"])
        heads, hd = int(d["NH"]), int(d["HD"])
        per_view = int(d.get("Sper_view", s))
        linears = 2 * s * (4 * dim * dim + 2 * dim * ff)
        attention = 4 * heads * nviews * per_view * per_view * hd
        return linears + attention

    def deepstack(d):
        n, mid, out = int(d["Nout"]), int(d["Dmid"]), int(d["Dout"])
        return 2 * n * (mid * mid + mid * out)

    def llm(d):
        s, dim, ff = int(d["S"]), int(d["D"]), int(d["FF"])
        qh, kvh, hd = int(d["NHQ"]), int(d["NHKV"]), int(d["HD"])
        qkv_out = (qh + 2 * kvh) * hd
        linears = 2 * s * (dim * qkv_out + dim * dim + 3 * dim * ff)
        attention = 4 * qh * s * s * hd
        return linears + attention

    def vlsa(d):
        t, dim, ff = int(d["T"]), int(d["D"]), int(d["ff_inner"])
        heads, hd = int(d["NH"]), int(d["HD"])
        return 2 * t * (4 * dim * dim + 2 * dim * ff) + 4 * heads * t * t * hd

    def vlln(d):
        return 5 * int(d["S"]) * int(d["D"])

    def dit(d, li):
        sa, dim, ff = int(d["Sa"]), int(d["D"]), int(d["FF"])
        heads, hd = 32, 48
        if li % 2:
            linears = 2 * sa * (4 * dim * dim + 2 * dim * ff)
            skv = sa
        else:
            # Cross-attention K/V are precomputed before the timed inference.
            linears = 2 * sa * (2 * dim * dim + 2 * dim * ff)
            skv = ntext if li % 4 == 0 else nimage
        return linears + 4 * heads * sa * skv * hd

    def state_encode(_d=None):
        return _linear_flops(1, fe._st_enc_l1_W, fe._st_enc_l2_W)

    def action_encode(_d=None):
        return _linear_flops(horizon, fe._ac_enc_W1_W,
                             fe._ac_enc_W2_W, fe._ac_enc_W3_W)

    def action_decode(_d=None):
        return _linear_flops(horizon, fe._ac_dec_l1_W, fe._ac_dec_l2_W)

    def timestep(_d=None):
        return _linear_flops(1, fe._ts_lin1_w, fe._ts_lin2_w)

    def adaln(_d=None):
        return _linear_flops(1, *fe._dit_ada_w)

    def output_proj(_d=None):
        return _linear_flops(1, fe._proj_out_1_w) + _linear_flops(
            horizon + 1, fe._proj_out_2_w)

    def cross_kv(_d=None):
        total = 0
        for j in range(16):
            rows = ntext if (2 * j) % 4 == 0 else nimage
            total += _linear_flops(rows, fe._dit_k_w[2 * j], fe._dit_v_w[2 * j])
        return total

    def action_step_other(_d=None):
        return (_linear_flops(horizon, fe._ac_enc_W1_W,
                              fe._ac_enc_W2_W, fe._ac_enc_W3_W)
                + _linear_flops(horizon + 1, fe._proj_out_2_w)
                + _linear_flops(horizon, fe._ac_dec_l1_W, fe._ac_dec_l2_W))

    return {
        "vit": vit, "deepstack": deepstack, "llm": llm, "vlsa": vlsa,
        "vlln": vlln,
        "dit": dit, "state_encode": state_encode,
        "action_encode": action_encode, "action_decode": action_decode,
        "timestep_embedding": timestep, "adaln_modulators": adaln,
        "dit_output_projection": output_proj, "cross_kv": cross_kv,
        "action_step_other": action_step_other,
        "shape": {"views": nviews, "vit_tokens": sv, "llm_tokens": se,
                  "text_tokens": ntext, "image_tokens": nimage,
                  "vit_tokens_per_view": sper},
    }


def _event_pair(stream: torch.cuda.Stream | None):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(stream)
    return start, end


def _install_layer_timers(module, records, flops):
    """Split subset-capable forwards into individual in-place layer calls."""
    specs = {
        "qwen3vl_vit_forward": ("vit", 24),
        "deepstack_merge_forward": ("deepstack", 3),
        "qwen3vl_llm_forward": ("llm", 16),
        "vlln_forward": ("vlln", 1),
        "vl_self_attn_forward": ("vlsa", 4),
        "dit_forward": ("dit", 32),
    }
    originals = {}
    for fn_name, (component, count) in specs.items():
        if not hasattr(module, fn_name):
            continue
        original = getattr(module, fn_name)
        originals[fn_name] = original

        @functools.wraps(original)
        def wrapped(*args, __original=original, __component=component,
                    __count=count, **kwargs):
            requested = kwargs.get("layers_subset")
            if requested is not None:
                return __original(*args, **kwargs)
            stream_id = int(kwargs.get("stream", 0))
            stream = (torch.cuda.current_stream() if stream_id == 0 else
                      torch.cuda.ExternalStream(stream_id))
            dims = kwargs.get("dims", {})
            if __component == "vlln":
                start, end = _event_pair(stream)
                result = __original(*args, **kwargs)
                end.record(stream)
                records["vlln"].append((start, end, flops["vlln"](dims)))
                return result
            for layer in range(__count):
                start, end = _event_pair(stream)
                __original(*args, layers_subset=[layer], **kwargs)
                end.record(stream)
                key = f"{__component}.layer_{layer:02d}"
                if __component == "dit":
                    layer_flops = flops[__component](dims, layer)
                else:
                    layer_flops = flops[__component](dims)
                records[key].append((start, end, layer_flops))

        setattr(module, fn_name, wrapped)
    return originals


def _summarize(records):
    torch.cuda.synchronize()
    rows = []
    for name, entries in records.items():
        ms = [float(start.elapsed_time(end)) for start, end, _ in entries]
        flop_values = [flops for _, _, flops in entries]
        avg_flops = int(statistics.mean(flop_values)) if flop_values else 0
        mean_ms = statistics.mean(ms)
        rows.append({
            "component": name,
            "samples": len(ms),
            "mean_ms": mean_ms,
            "p50_ms": statistics.median(ms),
            "min_ms": min(ms),
            "max_ms": max(ms),
            "estimated_flops_per_call": avg_flops,
            "effective_tflops": (avg_flops / (mean_ms * 1e9)
                                  if mean_ms > 0 else 0.0),
            "calls_flops_total": int(sum(flop_values)),
            "time_total_ms": float(sum(ms)),
        })
    return sorted(rows, key=lambda row: (-row["time_total_ms"], row["component"]))


def _record_call(records, name, fn, flop_count):
    stream = torch.cuda.current_stream()
    start, end = _event_pair(stream)
    fn()
    end.record(stream)
    records[name].append((start, end, flop_count))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="GR00T N1.7 checkpoint")
    ap.add_argument("--aux", required=True, help="256x256 *_llm_aux.pt")
    ap.add_argument("--fixture", default=None,
                    help="reference fixture used to load the matching robot state; "
                         "defaults to the aux filename without _llm_aux")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--views", type=int, default=2,
                    help="camera views; defaults to the published 2-view case")
    ap.add_argument("--output", default="groot_n17_profile")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This profiler must run on the CUDA server")
    aux_path = Path(args.aux)
    aux = torch.load(aux_path, map_location="cpu", weights_only=False)
    if isinstance(aux, list):
        aux = aux[0]
    fixture_path = (Path(args.fixture) if args.fixture else
                    aux_path.with_name(aux_path.stem.removesuffix("_llm_aux") + ".pt"))
    fixture = torch.load(fixture_path, map_location="cpu", weights_only=False)

    from flash_rt.frontends.torch.groot_n17_rtx_fp8 import (
        GrootN17TorchFrontendRtxFP8,
    )
    from flash_rt.models.groot_n17 import pipeline_rtx_fp8, pipeline_thor

    fe = GrootN17TorchFrontendRtxFP8(
        args.ckpt, num_views=args.views,
        embodiment_tag="oxe_droid_relative_eef_relative_joint")
    aux = {k: (v.cuda() if isinstance(v, torch.Tensor) else v)
           for k, v in aux.items()}
    fe.set_prompt(aux=aux, prompt="profile")
    device = fe.device
    state_dict = {"state." + k: v for k, v in fixture["inputs"]["state"].items()}
    state = fe.normalize_state(state_dict).to(device)
    noise = aux["initial_noise"].to(device).bfloat16().contiguous()

    # Warm up/capture production graphs before taking the baseline.
    for _ in range(max(args.warmup, 2)):
        fe.infer(state, aux=aux, initial_noise=noise,
                 action_horizon=args.horizon, use_dit_graph=True)
    torch.cuda.synchronize()

    # Production-path reference: same RTX FP8 frontend and whole-infer timing
    # as the existing 5090 benchmark, plus a synchronized stage split.
    baseline = defaultdict(list)
    for _ in range(args.iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        full_start = torch.cuda.Event(enable_timing=True)
        full_end = torch.cuda.Event(enable_timing=True)
        full_start.record()
        fe.infer(state, aux=aux, initial_noise=noise,
                 action_horizon=args.horizon, use_dit_graph=True)
        full_end.record()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        baseline["e2e_gpu_event_ms"].append(full_start.elapsed_time(full_end))
        baseline["e2e_wall_ms"].append((t1 - t0) * 1000.0)

        bb_start, bb_end = (torch.cuda.Event(enable_timing=True),
                            torch.cuda.Event(enable_timing=True))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        bb_start.record()
        fe._backbone_features = fe.run_backbone_graph(aux)
        bb_end.record()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        baseline["backbone_graph_gpu_event_ms"].append(bb_start.elapsed_time(bb_end))
        baseline["backbone_graph_wall_ms"].append((t1 - t0) * 1000.0)

        act_start, act_end = (torch.cuda.Event(enable_timing=True),
                              torch.cuda.Event(enable_timing=True))
        t0 = time.perf_counter()
        act_start.record()
        fe.infer(state, initial_noise=noise, action_horizon=args.horizon,
                 use_dit_graph=True)
        act_end.record()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        baseline["action_graph_gpu_event_ms"].append(act_start.elapsed_time(act_end))
        baseline["action_graph_wall_ms"].append((t1 - t0) * 1000.0)

    flops = _make_flop_functions(fe, aux, args.horizon)
    records = defaultdict(list)
    originals = _install_layer_timers(pipeline_rtx_fp8, records, flops)
    action_originals = _install_layer_timers(pipeline_thor, records, flops)
    dit_dims = {"Sa": args.horizon + 1, "D": 1536, "FF": 6144,
                "Skv_text": flops["shape"]["text_tokens"],
                "Skv_image": flops["shape"]["image_tokens"]}
    action_step_flops = (flops["action_step_other"]()
                         + sum(flops["dit"](dit_dims, layer)
                               for layer in range(32)))

    if not hasattr(fe, "_kdit_fwd"):
        raise RuntimeError("action CUDA graph setup did not expose the kernel forward")

    def run_eager_profile_pass():
        # Replay the same per-frame backbone computation, but split each
        # transformer into one-layer calls so event pairs can attribute time.
        fe._kbb_vit_h.copy_(
            aux["pixel_features"].to(device).half().reshape(-1, 1024))
        fe._kbb_llm_h.copy_(
            aux["llm_input_embeds"].to(device).half().reshape(-1, 2048))
        fe._kbb_forward(0)
        fe._backbone_features = fe._kbb_vlsa_h.unsqueeze(0)

        # Replay the same action kernel closures as the captured RTX call. The
        # per-layer DIT calls still use the prepared production weights.
        fe._ck_bb_src.copy_(fe._backbone_features.reshape(fe.Se, 2048).half())
        _record_call(records, "cross_kv_refresh", lambda: fe._cross_kv_fwd(0),
                     flops["cross_kv"]())
        fe._k_state_in.copy_(state.reshape(1, 132).to(device).bfloat16())
        fe._k_actions.copy_(noise.reshape(args.horizon, 132))
        _record_call(records, "state_encode", lambda: fe._kdit_fwd[0](0),
                     flops["state_encode"]())
        for step in range(fe._k_nsteps):
            start, end = _event_pair(torch.cuda.current_stream())
            fe._kdit_fwd[1](step, 0)
            end.record(torch.cuda.current_stream())
            records[f"action_step_{step}_inclusive"].append(
                (start, end, action_step_flops))

    for _ in range(args.warmup):
        run_eager_profile_pass()
    torch.cuda.synchronize()
    records.clear()
    for _ in range(args.iters):
        run_eager_profile_pass()
        torch.cuda.synchronize()

    for name, original in originals.items():
        setattr(pipeline_rtx_fp8, name, original)
    for name, original in action_originals.items():
        setattr(pipeline_thor, name, original)

    rows = _summarize(records)
    # Parent step events include the 32 nested DiT layer events. Derive the
    # action encoder/output/decode remainder so the leaf rows remain additive.
    for step in range(fe._k_nsteps):
        parent = records[f"action_step_{step}_inclusive"]
        residual_ms = []
        for iteration, (start, end, _) in enumerate(parent):
            nested = 0.0
            for layer in range(32):
                layer_events = records[f"dit.layer_{layer:02d}"]
                index = iteration * fe._k_nsteps + step
                nested += float(layer_events[index][0].elapsed_time(
                    layer_events[index][1]))
            residual_ms.append(max(0.0, float(start.elapsed_time(end)) - nested))
        mean_ms = statistics.mean(residual_ms)
        rows.append({
            "component": f"action_step_{step}_non_dit",
            "samples": len(residual_ms),
            "mean_ms": mean_ms,
            "p50_ms": statistics.median(residual_ms),
            "min_ms": min(residual_ms),
            "max_ms": max(residual_ms),
            "estimated_flops_per_call": flops["action_step_other"](),
            "effective_tflops": (flops["action_step_other"]() /
                                  (mean_ms * 1e9) if mean_ms > 0 else 0.0),
            "calls_flops_total": flops["action_step_other"]() * len(residual_ms),
            "time_total_ms": float(sum(residual_ms)),
        })
    rows.sort(key=lambda row: (-row["time_total_ms"], row["component"]))
    baseline_summary = {name: _stats(values) for name, values in baseline.items()}
    report = {
        "device": torch.cuda.get_device_name(),
        "frontend": type(fe).__name__,
        "profile_mode": "RTX FP8 frontend; graph baseline + eager per-layer diagnostic",
        "aux_contract": "pixel_features + llm_input_embeds; raw pixels absent",
        "shape": flops["shape"],
        "warmup": args.warmup,
        "iterations": args.iters,
        "baseline_graph_wall": baseline_summary,
        "eager_layer_rows": rows,
        "flop_note": "Linear/GEMM and attention MAC estimates use 2 FLOPs/MAC; elementwise, norm, quantize, copies, and Python are omitted.",
    }
    out_base = Path(args.output)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_base.with_suffix(".json")
    csv_path = out_base.with_suffix(".csv")
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"GPU: {report['device']}")
    print("Input shape:", json.dumps(report["shape"]))
    print("Graph replay baseline (wall ms, median):")
    for name, stat in baseline_summary.items():
        print(f"  {name}: {stat['p50_ms']:.3f}")
    print("Eager per-layer diagnostic (CUDA-event mean ms; same FP8 kernel path, "
          "but per-layer calls add launch overhead):")
    print(f"{'component':34s} {'mean ms':>9s} {'% eager':>9s} {'TFLOP/s*':>10s}")
    total_ms = sum(row["time_total_ms"] for row in rows
                   if not row["component"].endswith("_inclusive"))
    for row in rows:
        # Inclusive action-step rows contain the DIT layers and are shown for
        # reference only; leaf rows form the additive percentage breakdown.
        pct = (100.0 * row["time_total_ms"] / total_ms
               if total_ms and not row["component"].endswith("_inclusive") else 0.0)
        print(f"{row['component']:34s} {row['mean_ms']:9.4f} {pct:8.2f}% "
              f"{row['effective_tflops']:10.2f}")
    print("*Approximate useful FLOPs only; elementwise kernels and launch gaps are excluded.")
    print(f"Wrote {json_path} and {csv_path}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
