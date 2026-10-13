#!/usr/bin/env python3
"""RTX GR00T N1.7: measure layers *inside* replayed CUDA graphs.

This is a diagnostic graph with CUDA event nodes at layer boundaries.  A
separate, uninstrumented frontend supplies the production latency reference.
Python wrappers run while capturing, not while replaying.  Operation signatures
come from the same capture; dimensions are logical shapes, not layout proofs.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

import torch


STAGES = {
    "qwen3vl_vit_forward": ("vit", 24),
    "deepstack_merge_forward": ("deepstack", 3),
    "qwen3vl_llm_forward": ("llm", 16),
    "vlln_forward": ("vlln", 1),
    "vl_self_attn_forward": ("vlsa", 4),
    "dit_forward": ("dit", 32),
}


def stats(xs):
    return {"mean_ms": statistics.mean(xs), "p50_ms": statistics.median(xs),
            "min_ms": min(xs), "max_ms": max(xs)}


class CaptureTrace:
    def __init__(self):
        self.events = []
        self.operations = []
        self.scope = "outside_layer"
        self.dims = {}
        self.dit_step = 0
        self._originals = []

    def capturing(self):
        return torch.cuda.is_current_stream_capturing()

    def install_layers(self):
        from flash_rt.models.groot_n17 import pipeline_rtx_fp8, pipeline_thor

        for module in (pipeline_rtx_fp8, pipeline_thor):
            for name, (stage, count) in STAGES.items():
                if not hasattr(module, name):
                    continue
                original = getattr(module, name)
                self._originals.append((module, name, original))

                def wrapped(*args, _original=original, _stage=stage,
                            _count=count, **kwargs):
                    if not self.capturing() or kwargs.get("layers_subset") is not None:
                        return _original(*args, **kwargs)
                    step = self.dit_step if _stage == "dit" else None
                    if step is not None:
                        self.dit_step += 1
                    dims = {k: int(v) for k, v in kwargs.get("dims", {}).items()
                            if isinstance(v, (int, float))}
                    for layer in range(_count):
                        label = (f"dit.step_{step}.layer_{layer:02d}" if step is not None
                                 else f"{_stage}.layer_{layer:02d}")
                        previous = self.scope, self.dims
                        self.scope, self.dims = label, dims
                        # Graph-internal events encode dependencies only;
                        # external nodes retain timestamps across replay.
                        start = torch.cuda.Event(enable_timing=True, external=True)
                        end = torch.cuda.Event(enable_timing=True, external=True)
                        start.record()
                        try:
                            if _stage == "vlln":
                                _original(*args, **kwargs)
                            else:
                                _original(*args, layers_subset=[layer], **kwargs)
                        finally:
                            end.record()
                            self.scope, self.dims = previous
                        self.events.append({"scope": label, "stage": _stage,
                                            "dims": dims, "start": start, "end": end})

                setattr(module, name, wrapped)

    @staticmethod
    def _named_scalars(fn, args, kwargs):
        doc = getattr(fn, "__doc__", "") or ""
        first = doc.splitlines()[0] if doc else ""
        names = re.findall(r"([A-Za-z_]\w*):\s", first)
        if names and names[0] == "self":
            names = names[1:]
        values = dict(zip(names, args))
        values.update(kwargs)
        return {k: v for k, v in values.items()
                if k not in {"stream", "self", "eps"}
                and isinstance(v, (int, float, bool))
                and (isinstance(v, bool) or abs(v) <= 1_000_000)}

    def record(self, kind, name, fn, args, kwargs):
        if not self.capturing():
            return
        dims = self._named_scalars(fn, args, kwargs)
        row = {"scope": self.scope, "kind": kind, "operation": name,
               "shape_args": dims, "layer_dims": self.dims.copy()}
        if kind == "gemm" and all(k in dims for k in ("M", "N", "K")):
            m, n, k = (int(dims[key]) for key in ("M", "N", "K"))
            row.update({"M": m, "N": n, "K": k,
                        "logical_A": [m, k], "logical_B": [k, n],
                        "logical_output": [m, n]})
        self.operations.append(row)

    def install_operations(self):
        import flash_rt.flash_rt_kernels as fvk

        original_runner = fvk.GemmRunner
        trace = self

        class RecordingGemmRunner:
            def __init__(self, *args, **kwargs):
                self._runner = original_runner(*args, **kwargs)

            def __getattr__(self, name):
                fn = getattr(self._runner, name)
                if not callable(fn):
                    return fn

                def call(*args, **kwargs):
                    trace.record("gemm", name, fn, args, kwargs)
                    return fn(*args, **kwargs)
                return call

        self._originals.append((fvk, "GemmRunner", original_runner))
        fvk.GemmRunner = RecordingGemmRunner
        for name in dir(fvk):
            fn = getattr(fvk, name)
            if name.startswith("_") or not inspect.isbuiltin(fn):
                continue

            def call(*args, _fn=fn, _name=name, **kwargs):
                trace.record("kernel_api", _name, _fn, args, kwargs)
                return _fn(*args, **kwargs)

            self._originals.append((fvk, name, fn))
            setattr(fvk, name, call)

        from flash_rt.hardware.rtx.attn_backend_groot_n17 import (
            RtxFlashAttnBackendGrootN17,
        )
        cls = RtxFlashAttnBackendGrootN17
        original_run = cls.run

        def run(attn, site, layer_idx, *args, **kwargs):
            if trace.capturing():
                trace.operations.append({"scope": trace.scope,
                                         "kind": "attention", "operation": site,
                                         "site_index": layer_idx,
                                         "shape_args": {k: v for k, v in kwargs.items()
                                                        if k != "stream"},
                                         "layer_dims": trace.dims.copy()})
            return original_run(attn, site, layer_idx, *args, **kwargs)

        self._originals.append((cls, "run", original_run))
        cls.run = run

    def restore(self):
        for owner, name, original in reversed(self._originals):
            setattr(owner, name, original)


def event_ms(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--output", required=True, help="output directory")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    from flash_rt.frontends.torch.groot_n17_rtx_fp8 import (
        GrootN17TorchFrontendRtxFP8,
    )
    fixture = Path(args.fixture)
    ref = torch.load(fixture, weights_only=False, map_location="cpu")
    aux = torch.load(fixture.with_name(fixture.stem + "_llm_aux.pt"),
                     weights_only=False, map_location="cpu")
    aux = {k: v.cuda() if isinstance(v, torch.Tensor) else v
           for k, v in aux.items()}

    def make_frontend():
        fe = GrootN17TorchFrontendRtxFP8(
            args.ckpt, num_views=2,
            embodiment_tag="oxe_droid_relative_eef_relative_joint")
        fe.set_prompt(aux=aux, prompt="Put the blue block in the green bowl")
        state = fe.normalize_state(
            {"state." + k: v for k, v in ref["inputs"]["state"].items()}).cuda()
        return fe, state

    noise = aux["initial_noise"].bfloat16()
    def infer(fe, state):
        return fe.infer(state, aux=aux, initial_noise=noise,
                        num_inference_timesteps=4, action_horizon=40,
                        use_dit_graph=True)

    # Separate uninstrumented instance gives the actual production reference.
    fe, state = make_frontend()
    for _ in range(args.warmup):
        output = infer(fe, state)
    torch.cuda.synchronize()
    assert list(output.shape) == [1, 40, 132] and torch.isfinite(output).all()
    baseline = [event_ms(lambda: infer(fe, state)) for _ in range(args.iters)]
    del fe
    torch.cuda.empty_cache()

    trace = CaptureTrace()
    trace.install_layers()
    trace.install_operations()
    try:
        fe, state = make_frontend()
        infer(fe, state)  # lazily captures both graphs
        torch.cuda.synchronize()
        assert len(trace.events) == 24 + 3 + 16 + 1 + 4 + 32 * 4, (
            "unexpected captured layer count", len(trace.events))
        trace.restore()  # replay uses captured nodes; no wrappers on the hot path
        for _ in range(args.warmup):
            infer(fe, state)
        samples = defaultdict(list)
        backbone_ms, action_ms, full_ms = [], [], []
        for _ in range(args.iters):
            full_ms.append(event_ms(lambda: infer(fe, state)))
            backbone_ms.append(event_ms(lambda: fe.run_backbone_graph(aux)))
            action_ms.append(event_ms(lambda: fe.infer(
                state, initial_noise=noise, num_inference_timesteps=4,
                action_horizon=40, use_dit_graph=True)))
            for entry in trace.events:
                samples[entry["scope"]].append(
                    entry["start"].elapsed_time(entry["end"]))
    finally:
        trace.restore()

    layer_rows = []
    for entry in trace.events:
        label = entry["scope"]
        layer_rows.append({"scope": label, "stage": entry["stage"],
                           "calls_per_inference": 1,
                           **stats(samples[label]),
                           "layer_dims": json.dumps(entry["dims"], sort_keys=True)})
    write_csv(out / "layer_times.csv", layer_rows)

    # One capture contains precisely one inference. Preserve order and every
    # API call; aggregating by GEMM shape can be done without losing ownership.
    operation_rows = []
    for index, item in enumerate(trace.operations):
        operation_rows.append({"capture_index": index, "scope": item["scope"],
                               "kind": item["kind"], "operation": item["operation"],
                               "M": item.get("M", ""), "N": item.get("N", ""),
                               "K": item.get("K", ""),
                               "logical_A": json.dumps(item.get("logical_A", "")),
                               "logical_B": json.dumps(item.get("logical_B", "")),
                               "logical_output": json.dumps(item.get("logical_output", "")),
                               "shape_args": json.dumps(item["shape_args"], sort_keys=True),
                               "layer_dims": json.dumps(item["layer_dims"], sort_keys=True)})
    write_csv(out / "operation_shapes.csv", operation_rows)
    bb_layer_ms = sum(row["mean_ms"] for row in layer_rows if row["stage"] != "dit")
    dit_layer_ms = sum(row["mean_ms"] for row in layer_rows if row["stage"] == "dit")
    report = {"frontend": type(fe).__name__, "gpu": torch.cuda.get_device_name(),
              "warmup": args.warmup, "iterations": args.iters,
              "uninstrumented_e2e": stats(baseline),
              "instrumented_e2e": stats(full_ms),
              "instrumented_backbone": stats(backbone_ms),
              "instrumented_action": stats(action_ms),
              "backbone_layer_mean_sum_ms": bb_layer_ms,
              "action_dit_layer_mean_sum_ms": dit_layer_ms,
              "backbone_unattributed_mean_ms": statistics.mean(backbone_ms) - bb_layer_ms,
              "action_unattributed_mean_ms": statistics.mean(action_ms) - dit_layer_ms,
              "captured_layers": len(trace.events),
              "captured_api_calls": len(operation_rows),
              "note": "Instrumented graph includes timing event nodes. Residual includes "
                      "unwrapped ops, copies, event overhead and timing differences; "
                      "logical GEMM shapes do not prove physical matrix layout."}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    with torch.no_grad():
        main()
