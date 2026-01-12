#!/usr/bin/env python3
import os
import subprocess
import sys

def run(cmd):
    return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()

def get_visible():
    # Could be like "0,1" or GPU UUIDs like "GPU-xxxx,..."
    return os.environ.get("CUDA_VISIBLE_DEVICES", "(not set)")

def try_torch():
    try:
        import torch  # type: ignore
    except Exception:
        return None

    if not torch.cuda.is_available():
        return {"backend": "torch", "has_gpu": False}

    cur = torch.cuda.current_device()
    name = torch.cuda.get_device_name(cur)
    props = torch.cuda.get_device_properties(cur)
    # local index == index within visible devices
    return {
        "backend": "torch",
        "has_gpu": True,
        "local_device_index": cur,
        "name": name,
        "total_mem_gb": round(props.total_memory / (1024**3), 2),
        "cc": f"{props.major}.{props.minor}",
    }

def try_nvidia_smi(local_index_guess=0):
    try:
        out = run(["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"])
    except Exception as e:
        return {"backend": "nvidia-smi", "ok": False, "error": str(e)}

    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            gpus.append({"index": parts[0], "uuid": parts[1], "name": ",".join(parts[2:]).strip()})

    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    chosen = None

    if vis:
        tokens = [t.strip() for t in vis.split(",") if t.strip()]
        # If CUDA_VISIBLE_DEVICES lists UUIDs, match UUID; if numbers, match index.
        if tokens and tokens[0].startswith("GPU-"):
            if local_index_guess < len(tokens):
                want_uuid = tokens[local_index_guess]
                for g in gpus:
                    if g["uuid"] == want_uuid:
                        chosen = g
                        break
        else:
            # numeric indices
            if local_index_guess < len(tokens):
                want_index = tokens[local_index_guess]
                for g in gpus:
                    if g["index"] == want_index:
                        chosen = g
                        break
    else:
        # not set -> assume local 0 maps to physical 0
        if gpus:
            chosen = gpus[min(local_index_guess, len(gpus)-1)]

    return {"backend": "nvidia-smi", "ok": True, "all": gpus, "chosen": chosen}

def main():
    print(f"CUDA_VISIBLE_DEVICES = {get_visible()}")

    t = try_torch()
    if t is not None:
        if not t.get("has_gpu", False):
            print("GPU: (torch) no CUDA device available")
            return

        print(f"GPU (torch) local_index = {t['local_device_index']}")
        print(f"GPU name              = {t['name']}")
        print(f"Compute Capability    = {t['cc']}")
        print(f"Total VRAM (GB)       = {t['total_mem_gb']}")
        # Also show physical mapping via nvidia-smi if possible
        smi = try_nvidia_smi(local_index_guess=t["local_device_index"])
        if smi.get("ok") and smi.get("chosen"):
            c = smi["chosen"]
            print(f"nvidia-smi index      = {c['index']}")
            print(f"GPU UUID              = {c['uuid']}")
        return

    # No torch: fall back
    smi = try_nvidia_smi(local_index_guess=0)
    if not smi.get("ok"):
        print("GPU: cannot determine (no torch, and nvidia-smi failed)")
        print("Reason:", smi.get("error"))
        sys.exit(1)

    c = smi.get("chosen")
    if c:
        print(f"GPU (guess local_index=0) -> nvidia-smi index = {c['index']}")
        print(f"GPU name                                   = {c['name']}")
        print(f"GPU UUID                                   = {c['uuid']}")
    else:
        print("GPU: could not map visible device to a physical GPU.")
        print("All GPUs from nvidia-smi:")
        for g in smi.get("all", []):
            print(f"  index={g['index']} uuid={g['uuid']} name={g['name']}")

if __name__ == "__main__":
    main()
