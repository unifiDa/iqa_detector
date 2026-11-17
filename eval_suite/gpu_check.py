#!/usr/bin/env python3
"""gpu_check.py - checks that torch and onnxruntime see and use the GPU (pixi run -e gpu gpu-check)."""
import sys

ok = True
try:
    import torch
    print(f"torch {torch.__version__}  CUDA build {torch.version.cuda}  cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        arch = f"sm_{major}{minor}"
        archs = torch.cuda.get_arch_list()
        print(f"  GPU: {torch.cuda.get_device_name(0)} ({arch}); build architectures: {archs}")
        x = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
        torch.cuda.synchronize()
        print(f"  fp16 matmul on GPU: ok ({float((x @ x).float().abs().mean()):.3f})")
        if arch not in archs and f"compute_{major}{minor}" not in archs:
            ok = False
            print(f"  [!] build without {arch}")
    else:
        ok = False
except Exception as e:  # noqa: BLE001
    ok = False
    print(f"torch: {type(e).__name__}: {e}")

try:
    import onnxruntime as ort
    print(f"onnxruntime {ort.__version__}  available providers: {ort.get_available_providers()}")
    if "CUDAExecutionProvider" in ort.get_available_providers():
        try:
            ort.preload_dlls()
        except Exception:  # noqa: BLE001
            pass
        import numpy as np
        from onnx import TensorProto, helper
        g = helper.make_graph([helper.make_node("Relu", ["x"], ["y"])], "t",
                              [helper.make_tensor_value_info("x", TensorProto.FLOAT, [4])],
                              [helper.make_tensor_value_info("y", TensorProto.FLOAT, [4])])
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
        m.ir_version = 8
        s = ort.InferenceSession(m.SerializeToString(), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        s.run(None, {"x": np.array([-1, 0, 1, 2], np.float32)})
        print(f"  test session on: {s.get_providers()[0]}")
except Exception as e:  # noqa: BLE001
    print(f"onnxruntime: {type(e).__name__}: {e}")

print("\nGPU ready for B, C, D (torch)." if ok else "\nGPU NOT usable by torch: see above.")
sys.exit(0 if ok else 1)
