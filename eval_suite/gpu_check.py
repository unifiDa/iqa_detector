#!/usr/bin/env python3
"""gpu_check.py - verifica che torch e onnxruntime vedano e usino la GPU (pixi run -e gpu gpu-check)."""
import sys

ok = True
try:
    import torch
    print(f"torch {torch.__version__}  build CUDA {torch.version.cuda}  cuda disponibile: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        arch = f"sm_{major}{minor}"
        archs = torch.cuda.get_arch_list()
        print(f"  GPU: {torch.cuda.get_device_name(0)} ({arch}); architetture della build: {archs}")
        x = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
        torch.cuda.synchronize()
        print(f"  matmul fp16 su GPU: ok ({float((x @ x).float().abs().mean()):.3f})")
        if arch not in archs and f"compute_{major}{minor}" not in archs:
            ok = False
            print(f"  [!] build senza {arch}")
    else:
        ok = False
except Exception as e:  # noqa: BLE001
    ok = False
    print(f"torch: {type(e).__name__}: {e}")

try:
    import onnxruntime as ort
    print(f"onnxruntime {ort.__version__}  provider disponibili: {ort.get_available_providers()}")
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
        print(f"  sessione di prova su: {s.get_providers()[0]}")
except Exception as e:  # noqa: BLE001
    print(f"onnxruntime: {type(e).__name__}: {e}")

print("\nGPU pronta per B, C, D (torch)." if ok else "\nGPU NON utilizzabile da torch: vedi sopra.")
sys.exit(0 if ok else 1)
