#!/bin/bash
# FlashOptim TPU environment setup
# Run once on the TPU VM before starting experiments

set -e

echo "=== FlashOptim TPU Setup ==="

# PyTorch + XLA — pick the latest version supported by the VM's Python.
# v5e/v6e VMs (v2-alpha-tpuv6e/v5e) have Python 3.10+ → use 2.9.0
# v4 VMs (v2-alpha-tpuv4) have Python 3.8 → use 2.4.0 (last py38-compatible)
PYMINOR=$(python3 -c "import sys; print(sys.version_info.minor)")
if [ "$PYMINOR" -ge 9 ]; then
    TORCH_VER="2.9.0"
    XLA_VER="2.9.0"
    echo "Python 3.${PYMINOR} — installing torch ${TORCH_VER} + torch_xla ${XLA_VER}"
else
    # Python 3.8 (v4 VMs) — torch_xla[tpu] PyPI max is 2.3.0 for py3.8
    TORCH_VER="2.3.0"
    XLA_VER="2.3.0"
    echo "Python 3.${PYMINOR} (v4 VM) — installing torch ${TORCH_VER} + torch_xla ${XLA_VER}"
fi
pip install \
    torch==${TORCH_VER} \
    "torch_xla[tpu]==${XLA_VER}" \
    -f https://storage.googleapis.com/libtpu-releases/index.html \
    --quiet
# Verify installation and PJRT device access
python3 -c "import torch; import torch_xla; print('torch:', torch.__version__, '| torch_xla:', torch_xla.__version__)" || { echo "torch_xla import failed after install"; exit 1; }
# If the bundled libtpu can't access v4 hardware, fall back to system libtpu
python3 -c "
import torch_xla.core.xla_model as xm
try:
    xm.xla_device()
    print('PJRT TPU device: OK')
except Exception as e:
    import os
    print(f'PJRT with bundled libtpu failed ({e})')
    # Try system libtpu (v4 VMs have /usr/lib/libtpu.so)
    if os.path.exists('/usr/lib/libtpu.so'):
        print('Trying system libtpu at /usr/lib/libtpu.so')
        os.environ['TPU_LIBRARY_PATH'] = '/usr/lib/libtpu.so'
        os.environ['PJRT_DEVICE'] = 'TPU'
        try:
            import importlib
            import torch_xla
            importlib.reload(torch_xla.core.xla_model)
            import torch_xla.core.xla_model as xm2
            xm2.xla_device()
            print('System libtpu: OK')
        except Exception as e2:
            print(f'System libtpu also failed: {e2}')
            print('Training will fall back to CPU if TPU unavailable')
" 2>/dev/null || echo "TPU device check skipped"

# PyTorch Geometric — used for QM9 dataset loading only (not for model ops)
pip install torch_geometric --quiet

# PyG optional dependencies — install pure CPU versions (data loading only)
pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-$(python3 -c "import torch; print(torch.__version__)" | cut -d'+' -f1)+cpu.html --quiet 2>/dev/null || {
    echo "WARNING: torch_scatter/sparse CPU install failed — falling back to download-only mode"
    echo "Data loading will use networkx fallback"
    pip install networkx rdkit --quiet
}

# Other dependencies (rdkit enables real ChEMBL data in Phase 2)
pip install numpy pandas matplotlib tqdm rdkit --quiet

# Open Babel — required by phase21 HD Vina screen for PDBQT conversion
sudo apt-get install -y openbabel 2>/dev/null || pip install openbabel-wheel --quiet 2>/dev/null || echo "WARNING: openbabel install failed — phase21 Vina prep will error"

# AutoDock Vina Python bindings — required for docking in phases 21+
pip install vina meeko --quiet

echo ""
echo "=== Verifying TPU device ==="
python3 -c "
import torch
import torch_xla.core.xla_model as xm
device = xm.xla_device()
print('TPU device:', device)
x = torch.ones(3, 3).to(device)
print('Tensor on TPU:', x.device)
try:
    mem = xm.get_memory_info(device)
    total_kb = mem.get('kb_total', mem.get('bytes_limit', 0) // 1024)
    if total_kb:
        print(f'TPU memory: {total_kb / 1024 / 1024:.1f} GB total')
except Exception:
    print('TPU memory info unavailable on this runtime (non-fatal)')
"

# ── Patch torch_geometric QM9 to skip un-parseable molecules ────────────────
# torch_geometric's SDMolSupplier can yield None for malformed molecules in the
# QM9 SDF; without this patch the dataset processing crashes with AttributeError.
python3 - <<'PYEOF'
import os, site
for sp in site.getsitepackages() + [site.getusersitepackages()]:
    qm9 = os.path.join(sp, "torch_geometric", "datasets", "qm9.py")
    if not os.path.exists(qm9):
        continue
    src = open(qm9).read()
    old = "            if i in skip:\n                continue\n\n            N = mol.GetNumAtoms()"
    new = "            if i in skip:\n                continue\n\n            if mol is None:  # rdkit failed to parse — skip gracefully\n                continue\n\n            N = mol.GetNumAtoms()"
    if old in src:
        open(qm9, "w").write(src.replace(old, new, 1))
        print(f"QM9 None-mol patch applied: {qm9}")
    elif new in src:
        print(f"QM9 None-mol patch already present: {qm9}")
    else:
        print(f"WARNING: QM9 patch pattern not found in {qm9} — check manually")
    break
PYEOF

echo ""
echo "=== Setup complete. Run: bash run.sh ==="
