#!/bin/bash
# Prefetch Phase 2 datasets to GCS so Phase 2 starts instantly on the TPU VM.
# Runs on any machine with internet + gsutil — no TPU or GPU needed.
#
# Downloads:
#   - PDBbind v2020 plain-text index (~5 MB)
#   - ZINC FDA approved subset (~20 MB SDF)
#   - 5 MS target PDB structures (~1 MB each)
#
# Uploads everything to gs://aegismind-tpu-results/phase2_data/
# Phase 2 rsyncs from there on start.

set -e

GCS_DEST="gs://aegismind-tpu-results/phase2_data"
LOCAL="$(mktemp -d /tmp/phase2_prefetch.XXXX)"
trap "rm -rf $LOCAL" EXIT

echo "[$(date)] Phase 2 prefetch starting → $GCS_DEST"

# ── PDBbind v2020 index ───────────────────────────────────────────────────────
PDBBIND_URL="https://pdbbind.oss-cn-hangzhou.aliyuncs.com/download/PDBbind_v2020_plain_text_index.tar.gz"
PDBBIND_DEST="$LOCAL/pdbbind_index.tar.gz"
if gsutil -q stat "$GCS_DEST/pdbbind_index.tar.gz" 2>/dev/null; then
    echo "[$(date)] PDBbind index already in GCS — skipping"
else
    echo "[$(date)] Downloading PDBbind v2020 index..."
    wget -q -O "$PDBBIND_DEST" "$PDBBIND_URL" \
        && echo "[$(date)] PDBbind: $(du -sh $PDBBIND_DEST | cut -f1)" \
        || echo "[$(date)] WARNING: PDBbind download failed — Phase 2 will use mock data"
    [ -f "$PDBBIND_DEST" ] && gsutil cp "$PDBBIND_DEST" "$GCS_DEST/pdbbind_index.tar.gz"
fi

# ── ZINC FDA approved subset ──────────────────────────────────────────────────
ZINC_URL="https://zinc.docking.org/substances/subsets/fda.sdf?count=all"
ZINC_DEST="$LOCAL/zinc_fda.sdf"
if gsutil -q stat "$GCS_DEST/zinc_fda.sdf" 2>/dev/null; then
    echo "[$(date)] ZINC FDA already in GCS — skipping"
else
    echo "[$(date)] Downloading ZINC FDA subset..."
    wget -q -O "$ZINC_DEST" "$ZINC_URL" \
        && echo "[$(date)] ZINC FDA: $(du -sh $ZINC_DEST | cut -f1)" \
        || echo "[$(date)] WARNING: ZINC download failed — Phase 2 will screen fewer compounds"
    [ -f "$ZINC_DEST" ] && gsutil cp "$ZINC_DEST" "$GCS_DEST/zinc_fda.sdf"
fi

# ── MS target PDB structures ──────────────────────────────────────────────────
declare -A TARGETS=(
    ["LINGO1"]="7MHH"
    ["PCSK9"]="2P4E"
    ["CTSS"]="1MS6"
    ["GREM1"]="1XH0"
    ["HIF1A"]="1LQB"
    # AMR
    ["KPC3"]="5UL8"
    # HD — CHDI engagement targets (network hub analysis 2026-05-02)
    ["APEX1"]="4LWO"
    ["MSH3"]="3THW"
    ["CREBBP"]="4YGC"
)
mkdir -p "$LOCAL/structures"

for GENE in "${!TARGETS[@]}"; do
    PDB="${TARGETS[$GENE]}"
    DEST="$LOCAL/structures/${PDB}.pdb"
    GCS_KEY="$GCS_DEST/structures/${PDB}.pdb"
    if gsutil -q stat "$GCS_KEY" 2>/dev/null; then
        echo "[$(date)] $GENE ($PDB) already in GCS — skipping"
    else
        echo "[$(date)] Downloading $GENE structure ($PDB)..."
        wget -q -O "$DEST" "https://files.rcsb.org/download/${PDB}.pdb" \
            && gsutil cp "$DEST" "$GCS_KEY" \
            && echo "[$(date)] $GENE: OK" \
            || echo "[$(date)] WARNING: $GENE download failed"
    fi
done

echo ""
echo "[$(date)] Prefetch complete. GCS contents:"
gsutil ls -l "$GCS_DEST/" 2>/dev/null || true
echo "[$(date)] Phase 2 will rsync from $GCS_DEST on startup."
