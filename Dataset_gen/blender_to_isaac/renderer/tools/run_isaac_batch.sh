#!/bin/bash
# Full-sequence dataset, Isaac Sim only: N episodes, random iPhone start pose (within 10 cm, +-25 deg; right arm re-solved
# by IK, unreachable poses re-drawn), random phone box (14 Plus / 14 Pro Max print, random hue), random table colour.
# Tomo head camera 1920x1080 (fx 1070.79 fy 1070.53 cx 962.31 cy 550), saved 960x540, 30 fps (STRIDE=1).
#   N=250 bash tools/run_isaac_batch.sh
# Settings (env): N, STRIDE (1), SEED (0), ISAAC (path to isaac-sim python.sh), PY (python with numpy+opencv), OUT,
#   W (960), WARM (2)
# Resumable: finished episodes (manifest.json) are skipped.
# 2026-10-21 (Praveen + Claude): render straight at 960x540 (W=960: the same lens, scaled) with 2 warm-up updates
# instead of 1920x1080 + 8: 0.15 s/frame instead of 0.44 on the RTX 5050 laptop; depth identical (0 mm median, 99% within 2 mm),
# colour within ~4/255 (test: 38 frames of episode 0). W=1920 WARM=8 = the original settings.
ROOT=$(cd "$(dirname "$0")/.." && pwd)
N=${N:-250}; STRIDE=${STRIDE:-1}; SEED=${SEED:-0}; W=${W:-960}; WARM=${WARM:-2}
ISAAC=${ISAAC:-$HOME/isaac-sim/python.sh}; PY=${PY:-python3}
T=$ROOT/tools; OUT=${OUT:-$ROOT/dataset_isaac}; mkdir -p $OUT
PROMAX=$ROOT/assets/phone_textures/iphone_14_pro_max_texture_0.2x.png
[ -x "$ISAAC" ] || { echo "Isaac Sim python not found: $ISAAC  (set ISAAC=/path/to/isaac-sim/python.sh)"; exit 1; }
$PY -c "import numpy, cv2" || { echo "PY=$PY needs numpy and opencv-python"; exit 1; }
for i in $(seq 0 $((N - 1))); do
  ep=$OUT/fs_$(printf %03d $i)
  [ -f $ep/manifest.json ] && { echo "skip $ep"; continue; }
  ok=0
  for try in $(seq 0 19); do
    ARGS=$($PY - <<PYEOF
import numpy as n
r = n.random.default_rng($SEED * 100000 + $i * 10 + $try)
rad, ang = 0.10 * n.sqrt(r.random()), r.uniform(0, 2 * n.pi)
pose = "--dx 0 --dy 0 --dyaw 0" if $i == 0 else f"--dx {rad*n.cos(ang):.3f} --dy {rad*n.sin(ang):.3f} --dyaw {r.uniform(-25,25):.1f}"
tex = "--phone-tex $PROMAX" if r.random() < 0.5 else ""
hue = f"--phone-hue {r.uniform(0,180):.0f}" if r.random() < 0.5 else ""
tables = [None, (.55,.57,.60), (.85,.85,.82), (.75,.72,.66), (.45,.30,.18), (.70,.60,.45), (.15,.20,.35), (.18,.30,.20), (.35,.35,.35)]
t = tables[r.integers(len(tables))]
print(pose, tex, hue, "" if t is None else f"--table-rgb {t[0]},{t[1]},{t[2]}")
PYEOF
)
    rm -rf $ep; mkdir -p $ep
    "$ISAAC" $T/fs_isaac_gen.py --out $ep $ARGS --ik-only > /dev/null 2>&1
    ok=$($PY -c "import json; v=json.load(open('$ep/variant.json')); print(int(v['ik_max_pos_mm']<2 and v['limit_ok']))" 2>/dev/null)
    [ "$ok" = "1" ] && break; echo "fs_$i ($ARGS) not reachable, retry"
  done
  [ "$ok" = "1" ] || { echo "!!! fs_$i: no reachable pose in 20 tries, skipped"; rm -rf $ep; continue; }
  echo "=== fs_$i  $ARGS  $(date +%H:%M)"
  "$ISAAC" $T/fs_isaac_gen.py --out $ep $ARGS --stride $STRIDE --W $W --warm $WARM > $ep.log 2>&1 \
    && $PY $T/fs_pack.py $ep && rm -f $ep.log $ep/phone_print.png || echo "!!! fs_$i failed, see $ep.log"
done
echo "done: $(ls -d $OUT/fs_*/manifest.json 2>/dev/null | wc -l) episodes in $OUT"
