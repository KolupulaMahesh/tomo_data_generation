#!/usr/bin/env bash
# Normal / single-failure / multi-failure episodes of a scene, named normal1.., single1.., multi1.. (2026-11-17).
# Episode k (all categories in a row) = seed FIRST+k:
#   make_motion.py --seed      the .blend's Synthesizer: random phone place, balanced table colour / phone print
#   data_gen.generate          failures + recoveries (none for normal), robot limits checked, rendered in Isaac, packed
# -> <OUT>/normal1 .. normalN, single1 .., multi1 .. + task.yaml + generate.log. One Isaac Sim at a time.
#   bash generate_mix.sh <scene> <normal> <single> <multi> [first seed=0] --out DIR
# The failure / recovery part needs the scene's recovery task file (scenes/<scene>.yaml: recovery_task).
set -uo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$D"
POS=(); OUT=""
while [ $# -gt 0 ]; do
  case "$1" in --out) OUT=$2; shift ;; *) POS+=("$1") ;; esac
  shift
done
SCENE=${POS[0]}; NN=${POS[1]}; NS=${POS[2]}; NM=${POS[3]}; FIRST=${POS[4]:-0}
PY=$D/env/bin/python
DG=$(cd "$D/../../data_gen" && pwd)
OUT=${OUT:-$D/datasets/${SCENE}_$(date +%Y%m%d_%H%M)}
TASK=$("$PY" -c "import yaml, pathlib; p = pathlib.Path('scenes/$SCENE.yaml'); t = yaml.safe_load(p.read_text()).get('recovery_task'); print((p.parent / t).resolve() if t else '')")
if [ $((NS + NM)) -gt 0 ] && [ -z "$TASK" ]; then
  echo "scene $SCENE has no recovery_task in scenes/$SCENE.yaml: only normal episodes can be made"; exit 1
fi
if pgrep -f "isaacsim/kit/python" > /dev/null; then echo "another Isaac Sim is running - one at a time"; exit 1; fi
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)                               # absolute: the episode step runs in data_gen/
LOG=$OUT/generate.log
trap 'trap - INT TERM; echo "[generate] $(date +%T) STOPPED" | tee -a "$LOG"; pkill -TERM -P $$; kill -- -$$ 2>/dev/null; exit 130' INT TERM
run() { "$@" & wait $!; }
"$PY" make_task_file.py "$SCENE" "$OUT" -1 || { echo "task file FAILED"; exit 1; }
N=$((NN + NS + NM)); k=0; ok=0
echo "[generate] $(date '+%F %T') $SCENE: $NN normal, $NS single, $NM multi from seed $FIRST -> $OUT" | tee -a "$LOG"
for CAT in normal single multi; do
  case $CAT in normal) C=$NN ;; single) C=$NS ;; multi) C=$NM ;; esac
  for ((j = 1; j <= C; j++)); do
    S=$((FIRST + k)); k=$((k + 1)); NAME=$CAT$j
    EP=$D/runs/$SCENE/ep_$(printf %04d $S)/generic
    echo "[generate] $(date +%T) episode $k/$N ($NAME, seed $S)" | tee -a "$LOG"
    # (the phone's carry is kept >= 3 cm from the carton by data_gen/carry_fix.py: same phone place, hand path corrected)
    if ! run "$PY" make_motion.py "$SCENE" --seed $S >> "$LOG" 2>&1; then echo "  motion FAILED (see $LOG)" | tee -a "$LOG"; continue; fi
    TMP=$OUT/.work_$NAME; rm -rf "$TMP"
    ( cd "$DG" && run python3 -m data_gen.generate ${TASK:+--task "$TASK"} --scene "$SCENE" --motion "$EP/motion.npz" \
        --category $CAT --seed0 $S --episodes 1 --render --out "$TMP" ) >> "$LOG" 2>&1
    SRC=$TMP/dataset/fs_$(printf %03d $S)
    if [ ! -f "$SRC/manifest.json" ]; then echo "  generate FAILED (see $LOG)" | tee -a "$LOG"; continue; fi
    rm -rf "$OUT/$NAME"; mv "$SRC" "$OUT/$NAME"
    [ -f "$TMP/dataset/scene_objects.npz" ] && [ ! -f "$OUT/scene_objects.npz" ] && mv "$TMP/dataset/scene_objects.npz" "$OUT/"
    cp "$TMP"/ep_*/events.json "$OUT/$NAME/events_raw.json" 2>/dev/null
    rm -rf "$TMP" "$EP/raw"                                              # no Isaac episode file kept (user 2026-11-17: storage)
    ok=$((ok + 1))
    echo "  done -> $OUT/$NAME | $(grep -o '"success": [a-z]*' "$OUT/$NAME/validation.json" 2>/dev/null | head -1)" | tee -a "$LOG"
  done
done
echo "[generate] $(date '+%F %T') finished: $ok of $N episodes in $OUT" | tee -a "$LOG"
