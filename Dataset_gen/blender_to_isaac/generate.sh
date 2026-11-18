#!/usr/bin/env bash
# N dataset episodes of a scene (2026-10-29). Episode k = seed FIRST+k:
#   make_motion.py --seed  (every object the .blend's Synthesizer moves: random offset inside the .blend's range; the
#                           robot's motion re-solved by the rig; a random look)
#   robot_limits/check_motion.py  (the real robot's joint limits / speed / self-collision: report, limits.json)
#   render_episode.py      (Isaac: colour + depth from the head camera, live lens)
#   pack_episode.py        (the episode format the training reads)
# -> datasets/<scene>_<date>/fs_<seed>/ + task.yaml (+ its files) + generate.log. One Isaac Sim at a time.
#   bash generate.sh <scene> <N> [first seed=0] [last frame=-1: all] [--bars] [--out DIR]
#   e.g. bash generate.sh iPhone_Packing_local 50 0 505 --bars    (50 episodes up to the closed lid, live black bars)
# Ctrl-C (or the training window's STOP) ends the running step at once; finished episodes are kept.
set -uo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$D"
POS=(); BARS=""; OUT=${OUT:-}
while [ $# -gt 0 ]; do
  case "$1" in --bars) BARS="--bars" ;; --out) OUT=$2; shift ;; *) POS+=("$1") ;; esac
  shift
done
SCENE=${POS[0]}; N=${POS[1]}; FIRST=${POS[2]:-0}; LAST=${POS[3]:--1}
ISAAC=$HOME/isaacsim/python.sh
PY=$D/env/bin/python
OUT=${OUT:-$D/datasets/${SCENE}_$(date +%Y%m%d_%H%M)}
# another Isaac Sim = its own python under ~/isaacsim/kit (this script's name never matches that)
if pgrep -f "isaacsim/kit/python" > /dev/null; then echo "another Isaac Sim is running - one at a time"; exit 1; fi
mkdir -p "$OUT"
LOG=$OUT/generate.log
trap 'trap - INT TERM; echo "[generate] $(date +%T) STOPPED" | tee -a "$LOG"; pkill -TERM -P $$; kill -- -$$ 2>/dev/null; exit 130' INT TERM
run() { "$@" & wait $!; }                     # each step in the background: a stop does not wait for it to finish
"$PY" make_task_file.py "$SCENE" "$OUT" "$LAST" || { echo "task file FAILED"; exit 1; }
# the objects export (once per scene, Blender read-only): pack_episode.py writes the dataset's objects file from it (2026-11-08)
OBJ_EXPORT=$D/blender_export/$SCENE/objects_export.npz
if [ ! -f "$OBJ_EXPORT" ]; then
  BLENDER=$(command -v blender || echo "$HOME/blender/blender-5.2.2-linux-x64/blender")
  BLEND=$("$PY" -c "import yaml, pathlib, sys; p = pathlib.Path('scenes/$SCENE.yaml'); print((p.parent / yaml.safe_load(p.read_text())['blend']).resolve())")
  if [ -x "$BLENDER" ] && [ -f "$BLEND" ]; then
    echo "[generate] $(date +%T) objects export of $SCENE (once): $BLEND" | tee -a "$LOG"
    run "$BLENDER" -b "$BLEND" --python blender_scripts/export_scene_objects.py -- "$OBJ_EXPORT" --urdf "$("$PY" scene_robot.py "$SCENE" urdf)" >> "$LOG" 2>&1 \
      || echo "  objects export FAILED (see $LOG): this dataset gets no objects file" | tee -a "$LOG"
  else
    echo "[generate] no Blender ($BLENDER) or no .blend ($BLEND): this dataset gets no objects file" | tee -a "$LOG"
  fi
fi
echo "[generate] $(date '+%F %T') $SCENE: $N episodes from seed $FIRST, last frame $LAST ${BARS:+(live black bars)} -> $OUT" | tee -a "$LOG"
ok=0
for ((k = 0; k < N; k++)); do
  S=$((FIRST + k)); EP=$D/runs/$SCENE/ep_$(printf %04d $S)/generic; NAME=fs_$(printf %03d $S)
  echo "[generate] $(date +%T) episode $((k + 1))/$N (seed $S)" | tee -a "$LOG"
  if ! run "$PY" make_motion.py "$SCENE" --seed $S $BARS >> "$LOG" 2>&1; then echo "  motion FAILED (see $LOG)" | tee -a "$LOG"; continue; fi
  run "$PY" robot_limits/check_motion.py "$EP/motion.npz" --scene "$SCENE" --out "$EP/limits.json" > "$EP/limits.log" 2>&1; LIM=$?
  if ! run "$ISAAC" render_episode.py "$EP" --last-frame $LAST > "$EP/render.log" 2>&1; then echo "  render FAILED (see $EP/render.log)" | tee -a "$LOG"; continue; fi
  if ! run "$PY" pack_episode.py "$EP" "$OUT/$NAME" >> "$LOG" 2>&1; then echo "  pack FAILED (see $LOG)" | tee -a "$LOG"; continue; fi
  cp "$EP/limits.json" "$OUT/$NAME/limits.json" 2>/dev/null
  rm -rf "$EP/raw"                                                    # the per-frame renders (1.6 GB) are packed now
  ok=$((ok + 1))
  echo "  done -> $OUT/$NAME | robot limits: $([ $LIM = 0 ] && echo ok || echo 'broken somewhere, see limits.json')" | tee -a "$LOG"
done
echo "[generate] $(date '+%F %T') finished: $ok of $N episodes in $OUT" | tee -a "$LOG"
