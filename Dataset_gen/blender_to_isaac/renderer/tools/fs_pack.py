"""Pack one rendered full-sequence variant (fs_render.py output) into the ACT episode format used by the other TOMO datasets.
  ~/miniconda3/envs/tomo_qsa/bin/python fs_pack.py <episode_dir> [--keep-raw]
Writes rgb_jpg/rgb_off/depth_png/depth_off/depth_valid/frames/q/suction/phone_world/phone_uv/phone_visible .npy,
labels.npz, segments.json, manifest.json, preview.mp4 and removes raw/ (the big per-frame renders)."""
import sys, os, json, glob, shutil
import numpy as np, cv2

ep = os.path.abspath(sys.argv[1]); keep = "--keep-raw" in sys.argv
raw = os.path.join(ep, "raw"); L = np.load(os.path.join(ep, "labels_raw.npz")); var = json.load(open(os.path.join(ep, "variant.json")))
T = len(L["q"]); PW, PH = tuple(var.get("pack_wh", (336, 196)))
rgb_f = {int(os.path.basename(p)[4:-4]): p for p in glob.glob(os.path.join(raw, "rgb_*.png"))}
dep_f = {int(os.path.basename(p)[6:-4]): p for p in glob.glob(os.path.join(raw, "depth_*.npy"))}
frames = sorted(set(rgb_f) & set(dep_f))
assert frames, f"no rendered frames in {raw}"
idx = np.array(frames)                                              # rendered frames (all, or every Nth with --stride)

# phases of the recorded sequence (frames from the constraint / IO keys of full_sequnece_001)
PHASES = [("READY", 0), ("PICK_PHONE", 45), ("PLACE_PHONE", 155), ("CLOSE_CARTON", 291), ("APPLY_TAPE", 506),
          ("FLIP_BOX", 609), ("PACK_IN_HOLDER", 725), ("MOVE_BOX_OUT", 1058), ("RETURN_HOME", 1096)]
starts = np.array([s for _, s in PHASES])

rgbs, deps, uv, vis = [], [], [], []
fx, fy, cx, cy, FW, FH = var.get("K_full", (1070.789795, 1070.526978, 962.309387, 550.0, 1920, 1080))   # labels in native camera pixels
vw = cv2.VideoWriter(os.path.join(ep, "preview.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30.0 / max(1, int(np.median(np.diff(idx))) if len(idx) > 1 else 1), (PW, PH))
for f in frames:
    im = cv2.resize(cv2.imread(rgb_f[f], cv2.IMREAD_COLOR), (PW, PH), interpolation=cv2.INTER_AREA)
    ok, j = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 95]); rgbs.append(np.frombuffer(j.tobytes(), np.uint8)); vw.write(im)
    d = np.load(dep_f[f]); ok, p = cv2.imencode(".png", d); deps.append(np.frombuffer(p.tobytes(), np.uint8))
    C = L["cam_world"][f].copy(); C[:3, :3] /= np.linalg.norm(C[:3, :3], axis=0)
    pc = np.linalg.inv(C) @ np.append(L["phone_world"][f][:3, 3], 1.0); zc = -pc[2]
    if zc > 0.05:
        u, v = fx * pc[0] / zc + cx, cy - fy * pc[1] / zc; uv.append((u, v))
        ui, vi = int(u * PW / FW), int(v * PH / FH)
        vis.append(int(0 <= ui < PW and 0 <= vi < PH and abs(d[vi, ui] / 1000.0 - zc) < 0.05))
    else:
        uv.append((-1.0, -1.0)); vis.append(0)
vw.release()
n = len(frames)
np.save(os.path.join(ep, "rgb_jpg.npy"), np.concatenate(rgbs)); np.save(os.path.join(ep, "rgb_off.npy"), np.cumsum([0] + [len(x) for x in rgbs]).astype(np.int64))
np.save(os.path.join(ep, "depth_png.npy"), np.concatenate(deps)); np.save(os.path.join(ep, "depth_off.npy"), np.cumsum([0] + [len(x) for x in deps]).astype(np.int64))
np.save(os.path.join(ep, "depth_valid.npy"), np.ones(n, bool)); np.save(os.path.join(ep, "frames.npy"), np.arange(n, dtype=np.int64))
q = L["q"][idx].astype(np.float32); np.save(os.path.join(ep, "q.npy"), q)
suction = ((L["inf_pick"] > 0.5) | (L["inf_box_out"] > 0.5)).astype(np.float32)[idx]           # right-hand suction holds
np.save(os.path.join(ep, "suction.npy"), suction)
np.save(os.path.join(ep, "phone_world.npy"), L["phone_world"][idx].astype(np.float32))
np.save(os.path.join(ep, "phone_uv.npy"), np.array(uv, np.float32)); np.save(os.path.join(ep, "phone_visible.npy"), np.array(vis, np.int8))
np.save(os.path.join(ep, "box_world.npy"), L["box_world"][idx].astype(np.float32))
ph = np.searchsorted(starts, idx, side="right") - 1
skills = np.zeros((n, len(PHASES)), np.float32); skills[np.arange(n), ph] = 1
done = np.zeros(n, np.float32); done[-2:] = 1
np.savez(os.path.join(ep, "labels.npz"), suction=suction, skills_mask=skills, skill_names=np.array([p for p, _ in PHASES]),
         task_progress=(idx / max(T - 1, 1)).astype(np.float32), done=done, source_frame=idx.astype(np.int64),
         left_tape=(L["inf_tape"] > 0.5).astype(np.float32)[idx], box_flip=(L["inf_flip"] > 0.5).astype(np.float32)[idx],
         box_bones=L["box_bones"][idx], box_bone_names=L["box_bone_names"])
out_i = lambda f: int(np.searchsorted(idx, f, side="left"))
json.dump({"flags": [[out_i(s), p] for p, s in PHASES], "release_frame": out_i(290),
           "segments": [{"name": p, "start": out_i(s), "end": (out_i(PHASES[k + 1][1]) - 1 if k + 1 < len(PHASES) else n - 1)} for k, (p, s) in enumerate(PHASES)]},
          open(os.path.join(ep, "segments.json"), "w"), indent=2)
shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), "joint_names.json"), os.path.join(ep, "joint_names.json"))
json.dump({"episode": os.path.basename(ep), "source_episode": "full_sequnece_001", "domain": "full_sequence_blender",
           "variant": var, "rendered_frames": n, "complete": True}, open(os.path.join(ep, "manifest.json"), "w"), indent=2)
if not keep:
    shutil.rmtree(raw)
sz = sum(os.path.getsize(os.path.join(ep, x)) for x in os.listdir(ep) if os.path.isfile(os.path.join(ep, x))) / 1e6
print(f"[pack] {os.path.basename(ep)}: {n} frames, phone visible {np.mean(vis):.2f}, {sz:.0f} MB", flush=True)
