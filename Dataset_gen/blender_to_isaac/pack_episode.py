"""Pack one rendered episode (render_episode.py: raw/ + labels_raw.npz) into the episode format the training reads - the
files of the dataset generator's fs_pack.py (rgb_jpg/rgb_off, depth_png/depth_off, depth_valid, frames, q, suction,
phone_world/phone_uv/phone_visible, labels.npz, segments.json, joint_names.json, manifest.json, variant.json) - with
nothing about one scene in the code (2026-10-29): the steps, the vacuums' IO pins and the tracked object come from
episode.json (scenes/<scene>.yaml); the vacuums are the robot's IO byte the .blend's markers set (io0..io7).
Optionally the live head camera's black bars painted in (camera_bars). Plus preview_h264.mp4.

    env/bin/python pack_episode.py <episode dir: runs/<scene>/ep_N/generic> <out dir: <dataset>/fs_NNN>
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

import objects_file as OBJ

HERE = Path(__file__).resolve().parent


def bars_painter(bars, PW, FW):
    """camera_bars {colour_cols: [x0, x1), colour_top_rows: n, depth_cols: [x0, x1)} in full-frame pixels."""
    if not bars:
        return lambda im: im, lambda d: d
    s = PW / FW
    c0, c1 = [round(x * s) for x in bars["colour_cols"]]
    top = round(bars.get("colour_top_rows", 0) * s)
    d0, d1 = [round(x * s) for x in bars["depth_cols"]]

    def colour(im):
        im = im.copy()
        im[:, :c0] = 0
        im[:, c1:] = 0
        im[:top] = 0
        return im

    def depth(d):
        d = d.copy()
        d[:, :d0] = 0
        d[:, d1:] = 0
        return d
    return colour, depth


def main():
    ep, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ep / "episode.json").read_text())
    var = json.loads((ep / "variant.json").read_text())
    L = np.load(ep / "labels_raw.npz")
    frames = sorted(int(p.stem[4:]) for p in (ep / "raw").glob("rgb_*.png"))
    assert frames, f"no rendered frames in {ep / 'raw'}"
    idx = np.array(frames)                                               # the rendered frames (the .blend's numbers)
    first = int(L["frames"][0])                                          # labels_raw.npz row 0 = the .blend's first
    rows = idx - first                                                   # frame (not always 0: Blender's default is 1)
    n, T = len(frames), int(idx[-1]) + 1                                 # the episode ends at the last rendered frame
    PW, PH = var["pack_wh"]
    fx, fy, cx, cy, FW, FH = var["K_full"]
    paint, paint_d = bars_painter(cfg.get("camera_bars"), PW, FW)
    names = [str(x) for x in L["object_names"]]
    track = L["object_world"][:, names.index(cfg["track"])] if cfg.get("track") in names else None

    # steps (the scene file's, else one per marker), cut at the last rendered frame
    steps = cfg.get("steps") or [["START", 0]] + cfg.get("markers", [])
    steps = [(s, int(f)) for s, f in steps if int(f) <= idx[-1]]
    starts = np.array([f for _, f in steps])
    io = cfg.get("vacuum_io", {})
    relay = L["relay"].astype(np.uint8)
    vac = {k: ((relay >> int(b)) & 1).astype(np.float32) for k, b in io.items()}

    ff = imageio_ffmpeg.get_ffmpeg_exe()
    vw = subprocess.Popen([ff, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                           "-s", f"{PW}x{PH}", "-r", "30", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                           "-crf", "20", "-movflags", "+faststart", str(out / "preview_h264.mp4")], stdin=subprocess.PIPE)
    rgbs, deps, uv, vis = [], [], [], []
    for f, row in zip(frames, rows):
        im = paint(cv2.resize(cv2.imread(str(ep / "raw" / f"rgb_{f:04d}.png"), cv2.IMREAD_COLOR), (PW, PH),
                              interpolation=cv2.INTER_AREA))
        rgbs.append(np.frombuffer(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes(), np.uint8))
        vw.stdin.write(im.tobytes())
        d = paint_d(np.load(ep / "raw" / f"depth_{f:04d}.npy"))
        deps.append(np.frombuffer(cv2.imencode(".png", d)[1].tobytes(), np.uint8))
        if track is None:
            continue
        # the tracked object's centre in the camera: pixel (full-frame) + visible = in the picture and not hidden
        C = L["cam_world"][row].copy()
        C[:3, :3] /= np.linalg.norm(C[:3, :3], axis=0)
        pc = np.linalg.inv(C) @ np.append(track[row][:3, 3], 1.0)
        zc = -pc[2]
        if zc > 0.05:
            u, v = fx * pc[0] / zc + cx, cy - fy * pc[1] / zc
            ui, vi = int(u * PW / FW), int(v * PH / FH)
            uv.append((u, v))
            vis.append(int(0 <= ui < PW and 0 <= vi < PH and abs(d[vi, ui] / 1000.0 - zc) < 0.05))
        else:
            uv.append((-1.0, -1.0))
            vis.append(0)
    vw.stdin.close()
    vw.wait()

    np.save(out / "rgb_jpg.npy", np.concatenate(rgbs))
    np.save(out / "rgb_off.npy", np.cumsum([0] + [len(x) for x in rgbs]).astype(np.int64))
    np.save(out / "depth_png.npy", np.concatenate(deps))
    np.save(out / "depth_off.npy", np.cumsum([0] + [len(x) for x in deps]).astype(np.int64))
    np.save(out / "depth_valid.npy", np.ones(n, bool))
    np.save(out / "frames.npy", np.arange(n, dtype=np.int64))
    np.save(out / "q.npy", L["q"][rows].astype(np.float32))
    suction = vac.get("right", np.zeros(len(relay), np.float32))[rows]
    np.save(out / "suction.npy", suction)
    if track is not None:
        np.save(out / "phone_world.npy", track[rows].astype(np.float32))
        np.save(out / "phone_uv.npy", np.array(uv, np.float32))
        np.save(out / "phone_visible.npy", np.array(vis, np.int8))
    extra = {}
    for k in [str(x) for x in L["label_names"]]:                          # scenes/<scene>.yaml labels
        a = L[f"label_{k}"][rows]
        if a.ndim == 3:                                                     # a world matrix per frame: its own file
            np.save(out / f"{k}.npy", a.astype(np.float32))
        else:
            extra[k] = a.astype(np.float32)
    ph = np.searchsorted(starts, idx, side="right") - 1
    skills = np.zeros((n, len(steps)), np.float32)
    skills[np.arange(n), ph] = 1
    done = np.zeros(n, np.float32)
    done[-2:] = 1
    np.savez(out / "labels.npz", suction=suction, skills_mask=skills, skill_names=np.array([s for s, _ in steps]),
             task_progress=(idx / max(T - 1, 1)).astype(np.float32), done=done, source_frame=idx.astype(np.int64),
             relay=relay[rows], **{f"{k}_suction": v[rows] for k, v in vac.items() if k != "right"}, **extra)
    out_i = lambda f: int(np.searchsorted(idx, f, side="left"))            # noqa: E731  (.blend frame -> packed frame)
    r = vac.get("right", np.zeros(1))
    offs = np.flatnonzero((r[:-1] > 0) & (r[1:] == 0)) + 1 + first         # the right vacuum switching off (frame)
    json.dump({"flags": [[out_i(f), s] for s, f in steps],
               "release_frame": out_i(int(offs[0])) if len(offs) and offs[0] <= idx[-1] else None,
               "segments": [{"name": s, "start": out_i(f),
                             "end": (out_i(steps[k + 1][1]) - 1 if k + 1 < len(steps) else n - 1)}
                            for k, (s, f) in enumerate(steps)]}, open(out / "segments.json", "w"), indent=2)
    (out / "joint_names.json").write_text(json.dumps([str(x) for x in L["names"]]))
    shutil.copy(ep / "variant.json", out / "variant.json")
    shutil.copy(ep / "episode.json", out / "episode.json")
    json.dump({"episode": out.name, "source_episode": cfg["scene"], "domain": "blender_to_isaac (render_episode.py)",
               "seed": cfg.get("seed"), "offsets": cfg.get("offsets"), "camera_bars": cfg.get("camera_bars"),
               "variant": var, "rendered_frames": n, "last_source_frame": int(idx[-1]), "complete": True},
              open(out / "manifest.json", "w"), indent=2)
    # the dataset's objects file (objects_file.py, 2026-11-08): every object's shape once per dataset, every mesh's pose per frame of
    # this episode - what the YOLO and W2 + IK pipelines build the robot's high-level control from
    exp = HERE / "blender_export" / cfg["scene"] / "objects_export.npz"
    if exp.is_file():
        if not (out.parent / "scene_objects.npz").is_file():
            nm, nb = OBJ.scene_file(exp, out.parent / "scene_objects.npz", cfg["scene"])
            print(f"[pack] {out.parent / 'scene_objects.npz'}: {nm} meshes, {nb} change shape", flush=True)
        src = OBJ.episode_poses(exp, ep / "motion.npz", idx, out / "object_poses.npz")
        print(f"[pack] object_poses.npz: {len(src['motion'])} meshes from this episode's motion, {len(src['frame0'])} at their "
              f"own place, {len(src['export'])} from the export {src['export']}", flush=True)
    else:
        print(f"[pack] no {exp}: this dataset gets no objects file (run blender_scripts/export_scene_objects.py once for the "
              f"scene; generate.sh does it when Blender is found)", flush=True)
    sz = sum(p.stat().st_size for p in out.iterdir() if p.is_file()) / 1e6
    print(f"[pack] {out.name}: {n} frames (to source frame {idx[-1]}), right vacuum on {int(suction.sum())} frames, "
          + (f"phone visible {np.mean(vis):.2f}, " if vis else "") + f"{sz:.0f} MB", flush=True)


if __name__ == "__main__":
    main()
