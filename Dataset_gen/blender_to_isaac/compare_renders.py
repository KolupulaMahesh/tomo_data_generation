"""Compare two rendered episodes frame by frame - the picture (as the dataset packs it: JPEG 95 at 960x540) and the
depth - and make the side-by-side video. Each folder is a packed episode (rgb_jpg.npy ..., as fs_pack.py writes) or a raw
render (raw/rgb_XXXX.png + raw/depth_XXXX.npy). Two renders of the SAME setup differ by ~1.56 grey levels (Isaac's
ray-traced sampling noise, README); depth has no such noise.

    python compare_renders.py <A> <B> [label A] [label B]
-> <A>/compare_vs_<B name>.json + <A>/side_by_side_vs_<B name>_h264.mp4 (A | B | difference x4)
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

# ---------------- settings ----------------
HERE = Path(__file__).resolve().parent
PACK_WH = (960, 540)
NOISE = 1.56                                            # grey levels between two renders of the same setup


def jpeg(im):
    """As fs_pack.py."""
    im = cv2.resize(im, PACK_WH, interpolation=cv2.INTER_AREA)
    return np.frombuffer(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes(), np.uint8)


def episode(ep):
    """-> (frames, rgb(f) -> JPEG bytes, depth(f) -> uint16 mm, packed folder for side_by_side.py or None)."""
    if (ep / "raw").is_dir():
        fr = sorted(int(p.stem[4:]) for p in (ep / "raw").glob("rgb_*.png"))
        return (fr, lambda f: jpeg(cv2.imread(str(ep / "raw" / f"rgb_{f:04d}.png"), cv2.IMREAD_COLOR)),
                lambda f: np.load(ep / "raw" / f"depth_{f:04d}.npy"), None)
    rgb, ro = np.load(ep / "rgb_jpg.npy", mmap_mode="r"), np.load(ep / "rgb_off.npy")
    dep, do = np.load(ep / "depth_png.npy", mmap_mode="r"), np.load(ep / "depth_off.npy")
    src = [int(x) for x in np.load(ep / "labels.npz")["source_frame"]]
    at = {f: i for i, f in enumerate(src)}
    return (src, lambda f: np.asarray(rgb[ro[at[f]]:ro[at[f] + 1]]),
            lambda f: cv2.imdecode(np.asarray(dep[do[at[f]]:do[at[f] + 1]]), cv2.IMREAD_UNCHANGED), ep)


def save_packed(ep, blobs):
    out = ep / "packed"
    out.mkdir(exist_ok=True)
    np.save(out / "rgb_jpg.npy", np.concatenate(blobs))
    np.save(out / "rgb_off.npy", np.cumsum([0] + [len(x) for x in blobs]).astype(np.int64))
    return out


def main():
    A, B = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    labels = sys.argv[3:5] if len(sys.argv) > 4 else [A.name, B.name]
    fa, rgb_a, dep_a, pk_a = episode(A)
    fb, rgb_b, dep_b, pk_b = episode(B)
    frames = sorted(set(fa) & set(fb))
    ja, jb, colour, dpix, dmax = [], [], [], [], []
    box = None
    for f in frames:
        ja.append(rgb_a(f))
        jb.append(rgb_b(f))
        colour.append(float(cv2.absdiff(cv2.imdecode(ja[-1], cv2.IMREAD_COLOR),
                                        cv2.imdecode(jb[-1], cv2.IMREAD_COLOR)).mean()))
        dd = np.abs(dep_a(f).astype(np.int32) - dep_b(f).astype(np.int32))
        dpix.append(int((dd > 0).sum()))
        dmax.append(int(dd.max()))
        if dpix[-1] == max(dpix):
            ys, xs = np.nonzero(dd)
            box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None
    colour, dpix, dmax = np.array(colour), np.array(dpix), np.array(dmax)
    bad = np.flatnonzero(dpix)
    rep = {"A": str(A), "B": str(B), "frames": len(frames), "noise_between_two_same_renders": NOISE,
           "colour_mean_abs_diff_0_255": {"mean": round(float(colour.mean()), 3), "max": round(float(colour.max()), 3),
                                          "max_frame": frames[int(colour.argmax())]},
           "depth": {"frames_with_any_difference": int(len(bad)), "pixels_different_total": int(dpix.sum()),
                     "pixels_per_frame": PACK_WH[0] * PACK_WH[1], "max_mm": int(dmax.max()),
                     "worst_frame": frames[int(dpix.argmax())] if len(bad) else None,
                     "worst_frame_pixels": int(dpix.max()), "worst_frame_box_x0_y0_x1_y1": box,
                     "frames_different": [frames[i] for i in bad][:400]}}
    tag = B.name
    (A / f"compare_vs_{tag}.json").write_text(json.dumps(rep, indent=1))
    print("[compare]", json.dumps({k: v for k, v in rep.items() if k not in ("A", "B")})[:900])
    pa = pk_a or save_packed(A, ja)
    pb = pk_b or save_packed(B, jb)
    mp4 = A / f"side_by_side_vs_{tag}.mp4"
    subprocess.run([sys.executable, str(HERE / "side_by_side.py"), str(pa), str(pb), str(mp4), *labels], check=True)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(mp4),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart",
                    str(mp4.with_name(mp4.stem + "_h264.mp4"))], check=True)
    mp4.unlink()
    print(f"[compare] video -> {mp4.with_name(mp4.stem + '_h264.mp4')}")


if __name__ == "__main__":
    main()
