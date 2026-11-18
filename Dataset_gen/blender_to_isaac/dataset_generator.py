#!/usr/bin/env python3
"""TOMO Dataset Generator - a Blender scene in, a training dataset out (2026-10-29). Its own window, separate from the
training window (tomo_vla_train_gui); it runs only this folder's steps:
  Scene     choose a prepared scene, or add a Blender file: the two read-only exports (blender_scripts/export_rig.py and
            export_scene_usd.py; Blender only reads the file) + its settings file scenes/<name>.yaml from a template
            (what the .blend does not say: objects to leave out, steps, the training's task file)
  Generate  N episodes with generate.sh: random positions inside the .blend's own Synthesizer ranges, the robot's motion
            re-solved, the robot's limits checked, rendered in Isaac Sim with the live head-camera lens, packed in the
            format the training reads -> datasets/<scene>_<date>/ (open it in the training window)
Start: bash dataset_generator.sh
"""
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import yaml
from PySide6.QtCore import QProcess, Qt, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QKeySequence, QShortcut, QTextCharFormat, QTextCursor, QTextDocument
from PySide6.QtWidgets import (QApplication, QComboBox, QLineEdit, QTextEdit, QFileDialog, QFrame, QGridLayout, QGroupBox, QHBoxLayout,
                               QInputDialog, QLabel, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QSpinBox,
                               QVBoxLayout, QWidget)

from gui_theme import STYLE

HERE = Path(__file__).resolve().parent
BLENDER = shutil.which("blender") or str(Path.home() / "blender/blender-5.2.2-linux-x64/blender")
SEC_PER_FRAME, SEC_PER_EPISODE = 0.2, 100       # this laptop (2026-10-29: ~3 min per 506-frame episode, all steps)
TEMPLATE = """\
# {name}: added by the dataset generator window on {date}. What the generator needs that the .blend itself does not
# say - edit it; everything else (objects, materials, motion, the Synthesizer's ranges, the IO markers) comes from the
# .blend. iPhone_Packing_local.yaml is a filled-in example.
blend: {blend}

hide: []                          # objects to leave out of the pictures (Blender names)
hide_from_frame: {{}}               # {{object: frame}}: not drawn from that frame on
light_gain: 300                   # every scene light's intensity x this (Blender power -> Isaac)
dome: 2500                        # plus an even dome light of this intensity
look: {{}}                          # the fixed look (runs without a seed)
random_look: {{}}                   # colours / prints varied per episode (example: iPhone_Packing_local.yaml)
# steps: [[NAME, first frame], ...]   # without it: the sub-episodes of the .blend's MMRT recorder, else one step per marker
vacuum_io: {{right: 0, left: 2}}    # the robot's IO pins of the two vacuums (the same robot in every scene)
# Another robot than the first TOMO (scene_robot.py; gidora_packing.yaml is a filled-in example):
# robot: {{object: <root object in the .blend>, urdf: <path>, srdf: <path>, drawn: scene}}
# camera: {{object: <camera in the .blend>, lens: [fx, fy, cx, cy, 1920, 1080]}}
track: null                       # the object labelled as the target (phone_world / phone_uv / phone_visible)
labels: {{}}                        # extra labels: {{name: {{world: object}}}} or {{name: {{influence: [object, constraint]}}}}
camera_bars: false
task_yaml: null                   # the training's task file (scenes/{name}.task.yaml): needed to train on the dataset
task_files: []
"""


def nice(minutes):
    m = int(round(minutes))
    return f"{m // 60} h {m % 60:02d} min" if m >= 60 else f"{max(m, 1)} min"


def set_state(w, state):
    w.setProperty("state", state or "")
    w.style().unpolish(w)
    w.style().polish(w)


def field(text):
    k = QLabel(text.upper())
    k.setObjectName("fieldLabel")
    return k


def detail(text=""):
    d = QLabel(text)
    d.setObjectName("detail")
    d.setWordWrap(True)
    d.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return d


def pill(text=""):
    p = QLabel(text)
    p.setObjectName("statusPill")
    p.setWordWrap(True)
    return p


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("mainWindow")
        self.proc, self.queue, self.on_done, self.stopped, self.buf = None, [], None, False, ""
        self.out, self.n, self.made, self.t0, self.frames, self.ready = None, 0, 0, 0.0, 0, False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.header())
        body = QWidget()
        body.setObjectName("page")
        b = QVBoxLayout(body)
        b.setContentsMargins(24, 18, 24, 12)
        b.setSpacing(10)
        t = QLabel("Dataset from a Blender scene")
        s = QLabel("Choose a scene (or add a Blender file), how many episodes and how long, then Generate. Each "
                   "episode puts what the scene's own Synthesizer moves at a random place inside its ranges, and the "
                   "robot's motion follows. Open the finished dataset in the training window.")
        t.setObjectName("pageTitle")
        s.setObjectName("pageSubtitle")
        s.setWordWrap(True)
        b.addWidget(t)
        b.addWidget(s)
        b.addWidget(self.scene_box())
        b.addWidget(self.dataset_box())
        lay.addWidget(body)
        lay.addWidget(self.log_box(), 1)
        self.setStyleSheet(STYLE)
        self.setWindowTitle("TOMO Dataset Generator")
        self.setMinimumSize(1000, 760)
        self.resize(1250, 920)
        self.refresh_scenes()

    # ── layout ──
    def header(self):
        h = QFrame()
        h.setObjectName("consoleHeader")
        h.setFixedHeight(64)
        lay = QHBoxLayout(h)
        lay.setContentsMargins(20, 9, 18, 9)
        v = QVBoxLayout()
        v.setSpacing(0)
        a, b = QLabel("TOMO DATASET GENERATOR"), QLabel("Blender scene → Isaac Sim → training dataset")
        a.setObjectName("consoleBrand")
        b.setObjectName("consoleSubtitle")
        v.addWidget(a)
        v.addWidget(b)
        lay.addLayout(v)
        lay.addStretch()
        self.mode = QLabel("IDLE")
        self.mode.setObjectName("consoleMode")
        lay.addWidget(self.mode)
        return h

    def scene_box(self):
        box = QGroupBox("Scene")
        g = QGridLayout(box)
        self.scene = QComboBox()
        self.scene.currentTextChanged.connect(self.load_scene)
        self.add_b = QPushButton("Add a Blender file…")
        self.add_b.clicked.connect(self.add_scene)
        self.edit_b = QPushButton("Open its settings file")
        self.edit_b.clicked.connect(lambda: QDesktopServices.openUrl(
            QUrl.fromLocalFile(str(HERE / "scenes" / f"{self.scene.currentText()}.yaml"))))
        row = QHBoxLayout()
        row.addWidget(self.scene, 1)
        row.addWidget(self.add_b)
        row.addWidget(self.edit_b)
        self.preview_b = QPushButton("Preview in Isaac Sim")    # user 2026-11-18: check the action before generating
        self.preview_b.setToolTip("Builds the scene's own animation (as in the .blend, nothing random) as a playable "
                                  "Isaac file and opens it in Isaac Sim: press Play there")
        self.preview_b.clicked.connect(self.preview_scene)
        row.addWidget(self.preview_b)
        g.addWidget(field("Scene"), 0, 0)
        g.addLayout(row, 0, 1)
        self.info = detail()
        g.addWidget(self.info, 1, 0, 1, 2)
        g.setColumnStretch(1, 1)
        return box

    def dataset_box(self):
        box = QGroupBox("Dataset")
        g = QGridLayout(box)
        self.n_normal, self.n_single, self.n_multi = QSpinBox(), QSpinBox(), QSpinBox()
        for w, v, tip in ((self.n_normal, 5, "no failure: the task as in the .blend"),
                          (self.n_single, 3, "one failure (pick miss, flap not folded, lid not closed, ...), noticed and recovered"),
                          (self.n_multi, 2, "2-3 failures at different steps, each noticed and recovered")):
            w.setRange(0, 5000)
            w.setValue(v)
            w.setToolTip(tip)
            w.valueChanged.connect(self.estimate)
        self.first, self.until = QSpinBox(), QComboBox()
        self.first.setRange(0, 99999)
        self.first.setToolTip("Episode numbers run from here; the same number always gives the same episode")
        self.until.currentIndexChanged.connect(self.estimate)
        self.go = QPushButton("Generate")
        self.go.setObjectName("primaryButton")
        self.go.clicked.connect(self.generate)
        self.stop_b = QPushButton("Stop")
        self.stop_b.setObjectName("stopButton")
        self.stop_b.setEnabled(False)
        self.stop_b.clicked.connect(self.stop)
        self.open_b = QPushButton("Open the dataset folder")
        self.open_b.setEnabled(False)
        self.open_b.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.out))))
        for r, (k, w) in enumerate([("Normal episodes", self.n_normal), ("Single-failure episodes", self.n_single),
                                    ("Multi-failure episodes", self.n_multi), ("First episode number", self.first)]):
            g.addWidget(field(k), r, 0)
            g.addWidget(w, r, 1)
        self.est = detail()
        g.addWidget(self.est, 4, 0, 1, 2)
        row = QHBoxLayout()
        for w in (self.go, self.stop_b, self.open_b):
            row.addWidget(w)
        row.addStretch()
        g.addLayout(row, 5, 0, 1, 2)
        self.status, self.bar, self.text = pill("Not started"), QProgressBar(), detail()
        self.bar.setTextVisible(False)
        g.addWidget(self.status, 6, 0, 1, 2)
        g.addWidget(self.bar, 7, 0, 1, 2)
        g.addWidget(self.text, 8, 0, 1, 2)
        g.setColumnStretch(1, 1)
        return box

    def log_box(self):
        dock = QFrame()
        dock.setObjectName("logDock")
        lay = QVBoxLayout(dock)
        lay.setContentsMargins(14, 7, 14, 8)
        t = QLabel("RUNTIME LOG")
        t.setObjectName("logDockTitle")
        lay.addWidget(t)
        self.log_view = QPlainTextEdit()
        self.log_view.setObjectName("logView")
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        self.log_view.setPlaceholderText("Everything the steps print appears here.")
        lay.addWidget(self.log_view, 1)
        # find in the log (user 2026-11-18): Ctrl+F, Enter = next, Shift+Enter = previous, Esc = close
        self.find_bar = QFrame()
        fb = QHBoxLayout(self.find_bar)
        fb.setContentsMargins(0, 4, 0, 0)
        self.find_edit = QLineEdit()
        self.find_edit.setPlaceholderText("Find in the log (Enter = next, Shift+Enter = previous, Esc = close)")
        self.find_edit.textChanged.connect(lambda: self.find_in_log(0))
        self.find_edit.returnPressed.connect(lambda: self.find_in_log(1))
        self.find_count = QLabel("")
        prev_b, next_b, close_b = QPushButton("▲"), QPushButton("▼"), QPushButton("✕")
        prev_b.clicked.connect(lambda: self.find_in_log(-1))
        next_b.clicked.connect(lambda: self.find_in_log(1))
        close_b.clicked.connect(self.close_find)
        for w in (prev_b, next_b, close_b):
            w.setFixedWidth(34)
        fb.addWidget(QLabel("Find"))
        fb.addWidget(self.find_edit, 1)
        fb.addWidget(self.find_count)
        fb.addWidget(prev_b)
        fb.addWidget(next_b)
        fb.addWidget(close_b)
        self.find_bar.hide()
        lay.addWidget(self.find_bar)
        QShortcut(QKeySequence.StandardKey.Find, self, activated=self.open_find)
        QShortcut(QKeySequence("Shift+Return"), self.find_edit, activated=lambda: self.find_in_log(-1))
        QShortcut(QKeySequence("Escape"), self.find_edit, activated=self.close_find)
        return dock

    def open_find(self):
        self.find_bar.show()
        sel = self.log_view.textCursor().selectedText()
        if sel:
            self.find_edit.setText(sel)
        self.find_edit.setFocus()
        self.find_edit.selectAll()
        self.find_in_log(0)

    def close_find(self):
        self.find_bar.hide()
        self.log_view.setExtraSelections([])
        self.find_count.setText("")

    def find_in_log(self, direction):
        """highlight every match (yellow), the current one orange; direction 1 next, -1 previous, 0 from the top."""
        word, doc, view = self.find_edit.text(), self.log_view.document(), self.log_view
        if not word:
            view.setExtraSelections([])
            self.find_count.setText("")
            return
        hits, c = [], doc.find(word, 0)
        while not c.isNull() and len(hits) < 2000:
            hits.append(c)
            c = doc.find(word, c)
        if not hits:
            view.setExtraSelections([])
            self.find_count.setText("no match")
            return
        pos = view.textCursor().selectionStart() if direction else -1
        if direction > 0:
            cur = next((i for i, h in enumerate(hits) if h.selectionStart() > pos), 0)
        elif direction < 0:
            cur = next((i for i in range(len(hits) - 1, -1, -1) if hits[i].selectionStart() < pos), len(hits) - 1)
        else:
            cur = len(hits) - 1                                   # the newest line first
        sels = []
        for i, h in enumerate(hits):
            s = QTextEdit.ExtraSelection()
            s.cursor = h
            f = QTextCharFormat()
            f.setBackground(QColor("#ff9800" if i == cur else "#fff176"))
            f.setForeground(QColor("#000000"))
            s.format = f
            sels.append(s)
        view.setExtraSelections(sels)
        view.setTextCursor(hits[cur])
        view.ensureCursorVisible()
        self.find_count.setText(f"{cur + 1} of {len(hits)}" + ("+" if len(hits) >= 2000 else ""))

    def log(self, text):
        self.log_view.appendPlainText(text)
        if self.find_bar.isVisible() and self.find_edit.text():   # searching: don't jump away from the match
            return
        self.log_view.moveCursor(QTextCursor.MoveOperation.End)

    # ── the scene ──
    def refresh_scenes(self, select=None):
        names = sorted(p.stem for p in (HERE / "scenes").glob("*.yaml") if not p.stem.endswith(".task"))
        self.scene.blockSignals(True)
        self.scene.clear()
        self.scene.addItems(names)
        if select in names:
            self.scene.setCurrentText(select)
        self.scene.blockSignals(False)
        self.load_scene(self.scene.currentText())

    def load_scene(self, name):
        self.ready, self.frames, self.recovery = False, 0, False
        self.until.blockSignals(True)
        self.until.clear()
        self.until.addItem("Whole episode", -1)
        lines, problems = [], []
        try:
            cfg = yaml.safe_load((HERE / "scenes" / f"{name}.yaml").read_text()) or {} if name else None
        except (OSError, yaml.YAMLError) as e:
            cfg, problems = None, [f"scenes/{name}.yaml cannot be read: {e}"]
        exp_p, rig_p = HERE / "blender_export" / str(name) / "scene/export.json", HERE / "blender_export" / str(name) / "rig.json"
        if cfg is not None and (not exp_p.exists() or not rig_p.exists()):
            problems.append("not exported yet: add its Blender file again")
        elif cfg is not None:
            exp, rig = json.loads(exp_p.read_text()), json.loads(rig_p.read_text())
            self.frames = exp["frame_end"] - exp["frame_start"] + 1
            markers = sorted((m["frame"], m.get("marker_name") or m["name"])
                             for m in (rig.get("scene_mmrt") or {}).get("mmrt_markers") or rig.get("markers", []))
            lines.append(f"Blender file: {exp['blend']}  (Blender {exp['blender']})")
            lines.append(f"{self.frames} frames at {exp['fps']:g} fps ({self.frames / exp['fps']:.1f} s) · "
                         f"{len(exp['objects'])} objects · {len(markers)} markers")
            moved = []
            for o in ((rig.get("scene_mmrt") or {}).get("mmrt_synthesizer") or {}).get("objects", []):
                lo, hi, rl, rh = o["trans_min"], o["trans_max"], o["rot_min_deg"], o["rot_max_deg"]
                parts = ([f"{a} {lo[i]:+.3f}..{hi[i]:+.3f} m" for i, a in enumerate("xyz") if lo[i] != hi[i]]
                         + [f"{a} {rl[i]:+.0f}..{rh[i]:+.0f}°" for i, a in enumerate(("roll", "pitch", "yaw"))
                            if rl[i] != rh[i]])
                if parts:
                    moved.append(f"{o['obj']} {', '.join(parts)} (frames {o['frame_min']}–{o['frame_max']})")
            lines.append("Random per episode (the .blend's Synthesizer): "
                         + ("; ".join(moved) if moved else "NOTHING - every episode is the same motion, only the look "
                                                           "changes"))
            steps = cfg.get("steps") or [["START", 0]] + [[n, f] for f, n in markers]
            lines.append("Steps: " + " → ".join(s for s, _ in steps) + ("" if cfg.get("steps") else "  (one per marker)"))
            for (s, _), (_, nxt) in zip(steps, steps[1:]):
                self.until.addItem(f"Up to the end of {s} (frame {nxt - 1})", nxt - 1)
            for k, what in (("unsupported_animation", "Blender animates things the rig does not read"),
                            ("geometry_changes", "meshes change shape over time")):
                if exp.get(k):
                    problems.append(f"{what}: {exp[k][:3]}")
            self.recovery = bool(cfg.get("recovery_task"))
            lines.append("Failures + recoveries: " + ("yes (" + str(cfg["recovery_task"]).split("/")[-1] + ")" if self.recovery
                         else "not set up for this scene (recovery_task in its settings file): normal episodes only"))
            task = cfg.get("task_yaml")
            lines.append(f"Training task file: {task}" if task and (HERE / task).exists() else
                         "Training task file: none yet. The dataset can be made; to train on it, write "
                         f"scenes/{name}.task.yaml and name it in the settings file (task_yaml)")
        self.until.blockSignals(False)
        self.ready = cfg is not None and not problems
        self.info.setText("\n".join(lines + [f"✖  {p}" for p in problems]) if name else
                          "No scene yet: add a Blender file.")
        self.estimate()

    def play_file(self, name):
        return HERE / "blender_export" / name / "scene" / "scene_play.usdc"

    def play_steps(self, name):
        """the .blend's own motion (make_motion.py, nothing random) baked into scene_play.usdc next to scene.usdc: open it
        in Isaac Sim and press Play to check the action before generating (user 2026-11-18)."""
        return [(HERE / "env/bin/python", ["make_motion.py", name]),
                (Path.home() / "isaacsim/python.sh", [HERE / "render_episode.py", HERE / "runs" / name / "generic",
                                                      "--build-only", "--usd-out", self.play_file(name)])]

    def preview_scene(self):
        """open the scene's scene_play.usdc in Isaac Sim (built first when missing or older than the scene's export)."""
        name = self.scene.currentText()
        if not name:
            return
        usd, rig = self.play_file(name), HERE / "blender_export" / name / "rig.json"
        isaac = Path.home() / "isaacsim"

        def open_it(ok=True):
            if not ok or not usd.exists():
                self.status.setText(f"Preview {name}: FAILED (see the log)")
                set_state(self.status, "error")
                return
            # Isaac Sim's own messages go into this log too (user 2026-11-18: see what happens)
            if getattr(self, "isaac_proc", None) is not None and self.isaac_proc.state() != QProcess.ProcessState.NotRunning:
                self.log("Isaac Sim is already open from the Preview button: close it first to open another scene")
                return
            ip = QProcess(self)
            ip.setWorkingDirectory(str(isaac))
            ip.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
            ibuf = {"s": ""}

            def isaac_out():
                ibuf["s"] += bytes(ip.readAllStandardOutput()).decode(errors="replace")
                *lines, ibuf["s"] = ibuf["s"].split("\n")
                for ln in lines:
                    if ln.strip():
                        self.log("[isaac] " + ln)
                    if "[open_in_isaac] opened" in ln:
                        self.log(f"✔ Isaac Sim shows {name}: press Play (▶) on the left toolbar to watch the action")
                        self.status.setText(f"Preview {name}: open in Isaac Sim - press Play (▶) there")
            ip.readyReadStandardOutput.connect(isaac_out)
            ip.finished.connect(lambda *_: self.log(f"Isaac Sim closed (preview of {name})"))
            self.isaac_proc = ip
            ip.start(str(isaac / "isaac-sim.sh"), ["--exec", str(HERE / "open_in_isaac.py"), f"--/preview/file={usd}"])
            self.log(f"$ {isaac / 'isaac-sim.sh'} … {usd}   ({time.strftime('%F %T')})")
            self.log("Isaac Sim is starting (about a minute); its messages follow below")
            self.status.setText(f"Preview {name}: Isaac Sim is opening (about a minute) - press Play there")
            set_state(self.status, None)
        if usd.exists() and (not rig.exists() or usd.stat().st_mtime > rig.stat().st_mtime):
            self.log(f"Preview {name}: the playable file is already built "
                     f"({time.strftime('%F %T', time.localtime(usd.stat().st_mtime))}): {usd}")
            open_it()
            return
        self.log(f"Preview {name}: building {usd.name} (a minute or two), then Isaac Sim opens it")
        set_state(self.status, "busy")
        self.status.setText(f"Preview {name}: building the playable Isaac file …")
        self.run(self.play_steps(name), "PREVIEW", open_it)

    def add_scene(self):
        blend, _ = QFileDialog.getOpenFileName(self, "Choose the Blender file", str(HERE.parent.parent),
                                               "Blender files (*.blend)")
        if not blend:
            self.log("Add a Blender file: no file chosen")
            return
        self.log(f"Add a Blender file: {blend}")

        def used(n):
            return (HERE / "blender_export" / n).exists() or (HERE / "scenes" / f"{n}.yaml").exists()
        same = [p.stem for p in (HERE / "scenes").glob("*.yaml") if not p.stem.endswith(".task")
                and (HERE / "scenes" / os.path.normpath(str((yaml.safe_load(p.read_text()) or {}).get("blend", "")))
                     ).resolve() == Path(blend).resolve()]
        if same:                                                  # user 2026-11-18: say so (in the log too)
            self.log(f"  this .blend is already the scene {', '.join(same)}: choose it in the Scene list "
                     f"(or give a new name to add it again)")
        base = re.sub(r"\W+", "_", Path(blend).stem)
        free, k = base, 2
        while used(free):
            free, k = f"{base}_{k}", k + 1
        name, ok = QInputDialog.getText(self, "Scene name", "A short name for the scene (letters, digits, _):"
                                        + (f"\n(already added as {', '.join(same)})" if same else ""), text=free)
        name = re.sub(r"\W+", "_", name or "").strip("_")
        if not ok or not name:
            self.log("  cancelled")
            if same:
                self.scene.setCurrentText(same[0])
            return
        if used(name):
            self.log(f"  a scene called {name} already exists: nothing added")
            QMessageBox.warning(self, "Name in use", f"A scene called {name} already exists: choose another name.")
            return
        out = HERE / "blender_export" / name
        set_state(self.status, "busy")
        self.status.setText(f"Adding {name}: Blender reads {Path(blend).name} (three exports, a few minutes) …")
        # the rig (motion), the look (USD) and the objects (pack_episode.py writes the dataset's objects file from it)
        self.run([(BLENDER, ["-b", blend, "--python", "blender_scripts/export_rig.py", "--", out]),
                  (BLENDER, ["-b", blend, "--python", "blender_scripts/export_scene_usd.py", "--", out]),
                  (BLENDER, ["-b", blend, "--python", "blender_scripts/export_scene_objects.py", "--",
                             out / "objects_export.npz", "--urdf", "robot_limits/source/robot_description.urdf"])],
                 "EXPORTING", lambda ok: self.scene_added(ok, name, blend))

    def scene_added(self, ok, name, blend):
        if not ok:
            shutil.rmtree(HERE / "blender_export" / name, ignore_errors=True)    # new this time: nothing else lost
            set_state(self.status, "error")
            self.status.setText(f"✖  {name}: the export did not finish (see the log); nothing was added")
            return
        p = HERE / "scenes" / f"{name}.yaml"
        p.write_text(TEMPLATE.format(name=name, blend=os.path.relpath(blend, p.parent), date=time.strftime("%Y-%m-%d")))
        # failures / recoveries: the pick box, tape and carton found from the .blend's vacuum markers, sizes measured
        import subprocess
        r = subprocess.run([str(HERE / "env/bin/python"), "setup_failures.py", name, "--auto"], cwd=HERE,
                           capture_output=True, text=True)
        self.log(r.stdout + r.stderr)
        found = next((ln[len("[setup] found: "):] for ln in r.stdout.splitlines() if ln.startswith("[setup] found: ")), "")
        set_state(self.status, "ready")
        self.status.setText(f"✔  {name} added. " + (f"Failure episodes ready - {found}." if r.returncode == 0 else
                            "Normal episodes only: the objects held by the vacuum could not be found (see the log)."))
        self.refresh_scenes(select=name)
        # the Isaac Sim play scene's path, last in the log (user 2026-11-18; made by the Preview in Isaac Sim button)
        self.log("")
        self.log(f"Isaac Sim play scene: {self.play_file(name)}")

    # ── the dataset ──
    def estimate(self):
        last = self.until.currentData()
        frames = (self.frames if last is None or last < 0 else last + 1) if self.frames else 0
        per = (frames * SEC_PER_FRAME + SEC_PER_EPISODE) / 60
        for w in (self.n_single, self.n_multi):
            w.setEnabled(self.recovery and self.proc is None)
            if not self.recovery:
                w.setValue(0)
        nn, ns, nm = self.n_normal.value(), self.n_single.value(), self.n_multi.value()
        n = nn + ns + nm
        names = ", ".join(f"{c}1..{c}{k}" if k > 1 else f"{c}1" for c, k in (("normal", nn), ("single", ns), ("multi", nm)) if k)
        # minutes per episode, measured 2026-11-17 (normal 4.1, single 5.0 incl. Isaac render); multi has 2-3 recoveries
        mins = nn * 4.5 + ns * 5.5 + nm * 7.0
        self.est.setText(f"{n} episodes ({names}): about {nice(mins)} (one after the other)" if frames and n else "")
        self.go.setEnabled(self.ready and self.proc is None and n > 0)

    def generate(self):
        name, first = self.scene.currentText(), self.first.value()
        nn, ns, nm = self.n_normal.value(), self.n_single.value(), self.n_multi.value()
        n = nn + ns + nm
        if QMessageBox.question(self, "Generate", f"{self.est.text()}\n\nStart?") != QMessageBox.StandardButton.Yes:
            return
        self.out = HERE / "datasets" / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}"
        self.n, self.made, self.t0 = n, 0, time.time()
        self.bar.setMaximum(n)
        self.bar.setValue(0)
        set_state(self.bar, None)
        set_state(self.status, "busy")
        self.status.setText(f"Generating {n} episodes → {self.out}")
        self.text.setText("")
        self.open_b.setEnabled(False)
        self.run([("bash", [HERE / "generate_mix.sh", name, nn, ns, nm, first, "--out", self.out])], "GENERATING",
                 self.generated)

    def generated(self, ok):
        made = len([p for p in self.out.glob("*") if (p / "manifest.json").exists()]) if self.out.exists() else 0
        good = ok and made == self.n
        set_state(self.status, "ready" if good else "error" if not made else "busy")
        set_state(self.bar, "ready" if good else None)
        self.status.setText(("✔  " if good else "✖  " if not made else "")
                            + f"{made} of {self.n} episodes made → {self.out}"
                            + ("" if ok else "  (stopped or failed: see the log)"))
        self.text.setText(f"Took {nice((time.time() - self.t0) / 60)}")
        self.open_b.setEnabled(made > 0)

    def progress(self, line):
        if line.startswith("  done -> "):
            self.made += 1
            self.bar.setValue(self.made)
        m = re.search(r" episode (\d+)/(\d+) ", line)
        if line.startswith("[generate]") and m:
            left = (time.time() - self.t0) / max(1, self.made) * (self.n - self.made) / 60
            self.text.setText(f"Episode {m.group(1)} of {self.n} · {self.made} done"
                              + (f" · about {nice(left)} left" if self.made else ""))

    # ── running the steps ──
    def run(self, steps, mode, on_done):
        self.queue, self.on_done, self.stopped = list(steps), on_done, False
        self.mode.setText(mode)
        set_state(self.mode, "busy")
        for w in (self.go, self.add_b, self.preview_b, self.scene, self.n_normal, self.n_single, self.n_multi, self.first):
            w.setEnabled(False)
        self.stop_b.setEnabled(True)
        self.next_step()

    def next_step(self):
        if not self.queue:
            self.finish(True)
            return
        prog, args = self.queue.pop(0)
        args = [str(a) for a in args]
        self.log(f"$ {prog} {' '.join(args)}   ({time.strftime('%F %T')})")
        p = QProcess(self)
        p.setWorkingDirectory(str(HERE))
        p.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        p.readyReadStandardOutput.connect(self.read_output)
        p.finished.connect(self.step_finished)
        p.errorOccurred.connect(lambda e: e == QProcess.ProcessError.FailedToStart and self.step_finished(-1, None))
        self.proc, self.buf = p, ""
        p.start(str(prog), args)

    def read_output(self):
        if self.proc is None:
            return
        self.buf += bytes(self.proc.readAllStandardOutput()).decode(errors="replace")
        *lines, self.buf = self.buf.split("\n")
        for line in lines:
            self.log(line)
            self.progress(line)

    def step_finished(self, code, _status):
        if self.proc is None:
            return
        self.read_output()
        if self.buf:
            self.log(self.buf)
            self.progress(self.buf)
        if code != 0 or self.stopped:
            self.log("[stopped]" if self.stopped else f"[FAILED] exit code {code}")
            self.queue = []
            self.finish(False)
        else:
            self.next_step()

    def finish(self, ok):
        self.proc = None
        self.mode.setText("IDLE" if ok else "STOPPED" if self.stopped else "FAILED")
        set_state(self.mode, None if ok else "error")
        for w in (self.add_b, self.preview_b, self.scene, self.n_normal, self.n_single, self.n_multi, self.first):
            w.setEnabled(True)
        self.stop_b.setEnabled(False)
        cb, self.on_done = self.on_done, None
        if cb:
            cb(ok)
        self.estimate()

    def stop(self):
        if self.proc and QMessageBox.question(self, "Stop", "Stop now? Finished episodes are kept.") \
                == QMessageBox.StandardButton.Yes:
            self.stopped = True
            self.proc.terminate()                       # generate.sh ends the running step and stops

    def closeEvent(self, e):
        if self.proc:
            if QMessageBox.question(self, "Still running", "A step is running. Stop it and close?") \
                    != QMessageBox.StandardButton.Yes:
                e.ignore()
                return
            self.stopped = True
            self.proc.terminate()
            self.proc.waitForFinished(20000)
        e.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("TOMO Dataset Generator")
    app.setStyle("Fusion")
    w = Window()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
