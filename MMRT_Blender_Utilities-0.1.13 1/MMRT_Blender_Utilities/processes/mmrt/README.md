# MMRT Processing Pipeline

This directory contains the standalone MMRT motion-processing process used by
the Blender addon. It converts a recorded MMRT motion bundle into a deployable
behavior-tree application containing the behavior tree, blackboards, robot
environment meshes, and application configuration.

## Entry points

- `main.py` is the thin command-line launcher used by Blender and external
  tools.
- `pipeline.py` contains the command-line parser and
  `run_mmrt_pipeline(...)`, which orchestrates the complete processing flow.
- `joint_limit_plotter.py` is an independent utility for plotting recorded
  joint position, velocity, and acceleration data.

The Blender operator launches the packaged `mmrt_backend` executable as a child
process. The backend contains the Python interpreter and MMRT dependencies,
including NumPy, SciPy, PyBullet, Matplotlib, PyYAML, and Rich. During source
development, `main.py` can still be launched with an explicit external Python.

## Pipeline flow

```text
recording_data.mmrt.yaml
        │
        ├── URDF/Xacro ──┐
        ├── SRDF ────────┼── robot and scene metadata
        └── environment ─┘
        │
        ▼
Load recording and resolve bundle paths
        │
        ▼
Preprocess and smooth joint samples
        │
        ▼
Detect motion per atomic group
        │
        ▼
Merge timelines and split at recorded events
        │
        ▼
Generate, merge, and name waypoints
        │
        ▼
Build motion, IO, LED, and flag keypoints
        │
        ▼
Evaluate motion with optional PyBullet collision checking
        │
        ▼
Export behavior tree, blackboards, meshes, and app config
```

The stages are implemented in `run_mmrt_pipeline` in this order:

1. **Load and map** — Load the YAML recording, resolve its URDF, SRDF, and
   environment references, parse robot joints and SRDF joint groups, and load
   environment objects.
2. **Preprocess** — Convert recorded joint samples into the configured unit
   system and smooth values using the robot joint limits.
3. **Segment motion** — Calculate joint velocities, find active segments per
   atomic group, merge the group timelines, combine nearby segments, remove
   ignored regions, and split at intervening recording events.
4. **Generate waypoints** — Find local speed minima, apply the configured
   waypoint merge method, map atomic groups to move groups, and create stable
   waypoint and trajectory identifiers.
5. **Create keypoints** — Preserve IO, LED, general flag, and motion
   events in the timeline used for behavior-tree generation.
6. **Simulate and evaluate** — Run PyBullet checks when enabled and report
   collision frames. Collision diagnostics are logged independently from XML
   export; the pipeline can still produce an output application.
7. **Export** — Build the behavior tree from
   `definitions/behavior_tree_models.xml`, export blackboards, copy referenced
   environment meshes, and write the generated app configuration.

## Directory structure

```text
mmrt/
├── main.py                         # Thin external-process entry point
├── pipeline.py                     # CLI and pipeline orchestration
├── joint_limit_plotter.py          # Optional joint-limit plotting utility
├── definitions/
│   ├── application.py              # Generated app, planner, blackboard definitions
│   ├── environment.py              # Environment JSON schema
│   ├── pipeline.py                 # Processing defaults, trajectory options, CLI flags
│   ├── recording.py                # Recording schema, episodes, flags, LED values
│   ├── robot.py                    # Robot description kinds and joint types
│   └── behavior_tree_models.xml    # Behavior-tree model contract
└── common/
    ├── behavior_tree.py             # XML model loading and BT/blackboard export
    ├── console_effects.py           # Optional visual console progress effects
    ├── logging_setup.py             # Logging configuration and tracing helpers
    ├── motion_analysis.py            # Velocities, segmentation, waypoints, hashes
    ├── recording_data.py             # Recording parsing and app config generation
    ├── robot_description.py          # URDF/Xacro and SRDF parsing
    ├── simulation.py                 # PyBullet setup and collision checks
    └── visualization.py              # Segmentation and trajectory plots
```

The dependency-free classes in `definitions/` are the single source for values
used by both Blender and the standalone process: recording schema labels and
formats, episode modes, LED modes and colors, generated app paths, planner choices,
blackboard defaults, environment columns, robot joint types, CLI options,
processing thresholds, subtree IDs, and MoveTomoTrajectory options. Both sides
import these classes using their respective package paths. The Blender defaults
also apply to standalone runs: waypoint merging defaults to `FIRST`, and global
velocity and acceleration scaling default to `1.0`. Behavior-tree model
identifiers remain in the XML definition file as part of the runtime contract.

## Input recording bundle

The required input is the recording YAML file, normally accompanied by files
in the same directory:

```text
recording_data.mmrt.yaml       # Required motion recording and metadata
recording_data.mmrt.bin        # Optional binary recording payload
recording_data.urdf            # Required robot description
recording_data.srdf            # Required joint groups and collision pairs
recording_data.env.json        # Optional environment object data
meshes/                        # Robot/environment mesh sources
```

The YAML metadata is the source of truth for the URDF, SRDF, environment, robot
name, frame rate, frame range, units, joint samples, and recorded events. The
`--urdf`, `--srdf`, and `--env` arguments are supported as legacy overrides, but
bundle metadata is preferred.

Recorder episodes are stored under `metadata.episodes`:

```yaml
episodes:
  - name: PickEpisode
    id: pick
    start_frame: 0
    end_frame: 80
    logic: NONE
    repeat_count: 1
    retry_count: 1
    delay_enabled: false
    delay_msec: 0
```

Episode names and IDs are unique. Names become behavior-tree subtree IDs. Within
episodes, IDs prefix waypoint and trajectory names. Ranges are inclusive and non-overlapping.
Each episode may select one primary logic (`REPEAT`, `RETRY_UNTIL_SUCCESSFUL`, or
`KEEP_RUNNING_UNTIL_FAILURE`) and may independently enable `Delay`. Complete
motion and events are exported into the matching subtree; work outside episodes
remains in the outer `moveSegmentTree` sequence. Episode ranges are shaded in
the segmentation plot. Waypoint names use the episode ID and an incrementing
per-episode index, such as `pick_wp0`; the blackboard YAML comments each waypoint
with its source frame. Trajectory names retain zero-based IDs such as
`pick_trajectory0`. Motion outside episodes and recordings without episodes
retain frame-based waypoint names.

## Command-line usage

Run from the repository root with the Python environment used by the addon:

```powershell
python MMRT_Blender_Utilities\processes\mmrt\main.py `
  --record .\path\to\recording_data.mmrt.yaml `
  --behavior_tree_models_path MMRT_Blender_Utilities\processes\mmrt\definitions\behavior_tree_models.xml `
  --output .\path\to\app\trees\behavior_tree.bt.xml `
  --app_dir .\path\to\app `
  --tree_id MainTree
```

`--behavior_tree_models_path` specifies the extra model catalog loaded by the
pipeline. Use `--help` for all segmentation, waypoint,
collision, simulation, plotting, export, and logging options.

The builder initializes with only the built-in actions such as Sleep and Script,
using the default catalog path in `BehaviorTreeModelConfig`. The pipeline then
loads the supplied extra model catalog. No default model path argument is needed.
The supplied model catalog takes precedence
for duplicate action IDs. `--bt_trajectory_resolution` sets the resampling
resolution in radians (default `0.01`); the generated tree references
`trajectory_resolution` in the system settings blackboard.

Important options include:

| Option | Purpose |
| --- | --- |
| `--record` | Input MMRT YAML recording |
| `--behavior_tree_models_path` | Extra behavior-tree model XML |
| `--output` | Requested behavior-tree XML path/name |
| `--app_dir` | Root directory for the generated application |
| `--mmrt_speed_thr` | Motion speed threshold |
| `--mmrt_wp_decimation_window` | Waypoint merge window |
| `--mmrt_wp_decimation_method` | Waypoint merge method, such as `MEAN` |
| `--mmrt_collision_thr` | Collision distance threshold in meters |
| `--sim` | Enable PyBullet simulation |
| `--plot` / `--save_plot` | Show or save trajectory plots |
| `--no_export` | Run analysis without behavior-tree export |
| `--verbosity` | `silent`, `normal`, `debug`, or `verbose` |

## Generated application

When export is enabled, `--app_dir` receives this structure:

```text
app_dir/
├── trees/
│   ├── behavior_tree.bt.xml
│   └── episode_subtrees.bt.xml  # only when episodes are declared
├── blackboards/
│   ├── system_settings.btbb.yaml
│   ├── motion_waypoints.btbb.yaml
│   ├── environment_settings.btbb.yaml
│   └── episode_settings.btbb.yaml  # only when parameterized episode logic exists
├── meshes/
│   └── <referenced environment meshes>
└── config.apcfg.yaml
```

The behavior-tree model XML is read from the repository, while generated
blackboard and application filenames are centralized in
`recording_data.GeneratedAppFilenames` and `recording_data.GeneratedAppDirectories`.
The main tree references episode subtrees by name; those episode definitions
are emitted in `episode_subtrees.bt.xml`, and both generated tree files are
listed in `bt_config.tree_files` when episodes exist. Parameterized episode
values are loaded by `ConfigureTree` from `episode_settings.btbb.yaml` and are
listed in `bt_config.blackboard_files`.

## Development conventions

- Keep `main.py` limited to process entry-point wiring.
- Keep orchestration in `pipeline.py`; place reusable domain logic in the
  focused `common` modules.
- Add or change numerical defaults in `common/pipeline_defaults.py`.
- Add recording schema, enum, and generated-output naming changes in
  `common/recording_data.py`.
- Keep behavior-tree node identifiers synchronized with
  `definitions/behavior_tree_models.xml` and the deployed runtime.
- Run the following checks after changes:

  ```powershell
  python -m compileall -q MMRT_Blender_Utilities\processes\mmrt
  python MMRT_Blender_Utilities\processes\mmrt\main.py --help
  ```
