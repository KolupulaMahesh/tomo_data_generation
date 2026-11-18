# MMRT Blender Utilities

[![Version 0.1.0](https://img.shields.io/badge/version-0.1.0-blue.svg)](./__init__.py)
[![Blender 5.1+](https://img.shields.io/badge/Blender-5.1%2B-orange.svg)](https://www.blender.org/)
[![Windows x64](https://img.shields.io/badge/platform-windows--x64-orange.svg)](https://www.microsoft.com/windows/)
[![Linux x64](https://img.shields.io/badge/platform-linux--x64-orange.svg)](https://www.kernel.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](../README.md#license)

`MMRT_Blender_Utilities` is the robot-agnostic robotics addon in this repository. It supports
robot import, scene recording, motion processing, behavior-tree generation,
simulation, remote deployment, and randomized animation episodes.

- Blender target: 5.1 or newer; current development is tested with Blender 5.2.
- Addon version in `bl_info`: 0.1.0.
- Sidebar tab: **MMRT Utilities**.
- The addon can be installed without `Imaging_Data_Generator`.

For repository-wide installation and the other components, see the
[repository README](../README.md).

## Capabilities

- Import URDF and XACRO robot descriptions into Blender armatures.
- Store URDF/SRDF descriptions inside the `.blend` as Blender Text datablocks.
- Pair URDF joints with armature bones and detect rotation axes.
- Record robot joints, active-object transforms, environment geometry, timeline
  flags, I/O events, camera events, and LED events.
- Export environment meshes, metadata, robot attachments, and collision rules.
- Segment motion into atomic and SRDF-defined movegroups and extract waypoints.
- Generate a deployable MMRT behavior-tree application.
- Check current-animation joint position, velocity, and acceleration limits, with
  frame navigation and manual rechecking after edits. See the
  [joint limit checks guide](JOINT_LIMIT_CHECKS.md).
- Apply reversible [regional timing](REGIONAL_RETIMING.md) to slow violating
  motion while preserving source actions and coordinated poses.
- Launch optional PyBullet simulation and joint-trajectory evaluation.
- Upload the generated application to a remote host and control it through the
  ROS bridge.
- Generate batches of randomized animation and MMRT-recording episodes.
- Launch and stop the Groot 2 behavior-tree editor.

## Installation

Install the `MMRT_Blender_Utilities` folder as a Blender addon. The ZIP must
contain the addon folder at its top level:

```text
MMRT_Blender_Utilities/
└── __init__.py
```

Remove any older copy of this addon before installing the renamed package. In
Blender, use **Edit → Preferences → Add-ons → Install...**, select the ZIP, and
enable **MMRT Blender Utilities**.

For a development copy on Windows:

```powershell
$addons = "$env:APPDATA\Blender Foundation\Blender\5.2\scripts\addons"
Copy-Item -LiteralPath .\MMRT_Blender_Utilities -Destination $addons -Recurse -Force
```

Restart Blender or disable and re-enable the addon after copying source files.

## MMRT backend

The addon launches processing tools outside Blender using the bundled
platform-specific PyInstaller backend. In **MMRT Utilities → Configuration**,
click **Verify MMRT Backend** to test the selected backend. It does not run pip
and does not modify the system Python, `PATH`, or registry.

The **Advanced Settings** section contains the **Use Development Python**
toggle, the development Python executable, and the Groot 2 path. The bundled
backend is selected by default. Enable the toggle only when developing the
source scripts with an explicitly configured Python executable; the addon does
not install packages into that environment. Build instructions and the
dependency manifest are in [`../tools/mmrt_backend/README.md`](../tools/mmrt_backend/README.md).

The backend currently contains:

- `websockets`
- `paramiko`
- `numpy`
- `scipy`
- `pybullet`
- `matplotlib`
- `rich`
- `PyYAML`

Groot 2 and a ROS 2-enabled remote robot environment are optional. Set **Groot 2
Path** when the `groot2` executable is not on `PATH`.

## Panels and workflow

All panels are in the 3D Viewport sidebar (`N`) under **MMRT Utilities**.

### Configuration

Verify the selected backend, optionally enable and configure the development
Python, set the Groot 2 executable, and launch or stop Groot 2. Processing and
upload operators use the bundled backend by default.

### URDF Importer

1. Select a URDF/XACRO file.
2. Select the matching SRDF/XACRO file when available. The current MMRT flow
   requires SRDF data.
3. Optionally set end-link and camera-link names in Advanced Options to
   configure IK and camera tracking explicitly. With no camera-link names, the
   importer uses the first link whose name contains `camera`; with no end-link
   names, terminal non-camera links are detected automatically.
4. Choose whether imported visual meshes should be shown. Source geometry is
   archived in an `Original_Meshes` collection either way, including additional
   visual and collision meshes needed for portable recording export.
5. Click **Build Armature** to import the robot, or **Store Only** to embed only
   the descriptions without changing the scene.

The importer stores one description pair per robot, named
`<armature-object>_URDF` and `<armature-object>_SRDF`. **Store Only** derives the
same naming format from the URDF robot name. Mesh filenames and source metadata
are stored relative to the `.blend` by default; absolute and `package://` input
paths remain supported. When multiple links use the same source asset, their
Blender Objects share the imported Mesh datablocks. Recorder robot entries later
reference the associated descriptions and archived source-mesh collection.

### Scene Recorder

The recorder is organized into robots, active objects, environment objects, and
timeline markers.

#### Robots

Add each robot armature and assign the matching URDF/SRDF Text datablocks and
**Original Meshes** collection. Imported armatures populate these fields
automatically. Use **Refresh Joints** to load movable URDF joints, then
**Auto-Detect Axes**. Review the bone, axis, invert, offset, and Record settings
manually when names or axes do not match.

#### Active Objects

Record object transforms or individual bones. Choose World, Local, or Custom
reference space and enable the location/rotation channels that should be stored.

#### Environment Objects

Add meshes and curves that must be present during replay. Configure origin
handling, modifiers, export inclusion, unit conversion, robot-link attachment,
collision links, collision movegroups, and **Allow Collision** behavior.

#### Timeline markers

Each marker is a frame-level group and can contain multiple events. The recorder
panel shows the marker name, frame, and event icons in the marker list; selecting
a marker opens its event list and the selected event's settings. Marker names use
the form `{label}_{uid}` in Blender's timeline; the label is editable, while the
six-character hexadecimal UID is preserved automatically.

Markers can represent:

- motion segmentation and forced splits;
- forced waypoint names;
- named episode logic;
- camera capture and recording start/stop;
- eight-bit I/O commands;
- two-channel LED commands; and
- custom flags.

Choose the recorder output directory and base filename, environment export mode,
output format, rotation representation, angle units, location units, and fast
mode. Click **Start Recording** to sample the configured scene frame range.
Use **Print Current Frame Data** to inspect values at the current frame.

### MMRT behavior-tree generation

1. Open **MMRT**.
2. Keep **Use Scene Recorder Output** enabled, or select an existing `.mmrt.yaml`.
3. Confirm the selected recording folder contains its exported `.urdf`, `.srdf`,
   and `meshes/robot` assets. The generator reads these recorded resources and
   does not fall back to the Importer or the current scene.
4. Set the output directory and basename. Set the Main Tree ID and support tree
   IDs in **Behavior Tree Settings → Behavior Tree Output IDs**.
5. Tune segmentation, waypoint decimation, and collision checking in
   **MMRT Process Tuning**. Configure the planning pipeline, planner ID,
   MoveToTrajectory strategy, resampling, time parameterization, visualization,
   and global planning in **Behavior Tree Settings**. Trajectory visualization
   is off by default. These trajectory settings are written to
   `blackboards/system_settings.btbb.yaml` and referenced by the generated tree.

   Shared planning and logging defaults are defined in `BlackboardDefaults`
   (`processes/mmrt/definitions/application.py`). Trajectory defaults are defined
   in `MoveTomoTrajectoryConfig` (`processes/mmrt/definitions/pipeline.py`).
   The add-on properties and generated blackboards both use these definitions.
6. Click **Generate Behavior Tree XML**. The **Generation Log** shows live
   progress and the latest process output in a scrollable list. Turn off
   **Follow Latest** to inspect earlier lines, or use **Copy Full Log** to copy
   all captured output. Press `Esc` to cancel processing.
7. Optionally run **Evaluate Joint Limits**. Use **Backend Simulation and Plots**
   to save or show the segmentation plot or enable backend simulation.

The result is a complete app directory, not only an XML file. It contains the
behavior tree, blackboards, configuration, and copied environment meshes. The
recording's robot mesh archive remains in the recording folder and is not copied
into the generated app.

### Simulator and remote controller

The optional simulator evaluates generated trajectories and collision rules.

Robot collisions use the URDF's `<collision>` definitions. The recorder exports
archived collision meshes as OBJ, keeping each Blender hull as a separate `o`
object. Visual meshes continue to export as STL, even when the same source asset
is used for both. Mesh coordinates stay in the source asset frame; the URDF
applies their origins and scales.

At load time, MMRT expands each OBJ object/group into its own temporary URDF
collision element. This preserves authored hull boundaries even where hulls
touch or share vertex positions. The recording bundle is not modified. PyBullet
still uses a convex hull for each individual piece, so each authored piece should
be convex. Primitives and explicit `concave` declarations retain their normal
Bullet behavior. Visual meshes do not supply replacement collision geometry.

Old STL bundles only allow recovery of disconnected components; touching hull
boundaries have already been lost. Re-record/export from the Blender source to
get the OBJ collision assets and updated URDF references.

This loading behavior is part of the Python backend. Rebuild a bundled backend
to include source changes, or run with **Use Development Python** enabled.
The Remote Controller is split into two workflows:

1. **Send Application Folder** uploads either the generated MMRT application
   (`trees/`, `blackboards/`, `meshes/`, and `config.apcfg.yaml`) or a selected
   local application folder over SSH/SFTP.
2. **Robot Console Controller** uses the local ROS bridge to call the robot's
   ROS 2 console services: load application/config, configure and generate,
   run, pause, and stop.

Use **Start ROS Bridge server** to check the remote ROS environment and start
`rosbridge_server` over SSH when it is installed and the configured WebSocket
port is unused. Set `ROS_DOMAIN_ID` and enter any setup commands needed before
launch, such as sourcing the ROS and robot workspace setup files or exporting
`RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`. A `ros2` alias defined there is also
used by the server launch command. The SSH shell loads the remote user's
interactive Bash aliases, so the field can also invoke an alias such as
`tomo-ctrl-bash`. The server button changes to **Stop ROS Bridge server** after
startup and keeps the SSH session attached so remote launch output appears in
Blender's console. Clicking the button again sends Ctrl-C to the remote launch
and closes the session. Disabling the addon or closing Blender also stops the
managed server session. Then use
**Start ROS Bridge client** to connect Blender to that server. Configure remote
IP, rosbridge port, username, and password in the connection section. The send
section can use MMRT output or a custom local folder and has its own destination
directory. The controller can use that sender destination or a separate remote
application path, and switches between application loading and a standalone
robot-visible BT config YAML.
The bridge is considered connected after the configured startup timeout while
its process remains alive; a process that exits is considered disconnected.
The bridge forwards `/BTLog` and `/rosout` messages back to Blender; the panel
shows the ROS bridge status, service response, and a scrollable history of up
to 50 remote log entries.

The ROS service names, `Control.srv` command values, request field names,
rosbridge operations, topic subscriptions, and local bridge ports are defined
in [`common/robot_server_config.py`](common/robot_server_config.py).
This file is the protocol boundary to keep aligned with the robot server.

### Synthesizer

The Synthesizer creates randomized episodes from an animated Blender scene.

1. Set dry-run mode, episode prefix/count/start index, and output root.
2. Add objects under **Object Range Randomization Targets**.
3. Set translation and rotation ranges over selected frame intervals.
4. Add material targets and populate each alternative material pool.
5. Enable animation rendering and/or MMRT recording.
6. Click **Run Generation Batch**. Use **Reset Actions** after a dry run when
   action restoration is needed.

## Generated files

### Recorder output

For a base filename such as `recording_data`:

| File | Description |
| --- | --- |
| `recording_data.mmrt.yaml` | Motion metadata, labels, frames, and timeline events. |
| `recording_data.mmrt.bin` | Optional compact binary frame data. |
| `recording_data.env.json` | Environment metadata, units, attachments, and collisions. |
| `recording_data.urdf` | Resolved recorded URDF with portable relative mesh paths. |
| `recording_data.srdf` | Resolved recorded SRDF with portable relative mesh paths. |
| `meshes/robot/*` | Robot visual/collision source meshes referenced by the recorded descriptions. |
| `meshes/*.stl` | Exported environment meshes when enabled. |

### MMRT app output

```text
app/
├── trees/
│   ├── app.bt.xml
│   └── episode_subtrees.bt.xml  # only when episodes are declared
├── blackboards/
│   ├── system_settings.btbb.yaml
│   ├── motion_waypoints.btbb.yaml
│   ├── environment_settings.btbb.yaml
│   └── episode_settings.btbb.yaml  # only when parameterized episode logic exists
├── meshes/
│   └── ... copied environment meshes ...
└── config.apcfg.yaml
```

The complete directory is required for remote deployment. Sending only the XML
does not include the blackboards or environment assets.
When episodes are declared, the main tree references their definitions from
`trees/episode_subtrees.bt.xml`; the generated app config lists both XML files.

## Process manager and console output

External tools are launched through `processes/process_manager.py`. Processes
are registered by name, monitored by a Blender timer, and stopped when the
addon unloads, Blender exits, or a new `.blend` is loaded. Behavior-tree
generation captures stdout and stderr in the MMRT panel's Generation Log.
Other console-oriented jobs inherit the parent Blender process's standard
output, so launch Blender from a terminal or enable Blender's System Console
to follow those jobs live.

The process manager also provides:

- duplicate-process protection;
- named process lookup and stop operations;
- cleanup on `.blend` load and Blender exit; and
- Windows process-group creation with platform-specific termination behavior.

## Defaults and conventions

- Addon tab: `MMRT Utilities`
- Robot recognition: every visible armature, regardless of its name
- Active objects: added explicitly in the Recorder panel
- Environment candidates: visible mesh and curve objects
- Default recorder output directory: `//output/`
- Default recording units: radians and meters
- Default MMRT app directory: `//output/untitled_app/`
- Default main behavior-tree ID: `MainTree`

## Project structure

```text
MMRT_Blender_Utilities/
├── __init__.py
├── backend/                 # Generated platform-specific PyInstaller backend
├── common/
│   ├── config.py           # Shared defaults, enums, and UI configuration
│   ├── robot_server_config.py # Robot ROS bridge protocol contract
│   ├── blender/            # Blender context, operator, and UI helpers
│   ├── exporting/          # Environment and mesh export
│   ├── marker_utils/       # Marker types, handlers, events, and serialization
│   ├── mmrt/               # Generated-application paths
│   ├── recording/          # Recorder engine and output paths
│   ├── robot_description/  # URDF/Xacro parsing, storage, and kinematics
│   ├── synthesis/          # Randomized episode pipeline
│   └── system/             # Filesystem and Python-environment paths
├── operators/              # Blender operators
├── processes/
│   ├── process_manager.py
│   ├── helpers/            # SFTP sender, ROS bridge, and local bridge client
│   └── mmrt/               # MMRT processor and supporting modules
│       ├── main.py         # Thin command-line entry point
│       ├── pipeline.py     # Processing orchestration and export flow
│       ├── joint_limit_plotter.py
│       ├── common/         # Focused MMRT data, analysis, simulation, and export modules
│       └── definitions/    # Behavior-tree model definitions
├── properties/             # Blender PropertyGroups
└── ui/                     # Sidebar panels
```

## Troubleshooting

- **Incomplete recording bundle** — re-record after assigning each Recorder
  robot its armature, matching URDF/SRDF Text datablocks, and Original Meshes
  collection.
- **URDF/SRDF mismatch** — assign the same URDF and SRDF Text datablocks to every
  robot entry before recording.
- **Record file not found** — record motion first, or disable Scene Recorder
  output and select an existing `.mmrt.yaml`.
- **Incomplete MMRT app** — regenerate and verify `trees`, `blackboards`,
  `meshes`, the behavior tree, all three blackboards, and `config.apcfg.yaml`.
- **Processing/import errors** — click **Verify MMRT Backend**. If the bundle is
  missing or stale, rebuild it from `tools/mmrt_backend` for the current OS.
- **Remote-control errors** — verify SSH settings, the destination directory,
  ROS bridge state, and the configured robot console service.
- **No behavior-tree generation output** — check the **Generation Log** in the
  MMRT panel, including its status and any launch error. For other processes,
  start Blender from a terminal or enable the System Console.

## Development

The addon registers from [`__init__.py`](__init__.py). Add scene state in
`properties/`, operators in `operators/`, panels in `ui/`, and processing logic
in `common/` or `processes/`.

For the MMRT processing stages, module responsibilities, input bundle, CLI,
and generated application layout, see
[`processes/mmrt/README.md`](processes/mmrt/README.md).

When adding a timeline marker type, update `types.py`, `icons.py`, the handler in
`common/marker_utils/handlers/`, and its `registry.py` entry.

The identifiers in `processes/mmrt/definitions/behavior_tree_models.xml` mirror plugins and
data types registered by the deployed ROS/MMRT runtime. Treat that file as an
external compatibility contract; changing those identifiers requires a matching
runtime release.

## License

MIT License © 2025 Hỷ Tạ Chiu
