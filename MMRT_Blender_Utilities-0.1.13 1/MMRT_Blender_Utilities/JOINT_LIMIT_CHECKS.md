Joint limit checks while editing animation

The **MMRT Utilities → Joint Limit Checks (Current Animation)** panel checks the
current scene without recording or exporting it. It reports position, velocity,
and acceleration violations and lets you jump to their frames. The former
Evaluate Joint Limits button is now named **Plot Exported Joint Trajectories**;
that separate tool still plots an existing recording.

**Set up a check**

1. Update/reinstall the edited addon if Blender uses a separate installed copy,
   then restart Blender.
2. In the recorder, configure the robot armature, its URDF Text datablock, and
   the URDF-joint-to-bone pairings. Check the axis, inversion, and offset settings.
   The checker uses the same rotational measurement convention as the recorder.
3. Set the scene's Start/End range to the sequence you want to check.
4. Open **Joint Limit Checks (Current Animation)**. Select the matching robot's
   `joint_limits.yaml` in **Joint Limits YAML/JSON**. In this workspace, the two
   robot variants provide these files:

   - `TomO_files/tomo_re_desc/config/joint_limits.yaml`
   - `TomO_files/tomo_re_uhand_desc/config/joint_limits.yaml`

   Choose the file for your actual robot variant. Acceleration limits are usually
   supplied here; the checker does not invent missing limits.
5. Leave **Samples per Frame** at 4 initially and click **Stretch Violating Regions**.

URDF supplies lower/upper position bounds and maximum velocity. The selected
limits file can override those values and add acceleration bounds. Rotational
units are radians, radians/second, and radians/second², regardless of the recorder's
chosen export angle units. Speed and acceleration are checked in both directions:
`abs(velocity) <= max_velocity` and `abs(acceleration) <= max_acceleration`.

**Automatically slow violating regions**

The panel also provides **Stretch Violating Regions** and **Restore Original
Timing**. These retain recoverable original actions and use a common clock to slow
coordinated motion. See the [regional retiming guide](REGIONAL_RETIMING.md) for
setup, supported input, and restoration.

**Read the results and edit the motion**

Each result identifies the robot/joint, violating frame, measurement type, and
measured value. Click its frame button to go to that evaluated pose. Select a row
to see its permitted bounds and **Original timing frame**.

With time stretching, the timeline/output frame and source curve frame can differ.
For example, with Old=100 and New=200, output frame 100 evaluates source frame 50.
The original timing frame refers to the original action clock. Restore Original
Timing before editing and reapplying a retime.
The speed calculation uses output elapsed time and `fps / fps_base`; slowing the
animation by a factor of two halves velocity and quarters acceleration.

The checker samples between keys too. A reported frame may therefore be a
fractional frame with no keyframe of its own: inspect and adjust the neighboring
keys or handles. It does not insert keys or change your animation. It restores
the timeline frame and subframe after each check.

Use **Stretch Violating Regions** after editing; it runs the original-motion checks
and the required validation. The separate check button has been removed. There is no
background recheck timer. Edits mark the previous results stale without scanning.
Retiming still performs its required validation scans. Long scans occupy Blender's
main thread; use a shorter scene range while editing, then check the full range
before export.

**Statuses and the full report**

- **VIOLATIONS**: one or more sampled values exceeded a bound, beyond the configured
  absolute tolerance. The count is violating measurements, not distinct keyframes.
- **INCOMPLETE**: no sampled violations were found, but some checks could not be
  performed. Inspect the full report for missing/disabled limits or unpaired joints.
- **No sampled violations**: all configured, supported checks completed without a
  sampled violation. This describes the last scan, not a continuous mathematical proof.
- **Check failed**: configuration or evaluation failed. Old result rows are cleared.
- **Not checked or changed since last check**: results need refreshing. Rerun the
  check after finishing edits and before exporting.

The visible list is capped at 1,000 entries. **Copy Full Report** copies every
violation and coverage note as CSV text. The same report is stored in the Blender
Text datablock named `MMRT Joint Limit Report.csv` (Blender may append a suffix).
Open it in the Text Editor and use Save As to save an external report. Each new
check replaces that scene's report; copy/save it first if you need a history.
Reports include a UTC timestamp, effective FPS, tolerance, timeline/source frames,
measured values, and bounds. Lines beginning `# UNCHECKED:` explain coverage gaps.

**Limits file format**

The existing TomO files use the supported scalar-mapping YAML format, for example:

```yaml
joint_limits:
  Left_arm_joint1:
    has_velocity_limits: true
    max_velocity: 2.6179
    has_acceleration_limits: true
    max_acceleration: 8.7263
```

Use the real values from your robot configuration. JSON with the same structure
is also supported. Optional `min_position`, `max_position`, and
`has_position_limits` fields are recognized. Disabling a check or omitting a
required bound is reported as incomplete coverage, not a pass. Continuous joints
have no bounded position check by definition.

The YAML reader intentionally supports only mappings, numeric values, booleans,
and comments, so no extra Python dependency is needed inside Blender. Complex
YAML features such as anchors, lists, tags, and multiline values are unsupported.

**Scope and measurement limits**

The current Blender integration checks paired, enabled revolute and continuous
joints. Unpaired/disabled movable joints and unsupported joint types, including
prismatic joints, are explicitly reported as unchecked. This follows the existing
recorder's rotational bone mapping; a translation-joint mapping needs a separate
extension. The numerical analysis itself supports linear values, but the Blender
sampler does not infer a translation mapping from a rotation pairing.

Evaluated poses include drivers and constraints. Angles are unwrapped across
samples before differentiation to avoid false spikes at ±π. A whole-turn offset
already present at the first sample cannot be recovered from orientation matrices;
start from a known joint configuration. Motion exceeding half a revolution per
sample is ambiguous, so increase sampling for very fast rotations.

Velocity and acceleration are finite-difference estimates. Samples between keys
catch many interpolation overshoots, but narrow peaks can fall between samples.
Increase sampling near suspicious motion. Sharp slope changes at keys produce
large sampled acceleration; changing handles/interpolation or timing may be needed.
Limits are checked over the selected scene range only, without assuming a stop
before or after that range. A one-frame range cannot establish velocity or
acceleration and is reported as incomplete. This checker does not block export
or replace the robot controller's limit enforcement.

**Validation**

The automated tests cover limit parsing, malformed limits, negative velocities,
acceleration including endpoints, fractional FPS, time stretching, rotation
wraparound, missing bounds, and tolerance. Blender integration tests exercise
evaluated drivers, Bezier overshoot between valid keys, subframe navigation,
frame restoration, report generation, manual-only checking and stale-result invalidation, and addon registration.

From the repository root, run the numerical tests with:

```bash
python3 -m pytest MMRT_Blender_Utilities/tests/test_joint_limits.py
```

Run the Blender integration test in a fresh background session:

```bash
blender --background --factory-startup --disable-autoexec --python-exit-code 1 --python MMRT_Blender_Utilities/tests/blender_joint_limit_checks.py
```
