Regional retiming to meet joint velocity and acceleration limits

**Stretch Violating Regions** changes when the robot reaches each pose. It keeps
the source actions, poses, and motion path intact. Velocity and acceleration profiles
necessarily change when timing changes; the geometric motion is preserved.

**Use it**

1. Update the installed addon from this checkout and restart Blender.
2. Configure the recorder robot, joint pairings, URDF, and joint-limits file as
   described in [Joint Limit Checks](JOINT_LIMIT_CHECKS.md).
3. Set the scene range to include all keys in the coordinated object actions.
   Stop playback. No separate check button is needed.
4. In **Joint Limit Checks → Regional Timing — Preserve Motion Path**, leave
   **Limit Utilization** at 90% initially. This targets a margin below the stated
   limits. **Timing Transition (s)** controls how broadly slowdowns extend into
   neighboring motion; its default is 0.25 seconds of original motion time.
5. Select **Move Existing Keys** (the default) and click **Stretch Violating Regions**.
   The tool checks the original motion, retimes all coordinated actions, and validates the result. The scene end
   frame grows to accommodate the slower motion.
6. Review the refreshed results and play the animation. Recording and rendering
   evaluate the retimed motion directly. No separate motion export is required
   to perform the check or apply the retiming.

Stretching uses one sample per frame for its checks and candidate validation.
There is no additional double-density validation pass. **Keep Best Improvement** is enabled by default. If the attempt budget
is exhausted, the best verified partial result can remain applied: its peak
velocity and acceleration limit ratios must not worsen relative to the original,
and their sum must decrease. The comparison excludes explicitly skipped regions.
Raw violation counts are not used for ranking because longer timelines contain
more samples. Remaining violations stay visible and the result is labelled
**Partial retiming**. Disable this option to require a complete pass outside
skipped regions. **Maximum Attempts** defaults to 6 and allows 1–20.

Corresponding joint samples and world-space pose samples must still match the
source. Position-limit failures, missing coverage, and changed poses are not
accepted as partial improvements. If no acceptable improvement is found, the
original timing is restored. A later planner failure may retain an earlier
verified improvement. Restore Original Timing remains available after acceptance.

Each run prints elapsed time, attempt number, proposed end frame, validation
density, violation count, peak limit ratios, and the selected attempt. The same
messages are saved in a Blender Text datablock named **MMRT Retiming Log** (with
a numeric suffix for subsequent runs), selectable in the Text Editor. Initial
and corresponding-pose scans print progress every 1,000 samples. The operation
still runs synchronously; the UI may remain busy during evaluation.

**What changes**

**Move Existing Keys** is the default. It moves existing keyframe times and
Bezier handle times on working copies of the active actions. Key counts, values,
and interpolation modes remain unchanged: no per-frame keys are baked. Original
actions are retained for Restore Original Timing, and shared actions stay shared.
Bezier handles on working copies are set to Free to retain their original curve
geometry instead of allowing Blender to recalculate it after moving the keys.

All coordinated objects use one piecewise-linear time map. Overlapping nonconstant
curve segments must share a uniform stretch to preserve motion without inserting
keys. They are merged into common motion blocks, including constraint-influence
curves. Separated movements can stretch independently, but a long overlapping
curve can cause most or all of the sequence to stretch together. This method
therefore may stretch more of the timeline than the NLA method.

**NLA Timing Curve** remains an optional method. It retains the original source
keys and uses managed NLA strips for a smoother regional clock. Safe sections
outside its slowdown transitions retain their duration. Both methods preserve
coordinated robot/controller/prop timing and validate sampled poses and limits.

Interactable props do not need a separate opt-in: their active object actions,
including keyed Child Of constraint influences, use the same clock. Objects with
Child Of targets also have their world poses compared at every source sample,
in addition to the broader scene checkpoints. Thus the iPhone pickup/placement
timing follows the hands and box. This preserves sampled original motion; it
does not correct flying/contact errors already present in the source animation
or support unbaked physics and scene-time drivers.

With Move Existing Keys, the moved keys are visible in the active action in the
Dope Sheet/Graph Editor. With NLA Timing Curve, source keys stay at their original
action times. The report's **Original timing frame** refers to the original clock;
the output/timeline frame identifies the evaluated scene time.

The tool updates:

- The scene end frame and preview range.
- Timeline markers, including named recorder events and camera markers.
- Recorder episode start/end frames.

Markers and episode boundaries are integer-frame fields, so their mapped times
are rounded to the nearest frame. Events after the checked range are shifted by
the added duration; events before the range retain their original frame. Existing
global Old/New time stretching is retained and included in the calculations.

**Restore and edit again**

Click **Restore Original Timing** to discard the working keyframe copies (or
remove managed NLA tracks) and restore
original action/slot assignments, scene range, preview range, markers, episode
ranges, and the saved timeline position. Recovery survives saving/reopening the
`.blend` and renaming the source objects or actions.

Restore before editing the source motion and applying a new retime. Do not rename
or remove the managed NLA tracks, delete source actions/objects, add competing NLA
layers, or change marker/episode list identities while this timing state is active.
Restoration validates those references and reports a problem rather than guessing
which animation or events to overwrite. In direct-key mode, edits to the retimed working actions are discarded on restore;
restore first, edit the original animation, then retime again. In NLA mode, edits
to original source actions themselves are not undone by restoring timing.

**Supported input and cases that need attention**

This initial retimer works with plain active object actions, including shared
and slotted actions, and deterministic constraints whose motion is driven by those
coordinated actions. All animated scene objects participate, not only the selected
robot. Their action ranges must fit inside the scene range.

It rejects these cases before changing animation:

- Incomplete limit/joint coverage, unsupported joint types, or position-limit
  violations. Stretching cannot correct an out-of-range pose.
- Existing NLA stacks, drivers, or animated non-object datablocks. Convert
  coordinated motion to plain object actions before using this tool.
  In particular, frame-based drivers would
  otherwise continue to run on the scene clock instead of the retimed clock.
- Rigid bodies, particles, cloth, soft bodies, fluid, dynamic paint, and mesh
  cache modifiers. Errors identify the object and modifier. A simulation cache
  bake alone is **not** enough: the cache still follows scene time. Rigid motion
  must be converted to object actions with the physics removed. Deforming
  geometry caches require separate cache retiming, which this tool does not
  implement. Remove unrelated effects from the scene being retimed if necessary.
- Geometry Nodes using Scene Time, simulation zones, Bake nodes, animated image
  inputs, or node-tree animation/drivers, including inside nested groups.
  Static Geometry Nodes are allowed. Errors identify the node/group path where
  possible. The scan is conservative and also inspects unconnected nodes and
  hidden modifiers, since connections and visibility can change.
- Stepped transform jumps or sharp interior velocity corners when **Skip
  Unrepairable Transitions** is disabled. Finite stretching cannot remove a
  discontinuity.
- Active synthesis action previews. Reset those previews first.

A nonzero motion with a zero speed limit cannot be repaired by finite stretching.
The planner also bounds sampling and maximum stretch, and reports failure instead
of leaving a partially modified sequence.

**Skipping unrepairable transitions**

**Skip Unrepairable Transitions** is enabled by default. With the NLA method, a stepped segment retains
its duration; a sharp velocity corner protects both adjacent keyframe segments.
A small sampling buffer is also protected by the NLA method. In Move Existing
Keys mode, skipped regions are excluded from limit fitting/acceptance, but may
still stretch with an overlapping motion block to preserve coordinated curves. All objects use the same clock, so
these source-time regions are protected across the entire scene. Their absolute
timeline positions can shift when earlier regions stretch. Source poses and
interpolation remain unchanged.

Other regions are stretched and validated. The status reports **Partial retiming**
and lists the skipped source segments. The regular joint checker still reports
violations in those regions; this is not a fully limit-compliant result. If all
regions are skipped, timing stays unchanged. Failures outside the skipped regions
still roll back the operation. Position-limit errors and unsupported simulations
are not bypassed by this option.

If retiming reports incomplete coverage, expand the robot in the Recorder panel,
use **Refresh Joints**, and configure the named joint's **Bone**, **Record**,
**Axis**, **Inv**, and **Offset**. Refresh preserves existing mappings and excludes URDF mimic joints. Mimic joints
are outside the scope of the joint checker and do not block regional retiming.
Missing joints or missing velocity/acceleration limits must be resolved before
retiming. Skipping discontinuous segments does not bypass missing joint coverage.

When body and hands are separate armatures sharing the same URDF Text datablock,
joint coverage is combined across those recorder entries. Enable each joint on
its owning armature and leave unrelated rows disabled on the other parts. A
valid enabled mapping on another part covers those disabled rows; genuinely
missing joints and enabled mappings with invalid bones still report errors.

The existing synthesizer edits active actions and does not support these managed
NLA strips. Generation is therefore blocked while regional timing is active;
**Restore Original Timing** before synthesis. Recording and rendering the retimed
scene are supported. A workflow that bakes retiming into new active actions for
subsequent synthesis is not included in this version.

**How the timing is calculated**

For a constant stretch factor `s`, velocity scales by `1/s` and acceleration by
`1/s²`. Regional stretching also introduces acceleration through the changing
clock rate. The planner accounts for that extra term and broadens transitions
where needed. It constructs a monotone Bezier timing curve, then validates the
actual evaluated Blender animation rather than accepting the estimate alone.

These are sampled numerical checks, with finite-difference acceleration estimates.
They do not prove bounds at every possible instant. Increase sampling for narrow
peaks and inspect the results before exporting. Start/end checks cover the selected
sequence; they do not add a stop before its first pose or after its last pose.

**Tested behavior**

Tests cover shared-clock velocity bounds, localized slowdowns, missing/zero limits,
source-action preservation, synchronized objects, poses between clock keys, marker
and episode timing, preview restoration, rollback after validation failure,
save/reopen recovery, and existing global time stretching. The Blender integration
test also confirms that unsupported synthesis is rejected while retiming is active.

From the repository root:

```bash
python3 -m pytest MMRT_Blender_Utilities/tests/test_joint_limits.py MMRT_Blender_Utilities/tests/test_regional_retime.py
blender --background --factory-startup --disable-autoexec --python-exit-code 1 --python MMRT_Blender_Utilities/tests/blender_regional_retime.py
```

World-pose validation uses separate tolerances: translation 0.0001 scene units,
rotation 0.05 degrees, and scale/shear matrix difference 0.0001. These allow small
evaluation differences; they do not guarantee mathematically identical poses.

### Optional speedup
With Move Existing Keys, enable **Speed Up Under-Limit Blocks** and choose
**Maximum Speedup** (default 2×). The shared block uses the tightest velocity
and acceleration headroom across all checked joints and Limit Utilization.
Pauses and blocks overlapping skipped transitions are not compressed. Existing
keys, coordinated objects and event timing still share one clock. The log lists
block ranges and proposed speed multipliers. Overlapping curves still limit
how independently a head-only movement can be accelerated.

Speedup can also run on an already compliant animation. Each candidate is
validated at one sample per frame; speedup requires no remaining violations in accelerated blocks or their two-output-frame
boundary guards. Skipped violations inside those guards still block acceleration.
Velocity/acceleration violations elsewhere may remain, independently of Keep Best
Improvement; they stay visible in the results and the run is marked partial.
Position violations still reject the candidate everywhere.
Each retry halves the compression toward the slowdown-only proposal. If speedup
attempts fail, one additional slowdown-only fallback is tried (also when Maximum
Attempts is 1). That fallback may retain partial improvement when Keep Best
Improvement is enabled; remaining violations are reported. If neither compliant
timing nor an allowed fallback improvement is found, original timing is restored. Restore Original Timing before
changing the options and rerunning. Frame sampling does not certify subframe limits.
