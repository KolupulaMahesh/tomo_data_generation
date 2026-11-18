# Animator handoff

Install `MMRT_Blender_Utilities-0.1.7.zip` and restart Blender. Animator Mode is
on by default for scenes without a saved mode selection. The supported export
workflow uses Blender 5.2; tests ran on Blender 5.2.2.

The MMRT sidebar shows:

1. **Animator Workflow** — scene range, camera, configured robot/object summary.
2. **Joint Limit Checks (Current Animation)** — check limits, inspect violations,
   move existing keys with Stretch Violating Regions, and restore original timing.
3. **Export One Dataset Episode** — output root, prefix, episode index, optional
   scene copy, and one export button.

Animate the prepared robot scene normally. Click Stretch Violating Regions (which includes the checks),
review the motion and any remaining violations, then export. Do not restore
original timing before exporting if you want the stretched animation.

Advanced panels and their direct operators are disabled in Animator Mode:
robot importer, global/backend tools, recorder configuration and standalone
recording, waypoint tuning, MMRT processing, simulator, and synthesis batch
randomization. Their code, data, and required core services remain installed.
This is a simplified interface, not a security or access-control mechanism.

To configure robot mappings, objects, materials, or advanced settings, uncheck
**Animator Mode** in the Animator Workflow panel. All original panels/operators
return immediately. Re-enable Animator Mode and save the scene before handing it
to animators. This setting is stored per scene, so a file deliberately saved in
full mode will reopen that way.

## Files to give animators

- The add-on ZIP.
- A prepared `.blend` with the three robot recorder entries and any required
  interactive objects already configured, plus the desired camera/render settings.
- The matching joint-limits YAML. Set **Joint Limits YAML/JSON** to a path available
  on their machine; a project-relative path such as `//joint_limits.yaml` is useful.
- All linked `.blend` libraries, textures, and required mesh assets, using portable
  project-relative paths where possible. Installing the add-on does not transfer
  files referenced from your computer's `/home/leo/...` paths.

Each animator selects their own Output Root and a unique Prefix/Episode Index.
The export preserves retimed NLA and current materials and produces MP4, depth
EXRs, MMRT YAML, robot URDF/SRDF/meshes, and dataset metadata. Existing episode
folders are not overwritten. See CURRENT_DATASET_EPISODE.md for the full layout.

The add-on does not disable unrelated third-party Blender add-ons.
