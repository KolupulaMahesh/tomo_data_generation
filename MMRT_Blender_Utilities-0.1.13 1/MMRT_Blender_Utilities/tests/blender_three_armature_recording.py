"""Read-only recording startup and sample checks against the supplied TomO scene."""
import bpy,sys,os
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import MMRT_Blender_Utilities as addon
addon.register()
scene_path = Path(sys.argv[sys.argv.index('--')+1]) if '--' in sys.argv else Path(__file__).resolve().parents[3] / 'iPhone_Packing_recording_ready.blend'
bpy.ops.wm.open_mainfile(filepath=str(scene_path))
from MMRT_Blender_Utilities.common.recording.recorder import MMRTRecorder
r=MMRTRecorder(bpy.context)
r.run_pipeline()
assert r.robot_recording_summary() == 'tomo: 16 joints, tomo_right_hand: 15 joints, tomo_left_hand: 15 joints'
print('ROBOT_CHANNELS',dict(Counter(d['obj_name'] for d in r.descriptors if d['section']=='robots')),flush=True)
for frame in (0,290,1095):
 bpy.context.scene.frame_set(frame)
 row=r.evaluate_current_frame()
 for name in ('tomo','tomo_left_hand','tomo_right_hand'):
  assert any(k.startswith(name+'.location.') for k in row),name
 assert all(j.joint_name in row for robot in r.recorder.robots for j in robot.joints if j.record), list(row)[:70]
 print('FRAME',frame,'VALUES',len(row),flush=True)
print('PASS three robot recording pipeline and marker collection',flush=True)
from MMRT_Blender_Utilities import joint_limit_retime as retime
import numpy as np
scene = bpy.context.scene
objects = retime._preflight(scene, [])
phone = scene.objects['iPhone']
assert phone in objects
times = {float(frame) for frame in range(scene.frame_start, scene.frame_end+1, 100)}
for curve in retime._curves(phone.animation_data.action, phone.animation_data.action_slot):
 if 'constraints[' in curve.data_path:
  for key in curve.keyframe_points:
   times.update(float(key.co.x) + delta for delta in (-.25, 0, .25)
                if scene.frame_start <= key.co.x+delta <= scene.frame_end)
def phone_pose(frame):
 import math
 scene.frame_set(math.floor(frame), subframe=frame-math.floor(frame))
 return np.asarray(phone.evaluated_get(bpy.context.evaluated_depsgraph_get()).matrix_world).copy()
before = {frame: phone_pose(frame) for frame in sorted(times)}
scene[retime.STATE] = retime._snapshot(scene, objects)
try:
 strips = retime._make_strips(scene)
 old = np.linspace(scene.frame_start, scene.frame_end, 101)
 new = old[0] + (old-old[0])*2
 for strip in strips:
  retime._set_clock(strip, old, new, np.full(len(old), .5))
 for frame, matrix in before.items():
  assert np.allclose(matrix, phone_pose(old[0]+(frame-old[0])*2), atol=1e-4, rtol=0), frame
 print(f'PASS actual iPhone world poses at {len(before)} times including attachment transitions under shared clock',flush=True)
finally:
 retime.restore_timing(scene)
scene[retime.STATE] = retime._snapshot(scene, objects)
scene[retime.STATE]['method'] = 'KEYFRAMES'
counts = {obj.name: sum(len(c.keyframe_points) for c in retime._curves(obj.animation_data.action, None)) for obj in objects}
try:
 retime.make_working_actions(scene[retime.STATE])
 retime.apply_key_clock(scene[retime.STATE], old, new, retime._curves)
 for obj in objects:
  assert not obj.animation_data.nla_tracks
  assert sum(len(c.keyframe_points) for c in retime._curves(obj.animation_data.action, None)) == counts[obj.name]
 for frame, matrix in before.items():
  assert np.allclose(matrix, phone_pose(old[0]+(frame-old[0])*2), atol=1e-4, rtol=0), frame
 print('PASS direct existing-key retime preserves actual iPhone attachment poses and key counts',flush=True)
finally:
 retime.restore_timing(scene)
sys.stdout.flush();os._exit(0)
