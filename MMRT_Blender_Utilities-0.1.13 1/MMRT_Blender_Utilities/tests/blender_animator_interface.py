"""Check default animator controls, operator gating, and reversible full mode."""
import bpy,sys,os,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import MMRT_Blender_Utilities as addon
from MMRT_Blender_Utilities.common.blender import get_bl_idname
from MMRT_Blender_Utilities.animator_interface import MMRT_PT_AnimatorExport
from MMRT_Blender_Utilities.ui.recorder_panel import MMRT_PT_RecorderPanel

def op(name):
    module,method=get_bl_idname(name).split('.')
    return getattr(getattr(bpy.ops,module),method)

def run():
    addon.register()
    scene=bpy.context.scene
    assert scene.mmrt_animator_mode
    assert MMRT_PT_AnimatorExport.poll(bpy.context)
    assert not MMRT_PT_RecorderPanel.poll(bpy.context)
    assert not op('run_synthesizer').poll()
    assert not op('refresh_robot_joints').poll()
    assert op('export_current_dataset_episode').poll()
    assert op('check_joint_limits').poll()
    assert op('retime_joint_limits').poll()
    scene.mmrt_synthesizer.episode_prefix='preserved'
    scene.mmrt_animator_mode=False
    assert MMRT_PT_RecorderPanel.poll(bpy.context)
    assert op('run_synthesizer').poll()
    assert op('refresh_robot_joints').poll()
    assert not MMRT_PT_AnimatorExport.poll(bpy.context)
    scene.mmrt_animator_mode=True
    assert scene.mmrt_synthesizer.episode_prefix=='preserved'
    addon.unregister()
    addon.register()
    assert not op('run_synthesizer').poll()
    scene.mmrt_animator_mode=False
    assert op('run_synthesizer').poll()
    addon.unregister()
    print('PASS default animator interface, advanced operator gating, full-mode recovery, data preservation and re-registration',flush=True)
try:run()
except BaseException:
    traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
else:sys.stdout.flush();os._exit(0)
