"""Export the evaluated scene as one dataset episode, preserving retimed NLA."""
import ast
import os
from pathlib import Path
import pprint

import bpy

from .pipeline import BlenderSynthesizer, print_synthesis_progress
from ..config import EnvironmentExportMode
from ..recording.frame_domain import get_frame_domain


def _read_metadata(path):
    if not path.exists():
        return {}
    tree = ast.parse(path.read_text(encoding='utf-8'))
    if (len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign)
            or len(tree.body[0].targets) != 1
            or not isinstance(tree.body[0].targets[0], ast.Name)
            or tree.body[0].targets[0].id != 'metadata'):
        raise ValueError('Existing metadata.py is not a plain metadata dictionary; it will not be overwritten.')
    value = ast.literal_eval(tree.body[0].value)
    if not isinstance(value, dict) or not isinstance(value.get('episodes', {}), dict):
        raise ValueError('Existing dataset metadata has an unsupported structure.')
    return value


def _write_metadata(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text('metadata = ' + pprint.pformat(value, sort_dicts=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def export_current_episode(context):
    scene = context.scene
    config = scene.mmrt_synthesizer
    prefix = config.episode_prefix.strip()
    if not prefix or prefix in {'.', '..'} or any(c in prefix for c in '/\\\x00'):
        raise ValueError('Episode Prefix must be a filename, without directory separators.')
    if not config.output_root:
        raise ValueError('Set the dataset Output Root first.')
    if scene.camera is None:
        raise ValueError('Select a scene camera before exporting the video.')
    if scene.mmrt_recorder.is_running or bpy.app.is_job_running('RENDER'):
        raise ValueError('Wait for the current recording/render to finish.')
    if context.screen and context.screen.is_animation_playing:
        raise ValueError('Stop playback before exporting.')
    if not hasattr(scene, 'compositing_node_group'):
        raise ValueError('Current-episode export requires the Blender 5 compositor API.')
    root = Path(bpy.path.abspath(config.output_root)).resolve()
    name = f'{prefix}_{config.episode_start_index:03d}'
    directory = root / name
    metadata_path = root / 'metadata.py'
    existing = _read_metadata(metadata_path)
    if directory.exists() or name in existing.get('episodes', {}):
        raise ValueError(f'{name} already exists. Choose a new Prefix or Start Index; nothing was overwritten.')
    domain = get_frame_domain(scene)
    output_start, output_end = domain['stretched_start'], domain['stretched_end']
    if output_end > 1048574 or output_start < 0 or output_end < output_start:
        raise ValueError('Output frame range is invalid or exceeds Blender limits.')
    render = scene.render
    recorder_config = scene.mmrt_recorder.config
    original = {
        'tree': scene.compositing_node_group,
        'range': (scene.frame_start, scene.frame_end, scene.frame_step),
        'frame': (scene.frame_current, scene.frame_subframe),
        'filepath': render.filepath,
        'media_type': getattr(render.image_settings, 'media_type', None),
        'format': render.image_settings.file_format,
        'ffmpeg': {key: getattr(render.ffmpeg, key) for key in ('format', 'codec', 'constant_rate_factor', 'ffmpeg_preset')},
        'z': [(layer, layer.use_pass_z) for layer in scene.view_layers],
        'compositing': render.use_compositing,
        'lock': render.use_lock_interface,
        'extension': render.use_file_extension,
        'binary': recorder_config.export_binary,
        'env_mode': recorder_config.env_export_mode,
        'running': scene.mmrt_recorder.is_running,
    }
    working_tree = None
    directory.mkdir(parents=True, exist_ok=False)
    print_synthesis_progress(f'{name}: exporting CURRENT animation; no pose or material randomization')
    try:
        # Isolate output paths and mute other file-output nodes so an export
        # cannot write into a previous dataset through a saved compositor path.
        working_tree = original['tree'].copy() if original['tree'] else bpy.data.node_groups.new('MMRT Episode Compositor', 'CompositorNodeTree')
        scene.compositing_node_group = working_tree
        for node in working_tree.nodes:
            if node.bl_idname == 'CompositorNodeOutputFile':
                node.mute = True
        old_depth = working_tree.nodes.get('SynthDepthOutput')
        if old_depth:
            working_tree.nodes.remove(old_depth)
        engine = BlenderSynthesizer(dict(dry_run=False, random_seed=None, objects={},
            material_randomization=[], output_root=str(root), render_output='MP4',
            record_mmrt_fast_mode=True))
        depth = engine.configure_depth_output(directory)
        depth.mute = False
        depth.format.file_format = 'OPEN_EXR_MULTILAYER'
        depth.format.color_depth = '32'
        render.use_compositing = True
        render.use_file_extension = True
        scene.frame_step = 1
        # Match the existing dataset's self-contained text YAML.
        recorder_config.export_binary = False
        if recorder_config.env_export_mode != EnvironmentExportMode.OFF:
            recorder_config.env_export_mode = EnvironmentExportMode.AUTO
        mmrt_path = engine.record_mmrt(context, directory, name)
        engine.configure_render_paths(directory, name)
        if config.save_blend:
            print_synthesis_progress(f'{name}: saving scene copy')
            bpy.ops.wm.save_as_mainfile(filepath=str(directory / f'{name}.blend'), copy=True)
        # MMRT samples OUTPUT frames. Render exactly that same range, including
        # global Old/New mapping, without applying the mapping a second time.
        scene.frame_start, scene.frame_end = output_start, output_end
        print_synthesis_progress(f'{name}: rendering MP4 and depth, output frames {output_start}–{output_end}')
        result = bpy.ops.render.render(animation=True, write_still=False)
        if 'FINISHED' not in result:
            raise RuntimeError('Render did not finish; episode metadata was not published.')
        video = directory / f'{name}.mp4'
        if not video.is_file() or video.stat().st_size == 0:
            raise RuntimeError('Render returned without the expected MP4; episode metadata was not published.')
        depth_count = len(list((directory / 'depth').glob('depth_*.exr')))
        if depth_count != output_end-output_start+1:
            raise RuntimeError(f'Expected {output_end-output_start+1} depth frames, found {depth_count}; episode metadata was not published.')
        entry = dict(episode_name=name, episode_dir=str(directory), offsets_applied={},
                     materials_swapped=[], mmrt_record_filepath=mmrt_path,
                     export_mode='current_animation', regional_retiming_active='_mmrt_joint_retime' in scene,
                     source_frame_start=original['range'][0], source_frame_end=original['range'][1],
                     output_frame_start=output_start, output_frame_end=output_end,
                     fps=render.fps / render.fps_base)
        _write_metadata(directory / 'metadata.py', entry)
        # Re-read immediately before merging, never discard other episode entries.
        batch = _read_metadata(metadata_path)
        episodes = batch.setdefault('episodes', {})
        if name in episodes:
            raise RuntimeError('Dataset metadata changed during export; refusing to overwrite its episode.')
        batch.setdefault('blend_file', bpy.data.filepath)
        batch.setdefault('output_root', str(root))
        batch.setdefault('frame_start', original['range'][0])
        batch.setdefault('frame_end', original['range'][1])
        episodes[name] = entry
        _write_metadata(metadata_path, batch)
        print_synthesis_progress(f'{name}: complete: {directory}')
        return directory
    finally:
        scene.frame_start, scene.frame_end, scene.frame_step = original['range']
        render.filepath = original['filepath']
        if original['media_type'] is not None:
            render.image_settings.media_type = original['media_type']
        render.image_settings.file_format = original['format']
        for key, value in original['ffmpeg'].items():
            setattr(render.ffmpeg, key, value)
        render.use_compositing = original['compositing']
        render.use_lock_interface = original['lock']
        render.use_file_extension = original['extension']
        for layer, enabled in original['z']:
            layer.use_pass_z = enabled
        recorder_config.export_binary = original['binary']
        recorder_config.env_export_mode = original['env_mode']
        scene.mmrt_recorder.is_running = original['running']
        scene.compositing_node_group = original['tree']
        if working_tree and working_tree.users == 0:
            bpy.data.node_groups.remove(working_tree)
        scene.frame_set(original['frame'][0], subframe=original['frame'][1])
