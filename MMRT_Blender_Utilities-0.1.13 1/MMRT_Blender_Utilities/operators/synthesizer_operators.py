import bpy
from ..common.blender import get_bl_idname, show_report
from ..common.synthesis import BlenderSynthesizer
from ..common.synthesis.pipeline import ActionManager, MaterialManager

class MMRT_OT_RunSynthesizer(bpy.types.Operator):
    bl_idname = get_bl_idname("run_synthesizer")
    bl_label = "Run Generation Batch"
    bl_description = "Execute the synthesizer pipeline processing bounds"

    def execute(self, context):
        config_pg = context.scene.mmrt_synthesizer
        
        engine_config = {
            "dry_run": config_pg.dry_run,
            "episode_prefix": config_pg.episode_prefix,
            "episode_count": config_pg.episode_count,
            "episode_start_index": config_pg.episode_start_index,
            "random_seed": config_pg.random_seed if config_pg.random_seed != 0 else None,
            "output_root": config_pg.output_root if config_pg.output_root else None,
            "render_output": config_pg.render_output,
            "render_animation": config_pg.render_animation,
            "save_episode_blend": config_pg.save_blend,
            "record_mmrt": config_pg.record_mmrt,
            "record_mmrt_before_render": config_pg.record_mmrt_before_render,
            "record_mmrt_fast_mode": config_pg.record_mmrt_fast,
            "objects": {},
            "material_randomization": []
        }

        for item in config_pg.objects:
            if not item.obj: continue
            engine_config["objects"][item.obj.name] = {
                "frame_min": item.frame_min,
                "frame_max": item.frame_max,
                "parent_offset_object": item.parent_offset_obj.name if item.parent_offset_obj else None,
                "translation_range": {
                    "x": (item.trans_min[0], item.trans_max[0]), 
                    "y": (item.trans_min[1], item.trans_max[1]), 
                    "z": (item.trans_min[2], item.trans_max[2])
                },
                "rotation_range_deg": {
                    "x": (item.rot_min_deg[0], item.rot_max_deg[0]), 
                    "y": (item.rot_min_deg[1], item.rot_max_deg[1]), 
                    "z": (item.rot_min_deg[2], item.rot_max_deg[2])
                },
            }

        for index, mat in enumerate(config_pg.materials, start=1):
            if not mat.obj or not mat.target_material:
                show_report(
                    f"Material target {index}: select an object and its original material.",
                    "ERROR",
                )
                return {'CANCELLED'}
            pool = [item.material.name for item in mat.material_pool if item.material]
            if not pool:
                show_report(f"Material target {index}: the pool is empty.", "ERROR")
                return {'CANCELLED'}

            engine_config["material_randomization"].append({
                "object_name": mat.obj.name,
                "target_material_name": mat.target_material.name,
                "material_pool": pool,
            })

        try:
            synthesizer = BlenderSynthesizer(engine_config)
            synthesizer.execute_generation()
            show_report(f"Synthesis Complete: {config_pg.episode_count} episodes generated.", "INFO")
        except Exception as e:
            show_report(f"Synthesis Failed: {str(e)}", "ERROR")
            return {'CANCELLED'}

        return {'FINISHED'}

class MMRT_OT_ExportCurrentDatasetEpisode(bpy.types.Operator):
    bl_idname = get_bl_idname('export_current_dataset_episode')
    bl_label = 'Export Current Animation as Dataset Episode'
    bl_description = 'Export one MP4, depth sequence, MMRT bundle and dataset metadata using current animation and materials, including regional timing'

    def execute(self, context):
        from ..common.synthesis.current_episode import export_current_episode
        try:
            directory = export_current_episode(context)
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        self.report({'INFO'}, f'Current animation exported: {directory}')
        return {'FINISHED'}


class MMRT_OT_ResetSynthActions(bpy.types.Operator):
    bl_idname = get_bl_idname("reset_synth_actions")
    bl_label = "Reset Previews"
    bl_description = "Restore original actions and material assignments after dry runs"

    def execute(self, context):
        restored_count = 0
        errors = []

        # Include preview objects removed from the target list after a dry run.
        for obj in list(context.scene.objects):
            restored = False
            for manager in (ActionManager, MaterialManager):
                try:
                    restored = manager.restore_preview(obj) or restored
                except Exception as exc:
                    errors.append(f"{obj.name}: {exc}")
            restored_count += int(restored)

        bpy.context.view_layer.update()
        if errors:
            show_report("; ".join(errors), "ERROR")
            return {'CANCELLED'}

        show_report(f"Restored {restored_count} preview object(s).", "INFO")
        return {'FINISHED'}

class MMRT_OT_AddSynthObject(bpy.types.Operator):
    bl_idname = get_bl_idname("add_synth_object")
    bl_label = "Add Target"
    def execute(self, context):
        context.scene.mmrt_synthesizer.objects.add()
        return {'FINISHED'}

class MMRT_OT_RemoveSynthObject(bpy.types.Operator):
    bl_idname = get_bl_idname("remove_synth_object")
    bl_label = "Remove Target"
    def execute(self, context):
        idx = context.scene.mmrt_synthesizer.objects_index
        if idx >= 0 and idx < len(context.scene.mmrt_synthesizer.objects):
            context.scene.mmrt_synthesizer.objects.remove(idx)
        return {'FINISHED'}

class MMRT_OT_AddSynthMaterial(bpy.types.Operator):
    bl_idname = get_bl_idname("add_synth_material")
    bl_label = "Add Material Target"
    def execute(self, context):
        context.scene.mmrt_synthesizer.materials.add()
        return {'FINISHED'}

class MMRT_OT_RemoveSynthMaterial(bpy.types.Operator):
    bl_idname = get_bl_idname("remove_synth_material")
    bl_label = "Remove Material Target"
    def execute(self, context):
        idx = context.scene.mmrt_synthesizer.materials_index
        if idx >= 0 and idx < len(context.scene.mmrt_synthesizer.materials):
            context.scene.mmrt_synthesizer.materials.remove(idx)
        return {'FINISHED'}

class MMRT_OT_AddSynthPoolMaterial(bpy.types.Operator):
    bl_idname = get_bl_idname("add_synth_pool_material")
    bl_label = "Add Pool Material"
    def execute(self, context):
        config = context.scene.mmrt_synthesizer
        if config.materials and 0 <= config.materials_index < len(config.materials):
            mat_target = config.materials[config.materials_index]
            mat_target.material_pool.add()
        return {'FINISHED'}

class MMRT_OT_RemoveSynthPoolMaterial(bpy.types.Operator):
    bl_idname = get_bl_idname("remove_synth_pool_material")
    bl_label = "Remove Pool Material"
    def execute(self, context):
        config = context.scene.mmrt_synthesizer
        if config.materials and 0 <= config.materials_index < len(config.materials):
            mat_target = config.materials[config.materials_index]
            idx = mat_target.pool_index
            if idx >= 0 and idx < len(mat_target.material_pool):
                mat_target.material_pool.remove(idx)
        return {'FINISHED'}

classes = (
    MMRT_OT_RunSynthesizer,
    MMRT_OT_ExportCurrentDatasetEpisode,
    MMRT_OT_ResetSynthActions,
    MMRT_OT_AddSynthObject,
    MMRT_OT_RemoveSynthObject,
    MMRT_OT_AddSynthMaterial,
    MMRT_OT_RemoveSynthMaterial,
    MMRT_OT_AddSynthPoolMaterial,
    MMRT_OT_RemoveSynthPoolMaterial,
)

def register():
    for cls in classes: bpy.utils.register_class(cls)

def unregister():
    for cls in reversed(classes): bpy.utils.unregister_class(cls)
