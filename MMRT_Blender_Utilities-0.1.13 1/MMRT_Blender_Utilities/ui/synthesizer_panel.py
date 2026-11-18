import bpy
from ..common.blender import get_bl_idname
from ..common.config import UIConfig

class MMRT_UL_SynthObjectList(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        obj = item.obj
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row = layout.row(align=True)
            if obj:
                if obj.override_library:
                    icon_id = 'LIBRARY_DATA_OVERRIDE'
                elif obj.library:
                    icon_id = 'LINKED'
                else:
                    icon_id = 'OBJECT_DATA'
                
                row.label(text=obj.name, icon=icon_id)
                row.alignment = 'RIGHT'
                row.label(text=f"Frames: {item.frame_min}-{item.frame_max}", icon='TIME')
            else:
                row.label(text="(No Object Associated)", icon='ERROR')
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text="", icon='OBJECT_DATA')

class MMRT_UL_SynthMaterialList(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        obj = item.obj
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row = layout.row(align=True)
            if obj:
                mat_display = item.target_material.name if item.target_material else "(None Selected)"
                row.label(text=f"{obj.name} → {mat_display}", icon='MATERIAL')
            else:
                row.label(text="(No Object Associated)", icon='ERROR')
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text="", icon='MATERIAL')

class MMRT_UL_SynthPoolMaterialList(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            layout.prop(item, "material", text="", icon='MATERIAL')
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text="", icon='MATERIAL')

class MMRT_PT_Synthesizer(bpy.types.Panel):
    bl_idname = "MMRT_PT_SynthesizerPanel"
    bl_label = UIConfig.PANEL_SYNTHESIZER_LABEL
    bl_category = UIConfig.ADDON_CATEGORY
    bl_space_type = UIConfig.ADDON_SPACE
    bl_region_type = UIConfig.ADDON_REGION
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        config = scene.mmrt_synthesizer

        # --- GLOBAL SETTINGS ---
        box = layout.box()
        box.label(text="Global Generation Settings", icon='PREFERENCES')
        col = box.column(align=True)
        col.prop(config, "dry_run")
        col.prop(config, "episode_prefix")
        col.prop(config, "episode_count")
        col.prop(config, "episode_start_index")
        col.prop(config, "output_root")
        
        row = col.row()
        row.prop(config, "render_animation", text="Render")
        row.prop(config, "record_mmrt", text="Rec MMRT")

        current = layout.box()
        current.label(text='Current Animation → One Dataset Episode', icon='RENDER_ANIMATION')
        current.label(text='Keeps current timing, poses and materials.')
        current.label(text='Uses Output Root, Prefix and Start Index above.')
        current.prop(config, 'save_blend', text='Also Save Scene Copy')
        current.operator(get_bl_idname('export_current_dataset_episode'))
        
        # --- OBJECT CONFIGURATION ---
        layout.separator()
        box = layout.box()
        box.label(text="Object Range Randomization Targets", icon='OBJECT_DATA')
        
        row = box.row()
        row.template_list("MMRT_UL_SynthObjectList", "", config, "objects", config, "objects_index", rows=3)
        
        col = row.column(align=True)
        col.operator(get_bl_idname("add_synth_object"), text="", icon='ADD')
        col.operator(get_bl_idname("remove_synth_object"), text="", icon='REMOVE')

        if config.objects and 0 <= config.objects_index < len(config.objects):
            item = config.objects[config.objects_index]
            
            sep_box = box.box()
            sep_box.prop(item, "obj")
            sep_box.prop(item, "parent_offset_obj")
            
            f_row = sep_box.row(align=True)
            f_row.prop(item, "frame_min", text="Start Frame")
            f_row.prop(item, "frame_max", text="End Frame")
            
            sep_box.separator()

            sep_box.label(text="Translation Margins:", icon='ORIENTATION_GLOBAL')
            t_row = sep_box.row(align=True)
            col_t_min = t_row.column()
            col_t_min.prop(item, "trans_min", text="Min")
            col_t_max = t_row.column()
            col_t_max.prop(item, "trans_max", text="Max")
            
            sep_box.separator()
            
            sep_box.label(text="Rotation Margins (Deg):", icon='ORIENTATION_GIMBAL')
            r_row = sep_box.row(align=True)
            col_r_min = r_row.column()
            col_r_min.prop(item, "rot_min_deg", text="Min")
            col_r_max = r_row.column()
            col_r_max.prop(item, "rot_max_deg", text="Max")

        # --- MATERIAL CONFIGURATION ---
        layout.separator()
        box = layout.box()
        box.label(text="Material Swap Randomization Targets", icon='MATERIAL')
        
        row = box.row()
        row.template_list("MMRT_UL_SynthMaterialList", "", config, "materials", config, "materials_index", rows=3)
        
        col = row.column(align=True)
        col.operator(get_bl_idname("add_synth_material"), text="", icon='ADD')
        col.operator(get_bl_idname("remove_synth_material"), text="", icon='REMOVE')

        if config.materials and 0 <= config.materials_index < len(config.materials):
            mat_item = config.materials[config.materials_index]
            
            sep_box = box.box()
            sep_box.prop(mat_item, "obj")
            
            if mat_item.obj:
                sep_box.prop(mat_item, "target_material")
                
                sep_box.separator()
                sep_box.label(text="Alternative Material Selection Pool:", icon='ASSET_MANAGER')
                
                p_row = sep_box.row()
                p_row.template_list("MMRT_UL_SynthPoolMaterialList", "", mat_item, "material_pool", mat_item, "pool_index", rows=3)
                
                p_col = p_row.column(align=True)
                p_col.operator(get_bl_idname("add_synth_pool_material"), text="", icon='ADD')
                p_col.operator(get_bl_idname("remove_synth_pool_material"), text="", icon='REMOVE')

        # --- EXECUTE ENGINE ACTIONS ---
        layout.separator()
        row = layout.row(align=True)
        row.scale_y = 1.2
        row.operator(get_bl_idname("run_synthesizer"), icon='PLAY')
        row.operator(get_bl_idname("reset_synth_actions"), icon='FILE_REFRESH')

classes = (
    MMRT_UL_SynthObjectList,
    MMRT_UL_SynthMaterialList,
    MMRT_UL_SynthPoolMaterialList,
    MMRT_PT_Synthesizer,
)

def register():
    for cls in classes: bpy.utils.register_class(cls)

def unregister():
    for cls in reversed(classes): bpy.utils.unregister_class(cls)
