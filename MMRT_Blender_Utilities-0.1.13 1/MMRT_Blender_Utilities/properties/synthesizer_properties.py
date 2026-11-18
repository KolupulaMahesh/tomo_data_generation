import bpy
from ..common.config import SynthesisRenderOutput

class SynthesizerMaterialPoolItem(bpy.types.PropertyGroup):
    material: bpy.props.PointerProperty(name="Material", type=bpy.types.Material)

class SynthesizerObject(bpy.types.PropertyGroup):
    obj: bpy.props.PointerProperty(name="Object", type=bpy.types.Object)
    parent_offset_obj: bpy.props.PointerProperty(name="Offset Parent", type=bpy.types.Object)
    
    # Range of frames to check for animation keyframes
    frame_min: bpy.props.IntProperty(name="Frame Start", default=1)
    frame_max: bpy.props.IntProperty(name="Frame End", default=250)
    
    # Relative Transforms Ranges
    trans_min: bpy.props.FloatVectorProperty(name="Trans Min", size=3, default=(0,0,0))
    trans_max: bpy.props.FloatVectorProperty(name="Trans Max", size=3, default=(0,0,0))
    rot_min_deg: bpy.props.FloatVectorProperty(name="Rot Min (Deg)", size=3, default=(0,0,0))
    rot_max_deg: bpy.props.FloatVectorProperty(name="Rot Max (Deg)", size=3, default=(0,0,0))

class SynthesizerMaterial(bpy.types.PropertyGroup):
    obj: bpy.props.PointerProperty(name="Target Object", type=bpy.types.Object)
    target_material: bpy.props.PointerProperty(
        name="Target Material",
        description="Original material to replace on this object; reselect after upgrading",
        type=bpy.types.Material,
    )
    material_pool: bpy.props.CollectionProperty(type=SynthesizerMaterialPoolItem)
    pool_index: bpy.props.IntProperty()

class SynthesizerConfig(bpy.types.PropertyGroup):
    dry_run: bpy.props.BoolProperty(name="Dry Run", default=True)
    episode_prefix: bpy.props.StringProperty(name="Prefix", default="box_picking")
    episode_count: bpy.props.IntProperty(name="Count", default=10, min=1)
    episode_start_index: bpy.props.IntProperty(name="Start Index", default=1, min=1)
    random_seed: bpy.props.IntProperty(name="Seed (0=Random)", default=0)
    
    output_root: bpy.props.StringProperty(name="Output Root", subtype='DIR_PATH')
    render_output: bpy.props.EnumProperty(
        name="Format",
        items=[(SynthesisRenderOutput.MP4, "MP4", ""), (SynthesisRenderOutput.BLEND, "Blend Native", "")]
    )
    render_animation: bpy.props.BoolProperty(name="Render Animation", default=True)
    save_blend: bpy.props.BoolProperty(name="Save Episode Blend", default=False)
    
    record_mmrt: bpy.props.BoolProperty(name="Record MMRT", default=True)
    record_mmrt_before_render: bpy.props.BoolProperty(name="Record Before Render", default=True)
    record_mmrt_fast: bpy.props.BoolProperty(name="Fast Mode", default=True)

    objects: bpy.props.CollectionProperty(type=SynthesizerObject)
    objects_index: bpy.props.IntProperty()

    materials: bpy.props.CollectionProperty(type=SynthesizerMaterial)
    materials_index: bpy.props.IntProperty()

classes = (
    SynthesizerMaterialPoolItem,
    SynthesizerObject,
    SynthesizerMaterial,
    SynthesizerConfig,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.mmrt_synthesizer = bpy.props.PointerProperty(type=SynthesizerConfig)

def unregister():
    del bpy.types.Scene.mmrt_synthesizer
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
