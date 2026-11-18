import bpy
import os
import json
import math
from pathlib import PurePosixPath

from ..blender.utils import show_report
from ..config import ObjectOriginMode
from ...processes.mmrt.definitions import (
    EnvironmentSchema,
    GeneratedAppDirectories,
    MeasurementUnits as MeasurementConstants,
)
from ..recording.paths import get_recorder_environment_filepath
from ..system.paths import normalize_output_dir
from .mesh import (
    bake_custom_transform,
    bake_world_transform,
    create_export_object,
    write_stl,
)


def export_environment(context):
    recorder = context.scene.mmrt_recorder
    if len(recorder.env_objs) == 0:
        show_report("No environment objects to export", level="WARNING")
        return
    
    output_dir = normalize_output_dir(recorder.config.output_dir)
    mesh_dir = os.path.join(output_dir, GeneratedAppDirectories.MESH_DIR)
    os.makedirs(mesh_dir, exist_ok=True)

    for item in recorder.env_objs:
        src = item.obj
        if not src or not item.include_for_export:
            continue

        temp = create_export_object(
            src,
            apply_modifiers=item.apply_modifiers
        )

        temp.select_set(True)
        bpy.context.view_layer.objects.active = temp
        if item.origin_mode == ObjectOriginMode.WORLD:
            bake_world_transform(temp)
        elif item.origin_mode == ObjectOriginMode.CUSTOM and item.custom_ref:
            if item.custom_ref == src:
                # avoid identity confusion
                pass
            bake_custom_transform(temp, item.custom_ref)

        filepath = os.path.join(mesh_dir, f"{src.name}.stl")
        write_stl(filepath, temp)
        bpy.data.objects.remove(temp, do_unlink=True)


def export_environment_metadata(context):
    recorder = context.scene.mmrt_recorder
    cfg = recorder.config
    env_path = get_recorder_environment_filepath(context)
    env_data = []

    # -----------------------------
    # Unit handling
    # -----------------------------
    angle_is_rad = cfg.angle_unit == MeasurementConstants.RADIAN_UNIT
    length_is_mm = cfg.length_unit == MeasurementConstants.MILLIMETER_UNIT

    angle_unit_str = MeasurementConstants.RADIAN_UNIT if angle_is_rad else MeasurementConstants.DEGREE_UNIT
    length_unit_str = MeasurementConstants.MILLIMETER_UNIT if length_is_mm else MeasurementConstants.METER_UNIT

    length_mul = 1000.0 if length_is_mm else 1.0    

    loc_dp = (
        MeasurementConstants.MILLIMETER_DECIMAL_PLACES
        if length_is_mm else
        MeasurementConstants.METER_DECIMAL_PLACES
    )

    rot_dp = (
        MeasurementConstants.RAD_DECIMAL_PLACES
        if angle_is_rad else
        MeasurementConstants.DEG_DECIMAL_PLACES
    )

    # -----------------------------
    # Collect env objects
    # -----------------------------
    for item in recorder.env_objs:
        obj = item.obj
        if not obj or not item.include_for_export:
            continue
        ref = item.custom_ref
        follow_robot_obj = next(
            (
                robot.obj for robot in recorder.robots
                if robot.obj and robot.obj.name == item.follow_robot
            ),
            None,
        )
        transform_ref = follow_robot_obj or (
            ref if item.origin_mode == ObjectOriginMode.CUSTOM else None
        )
        if item.origin_mode == ObjectOriginMode.WORLD:
            # baked → identity
            loc_vals = (0.0, 0.0, 0.0)
            rot_vals = (0.0, 0.0, 0.0)
            scale_vals = (1.0, 1.0, 1.0)

        elif transform_ref:
            if transform_ref == obj:
                # avoid identity confusion
                loc_vals = (0, 0, 0)
                rot_vals = (0, 0, 0)
                scale_vals = (1, 1, 1)
            else:
                ref_inv = transform_ref.matrix_world.inverted()
                mat = ref_inv @ obj.matrix_world
                loc = mat.to_translation()
                rot = mat.to_euler()
                scale = mat.to_scale()

                loc_vals = (loc.x, loc.y, loc.z)
                rot_vals = (rot.x, rot.y, rot.z)
                scale_vals = (scale.x, scale.y, scale.z)
        
        else:  # LOCAL
            mat = obj.matrix_world
            loc = mat.to_translation()
            rot = mat.to_euler()
            scale = mat.to_scale()

            loc_vals = (loc.x, loc.y, loc.z)
            rot_vals = (rot.x, rot.y, rot.z)
            scale_vals = (scale.x, scale.y, scale.z)

        # --- Convert units ---
        loc_converted = [round(v * length_mul, loc_dp) for v in loc_vals]

        if angle_is_rad:
            rot_converted = [round(v, rot_dp) for v in rot_vals]
        else:
            rot_converted = [round(math.degrees(v), rot_dp) for v in rot_vals]

        scale_converted = [round(v, 6) for v in scale_vals]

        env_data.append([
            obj.name,
            loc_converted,
            rot_converted,
            scale_converted,
            item.origin_mode,
            item.apply_modifiers,
            PurePosixPath(GeneratedAppDirectories.MESH_DIR, f"{obj.name}.stl").as_posix(),
            obj.type,
            ref.name if item.origin_mode == ObjectOriginMode.CUSTOM and ref else "",
            item.follow_robot,
            item.follow_link.strip(),
            ",".join(rule.name for rule in item.collision_links),
            ",".join(rule.name for rule in item.collision_movegroups),
            item.allow_collision,
        ])

    # -----------------------------
    # Write JSON
    # -----------------------------
    data = {
        EnvironmentSchema.UNITS_KEY: {
            EnvironmentSchema.ANGLE_UNIT_KEY: angle_unit_str,
            EnvironmentSchema.LENGTH_UNIT_KEY: length_unit_str
        },
        EnvironmentSchema.SCHEMA_KEY: EnvironmentSchema.COLUMNS,
        EnvironmentSchema.ENVIRONMENT_KEY: env_data
    }
    
    os.makedirs(os.path.dirname(env_path), exist_ok=True)
    with open(env_path, "w") as f:
        f.write("{\n")

        # units
        f.write(f'  "{EnvironmentSchema.UNITS_KEY}": ')
        f.write(json.dumps(data[EnvironmentSchema.UNITS_KEY], separators=(',', ':')))
        f.write(",\n")

        # schema
        f.write(f'  "{EnvironmentSchema.SCHEMA_KEY}": ')
        f.write(json.dumps(data[EnvironmentSchema.SCHEMA_KEY], separators=(',', ':')))
        f.write(",\n")

        # environment (multi-line but compact rows)
        f.write(f'  "{EnvironmentSchema.ENVIRONMENT_KEY}": [\n')

        for i, row in enumerate(data[EnvironmentSchema.ENVIRONMENT_KEY]):
            f.write("    ")
            f.write(json.dumps(row, separators=(',', ':')))
            if i < len(data[EnvironmentSchema.ENVIRONMENT_KEY]) - 1:
                f.write(",")
            f.write("\n")

        f.write("  ]\n")
        f.write("}")

    return env_path
