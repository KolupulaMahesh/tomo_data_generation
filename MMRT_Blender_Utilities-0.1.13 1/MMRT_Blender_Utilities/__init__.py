bl_info = {
    "name": "MMRT Blender Utilities",
    "blender": (5, 1, 0),
    "category": "Robotics",
    "author": "tachiuhy",
    "version": (0, 1, 13),
    "description": (
        "A collection of utilities for behavior recording, motion planning, and simulation in Blender, "
        "designed for robotics applications."
        "\n\n"
        "This addon is intended to streamline the workflow for robotics developers using Blender for visualization, "
        "simulation, and motion planning tasks."
    ),
}


from .properties import global_settings_properties, recorder_properties, tuning_properties, importer_properties, mmrt_properties, simulator_properties, synthesizer_properties
from .operators import global_settings_operators, recorder_operators, tuning_operators, importer_operators, mmrt_operators, simulator_operators, message_operators, synthesizer_operators
from .ui import global_settings_panel, recorder_panel, tuning_panel, importer_panel, mmrt_panel, simulator_panel, synthesizer_panel
from .processes.process_manager import ProcessManager
from . import joint_limit_checks
from . import animator_interface
import bpy


def _configure_animator_interface():
    # Keep properties and core services available to retiming/export. Only
    # advanced panels and direct user operators are disabled in Animator Mode.
    for module in (global_settings_panel, recorder_panel, tuning_panel,
                   importer_panel, mmrt_panel, simulator_panel, synthesizer_panel):
        for cls in module.classes:
            if issubclass(cls, bpy.types.Panel):
                animator_interface.restrict_to_full_mode(cls)
    for module in (global_settings_operators, recorder_operators, tuning_operators,
                   importer_operators, mmrt_operators, simulator_operators,
                   synthesizer_operators):
        for cls in module.classes:
            if cls is not synthesizer_operators.MMRT_OT_ExportCurrentDatasetEpisode:
                animator_interface.restrict_to_full_mode(cls)

def register():
    _configure_animator_interface()
    ProcessManager.register_handlers()

    global_settings_properties.register()
    global_settings_panel.register()
    global_settings_operators.register()

    message_operators.register()

    #
    importer_properties.register()
    importer_operators.register()
    importer_panel.register()

    recorder_properties.register()
    recorder_operators.register()
    recorder_panel.register()

    #
    tuning_properties.register()
    tuning_operators.register()
    tuning_panel.register()

    mmrt_properties.register()
    mmrt_operators.register()
    mmrt_panel.register()

    # 
    simulator_properties.register()
    simulator_operators.register()
    simulator_panel.register()

    synthesizer_properties.register()
    synthesizer_operators.register()
    synthesizer_panel.register()
    joint_limit_checks.register()
    animator_interface.register()

def unregister():
    animator_interface.unregister()
    joint_limit_checks.unregister()
    ProcessManager.stop_all_processes()
    ProcessManager.unregister_handlers()

    synthesizer_panel.unregister()
    synthesizer_operators.unregister()
    synthesizer_properties.unregister()

    simulator_panel.unregister()
    simulator_operators.unregister()
    simulator_properties.unregister()

    mmrt_panel.unregister()
    mmrt_operators.unregister()
    mmrt_properties.unregister()

    tuning_panel.unregister()
    tuning_operators.unregister()
    tuning_properties.unregister()

    recorder_panel.unregister()
    recorder_operators.unregister()
    recorder_properties.unregister()

    importer_panel.unregister()
    importer_operators.unregister()
    importer_properties.unregister()

    message_operators.unregister()

    global_settings_operators.unregister()
    global_settings_panel.unregister()
    global_settings_properties.unregister()

if __name__ == "__main__":
    register()
