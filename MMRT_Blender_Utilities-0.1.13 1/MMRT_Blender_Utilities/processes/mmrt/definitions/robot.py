"""Robot description identifiers shared by Blender and the MMRT process."""


class RobotJointTypes:
    FIXED = "fixed"
    MIMIC = "mimic"
    PRISMATIC = "prismatic"
    REVOLUTE = "revolute"

    NON_MOVING = frozenset((FIXED, MIMIC))


class RobotDescriptionKind:
    URDF = "urdf"
    SRDF = "srdf"
