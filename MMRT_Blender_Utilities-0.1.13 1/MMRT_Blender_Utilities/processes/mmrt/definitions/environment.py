"""Environment JSON contract shared by the exporter and MMRT process."""

from .recording import MDataLabel


class EnvironmentSchema:
    UNITS_KEY = "units"
    SCHEMA_KEY = "schema"
    ENVIRONMENT_KEY = "environment"
    COLLISION_RULES_KEY = "collision_rules"

    ANGLE_UNIT_KEY = "angle"
    LENGTH_UNIT_KEY = "length"
    ROTATION_FORMAT_KEY = MDataLabel.ROTATION_FORMAT_KEY
    LEGACY_ROTATION_KEY = "rotation"

    NAME = "name"
    LOCATION = "loc"
    ROTATION = "rot"
    SCALE = "scale"
    ORIGIN = "origin"
    MODIFIER = "mod"
    MESH = "mesh"
    TYPE = "type"
    REFERENCE = "ref"
    FOLLOW_ROBOT = "follow_robot"
    FOLLOW_LINK = "follow_link"
    COLLISION_LINKS = "collision_links"
    COLLISION_MOVEGROUP = "collision_movegroup"
    ALLOW_COLLISION = "allow_collision"

    COLUMNS = (
        NAME, LOCATION, ROTATION, SCALE, ORIGIN, MODIFIER, MESH, TYPE,
        REFERENCE, FOLLOW_ROBOT, FOLLOW_LINK, COLLISION_LINKS,
        COLLISION_MOVEGROUP, ALLOW_COLLISION,
    )
