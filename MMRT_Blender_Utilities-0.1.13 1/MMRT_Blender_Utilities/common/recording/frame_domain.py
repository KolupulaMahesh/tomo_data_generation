"""Frame-domain helpers shared by the recorder and its UI."""


class EpisodeRangeStatus:
    OUTSIDE = "OUTSIDE"
    PARTIAL = "PARTIAL"
    INSIDE = "INSIDE"


def get_frame_domain(scene):
    """Return the original and time-remapped frame domains for ``scene``.

    Blender evaluates animation time as ``current * Old / New``.  Therefore a
    source-domain frame maps to ``source * New / Old`` in the recorder/output
    domain.  Samples, markers, and episodes all use that output coordinate.
    """
    source_start = int(scene.frame_start)
    source_end = int(scene.frame_end)
    map_old = max(1, int(scene.render.frame_map_old))
    map_new = max(1, int(scene.render.frame_map_new))
    stretch_factor = map_new / map_old

    return {
        "source_start": source_start,
        "source_end": source_end,
        "stretched_start": round(source_start * stretch_factor),
        "stretched_end": round(source_end * stretch_factor),
        "map_old": map_old,
        "map_new": map_new,
        "stretch_factor": stretch_factor,
    }


def remap_frame(frame, *, stretch_factor):
    """Map a source-domain frame into the stretched recorder domain."""
    return round(frame * stretch_factor)


def unmap_frame(frame, *, stretch_factor):
    """Map a stretched recorder frame back into source animation time."""
    return frame / stretch_factor


def episode_range_status(start_frame, end_frame, *, record_start, record_end):
    """Classify an episode against the original recorder range."""
    if end_frame < record_start or start_frame > record_end:
        return EpisodeRangeStatus.OUTSIDE
    if start_frame < record_start or end_frame > record_end:
        return EpisodeRangeStatus.PARTIAL
    return EpisodeRangeStatus.INSIDE
