"""Isaac Sim GUI: open a .usd(c) once the app is up and set the timeline to its frames (user 2026-11-18: preview the
.blend's action before generating). isaac-sim.sh ignores a bare file name, so the GUI starts it as
    ~/isaacsim/isaac-sim.sh --exec open_in_isaac.py --/preview/file=<file.usdc>
"""
import asyncio
import sys

import carb
import omni.kit.app
import omni.timeline
import omni.usd

PATH = carb.settings.get_settings().get("/preview/file") or sys.argv[-1]


async def _open():
    app = omni.kit.app.get_app()
    for _ in range(60):                                          # let the window and the extensions come up
        await app.next_update_async()
    ok, err = await omni.usd.get_context().open_stage_async(PATH)
    if not ok:
        carb.log_error(f"[open_in_isaac] could not open {PATH}: {err}")
        print(f"[open_in_isaac] could not open {PATH}: {err}", flush=True)
        return
    for _ in range(10):
        await app.next_update_async()
    st = omni.usd.get_context().get_stage()
    tl = omni.timeline.get_timeline_interface()
    tl.set_time_codes_per_second(st.GetTimeCodesPerSecond())
    tl.set_start_time(st.GetStartTimeCode() / st.GetTimeCodesPerSecond())
    tl.set_end_time(st.GetEndTimeCode() / st.GetTimeCodesPerSecond())
    tl.set_current_time(0.0)
    msg = f"[open_in_isaac] opened {PATH}: press Play to watch the action"
    carb.log_warn(msg)
    print(msg, flush=True)                                       # (the dataset generator window shows this line)

asyncio.ensure_future(_open())
