# MMRT motion segmentation and waypoint visualization.
import matplotlib
import matplotlib.patches as patches
import matplotlib.cm as cm
from common.recording_data import RecordingData

class Visualization:
    """
    Plot per-group speed with segments and binned segments, with per-group waypoints.
    """

    def __init__(
        self,
        speed_dict,
        segment_dict,
        binned_segments=None,
        atomic_waypoints=None,
        mmrt_data: RecordingData|None=None,
        episodes=None,
        collision_frames=None,
        collision_frames_by_atomic_group=None,
    ):
        self.speed_dict = speed_dict
        self.segment_dict = segment_dict
        self.binned_segments = binned_segments or []
        self.atomic_waypoints = atomic_waypoints or {}
        self.mmrt_data = mmrt_data
        self.episodes = episodes if episodes is not None else getattr(mmrt_data, "episodes", [])
        self.collision_frames = sorted(set(collision_frames or []))
        self.collision_frames_by_atomic_group = collision_frames_by_atomic_group

    def _relative_frame(self, frame):
        return frame - getattr(self.mmrt_data, "frame_start", 0)

    def _plot_episodes(self, ax, label_state):
        """Shade episode ranges on a plot using the same frame domain as events."""
        for episode in self.episodes:
            start = self._relative_frame(episode["start_frame"])
            end = self._relative_frame(episode["end_frame"]) + 1
            ax.axvspan(start, end, color="green", alpha=0.10, zorder=0)
            if not label_state[0]:
                ax.text(
                    (start + end) / 2, 0.98, episode["name"],
                    transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=8, color="darkgreen",
                )
        if self.episodes:
            label_state[0] = True

    def _collision_frames_for_group(self, group_name):
        """Return collision frames that belong to one plotted group."""
        if self.collision_frames_by_atomic_group is not None:
            # Contact ownership also applies when the affected group is stationary.
            return sorted(set(self.collision_frames_by_atomic_group.get(group_name, [])))

        # Compatibility for callers providing only the legacy frame list.
        group_segments = self.segment_dict.get(group_name, [])
        if not group_segments:
            return []

        return [
            frame for frame in self.collision_frames
            if any(
                start <= self._relative_frame(frame) <= end
                for start, end in group_segments
            )
        ]

    def _plot_collision_frames(self, ax, collision_frames):
        """Draw one legend entry for collision frames on the owning axis."""
        for index, frame in enumerate(collision_frames):
            ax.axvline(
                self._relative_frame(frame),
                color="red",
                linewidth=1.5,
                alpha=0.9,
                zorder=6,
                label="Collision" if index == 0 else None,
            )

    def plot(self, title="Motion Segmentation Visualization", save_path=None, show=True):
        # Saved pipeline plots must not depend on an installed Tk runtime.  An
        # interactive request retains the original Tk-based behavior.
        matplotlib.use("TkAgg" if show else "Agg")
        import matplotlib.pyplot as plt

        n = len(self.speed_dict)
        # Determine adaptive width based on number of frames
        try:
            num_frames = max(len(speed) for speed in self.speed_dict.values())
        except ValueError:
            num_frames = 1000  # safe fallback

        # Width scales with input size
        adaptive_width = min(max(12, (num_frames / 1000) * 5), 60) * 2

        fig, axs = plt.subplots(
            n + 1, 1,
            figsize=(adaptive_width, 2 * (n + 1)),
            sharex=True
        )


        if n == 0:
            raise ValueError("No speed data to plot.")
        elif n == 1:
            axs = [axs[0], axs[1]]

        # Plot each atomic group's speed
        episode_label_state = [False]
        for ax, (group_name, speed) in zip(axs[:-1], self.speed_dict.items()):
            ax.plot(range(len(speed)), speed, label=f"{group_name} speed")
            self._plot_episodes(ax, episode_label_state)
            self._plot_collision_frames(
                ax,
                self._collision_frames_for_group(group_name),
            )
            segments = self.segment_dict.get(group_name, [])
            for s, e in segments:
                ax.axvspan(s, e + 1, color='orange', alpha=0.3)
                ax.text((s + e) / 2, max(speed) * 0.9, f"{s}-{e}", ha='center', fontsize=7)

            # --- plot atomic group waypoint dashed lines ---
            group_wps = self.atomic_waypoints.get(group_name, [])
            for wp in group_wps:
                ax.axvline(wp, linestyle='--', color='blue', linewidth=1)

            ax.set_ylabel("Speed")
            ax.set_title(group_name)
            ax.legend()
            ax.grid(True)

        # Plot binned segments + waypoints
        bin_ax = axs[-1]
        bin_ax.set_title("Binned Move Groups + Waypoints")
        bin_ax.set_ylim(0, 1)
        bin_ax.set_yticks([])
        bin_ax.set_xlabel("Frame")
        bin_ax.grid(True)
        self._plot_episodes(bin_ax, episode_label_state)

        color_map = cm.get_cmap('tab20')
        group_names = sorted({mg for _, _, mg, _ in self.binned_segments})
        color_lookup = {name: color_map(i % 20) for i, name in enumerate(group_names)}

        for s, e, mg, wps in self.binned_segments:
            rect = patches.Rectangle((s, 0.1), e - s, 0.8, color=color_lookup[mg], alpha=0.6)
            bin_ax.add_patch(rect)

            # --- Plot waypoint markers as dashed vertical lines ---
            if wps:
                for wp in wps:
                    bin_ax.axvline(wp, ymin=0.1, ymax=0.9, linestyle='--', color='black', linewidth=1)

        # === Plot Flags (general + IO) ===
        if self.mmrt_data:
            # General flags (triangle markers)
            for frame, codes in self.mmrt_data.flags_general.items():
                for code in codes:
                    relative_frame = self._relative_frame(frame)
                    bin_ax.scatter(relative_frame, 0.05, marker="^", s=40, color="red", zorder=5)
                    bin_ax.text(relative_frame, 0.02, code, ha='center', va='top', fontsize=6, color='red')

            # IO events (square markers)
            for frame, [byte_val, delay_ms] in self.mmrt_data.flags_io.items():
                relative_frame = self._relative_frame(frame)
                bin_ax.scatter(relative_frame, 0.05, marker="s", s=40, color="purple", zorder=5)
                bin_ax.text(relative_frame, 0.02, f"{byte_val:02X}", ha='center', va='top', fontsize=6, color='purple')

        legend_handles = [patches.Patch(color=color_lookup[name], label=name) for name in group_names]
        bin_ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4)

        plt.suptitle(title)
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])

        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()

        return fig
