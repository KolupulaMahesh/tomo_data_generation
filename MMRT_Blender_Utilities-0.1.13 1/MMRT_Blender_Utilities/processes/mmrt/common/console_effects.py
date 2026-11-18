# MMRT console effects and progress animations.
import time
import random
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.panel import Panel
from rich.rule import Rule
from rich.live import Live
from definitions import MMRTProcessDefaults

console = Console()

def matrix_stream(duration=3.0):
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789#$%&@"
    t_end = time.time() + duration
    while time.time() < t_end:
        line = "".join(random.choice(chars) for _ in range(80))
        console.print(f"[green]{line}[/green]", highlight=False, overflow="ignore")
        time.sleep(0.03)

def boot_sequence():
    console.clear()
    console.rule("[bold cyan]Cognitive Motion Pipeline v2.3[/bold cyan]")
    console.print(Panel("⚙️  Initializing Multi-Modal Robotics Translator (MMRT)...", border_style="yellow"))
    time.sleep(1.0)

    modules = [
        "Loading neural motion graphs...",
        "Calibrating scene reconstruction...",
        "Activating proprioceptive sensors...",
        "Compiling behavior trees...",
        "Synchronizing cognitive map...",
    ]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=36),
        transient=True,
    ) as progress:
        task = progress.add_task("Booting modules...", total=len(modules))
        for step in modules:
            progress.update(task, description=f"[cyan]{step}")
            time.sleep(random.uniform(0.4, 1.0))
            progress.advance(task)

    console.print("[bold green]✓ All modules online[/bold green]")
    console.print(Panel(">> LAUNCHING PIPELINE CORE...", border_style="cyan"))
    matrix_stream(0.5)
    console.print("\n[bold bright_green]SYSTEM READY[/bold bright_green]\n")

def fake_learning_phase():
    console.rule("[magenta]Learning Primitive Behaviors[/magenta]")
    for epoch in range(1, 11):
        acc = 80 + random.random() * 15
        loss = 0.1 + random.random() * 0.2
        console.print(
            f"[cyan]Epoch {epoch}/10[/cyan]  "
            f"Loss: [yellow]{loss:.4f}[/yellow]  "
            f"Accuracy: [green]{acc:.2f}%[/green]"
        )
        time.sleep(0.3)
    console.print("[bold green]Training phase complete.[/bold green]\n")


# ─────────────────────────────────────────────
#  Stage effects (single-line animated visuals)
# ─────────────────────────────────────────────

def effect_data_mapping(duration: float = 2.0):
    """Green matrix data scan effect."""
    console.rule("[cyan]Loading Motion Data and Mappings[/cyan]")
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789@#$%&"
    t_end = time.time() + duration
    with Live(refresh_per_second=30, transient=True) as live:
        while time.time() < t_end:
            line = "".join(random.choice(chars) for _ in range(100))
            live.update(f"[green]{line}[/green]")
            time.sleep(0.04)
    console.print("[bold green]✓ Data mapping complete[/bold green]\n")


def effect_signal_processing(duration: float = 2.0):
    """Blue waveform effect representing smoothing and filtering."""
    console.rule("[blue]Preprocessing and Smoothing Joint Data[/blue]")
    chars = ["▁","▂","▃","▄","▅","▆","▇","█"]
    t_end = time.time() + duration
    with Live(refresh_per_second=25, transient=True) as live:
        while time.time() < t_end:
            wave = "".join(random.choice(chars) for _ in range(100))
            live.update(f"[blue]{wave}[/blue]")
            time.sleep(0.05)
    console.print("[bold green]✓ Signal preprocessing complete[/bold green]\n")


def effect_segmentation(duration: float = 2.2):
    """Yellow segmentation effect."""
    console.rule("[yellow]Segmenting Motions by Atomic Groups[/yellow]")
    chars = "░▒▓█"
    t_end = time.time() + duration
    with Live(refresh_per_second=20, transient=True) as live:
        while time.time() < t_end:
            clusters = "".join(random.choice(chars) for _ in range(100))
            live.update(f"[yellow]{clusters}[/yellow]")
            time.sleep(0.05)
    console.print("[bold green]✓ Segmentation complete[/bold green]\n")


def effect_waypoint_learning(duration: float = 2.0):
    """Magenta neural dots effect for waypoint discovery."""
    console.rule("[magenta]Generating Waypoints and Mapping Move Groups[/magenta]")
    chars = ["·","•","∙","⋅","◦"]
    t_end = time.time() + duration
    with Live(refresh_per_second=20, transient=True) as live:
        while time.time() < t_end:
            dots = "".join(random.choice(chars) for _ in range(100))
            live.update(f"[magenta]{dots}[/magenta]")
            time.sleep(0.06)
    console.print("[bold green]✓ Waypoints identified[/bold green]\n")


def effect_simulation(duration: float = 2.5):
    """Cyan neural activation simulation effect."""
    console.rule("[cyan]Running PyBullet Simulation and Collision Check[/cyan]")
    chars = [" ", ".", "o", "O", "@", "#"]
    t_end = time.time() + duration
    with Live(refresh_per_second=25, transient=True) as live:
        while time.time() < t_end:
            row = "".join(random.choice(chars) for _ in range(100))
            live.update(f"[cyan]{row}[/cyan]")
            time.sleep(0.04)
    console.print("[bold green]✓ Simulation complete[/bold green]\n")


def effect_compilation(duration: float = 2.0):
    """White spinner-like export visualization."""
    console.rule("[white]Compiling and Exporting Behavior Tree[/white]")
    spinners = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
    t_end = time.time() + duration
    with Live(refresh_per_second=30, transient=True) as live:
        while time.time() < t_end:
            spin_line = "".join(random.choice(spinners) for _ in range(100))
            live.update(f"[white]{spin_line}[/white]")
            time.sleep(0.05)
    console.print("[bold green]✓ Behavior Tree export complete[/bold green]\n")

def effect_bt_crafting(duration: float = 3.0):
    console.rule("[white]Constructing Behavior Tree[/white]")

    stages = [
        "Parsing motion segments...",
        "Synthesizing trajectory nodes...",
        "Embedding control logic...",
        "Assembling condition checks...",
        "Finalizing XML structure...",
    ]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        transient=True,
    ) as progress:
        task = progress.add_task("Crafting...", total=len(stages))
        for step in stages:
            progress.update(task, description=f"[cyan]{step}")
            time.sleep(duration / len(stages))
            progress.advance(task)

    console.print("[bold green]✓ Behavior Tree successfully crafted[/bold green]\n")

def effect_bt_crafting_hybrid(
    duration_tree: float = 10.0,
    duration_compile: float = 3.0,
    configure_tree_name: str = MMRTProcessDefaults.CONFIGURE_TREE_NAME,
    move_segment_tree_name: str = MMRTProcessDefaults.MOVE_SEGMENT_TREE_NAME,
):
    """
    Cinematic hybrid animation:
    1️⃣ Stage 1: Behavior Tree grows node-by-node.
    2️⃣ Stage 2: Compilation progress bar animates until complete.
    """

    from rich.live import Live
    from rich.text import Text
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

    console.rule("[white]Constructing and Compiling Behavior Tree[/white]")

    # ─────────────────────────────────────────────
    # Stage 1: Behavior Tree Growth (AI building)
    # ─────────────────────────────────────────────
    nodes = [
        "🧩 RootNode [Sequence]",
        f" ├── ⚙️ {configure_tree_name} [Subtree]",
        " │     ├── LoadBlackboard",
        " │     ├── SetVelAcc",
        " │     ├── SetPlanner",
        " │     ├── SetCollisionPairs",
        " │     └── ValidateSystemParams",
        " ├── 🧠 CognitivePerception [Selector]",
        " │     ├── VisionCheck [Condition: camera_online]",
        " │     ├── DepthFusion [Action]",
        " │     ├── ObjectDetection [Action]",
        " │     └── SceneContextReconstruction [Parallel]",
        " │           ├── MapUpdate [Action]",
        " │           └── SemanticLabeling [Action]",
        f" ├── 🤖 {move_segment_tree_name} [Subtree]",
        " │     ├── GraspPreparation [Sequence]",
        " │     │     ├── MoveToPreGraspPose [Action]",
        " │     │     ├── OpenGripper [Action]",
        " │     │     └── AlignEndEffector [Condition: alignment_error < 2mm]",
        " │     ├── ExecuteTrajectory [Parallel]",
        " │     │     ├── ExecuteRobotTrajectory [Action]",
        " │     │     ├── CollisionMonitor [Condition]",
        " │     │     └── VelocityProfileAdjust [Decorator]",
        " │     ├── ManipulationPhase [Selector]",
        " │     │     ├── ApplyForceControl [Condition: contact_detected]",
        " │     │     ├── TorqueCompensation [Action]",
        " │     │     └── CartesianAdjustment [Action]",
        " │     ├── ReleaseSequence [Sequence]",
        " │     │     ├── MoveToReleasePose [Action]",
        " │     │     ├── OpenGripper [Action]",
        " │     │     └── VerifyRelease [Condition]",
        " │     └── PostMotionCheck [Parallel]",
        " │           ├── SensorHealthMonitor [Action]",
        " │           ├── MotorTemperatureCheck [Condition]",
        " │           └── UpdateBlackboard [Action]",
        " ├── ⚡ SafetyAndRecovery [Selector]",
        " │     ├── CollisionDetected [Condition]",
        " │     ├── StopAllMotion [Action]",
        " │     ├── ClearPathObstruction [Action]",
        " │     └── ReplanTrajectory [Subtree]",
        " │           ├── ComputeSafePath [Action]",
        " │           ├── ValidateNewTrajectory [Condition]",
        " │           └── ResumeExecution [Action]",
        " └── 📤 SystemFinalization [Sequence]",
        "       ├── SaveSessionData [Action]",
        "       ├── ExportBlackboardToFile [Action]",
        "       ├── GenerateBTXML [Action]",
        "       └── ShutdownRobotCore [Action]",
    ]

    t_start = time.time()
    t_end = t_start + duration_tree
    reveal_speed = duration_tree / len(nodes)
    revealed = 0

    with Live(console=console, refresh_per_second=15, transient=True) as live:
        while time.time() < t_end:
            elapsed = time.time() - t_start
            new_revealed = min(int(elapsed / reveal_speed), len(nodes))
            if new_revealed != revealed:
                revealed = new_revealed
            glow = "".join(random.choice([".", "·", "•", "∙"]) for _ in range(40))
            partial_tree = "\n".join(nodes[:revealed])
            live.update(Text.from_markup(f"[bright_white]{partial_tree}[/bright_white]\n[cyan]{glow}[/cyan]"))
            time.sleep(0.1)

        # ensure full tree shown at end
        live.update(Text.from_markup("\n".join(nodes)))

    console.print("[bold cyan]AI synthesis complete.[/bold cyan]\n")
    time.sleep(0.5)

    # ─────────────────────────────────────────────
    # Stage 2: Compilation / Deployment Phase
    # ─────────────────────────────────────────────
    stages = [
        "Parsing motion segments...",
        "Synthesizing trajectory nodes...",
        "Embedding control logic...",
        "Assembling condition checks...",
        "Optimizing tree connections...",
        "Finalizing XML structure...",
        "Exporting Behavior Tree file...",
    ]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        transient=True,
    ) as progress:
        task = progress.add_task("Compiling...", total=len(stages))
        for step in stages:
            progress.update(task, description=f"[cyan]{step}")
            time.sleep(duration_compile / len(stages))
            progress.advance(task)

    console.print("[bold green]✓ Behavior Tree successfully crafted and compiled[/bold green]\n")
