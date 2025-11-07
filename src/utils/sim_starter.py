import socket
import gym
import uuid
import subprocess
import os
import time
import signal
import atexit
import sys
from typing import Optional, List

# Valid scene names matching the environment classes in __init__.py
VALID_SCENES = {
    "generated_road": "generated_road",
    "warehouse": "warehouse",
    "sparkfun_avc": "sparkfun_avc",
    "generated_track": "generated_track",
    "roboracingleague_1": "roboracingleague_1",
    "waveshare": "waveshare",
    "mini_monaco": "mini_monaco",
    "warren": "warren",
    "circuit_launch": "circuit_launch",
    "mountain_track": "mountain_track",
    "thunderhill": "thunderhill",
}

# Mapping from gym environment names to scene names
ENV_NAME_TO_SCENE = {
    "donkey-generated-roads-v0": "generated_road",
    "donkey-warehouse-v0": "warehouse",
    "donkey-avc-sparkfun-v0": "sparkfun_avc",
    "donkey-generated-track-v0": "generated_track",
    "donkey-mountain-track-v0": "mountain_track",
    "donkey-roboracingleague-track-v0": "roboracingleague_1",
    "donkey-waveshare-v0": "waveshare",
    "donkey-minimonaco-track-v0": "mini_monaco",
    "donkey-warren-track-v0": "warren",
    "donkey-thunderhill-track-v0": "thunderhill",
    "donkey-circuit-launch-track-v0": "circuit_launch",
}

# Default simulator executable path
DEFAULT_SIM_PATH = "./simulator/build/linux/sim.x86_64"

# Track all launched simulator processes
_launched_processes: List[subprocess.Popen] = []


def _cleanup_processes():
    """Kill all tracked simulator processes."""
    for proc in _launched_processes:
        if proc.poll() is None:  # Process is still running
            try:
                proc.terminate()
                # Wait a bit for graceful shutdown
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    # Force kill if it doesn't terminate gracefully
                    proc.kill()
                    proc.wait()
            except Exception as e:
                print(f"Error killing process {proc.pid}: {e}")
    _launched_processes.clear()


def _signal_handler(signum, frame):
    """Handle termination signals."""
    _cleanup_processes()
    sys.exit(0)


# Register signal handlers for cleanup
signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)

# Register atexit handler as fallback
atexit.register(_cleanup_processes)

def is_port_in_use(port: int, host: str = "0.0.0.0") -> bool:
    """
    Check if the given port is in use.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((host, port))
            return True
        except ConnectionRefusedError:
            return False


def launch_simulator(
    scene: str,
    sim_path: str = DEFAULT_SIM_PATH,
    port: int = 9091,
    host: str = "0.0.0.0",
    debug: bool = False,
    logfile: str = "-",
) -> Optional[subprocess.Popen]:
    """
    Launch the DonkeySim simulator with a given scene.
    
    Args:
        scene: Scene name to load. Must be one of:
            - generated_road
            - warehouse
            - sparkfun_avc
            - generated_track
            - roboracingleague_1
            - waveshare
            - mini_monaco
            - warren
            - circuit_launch
            - mountain_track
        sim_path: Path to the simulator executable (default: ./SimLinux/im.x86_64)
        port: Port number for the simulator (default: 9091)
        host: Host address to bind to (default: 0.0.0.0)
        debug: Enable debug mode with logging (default: False)
        logfile: Log file path, use "-" for stdout (default: -)
    
    Returns:
        subprocess.Popen object if successful, None if simulator path doesn't exist
    
    Raises:
        ValueError: If scene name is not valid
    """
    # Validate scene name
    if scene not in VALID_SCENES:
        valid_scenes = ", ".join(sorted(VALID_SCENES.keys()))
        raise ValueError(
            f"Invalid scene name '{scene}'. Valid scenes are: {valid_scenes}"
        )
    
    # Check if simulator executable exists
    if not os.path.exists(sim_path):
        raise FileNotFoundError(f"Simulator path '{sim_path}' cannot be found.")
    
    # check if the port is already in use
    while is_port_in_use(port):
        port += 1
    print(f"Using port {port}")

    # Build command
    cmd = [
        sim_path,
        "--scene", scene,
        "--port", str(port),
        "--host", host,
    ]
    
    # Only add logfile parameter in debug mode
    if debug:
        cmd.extend(["-logfile", logfile])
    
    # Launch simulator
    proc = subprocess.Popen(cmd)
    _launched_processes.append(proc)
    print(f"DonkeySim launched with scene '{scene}' on port {port} (PID: {proc.pid})")
    return proc, port


def start_sim(env_name: str = "donkey-circuit-launch-track-v0", port: int = 9091, conf: Optional[dict] = None, debug: bool = False):
    """
    Start the simulator on the given port and create a gym environment.
    Each call launches a new simulator instance.
    
    Args:
        env_name: Gym environment name (default: donkey-circuit-launch-track-v0)
        port: Port number for the simulator (default: 9091)
        conf: Optional configuration dictionary to override defaults
        debug: Enable debug mode with logging (default: False)
    
    Returns:
        Gym environment instance
    
    Raises:
        ValueError: If env_name is not recognized or scene name is invalid
    """
    # Map environment name to scene name
    if env_name not in ENV_NAME_TO_SCENE:
        valid_envs = ", ".join(sorted(ENV_NAME_TO_SCENE.keys()))
        raise ValueError(
            f"Unknown environment name '{env_name}'. Valid environments are: {valid_envs}"
        )
    
    scene = ENV_NAME_TO_SCENE[env_name]
    
    # Get simulator path from conf if provided, otherwise use default
    sim_path = DEFAULT_SIM_PATH
    if conf is not None and "exe_path" in conf:
        sim_path = conf["exe_path"]
    
    # Launch simulator instance
    proc, port = launch_simulator(scene=scene, sim_path=sim_path, port=port, debug=debug)
    
    # Wait for simulator to start (if it was launched)
    if proc is not None:
        start_delay = conf.get("start_delay", 5.0) if conf is not None else 5.0
        time.sleep(start_delay)
    
    # Prepare configuration for gym environment
    default_conf = {
        "host": "127.0.0.1",
        "port": port,
        "body_style": "donkey",
        "body_rgb": (128, 128, 128),
        "car_name": f"agent_{port}",
        "font_size": 100,
        "racer_name": "PPO_Puffer",
        "country": "USA",
        "bio": "Learning to drive w PufferLib PPO",
        "guid": str(uuid.uuid4()),
        "max_cte": 10,
    }
    
    # Merge user-provided conf with defaults
    if conf is not None:
        # First remove exe_path from the user's conf before merging
        # This prevents it from being included in merged_conf
        conf_without_exe = {k: v for k, v in conf.items() if k != "exe_path"}
        merged_conf = {**default_conf, **conf_without_exe}
        # Ensure port is set correctly
        merged_conf["port"] = port
    else:
        merged_conf = default_conf
    
    # Double-check that exe_path is not in the config
    # This is critical to prevent DonkeyEnv from launching another simulator
    if "exe_path" in merged_conf:
        del merged_conf["exe_path"]
    
    # Create gym environment
    env = gym.make(env_name, conf=merged_conf)
    env.reset()
    return env
