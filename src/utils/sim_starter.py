import socket
import gym
import uuid
import subprocess
import os
import time
import signal
import atexit
import sys
import re
import traceback
from typing import Optional, List
import threading

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

# Track which ports have active simulators (port -> (proc, actual_port))
# actual_port may differ from requested port if simulator auto-increments
_port_to_simulator: dict[int, tuple[subprocess.Popen, int]] = {}

# Cache gym environments by (env_name, port) to avoid creating duplicate connections
# This prevents multiple cars from connecting to the same simulator
_env_cache: dict[tuple[str, int], gym.Env] = {}


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
    _port_to_simulator.clear()
    _env_cache.clear()


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
    
    This function attempts to bind to the port. If binding fails with
    "Address already in use", the port is in use. This correctly detects
    ports that are bound (even if not yet listening), preventing multiple
    simulators from trying to use the same port.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            # Try to bind to the port. If it's in use, this will raise OSError
            s.bind((host, port))
            # Successfully bound - port is free
            return False
        except OSError:
            # Failed to bind - port is in use
            return True


def launch_simulator(
    scene: str,
    sim_path: str = DEFAULT_SIM_PATH,
    port: int = 9091,
    host: str = "0.0.0.0",
    logfile: str = "-",
    debug: bool = False,
    log_filename_prefix: str = "sim_output",
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
        log_filename_prefix: Prefix for the log filename (default: sim_output)
    
    Returns:
        subprocess.Popen object if successful, None if simulator path doesn't exist
    
    Raises:
        ValueError: If scene name is not valid
    """
    # Save requested port before it might be modified
    requested_port = port
    
    # Validate scene name
    if scene not in VALID_SCENES:
        valid_scenes = ", ".join(sorted(VALID_SCENES.keys()))
        raise ValueError(
            f"Invalid scene name '{scene}'. Valid scenes are: {valid_scenes}"
        )
    
    # Check if simulator executable exists
    if not os.path.exists(sim_path):
        raise FileNotFoundError(f"Simulator path '{sim_path}' cannot be found.")
    
    # Check if we already have a simulator running on this port
    if port in _port_to_simulator:
        proc, actual_port = _port_to_simulator[port]
        if proc.poll() is None:  # Process is still running
            if debug:
                print(f"[DEBUG] Reusing existing simulator on port {port} (PID: {proc.pid}, actual port: {actual_port})")
            return proc, actual_port
        else:
            # Process died, remove from tracking
            if debug:
                print(f"[DEBUG] Previous simulator on port {port} has died, launching new one")
            del _port_to_simulator[port]
    
    # Check if port is already in use (might indicate another simulator is running)
    if is_port_in_use(port):
        if debug:
            print(f"[DEBUG] WARNING: Port {port} is already in use! This might indicate another simulator is running.")
            print(f"[DEBUG] This launch_simulator call will still proceed, but the simulator may choose a different port.")

    # Build command
    abs_sim_path = os.path.abspath(sim_path)
    cmd = [
        sim_path,
        "--scene", scene,
        "--port", str(port),
        "--host", host,
        "-logfile", logfile,
    ]
    print(f"Launching simulator with command:\n  {' '.join(cmd)}")
    
    # Launch simulator with output capture
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
        errors='replace'
    )
    pid = proc.pid
    _launched_processes.append(proc)
    print(f"Simulator process started (PID: {pid})")
    
    # Read output to find the actual port the simulator is using
    # Look for "Simulation Server Listening on: HOSTNAME:PORT"
    max_wait_time = 10.0  # Maximum time to wait for port message
    start_time = time.time()
    port_pattern = re.compile(r"Simulation Server Listening on: [^:]+:(\d+)")
    
    # Create log file in the project's logs folder
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    log_dir = os.path.join(project_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"{log_filename_prefix}_{port}_{int(time.time())}.log")
    print(f"Logging simulator output to: {log_filename}")
    
    log_file = open(log_filename, "w", encoding="utf-8")

    while time.time() - start_time < max_wait_time:
        if proc.poll() is not None:
            # Process has terminated, read remaining output
            output = proc.stdout.read()
            if output:
                log_file.write(output)
                match = port_pattern.search(output)
                if match:
                    port = int(match.group(1))
                    break
            log_file.close()
            raise RuntimeError(f"Simulator process terminated before starting (exit code: {proc.returncode})")
        
        # Try to read a line
        line = proc.stdout.readline()
        if line:
            log_file.write(line)
            log_file.flush()
            match = port_pattern.search(line)
            if match:
                port = int(match.group(1))
                print(f"Simulator listening on port {port} (PID: {pid})")
                break
        else:
            # No data available yet, sleep briefly to avoid busy-waiting
            time.sleep(0.1)
            
    # Start a background thread to continue logging output
    def log_worker():
        try:
            for line in proc.stdout:
                log_file.write(line)
                log_file.flush()
        except Exception as e:
            print(f"Error in simulator logger thread: {e}")
        finally:
            log_file.close()

    t = threading.Thread(target=log_worker, daemon=True)
    t.start()
    
    if time.time() - start_time >= max_wait_time:
        # Fallback: use the requested port if we couldn't parse it
        print(f"Warning: Could not parse port from simulator output, using requested port {port} (PID: {pid})")
    
    print(f"DonkeySim launched with scene '{scene}' on port {port} (PID: {pid})")
    
    # Track this simulator by both the requested port and actual port
    # This allows reuse when the same port is requested again
    _port_to_simulator[requested_port] = (proc, port)
    if port != requested_port:
        # Also track by actual port if it differs
        _port_to_simulator[port] = (proc, port)
    
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
    if debug:
        print(f"\n[DEBUG] start_sim called for env_name={env_name}, port={port}")
        print("[DEBUG] Call stack:")
        for line in traceback.format_stack()[-5:-1]:  # Show last 4 frames (excluding this one)
            print(f"  {line.strip()}")
        print()
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
    
    # Get log filename prefix from conf if provided
    log_filename_prefix = conf.get("log_filename_prefix", "sim_output") if conf else "sim_output"
    
    # Launch simulator instance (this may reuse an existing simulator)
    proc, port = launch_simulator(scene=scene, sim_path=sim_path, port=port, debug=debug, log_filename_prefix=log_filename_prefix)
    
    # Check if we already have an environment for this (env_name, port) combination
    # Do this after launch_simulator so we use the actual port
    cache_key = (env_name, port)
    if cache_key in _env_cache:
        env = _env_cache[cache_key]
        # Verify the environment is still valid (hasn't been closed)
        if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'viewer'):
            if debug:
                print(f"[DEBUG] Reusing cached environment for {env_name} on port {port}")
            return env
        else:
            # Environment was closed, remove from cache
            if debug:
                print(f"[DEBUG] Cached environment was closed, creating new one")
            del _env_cache[cache_key]
    
    # Wait for simulator to start (if it was launched)
    if proc is not None:
        start_delay = conf.get("start_delay", 5.0) if conf is not None else 5.0
        time.sleep(start_delay)
    
    # Get policy name from conf if provided, otherwise default to "agent"
    policy_name = conf.get("policy_name", "agent") if conf else "agent"
    
    # Prepare configuration for gym environment
    default_conf = {
        "host": "127.0.0.1",
        "port": port,
        "body_style": "donkey",
        "body_rgb": (128, 128, 128),
        "car_name": policy_name,
        "font_size": 100,
        "racer_name": f"{policy_name.upper()}_Puffer",
        "country": "USA",
        "bio": f"Learning to drive w PufferLib {policy_name.upper()}",
        "guid": str(uuid.uuid4()),
        "max_cte": 10,
    }
    
    # Merge user-provided conf with defaults
    if conf is not None:
        # Remove exe_path, policy_name, and log_filename_prefix from the user's conf before merging
        # exe_path prevents DonkeyEnv from launching another simulator
        # policy_name is only used for naming, not a valid gym config parameter
        # log_filename_prefix is for sim_starter only
        conf_filtered = {k: v for k, v in conf.items() if k not in ["exe_path", "policy_name", "log_filename_prefix"]}
        merged_conf = {**default_conf, **conf_filtered}
        # Ensure port is set correctly
        merged_conf["port"] = port
    else:
        merged_conf = default_conf
    
    # Double-check that exe_path and policy_name are not in the config
    # This is critical to prevent DonkeyEnv from launching another simulator
    if "exe_path" in merged_conf:
        if debug:
            print(f"[DEBUG] WARNING: exe_path found in merged_conf, removing it. Keys: {list(merged_conf.keys())}")
        del merged_conf["exe_path"]
    if "policy_name" in merged_conf:
        del merged_conf["policy_name"]
    
    if debug:
        print(f"[DEBUG] Creating gym environment with conf keys: {list(merged_conf.keys())}")
        print(f"[DEBUG] Conf port: {merged_conf.get('port')}, conf host: {merged_conf.get('host')}")
        print(f"[DEBUG] Calling gym.make({env_name}, conf=merged_conf)")
    
    # Create gym environment
    env = gym.make(env_name, conf=merged_conf)
    
    if debug:
        print(f"[DEBUG] gym.make completed, calling env.reset()")
    env.reset()
    
    if debug:
        print(f"[DEBUG] env.reset() completed")
        print(f"[DEBUG] Cached environment for {env_name} on port {port}")
    
    # Cache the environment for reuse
    _env_cache[cache_key] = env
    
    return env
