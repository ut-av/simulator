using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Std;

/// <summary>
/// ROSSimApi - Implements ROS topics for simulator control similar to SandboxServer.
/// This class should be attached to a ROS prefab and provides topics for:
/// - Spawning/despawning cars
/// - Controlling car physics
/// - Resetting simulation state
/// - Configuring sensors
/// </summary>
public class ROSSimApi : MonoBehaviour
{
    private ROSConnection ros;
    private CarSpawner carSpawner;

    // ROS Topic names
    private const string SPAWN_CAR_TOPIC = "/sim/spawn_car";
    private const string DESPAWN_CAR_TOPIC = "/sim/despawn_car";
    private const string RESET_CAR_TOPIC = "/sim/reset_car";
    private const string CAR_CONTROL_TOPIC = "/sim/car_control";
    private const string SET_CAR_POSITION_TOPIC = "/sim/set_car_position";
    private const string SIM_STATE_TOPIC = "/sim/state";
    private const string SIM_RESET_TOPIC = "/sim/reset";

    // Message structure classes for ROS communication
    [System.Serializable]
    public class SpawnCarMsg
    {
        public int car_id;
        public float x;
        public float y;
        public float z;
        public float roll;
        public float pitch;
        public float yaw;
    }

    [System.Serializable]
    public class DespawnCarMsg
    {
        public int car_id;
    }

    [System.Serializable]
    public class ResetCarMsg
    {
        public int car_id;
    }

    [System.Serializable]
    public class CarControlMsg
    {
        public int car_id;
        public float steering;
        public float throttle;
        public float brake;
    }

    [System.Serializable]
    public class SetCarPositionMsg
    {
        public int car_id;
        public float x;
        public float y;
        public float z;
        public float roll;
        public float pitch;
        public float yaw;
    }

    [System.Serializable]
    public class SimStateMsg
    {
        public int active_car_count;
        public float simulation_time;
        public float time_scale;
    }

    [System.Serializable]
    public class SimResetMsg
    {
        public bool reset_all_cars;
        public bool reload_scene;
    }

    private Dictionary<int, GameObject> carInstances = new Dictionary<int, GameObject>();
    private int nextCarId = 0;
    private bool debugLogging = false;

    void Start()
    {
        // Initialize ROS connection
        if (!GlobalState.rosEnabled)
        {
            Debug.LogWarning("ROSSimApi: ROS is not enabled. Check that --ros flag was set.");
            return;
        }

        ros = GlobalState.rosConnection;
        if (ros == null)
        {
            Debug.LogError("ROSSimApi: Failed to get ROSConnection instance");
            return;
        }

        carSpawner = GameObject.FindFirstObjectByType<CarSpawner>();
        if (carSpawner == null)
        {
            Debug.LogWarning("ROSSimApi: CarSpawner not found in scene. Car spawning will not work.");
        }

        // Register ROS topic subscribers
        RegisterTopicSubscribers();

        Debug.Log("ROSSimApi initialized successfully");
    }

    private void RegisterTopicSubscribers()
    {
        if (ros == null)
        {
            Debug.LogError("ROSSimApi: Cannot register subscribers - ROSConnection is null");
            return;
        }

        // Subscribe to spawn car topic
        ros.Subscribe<RosMessageTypes.Std.Int32Msg>(SPAWN_CAR_TOPIC, (msg) => OnSpawnCarMessage(msg));

        // Subscribe to despawn car topic
        ros.Subscribe<RosMessageTypes.Std.Int32Msg>(DESPAWN_CAR_TOPIC, (msg) => OnDespawnCarMessage(msg));

        // Subscribe to reset car topic
        ros.Subscribe<RosMessageTypes.Std.Int32Msg>(RESET_CAR_TOPIC, (msg) => OnResetCarMessage(msg));

        // Subscribe to simulation reset topic
        ros.Subscribe<RosMessageTypes.Std.BoolMsg>(SIM_RESET_TOPIC, (msg) => OnSimResetMessage(msg));

        Debug.Log("ROSSimApi: Topic subscribers registered");
    }

    /// <summary>
    /// Handles spawn car requests from ROS
    /// </summary>
    private void OnSpawnCarMessage(RosMessageTypes.Std.Int32Msg msg)
    {
        Debug.Log($"ROSSimApi: Received spawn car request");

        if (carSpawner == null)
        {
            Debug.LogError("ROSSimApi: Cannot spawn car - CarSpawner not found");
            return;
        }

        // Get spawn position
        var (spawnPos, spawnRot) = carSpawner.GetCarStartPosRot();

        // Create a JsonTcpClient wrapper for ROS-controlled cars
        // For now, we'll spawn with null client to use autonomous mode
        GameObject carObj = carSpawner.Spawn(null, false);

        if (carObj != null)
        {
            carInstances[nextCarId] = carObj;
            Debug.Log($"ROSSimApi: Spawned car with ID {nextCarId} at position {spawnPos}");
            nextCarId++;
        }
        else
        {
            Debug.LogError("ROSSimApi: Failed to spawn car");
        }
    }

    /// <summary>
    /// Handles despawn car requests from ROS
    /// </summary>
    private void OnDespawnCarMessage(RosMessageTypes.Std.Int32Msg msg)
    {
        int carId = msg.data;
        Debug.Log($"ROSSimApi: Received despawn car request for car {carId}");

        if (!carInstances.ContainsKey(carId))
        {
            Debug.LogWarning($"ROSSimApi: Car {carId} not found");
            return;
        }

        GameObject carObj = carInstances[carId];
        if (carObj != null && carSpawner != null)
        {
            carSpawner.RemoveCar(carObj.GetComponent<tk.JsonTcpClient>());
            carInstances.Remove(carId);
            Debug.Log($"ROSSimApi: Despawned car {carId}");
        }
    }

    /// <summary>
    /// Handles reset car requests from ROS
    /// </summary>
    private void OnResetCarMessage(RosMessageTypes.Std.Int32Msg msg)
    {
        int carId = msg.data;
        Debug.Log($"ROSSimApi: Received reset car request for car {carId}");

        if (!carInstances.ContainsKey(carId))
        {
            Debug.LogWarning($"ROSSimApi: Car {carId} not found");
            return;
        }

        GameObject carObj = carInstances[carId];
        if (carObj != null)
        {
            // Reset car position and velocity
            var (resetPos, resetRot) = carSpawner.GetCarStartPosRot();
            carObj.transform.position = resetPos;
            carObj.transform.rotation = resetRot;

            // Reset car physics
            ICar carController = carObj.GetComponent<ICar>();
            if (carController != null)
            {
                carController.RequestThrottle(0.0f);
                carController.RequestSteering(0.0f);
                carController.RequestFootBrake(1.0f);
                carController.RequestHandBrake(1.0f);
            }

            Debug.Log($"ROSSimApi: Reset car {carId}");
        }
    }

    /// <summary>
    /// Handles simulation reset requests from ROS
    /// </summary>
    private void OnSimResetMessage(RosMessageTypes.Std.BoolMsg msg)
    {
        Debug.Log($"ROSSimApi: Received simulation reset request");

        // Reset all cars
        foreach (var carId in new List<int>(carInstances.Keys))
        {
            OnResetCarMessage(new RosMessageTypes.Std.Int32Msg { data = carId });
        }

        // Reset global simulation state
        Time.timeScale = 1.0f;
        GlobalState.timeScale = 1.0f;

        Debug.Log("ROSSimApi: Simulation reset complete");
    }

    /// <summary>
    /// Publishes current simulation state to ROS
    /// </summary>
    private void PublishSimState()
    {
        if (ros == null || !GlobalState.rosEnabled)
            return;

        // Create a simple string message with the state info
        string stateInfo = $"Cars: {carInstances.Count}, Time: {Time.time:F2}, TimeScale: {GlobalState.timeScale}";
        
        // Log the state instead of publishing for now
        // TODO: Define custom message type for simulation state if needed
        if (debugLogging)
        {
            Debug.Log($"ROSSimApi State: {stateInfo}");
        }
    }

    void Update()
    {
        // Periodically publish simulation state
        if (Time.frameCount % 60 == 0) // Publish every 60 frames (~1 second at 60 FPS)
        {
            PublishSimState();
        }
    }

    /// <summary>
    /// Sets a car's position (can be called from ROS or other systems)
    /// </summary>
    public void SetCarPosition(int carId, Vector3 position, Quaternion rotation)
    {
        if (!carInstances.ContainsKey(carId))
        {
            Debug.LogWarning($"ROSSimApi: Car {carId} not found");
            return;
        }

        GameObject carObj = carInstances[carId];
        if (carObj != null)
        {
            carObj.transform.position = position;
            carObj.transform.rotation = rotation;
            Debug.Log($"ROSSimApi: Set car {carId} position to {position}");
        }
    }

    /// <summary>
    /// Controls a car's input (can be called from ROS or other systems)
    /// </summary>
    public void ControlCar(int carId, float steering, float throttle, float brake)
    {
        if (!carInstances.ContainsKey(carId))
        {
            Debug.LogWarning($"ROSSimApi: Car {carId} not found");
            return;
        }

        GameObject carObj = carInstances[carId];
        if (carObj != null)
        {
            ICar carController = carObj.GetComponent<ICar>();
            if (carController != null)
            {
                carController.RequestSteering(steering);
                carController.RequestThrottle(throttle);
                carController.RequestFootBrake(brake);
            }
        }
    }

    /// <summary>
    /// Gets a car object by ID
    /// </summary>
    public GameObject GetCar(int carId)
    {
        if (carInstances.ContainsKey(carId))
            return carInstances[carId];
        return null;
    }

    /// <summary>
    /// Gets the total number of spawned cars
    /// </summary>
    public int GetCarCount()
    {
        return carInstances.Count;
    }

    /// <summary>
    /// Gets all spawned car IDs
    /// </summary>
    public List<int> GetAllCarIds()
    {
        return new List<int>(carInstances.Keys);
    }
}
