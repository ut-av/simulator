using System;
using System.Reflection;
using UnityEngine;

/// <summary>
/// ROS-based car control driver that subscribes to /ackermann_curvature_drive topic.
/// Similar to the VESC driver used on real cars, this receives velocity and curvature
/// commands and applies them to the simulated car.
/// Uses reflection to avoid a direct compile-time dependency on the ROS-TCP-Connector package.
/// </summary>
public class UtavDriver : MonoBehaviour
{
    [Header("ROS Configuration")]
    [Tooltip("Name of the ROS topic to subscribe to for drive commands.")]
    public string ackermannTopicName = "/ackermann_curvature_drive";
    
    [Tooltip("Name of the ROS topic to publish odometry on.")]
    public string odomTopicName = "/odom";
    
    [Tooltip("Name of the ROS topic to publish drive commands on.")]
    public string vescDriveTopicName = "/vesc_drive";
    
    [Tooltip("Publish rate for odometry and drive messages in Hz.")]
    public float publishRateHz = 20f;

    [Header("Car Configuration")]
    [Tooltip("Reference to the car object to control.")]
    public GameObject carObj;
    
    [Tooltip("Wheelbase of the car in meters (distance between front and rear axles).")]
    public float wheelbase = 0.324f; // Default from VESC config
    
    [Tooltip("Maximum steering angle in degrees.")]
    public float maxSteeringAngle = 25.0f;
    
    [Tooltip("Maximum speed in m/s.")]
    public float maxSpeed = 5.0f;
    
    [Tooltip("Scale factor for velocity to throttle conversion.")]
    public float velocityToThrottleScale = 0.2f;

    [Header("Debug")]
    [Tooltip("Enable debug logging.")]
    public bool debugLog = false;

    // Car interface
    private ICar car;

    // ROS connection via reflection
    private object _rosConnectionInstance;
    private MethodInfo _subscribeMethod;
    private MethodInfo _registerPublisherMethod;
    private MethodInfo _publishMethod;
    
    // Message types
    private Type _ackermannMsgType;
    private Type _odomMsgType;
    private Type _twistStampedMsgType;
    private Type _headerMsgType;
    private Type _poseMsgType;
    private Type _poseWithCovarianceMsgType;
    private Type _twistMsgType;
    private Type _twistWithCovarianceMsgType;
    private Type _pointMsgType;
    private Type _quaternionMsgType;
    private Type _vector3MsgType;
    
    // Field accessors for AckermannCurvatureDriveMsg
    private FieldInfo _velocityField;
    private FieldInfo _curvatureField;

    // State tracking
    private bool _subscriberRegistered;
    private bool _odomPublisherRegistered;
    private bool _drivePublisherRegistered;
    private bool _missingRosConnectionLogged;
    private bool _missingMessageTypesLogged;
    
    // Last received command
    private float _lastVelocity = 0f;
    private float _lastCurvature = 0f;
    private float _lastCommandTime = 0f;
    private float _lastSteeringAngle = 0f;
    
    // Odometry state
    private Vector3 _position = Vector3.zero;
    private Quaternion _orientation = Quaternion.identity;
    private float _lastUpdateTime = 0f;
    
    // Publishing
    private float _nextPublishTime = 0f;
    
    // Command timeout
    private const float COMMAND_TIMEOUT = 0.5f;

    private void Awake()
    {
        if (carObj != null)
        {
            car = carObj.GetComponent<ICar>();
            if (car == null)
            {
                Debug.LogError("UtavDriver: Car object does not have an ICar component.");
            }
        }
        else
        {
            Debug.LogError("UtavDriver: Car object reference is null.");
        }

        TryInitialize();
    }

    private void OnEnable()
    {
        _subscriberRegistered = false;
        TryInitialize();
    }

    private void OnDisable()
    {
        _subscriberRegistered = false;
        
        // Stop the car when disabled
        if (car != null)
        {
            car.RequestThrottle(0.0f);
            car.RequestSteering(0.0f);
            car.RequestFootBrake(1.0f);
        }
    }

    private void FixedUpdate()
    {
        if (!EnsureReady())
        {
            return;
        }

        // Check for command timeout
        if (Time.time - _lastCommandTime > COMMAND_TIMEOUT)
        {
            // No recent commands, stop the car
            if (car != null)
            {
                car.RequestThrottle(0.0f);
                car.RequestSteering(0.0f);
            }
        }
        
        // Update odometry
        UpdateOdometry();
        
        // Publish at specified rate
        float now = Time.time;
        if (publishRateHz > 0f && now >= _nextPublishTime)
        {
            PublishOdometry();
            PublishDriveCommand();
            _nextPublishTime = now + (1f / publishRateHz);
        }
    }

    private bool EnsureReady()
    {
        if (!GlobalState.rosEnabled)
        {
            return false;
        }

        if (car == null)
        {
            return false;
        }

        if (_rosConnectionInstance == null || _subscribeMethod == null)
        {
            TryInitialize();
        }

        if (_rosConnectionInstance == null || _subscribeMethod == null)
        {
            return false;
        }

        if (!_subscriberRegistered)
        {
            try
            {
                // Create a delegate for the callback
                // Subscribe<T>(string topic, Action<T> callback)
                Type actionType = typeof(Action<>).MakeGenericType(_ackermannMsgType);
                Delegate callback = Delegate.CreateDelegate(actionType, this, "OnAckermannCurvatureDriveReceived");
                
                _subscribeMethod.Invoke(_rosConnectionInstance, new object[] { ackermannTopicName, callback });
                
                if (debugLog)
                {
                    Debug.Log($"UtavDriver: Subscribed to '{ackermannTopicName}'");
                }
            }
            catch (TargetInvocationException tie)
            {
                Debug.LogWarning($"UtavDriver: Subscribe for '{ackermannTopicName}' threw: {tie.InnerException?.Message ?? tie.Message}");
            }
            catch (Exception e)
            {
                Debug.LogWarning($"UtavDriver: Subscribe for '{ackermannTopicName}' failed: {e.Message}");
            }
            finally
            {
                _subscriberRegistered = true;
            }
        }

        return true;
    }

    private void TryInitialize()
    {
        if (!GlobalState.rosEnabled)
        {
            return;
        }

        if (_rosConnectionInstance != null && _subscribeMethod != null)
        {
            return;
        }

        Type rosConnectionType = FindType("Unity.Robotics.ROSTCPConnector.ROSConnection");
        if (rosConnectionType == null)
        {
            if (!_missingRosConnectionLogged)
            {
                Debug.LogWarning("UtavDriver: ROSConnection type not found. Install the Unity Robotics ROS-TCP-Connector package to enable ROS control.");
                _missingRosConnectionLogged = true;
            }
            return;
        }

        MethodInfo getOrCreateInstance = rosConnectionType.GetMethod("GetOrCreateInstance", BindingFlags.Public | BindingFlags.Static);
        if (getOrCreateInstance == null)
        {
            if (!_missingRosConnectionLogged)
            {
                Debug.LogWarning("UtavDriver: GetOrCreateInstance method not found on ROSConnection.");
                _missingRosConnectionLogged = true;
            }
            return;
        }

        _rosConnectionInstance = getOrCreateInstance.Invoke(null, null);
        if (_rosConnectionInstance == null)
        {
            if (!_missingRosConnectionLogged)
            {
                Debug.LogWarning("UtavDriver: Failed to obtain ROSConnection instance.");
                _missingRosConnectionLogged = true;
            }
            return;
        }

        _missingRosConnectionLogged = false;

        // Find the AckermannCurvatureDriveMsg type
        // The message type should be generated by the ROS-TCP-Connector
        _ackermannMsgType = FindType("RosMessageTypes.AmrlMsgs.AckermannCurvatureDriveMsgMsg");
        
        if (_ackermannMsgType == null)
        {
            // Try alternative naming convention
            _ackermannMsgType = FindType("RosMessageTypes.AmrlMsgs.AckermannCurvatureDriveMsg");
        }

        if (_ackermannMsgType == null)
        {
            if (!_missingMessageTypesLogged)
            {
                Debug.LogWarning("UtavDriver: Could not locate amrl_msgs/AckermannCurvatureDriveMsg message type. Generate it with the ROS-TCP-Connector.");
                Debug.LogWarning("UtavDriver: Use the MessageGeneration tool to generate messages from amrl_msgs package.");
                _missingMessageTypesLogged = true;
            }
            ResetRosState();
            return;
        }

        _velocityField = _ackermannMsgType.GetField("velocity");
        _curvatureField = _ackermannMsgType.GetField("curvature");

        if (_velocityField == null || _curvatureField == null)
        {
            if (!_missingMessageTypesLogged)
            {
                Debug.LogWarning("UtavDriver: Expected fields (velocity, curvature) not found on AckermannCurvatureDriveMsg.");
                _missingMessageTypesLogged = true;
            }
            ResetRosState();
            return;
        }

        _missingMessageTypesLogged = false;

        // Find the Subscribe method
        MethodInfo subscribeGeneric = null;
        MethodInfo[] methods = rosConnectionType.GetMethods(BindingFlags.Instance | BindingFlags.Public);

        for (int i = 0; i < methods.Length; i++)
        {
            MethodInfo method = methods[i];
            if (!method.IsGenericMethodDefinition)
            {
                continue;
            }

            ParameterInfo[] parameters = method.GetParameters();
            if (method.Name == "Subscribe" && parameters.Length == 2)
            {
                subscribeGeneric = method;
                break;
            }
        }

        if (subscribeGeneric == null)
        {
            Debug.LogWarning("UtavDriver: Could not find generic Subscribe method on ROSConnection.");
            ResetRosState();
            return;
        }

        _subscribeMethod = subscribeGeneric.MakeGenericMethod(_ackermannMsgType);
        _subscriberRegistered = false;
        
        // Find RegisterPublisher and Publish methods for publishing
        MethodInfo registerPublisherGeneric = null;
        MethodInfo publishGeneric = null;
        
        for (int i = 0; i < methods.Length; i++)
        {
            MethodInfo method = methods[i];
            if (!method.IsGenericMethodDefinition)
            {
                continue;
            }

            ParameterInfo[] parameters = method.GetParameters();
            if (method.Name == "RegisterPublisher" && parameters.Length == 1)
            {
                registerPublisherGeneric = method;
            }
            else if (method.Name == "Publish" && parameters.Length == 2)
            {
                publishGeneric = method;
            }
        }
        
        // Initialize message types for publishing
        if (registerPublisherGeneric != null && publishGeneric != null)
        {
            InitializePublishingMessageTypes();
            
            if (_odomMsgType != null)
            {
                _registerPublisherMethod = registerPublisherGeneric.MakeGenericMethod(_odomMsgType);
                _publishMethod = publishGeneric.MakeGenericMethod(_odomMsgType);
                _odomPublisherRegistered = false;
                _drivePublisherRegistered = false;
            }
        }

        if (debugLog)
        {
            Debug.Log("UtavDriver: ROS initialization successful.");
        }
    }
    
    private void InitializePublishingMessageTypes()
    {
        // Find nav_msgs/Odometry
        _odomMsgType = FindType("RosMessageTypes.Nav.OdometryMsg");
        if (_odomMsgType == null)
        {
            _odomMsgType = FindType("RosMessageTypes.NavMsgs.OdometryMsg");
        }
        
        // Find geometry_msgs/TwistStamped
        _twistStampedMsgType = FindType("RosMessageTypes.Geometry.TwistStampedMsg");
        if (_twistStampedMsgType == null)
        {
            _twistStampedMsgType = FindType("RosMessageTypes.GeometryMsgs.TwistStampedMsg");
        }
        
        // Find supporting types
        _headerMsgType = FindType("RosMessageTypes.Std.HeaderMsg");
        if (_headerMsgType == null)
        {
            _headerMsgType = FindType("RosMessageTypes.StdMsgs.HeaderMsg");
        }
        
        _poseMsgType = FindType("RosMessageTypes.Geometry.PoseMsg");
        if (_poseMsgType == null)
        {
            _poseMsgType = FindType("RosMessageTypes.GeometryMsgs.PoseMsg");
        }
        
        _poseWithCovarianceMsgType = FindType("RosMessageTypes.Geometry.PoseWithCovarianceMsg");
        if (_poseWithCovarianceMsgType == null)
        {
            _poseWithCovarianceMsgType = FindType("RosMessageTypes.GeometryMsgs.PoseWithCovarianceMsg");
        }
        
        _twistMsgType = FindType("RosMessageTypes.Geometry.TwistMsg");
        if (_twistMsgType == null)
        {
            _twistMsgType = FindType("RosMessageTypes.GeometryMsgs.TwistMsg");
        }
        
        _twistWithCovarianceMsgType = FindType("RosMessageTypes.Geometry.TwistWithCovarianceMsg");
        if (_twistWithCovarianceMsgType == null)
        {
            _twistWithCovarianceMsgType = FindType("RosMessageTypes.GeometryMsgs.TwistWithCovarianceMsg");
        }
        
        _pointMsgType = FindType("RosMessageTypes.Geometry.PointMsg");
        if (_pointMsgType == null)
        {
            _pointMsgType = FindType("RosMessageTypes.GeometryMsgs.PointMsg");
        }
        
        _quaternionMsgType = FindType("RosMessageTypes.Geometry.QuaternionMsg");
        if (_quaternionMsgType == null)
        {
            _quaternionMsgType = FindType("RosMessageTypes.GeometryMsgs.QuaternionMsg");
        }
        
        _vector3MsgType = FindType("RosMessageTypes.Geometry.Vector3Msg");
        if (_vector3MsgType == null)
        {
            _vector3MsgType = FindType("RosMessageTypes.GeometryMsgs.Vector3Msg");
        }
        
        if (_odomMsgType == null || _twistStampedMsgType == null)
        {
            if (!_missingMessageTypesLogged)
            {
                Debug.LogWarning("UtavDriver: Could not locate nav_msgs/Odometry or geometry_msgs/TwistStamped. Generate them with ROS-TCP-Connector.");
                _missingMessageTypesLogged = true;
            }
        }
    }

    private void ResetRosState()
    {
        _rosConnectionInstance = null;
        _subscribeMethod = null;
        _registerPublisherMethod = null;
        _publishMethod = null;
        _ackermannMsgType = null;
        _odomMsgType = null;
        _twistStampedMsgType = null;
        _headerMsgType = null;
        _poseMsgType = null;
        _poseWithCovarianceMsgType = null;
        _twistMsgType = null;
        _twistWithCovarianceMsgType = null;
        _pointMsgType = null;
        _quaternionMsgType = null;
        _vector3MsgType = null;
        _velocityField = null;
        _curvatureField = null;
        _subscriberRegistered = false;
        _odomPublisherRegistered = false;
        _drivePublisherRegistered = false;
    }

    private static Type FindType(string fullName)
    {
        Assembly[] assemblies = AppDomain.CurrentDomain.GetAssemblies();
        for (int i = 0; i < assemblies.Length; i++)
        {
            Type type = assemblies[i].GetType(fullName);
            if (type != null)
            {
                return type;
            }
        }

        return null;
    }

    private void UpdateOdometry()
    {
        if (car == null)
        {
            return;
        }

        Transform carTransform = car.GetTransform();
        if (carTransform == null)
        {
            return;
        }

        // Update position and orientation from car transform
        _position = carTransform.position;
        _orientation = carTransform.rotation;
        _lastUpdateTime = Time.time;
    }
    
    private void PublishOdometry()
    {
        if (_rosConnectionInstance == null || _odomMsgType == null || car == null)
        {
            return;
        }
        
        // Register publisher if not already done
        if (!_odomPublisherRegistered && _registerPublisherMethod != null)
        {
            try
            {
                _registerPublisherMethod.Invoke(_rosConnectionInstance, new object[] { odomTopicName });
                _odomPublisherRegistered = true;
                
                if (debugLog)
                {
                    Debug.Log($"UtavDriver: Registered publisher for '{odomTopicName}'");
                }
            }
            catch (Exception e)
            {
                Debug.LogWarning($"UtavDriver: Failed to register odom publisher: {e.Message}");
                return;
            }
        }
        
        if (!_odomPublisherRegistered)
        {
            return;
        }
        
        try
        {
            // Create odometry message
            object odomMsg = Activator.CreateInstance(_odomMsgType);
            
            // Create and set header
            object headerMsg = Activator.CreateInstance(_headerMsgType);
            SetHeaderTimestamp(headerMsg);
            _headerMsgType.GetField("frame_id").SetValue(headerMsg, "odom");
            _odomMsgType.GetField("header").SetValue(odomMsg, headerMsg);
            
            // Set child_frame_id
            _odomMsgType.GetField("child_frame_id").SetValue(odomMsg, "base_link");
            
            // Create and set pose with covariance
            object poseWithCov = Activator.CreateInstance(_poseWithCovarianceMsgType);
            object pose = Activator.CreateInstance(_poseMsgType);
            
            // Set position
            object position = Activator.CreateInstance(_pointMsgType);
            _pointMsgType.GetField("x").SetValue(position, (double)_position.x);
            _pointMsgType.GetField("y").SetValue(position, (double)_position.y);
            _pointMsgType.GetField("z").SetValue(position, (double)_position.z);
            _poseMsgType.GetField("position").SetValue(pose, position);
            
            // Set orientation
            object orientation = Activator.CreateInstance(_quaternionMsgType);
            _quaternionMsgType.GetField("x").SetValue(orientation, (double)_orientation.x);
            _quaternionMsgType.GetField("y").SetValue(orientation, (double)_orientation.y);
            _quaternionMsgType.GetField("z").SetValue(orientation, (double)_orientation.z);
            _quaternionMsgType.GetField("w").SetValue(orientation, (double)_orientation.w);
            _poseMsgType.GetField("orientation").SetValue(pose, orientation);
            
            _poseWithCovarianceMsgType.GetField("pose").SetValue(poseWithCov, pose);
            
            // Set pose covariance (similar to real car)
            double[] poseCovariance = new double[36];
            poseCovariance[0] = 0.001;   // x
            poseCovariance[7] = 0.001;   // y
            poseCovariance[14] = 1000000.0; // z (not measured)
            poseCovariance[21] = 1000000.0; // roll (not measured)
            poseCovariance[28] = 1000000.0; // pitch (not measured)
            poseCovariance[35] = 0.03;   // yaw
            _poseWithCovarianceMsgType.GetField("covariance").SetValue(poseWithCov, poseCovariance);
            
            _odomMsgType.GetField("pose").SetValue(odomMsg, poseWithCov);
            
            // Create and set twist with covariance
            object twistWithCov = Activator.CreateInstance(_twistWithCovarianceMsgType);
            object twist = Activator.CreateInstance(_twistMsgType);
            
            // Get velocity from car
            Vector3 velocity = car.GetVelocity();
            
            // Set linear velocity
            object linear = Activator.CreateInstance(_vector3MsgType);
            _vector3MsgType.GetField("x").SetValue(linear, (double)velocity.x);
            _vector3MsgType.GetField("y").SetValue(linear, (double)velocity.y);
            _vector3MsgType.GetField("z").SetValue(linear, (double)velocity.z);
            _twistMsgType.GetField("linear").SetValue(twist, linear);
            
            // Calculate angular velocity from velocity and curvature
            float angularZ = _lastVelocity * _lastCurvature;
            
            object angular = Activator.CreateInstance(_vector3MsgType);
            _vector3MsgType.GetField("x").SetValue(angular, 0.0);
            _vector3MsgType.GetField("y").SetValue(angular, 0.0);
            _vector3MsgType.GetField("z").SetValue(angular, (double)angularZ);
            _twistMsgType.GetField("angular").SetValue(twist, angular);
            
            _twistWithCovarianceMsgType.GetField("twist").SetValue(twistWithCov, twist);
            
            // Set twist covariance (similar to real car)
            double[] twistCovariance = new double[36];
            twistCovariance[0] = 0.001;   // vx
            twistCovariance[7] = 0.001;   // vy
            twistCovariance[14] = 0.001;  // vz
            twistCovariance[21] = 1000000.0; // wx (not measured)
            twistCovariance[28] = 1000000.0; // wy (not measured)
            twistCovariance[35] = 0.03;   // wz
            _twistWithCovarianceMsgType.GetField("covariance").SetValue(twistWithCov, twistCovariance);
            
            _odomMsgType.GetField("twist").SetValue(odomMsg, twistWithCov);
            
            // Publish
            _publishMethod.Invoke(_rosConnectionInstance, new object[] { odomTopicName, odomMsg });
        }
        catch (Exception e)
        {
            if (debugLog)
            {
                Debug.LogError($"UtavDriver: Error publishing odometry: {e.Message}");
            }
        }
    }
    
    private void PublishDriveCommand()
    {
        if (_rosConnectionInstance == null || _twistStampedMsgType == null)
        {
            return;
        }
        
        // Register publisher if not already done
        if (!_drivePublisherRegistered && _registerPublisherMethod != null)
        {
            try
            {
                // Need to create a new RegisterPublisher method for TwistStamped
                Type rosConnectionType = _rosConnectionInstance.GetType();
                MethodInfo[] methods = rosConnectionType.GetMethods(BindingFlags.Instance | BindingFlags.Public);
                MethodInfo registerPublisherGeneric = null;
                
                for (int i = 0; i < methods.Length; i++)
                {
                    MethodInfo method = methods[i];
                    if (method.IsGenericMethodDefinition && method.Name == "RegisterPublisher")
                    {
                        ParameterInfo[] parameters = method.GetParameters();
                        if (parameters.Length == 1)
                        {
                            registerPublisherGeneric = method;
                            break;
                        }
                    }
                }
                
                if (registerPublisherGeneric != null)
                {
                    MethodInfo registerTwistPublisher = registerPublisherGeneric.MakeGenericMethod(_twistStampedMsgType);
                    registerTwistPublisher.Invoke(_rosConnectionInstance, new object[] { vescDriveTopicName });
                    _drivePublisherRegistered = true;
                    
                    if (debugLog)
                    {
                        Debug.Log($"UtavDriver: Registered publisher for '{vescDriveTopicName}'");
                    }
                }
            }
            catch (Exception e)
            {
                Debug.LogWarning($"UtavDriver: Failed to register drive publisher: {e.Message}");
                return;
            }
        }
        
        if (!_drivePublisherRegistered)
        {
            return;
        }
        
        try
        {
            // Create TwistStamped message
            object twistStampedMsg = Activator.CreateInstance(_twistStampedMsgType);
            
            // Create and set header
            object headerMsg = Activator.CreateInstance(_headerMsgType);
            SetHeaderTimestamp(headerMsg);
            _headerMsgType.GetField("frame_id").SetValue(headerMsg, "base_link");
            _twistStampedMsgType.GetField("header").SetValue(twistStampedMsg, headerMsg);
            
            // Create twist
            object twist = Activator.CreateInstance(_twistMsgType);
            
            // Set linear velocity
            object linear = Activator.CreateInstance(_vector3MsgType);
            _vector3MsgType.GetField("x").SetValue(linear, (double)_lastVelocity);
            _vector3MsgType.GetField("y").SetValue(linear, 0.0);
            _vector3MsgType.GetField("z").SetValue(linear, 0.0);
            _twistMsgType.GetField("linear").SetValue(twist, linear);
            
            // Calculate angular velocity
            float angularZ = _lastVelocity * _lastCurvature;
            
            object angular = Activator.CreateInstance(_vector3MsgType);
            _vector3MsgType.GetField("x").SetValue(angular, 0.0);
            _vector3MsgType.GetField("y").SetValue(angular, 0.0);
            _vector3MsgType.GetField("z").SetValue(angular, (double)angularZ);
            _twistMsgType.GetField("angular").SetValue(twist, angular);
            
            _twistStampedMsgType.GetField("twist").SetValue(twistStampedMsg, twist);
            
            // Publish
            Type rosConnectionType = _rosConnectionInstance.GetType();
            MethodInfo[] methods = rosConnectionType.GetMethods(BindingFlags.Instance | BindingFlags.Public);
            MethodInfo publishGeneric = null;
            
            for (int i = 0; i < methods.Length; i++)
            {
                MethodInfo method = methods[i];
                if (method.IsGenericMethodDefinition && method.Name == "Publish")
                {
                    ParameterInfo[] parameters = method.GetParameters();
                    if (parameters.Length == 2)
                    {
                        publishGeneric = method;
                        break;
                    }
                }
            }
            
            if (publishGeneric != null)
            {
                MethodInfo publishTwist = publishGeneric.MakeGenericMethod(_twistStampedMsgType);
                publishTwist.Invoke(_rosConnectionInstance, new object[] { vescDriveTopicName, twistStampedMsg });
            }
        }
        catch (Exception e)
        {
            if (debugLog)
            {
                Debug.LogError($"UtavDriver: Error publishing drive command: {e.Message}");
            }
        }
    }
    
    private void SetHeaderTimestamp(object headerMsg)
    {
        if (_headerMsgType == null)
        {
            return;
        }
        
        // Get current time in ROS format (seconds and nanoseconds since epoch)
        double timeSeconds = Time.time;
        int sec = (int)Math.Floor(timeSeconds);
        uint nanosec = (uint)((timeSeconds - sec) * 1_000_000_000);
        
        // Find the stamp field (builtin_interfaces/Time)
        FieldInfo stampField = _headerMsgType.GetField("stamp");
        if (stampField != null)
        {
            object stampMsg = stampField.GetValue(headerMsg);
            if (stampMsg == null)
            {
                // Create new Time message
                Type timeType = stampField.FieldType;
                stampMsg = Activator.CreateInstance(timeType);
            }
            
            Type stampType = stampMsg.GetType();
            stampType.GetField("sec").SetValue(stampMsg, sec);
            stampType.GetField("nanosec").SetValue(stampMsg, nanosec);
            
            stampField.SetValue(headerMsg, stampMsg);
        }
    }

    // This method is called via reflection when a message is received
    private void OnAckermannCurvatureDriveReceived(object msg)
    {
        if (car == null)
        {
            return;
        }

        try
        {
            // Extract velocity and curvature from the message
            float velocity = (float)_velocityField.GetValue(msg);
            float curvature = (float)_curvatureField.GetValue(msg);

            _lastVelocity = velocity;
            _lastCurvature = curvature;
            _lastCommandTime = Time.time;

            // Convert velocity to throttle
            // Throttle is typically in range [-1, 1]
            float throttle = Mathf.Clamp(velocity * velocityToThrottleScale, -1.0f, 1.0f);

            // Convert curvature to steering angle
            // From the VESC driver: steering_angle = atan(wheelbase / turn_radius)
            // where turn_radius = velocity / (velocity * curvature) = 1 / curvature
            // So: steering_angle = atan(wheelbase * curvature)
            float steeringAngle = 0.0f;
            if (Mathf.Abs(curvature) > 1e-6f)
            {
                float turnRadius = 1.0f / curvature;
                steeringAngle = Mathf.Atan(wheelbase / turnRadius) * Mathf.Rad2Deg;
            }

            // Clamp steering angle to max
            steeringAngle = Mathf.Clamp(steeringAngle, -maxSteeringAngle, maxSteeringAngle);
            
            // Store for publishing
            _lastSteeringAngle = steeringAngle;

            // Apply commands to the car
            car.RequestSteering(steeringAngle);
            car.RequestThrottle(throttle);

            if (debugLog && Time.frameCount % 30 == 0) // Log every ~0.5 seconds at 60fps
            {
                Debug.Log($"UtavDriver: velocity={velocity:F2} m/s, curvature={curvature:F3}, " +
                         $"throttle={throttle:F2}, steering={steeringAngle:F1}°");
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"UtavDriver: Error processing AckermannCurvatureDriveMsg: {e.Message}");
        }
    }

    private void OnGUI()
    {
        if (!debugLog || !GlobalState.rosEnabled)
        {
            return;
        }

        // Display drive state in the bottom left corner
        string pubStatus = _odomPublisherRegistered && _drivePublisherRegistered ? "Active" : "Inactive";
        string displayText = $"UtavDriver\n" +
                            $"Velocity: {_lastVelocity:F2} m/s\n" +
                            $"Curvature: {_lastCurvature:F3}\n" +
                            $"Last Cmd: {Time.time - _lastCommandTime:F2}s ago\n" +
                            $"Publishing: {pubStatus} @ {publishRateHz:F0} Hz";
        
        GUIStyle panelStyle = new GUIStyle(GUI.skin.box);
        panelStyle.padding = new RectOffset(10, 10, 10, 10);
        panelStyle.alignment = TextAnchor.UpperLeft;
        
        GUIStyle textStyle = new GUIStyle(GUI.skin.label);
        textStyle.fontSize = 12;
        textStyle.normal.textColor = Color.white;
        
        GUIContent content = new GUIContent(displayText);
        Vector2 textSize = textStyle.CalcSize(content);
        
        float padding = 10f;
        Rect panelRect = new Rect(
            padding,
            Screen.height - textSize.y - 40,
            textSize.x + 20,
            textSize.y + 20
        );
        
        GUI.Box(panelRect, "", panelStyle);
        
        Rect textRect = new Rect(
            panelRect.x + 10,
            panelRect.y + 10,
            textSize.x,
            textSize.y
        );
        GUI.Label(textRect, displayText, textStyle);
    }
}

