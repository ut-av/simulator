using System;
using System.Reflection;
using UnityEngine;

/// <summary>
/// Publishes rosgraph_msgs/Clock messages based on the Unity simulation clock.
/// Uses reflection to avoid a direct compile-time dependency on the ROS-TCP-Connector package.
/// </summary>
public class RosClock : MonoBehaviour
{
    [Tooltip("Name of the ROS topic to publish clock messages on.")]
    public string topicName = "/clock";

    [Tooltip("How often (Hz) to publish the clock message. Set to <= 0 to publish every frame.")]
    public float publishRateHz = 30f;

    [Tooltip("Publish from FixedUpdate instead of Update.")]
    public bool publishInFixedUpdate = true;

    private object _rosConnectionInstance;
    private MethodInfo _registerPublisherMethod;
    private MethodInfo _publishMethod;
    private Type _clockMsgType;
    private Type _timeMsgType;
    private FieldInfo _clockField;
    private FieldInfo _secField;
    private FieldInfo _nanosecField;

    private bool _publisherRegistered;
    private bool _missingRosConnectionLogged;
    private bool _missingMessageTypesLogged;

    private float _nextPublishTime;

    private void Awake()
    {
        TryInitialize();
    }

    private void OnEnable()
    {
        _nextPublishTime = 0f;
        _publisherRegistered = false;
        TryInitialize();
    }

    private void OnDisable()
    {
        _publisherRegistered = false;
    }

    private void Update()
    {
        if (!publishInFixedUpdate)
        {
            TryPublish(GetUnityTimeSeconds());
        }
    }

    private void FixedUpdate()
    {
        if (publishInFixedUpdate)
        {
            TryPublish(GetUnityTimeSeconds());
        }
    }

    private void TryPublish(double unityTimeSeconds)
    {
        if (!EnsureReady())
        {
            return;
        }

        float now = Time.unscaledTime;
        if (publishRateHz > 0f && now < _nextPublishTime)
        {
            return;
        }

        if (publishRateHz > 0f)
        {
            _nextPublishTime = now + (1f / publishRateHz);
        }

        int sec = (int)Math.Floor(unityTimeSeconds);
        double fractional = unityTimeSeconds - sec;
        long nanosecLong = (long)Math.Round(fractional * 1_000_000_000d);

        if (nanosecLong >= 1_000_000_000L)
        {
            sec += 1;
            nanosecLong -= 1_000_000_000L;
        }

        if (nanosecLong < 0)
        {
            nanosecLong = 0;
        }

        object timeMsg = Activator.CreateInstance(_timeMsgType);
        _secField.SetValue(timeMsg, sec);
        _nanosecField.SetValue(timeMsg, (uint)nanosecLong);

        object clockMsg = Activator.CreateInstance(_clockMsgType);
        _clockField.SetValue(clockMsg, timeMsg);

        try
        {
            _publishMethod.Invoke(_rosConnectionInstance, new object[] { topicName, clockMsg });
        }
        catch (TargetInvocationException tie)
        {
            Debug.LogError($"RosClock: Failed to publish clock message: {tie.InnerException?.Message ?? tie.Message}");
        }
        catch (Exception e)
        {
            Debug.LogError($"RosClock: Failed to publish clock message: {e.Message}");
        }
    }

    private bool EnsureReady()
    {
        if (!GlobalState.rosEnabled)
        {
            return false;
        }

        if (_rosConnectionInstance == null || _publishMethod == null)
        {
            TryInitialize();
        }

        if (_rosConnectionInstance == null || _publishMethod == null)
        {
            return false;
        }

        if (!_publisherRegistered)
        {
            try
            {
                _registerPublisherMethod.Invoke(_rosConnectionInstance, new object[] { topicName });
            }
            catch (TargetInvocationException tie)
            {
                Debug.LogWarning($"RosClock: RegisterPublisher for '{topicName}' threw: {tie.InnerException?.Message ?? tie.Message}");
            }
            catch (Exception e)
            {
                Debug.LogWarning($"RosClock: RegisterPublisher for '{topicName}' failed: {e.Message}");
            }
            finally
            {
                _publisherRegistered = true;
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

        if (_rosConnectionInstance != null && _publishMethod != null)
        {
            return;
        }

        Type rosConnectionType = FindType("Unity.Robotics.ROSTCPConnector.ROSConnection");
        if (rosConnectionType == null)
        {
            if (!_missingRosConnectionLogged)
            {
                Debug.LogWarning("RosClock: ROSConnection type not found. Install the Unity Robotics ROS-TCP-Connector package to enable clock publishing.");
                _missingRosConnectionLogged = true;
            }
            return;
        }

        MethodInfo getOrCreateInstance = rosConnectionType.GetMethod("GetOrCreateInstance", BindingFlags.Public | BindingFlags.Static);
        if (getOrCreateInstance == null)
        {
            if (!_missingRosConnectionLogged)
            {
                Debug.LogWarning("RosClock: GetOrCreateInstance method not found on ROSConnection.");
                _missingRosConnectionLogged = true;
            }
            return;
        }

        _rosConnectionInstance = getOrCreateInstance.Invoke(null, null);
        if (_rosConnectionInstance == null)
        {
            if (!_missingRosConnectionLogged)
            {
                Debug.LogWarning("RosClock: Failed to obtain ROSConnection instance.");
                _missingRosConnectionLogged = true;
            }
            return;
        }

        _missingRosConnectionLogged = false;

        _clockMsgType = FindType("RosMessageTypes.Rosgraph.ClockMsg");
        _timeMsgType = FindType("RosMessageTypes.BuiltinInterfaces.TimeMsg");

        if (_clockMsgType == null || _timeMsgType == null)
        {
            if (!_missingMessageTypesLogged)
            {
                Debug.LogWarning("RosClock: Could not locate rosgraph_msgs/Clock or builtin_interfaces/Time message types. Generate them with the ROS-TCP-Connector.");
                _missingMessageTypesLogged = true;
            }
            ResetRosState();
            return;
        }

        _clockField = _clockMsgType.GetField("clock");
        _secField = _timeMsgType.GetField("sec");
        _nanosecField = _timeMsgType.GetField("nanosec");

        if (_clockField == null || _secField == null || _nanosecField == null)
        {
            if (!_missingMessageTypesLogged)
            {
                Debug.LogWarning("RosClock: Expected fields not found on ClockMsg or TimeMsg.");
                _missingMessageTypesLogged = true;
            }
            ResetRosState();
            return;
        }

        _missingMessageTypesLogged = false;

        MethodInfo registerPublisherGeneric = null;
        MethodInfo publishGeneric = null;
        MethodInfo[] methods = rosConnectionType.GetMethods(BindingFlags.Instance | BindingFlags.Public);

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

        if (registerPublisherGeneric == null || publishGeneric == null)
        {
            Debug.LogWarning("RosClock: Could not find generic RegisterPublisher/Publish methods on ROSConnection.");
            ResetRosState();
            return;
        }

        _registerPublisherMethod = registerPublisherGeneric.MakeGenericMethod(_clockMsgType);
        _publishMethod = publishGeneric.MakeGenericMethod(_clockMsgType);
        _publisherRegistered = false;
    }

    private void ResetRosState()
    {
        _rosConnectionInstance = null;
        _registerPublisherMethod = null;
        _publishMethod = null;
        _clockMsgType = null;
        _timeMsgType = null;
        _clockField = null;
        _secField = null;
        _nanosecField = null;
        _publisherRegistered = false;
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

    private static double GetUnityTimeSeconds()
    {
#if UNITY_2020_2_OR_NEWER
        return Time.timeAsDouble;
#else
        return Time.time;
#endif
    }
}

