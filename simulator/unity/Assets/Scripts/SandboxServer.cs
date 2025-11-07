using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using tk;
using System.Net;
using System.Net.Sockets;
using System;

/// <summary>
/// Unified Sandbox Server - Handles Track API and Menu API simultaneously on a single port.
/// Routes clients to Track API or Menu API based on their message patterns.
/// </summary>
[RequireComponent(typeof(tk.TcpServer))]
public class SandboxServer : MonoBehaviour
{
    public string host;
    public int port;

    tk.TcpServer _server = null;

    public GameObject clientTemplateObj = null;
    public Transform spawn_pt;
    
    bool argHost = false;
    bool argPort = false;

    // Track which clients are using which API
    private Dictionary<tk.TcpClient, ClientAPIType> clientAPITypes = new Dictionary<tk.TcpClient, ClientAPIType>();

    private enum ClientAPIType
    {
        Unknown,
        TrackAPI,
        MenuAPI
    }

    public void CheckCommandLineConnectArgs()
    {
        string[] args = System.Environment.GetCommandLineArgs();

        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--host")
            {
                host = args[i + 1];
                argHost = true;
            }
            else if (args[i] == "--port")
            {
                port = int.Parse(args[i + 1]);
                argPort = true;
            }
            // Note: --scene/--environment is handled by CommandLineArgsHandler component
        }

        if (argHost == false) { host = GlobalState.host; }
        if (argPort == false) { port = GlobalState.port; }
    }

    private void Awake()
    {
        _server = GetComponent<tk.TcpServer>();
    }

    // Start is called before the first frame update
    void Start()
    {
        CheckCommandLineConnectArgs();

        Debug.Log($"Unified Sandbox Server starting on {host}:{port}");
        Debug.Log("Supports: Track API and Menu API simultaneously");
        
        _server.onClientConntedCB += new tk.TcpServer.OnClientConnected(OnClientConnected);
        _server.onClientDisconntedCB += new tk.TcpServer.OnClientDisconnected(OnClientDisconnected);

        _server.Run(host, port);
    }

    // It's our responsibility to create a GameObject with a TcpClient
    // and return it to the server.
    public tk.TcpClient OnClientConnected()
    {
        if (clientTemplateObj == null)
        {
            Debug.LogError("Sandbox Server: client template object was null.");
            return null;
        }

        if (_server.debug)
            Debug.Log("Sandbox Server: creating client obj");

        GameObject go = GameObject.Instantiate(clientTemplateObj) as GameObject;

        go.transform.parent = this.transform;

        if (spawn_pt != null)
            go.transform.position = spawn_pt.position + UnityEngine.Random.insideUnitSphere * 2;

        tk.TcpClient client = go.GetComponent<tk.TcpClient>();

        // Initialize client with routing capability
        InitClient(client);

        return client;
    }

    private void InitClient(tk.TcpClient client)
    {
        tk.JsonTcpClient jsonClient = client.gameObject.GetComponent<tk.JsonTcpClient>();
        if (jsonClient == null)
        {
            Debug.LogError("Sandbox Server: JsonTcpClient component not found on client template.");
            return;
        }

        // Mark client as unknown initially - will be determined by first message
        clientAPITypes[client] = ClientAPIType.Unknown;

        // Register routing handlers that will determine the API type based on first message
        // These handlers will initialize the appropriate API handler, which will then register its own handlers
        jsonClient.dispatchInMainThread = true; // Need main thread for routing decisions
        
        // Menu API routing - these messages uniquely identify Menu API clients
        jsonClient.dispatcher.Register("load_scene", new tk.Delegates.OnMsgRecv((json) => RouteToMenuAPI(client, json)));
        jsonClient.dispatcher.Register("get_scene_names", new tk.Delegates.OnMsgRecv((json) => RouteToMenuAPI(client, json)));
        
        // Track API routing - these messages uniquely identify Track API clients
        jsonClient.dispatcher.Register("control", new tk.Delegates.OnMsgRecv((json) => RouteToTrackAPI(client, json)));
        jsonClient.dispatcher.Register("reset_car", new tk.Delegates.OnMsgRecv((json) => RouteToTrackAPI(client, json)));
        jsonClient.dispatcher.Register("exit_scene", new tk.Delegates.OnMsgRecv((json) => RouteToTrackAPI(client, json)));
        jsonClient.dispatcher.Register("step_mode", new tk.Delegates.OnMsgRecv((json) => RouteToTrackAPI(client, json)));
        jsonClient.dispatcher.Register("regen_road", new tk.Delegates.OnMsgRecv((json) => RouteToTrackAPI(client, json)));
        jsonClient.dispatcher.Register("car_config", new tk.Delegates.OnMsgRecv((json) => RouteToTrackAPI(client, json)));
        jsonClient.dispatcher.Register("cam_config", new tk.Delegates.OnMsgRecv((json) => RouteToTrackAPI(client, json)));
        jsonClient.dispatcher.Register("cam_config_b", new tk.Delegates.OnMsgRecv((json) => RouteToTrackAPI(client, json)));
        jsonClient.dispatcher.Register("lidar_config", new tk.Delegates.OnMsgRecv((json) => RouteToTrackAPI(client, json)));
        jsonClient.dispatcher.Register("set_position", new tk.Delegates.OnMsgRecv((json) => RouteToTrackAPI(client, json)));
        jsonClient.dispatcher.Register("node_position", new tk.Delegates.OnMsgRecv((json) => RouteToTrackAPI(client, json)));
        
        // Handle ambiguous messages that could be from Menu or Track API
        // We'll route based on scene context
        jsonClient.dispatcher.Register("get_protocol_version", new tk.Delegates.OnMsgRecv((json) => RouteAmbiguousMessage(client, json)));
        jsonClient.dispatcher.Register("quit_app", new tk.Delegates.OnMsgRecv((json) => RouteAmbiguousMessage(client, json)));
        jsonClient.dispatcher.Register("connected", new tk.Delegates.OnMsgRecv((json) => RouteAmbiguousMessage(client, json)));
    }
    
    private void RouteAmbiguousMessage(tk.TcpClient client, JSONObject json)
    {
        // For ambiguous messages, check if we've already determined the API type
        if (clientAPITypes.ContainsKey(client) && clientAPITypes[client] != ClientAPIType.Unknown)
        {
            return; // Already routed, let the API handler process it
        }
        
        // Try to determine based on scene context
        CarSpawner spawner = GameObject.FindFirstObjectByType<CarSpawner>();
        tk.TcpMenuHandler menuHandler = GameObject.FindFirstObjectByType<tk.TcpMenuHandler>();
        
        if (spawner != null && menuHandler == null)
        {
            // Track scene - route to Track API (it will re-dispatch)
            RouteToTrackAPI(client, json);
        }
        else if (menuHandler != null && spawner == null)
        {
            // Menu scene - route to Menu API (it will re-dispatch)
            RouteToMenuAPI(client, json);
        }
        else
        {
            // Both or neither - default to Track API if spawner exists, otherwise Menu
            if (spawner != null)
            {
                RouteToTrackAPI(client, json);
            }
            else if (menuHandler != null)
            {
                RouteToMenuAPI(client, json);
            }
        }
    }

    private void RouteToMenuAPI(tk.TcpClient client, JSONObject json)
    {
        if (clientAPITypes.ContainsKey(client) && clientAPITypes[client] != ClientAPIType.Unknown)
        {
            return; // Already routed
        }
        
        clientAPITypes[client] = ClientAPIType.MenuAPI;
        
        if (_server.debug)
            Debug.Log("Sandbox Server: Routing client to Menu API");

        // Initialize Menu API handler - it will register its own handlers
        tk.TcpMenuHandler menuHandler = GameObject.FindFirstObjectByType<tk.TcpMenuHandler>();
        if (menuHandler != null)
        {
            tk.JsonTcpClient jsonClient = client.gameObject.GetComponent<tk.JsonTcpClient>();
            menuHandler.Init(jsonClient);
            
            // Re-dispatch the message so the API handler can process it
            string msgType = json.GetField("msg_type").str;
            jsonClient.dispatcher.Dipatch(msgType, json);
        }
        else
        {
            Debug.LogWarning("Sandbox Server: TcpMenuHandler not found in scene.");
        }
    }

    private void RouteToTrackAPI(tk.TcpClient client, JSONObject json)
    {
        if (clientAPITypes.ContainsKey(client) && clientAPITypes[client] != ClientAPIType.Unknown)
        {
            return; // Already routed
        }
        
        clientAPITypes[client] = ClientAPIType.TrackAPI;
        
        if (_server.debug)
            Debug.Log("Sandbox Server: Routing client to Track API");

        // Initialize Track API handler (spawn car) - TcpCarHandler will register its own handlers
        CarSpawner spawner = GameObject.FindFirstObjectByType<CarSpawner>();
        if (spawner != null)
        {
            tk.JsonTcpClient jsonClient = client.gameObject.GetComponent<tk.JsonTcpClient>();
            spawner.Spawn(jsonClient, false);
            
            // Re-dispatch the message so the API handler can process it
            // Note: Car spawning is asynchronous, so handlers may not be registered yet
            // But the message will be processed once handlers are registered
            string msgType = json.GetField("msg_type").str;
            jsonClient.dispatcher.Dipatch(msgType, json);
        }
        else
        {
            Debug.LogWarning("Sandbox Server: CarSpawner not found in scene. Track API clients need a track scene.");
        }
    }

    public void OnSceneLoaded(bool bFrontEnd)
    {
        // Reinitialize Track API clients when scene loads
        List<tk.TcpClient> clients = _server.GetClients();

        foreach (tk.TcpClient client in clients)
        {
            if (clientAPITypes.ContainsKey(client) && clientAPITypes[client] == ClientAPIType.TrackAPI)
            {
                if (_server.debug)
                    Debug.Log("Sandbox Server: Reinitializing Track API client.");

                CarSpawner spawner = GameObject.FindFirstObjectByType<CarSpawner>();
                if (spawner != null)
                {
                    tk.JsonTcpClient jsonClient = client.gameObject.GetComponent<tk.JsonTcpClient>();
                    if (jsonClient != null)
                    {
                        spawner.Spawn(jsonClient, false);
                    }
                }
            }
        }

        if (GlobalState.paceCar && !bFrontEnd)
        {
            CarSpawner spawner = GameObject.FindFirstObjectByType<CarSpawner>();
            if (spawner)
            {
                spawner.EnsureOneCar();
            }
        }
    }

    public void OnClientDisconnected(tk.TcpClient client)
    {
        // Clean up based on API type
        if (clientAPITypes.ContainsKey(client))
        {
            if (clientAPITypes[client] == ClientAPIType.TrackAPI)
            {
                CarSpawner spawner = GameObject.FindFirstObjectByType<CarSpawner>();
                if (spawner != null)
                {
                    tk.JsonTcpClient jsonClient = client.gameObject.GetComponent<tk.JsonTcpClient>();
                    if (jsonClient != null)
                    {
                        spawner.RemoveCar(jsonClient);
                    }
                }
            }
            
            clientAPITypes.Remove(client);
        }

        GameObject.Destroy(client.gameObject);
    }

    private void OnGUI()
    {
        // Display host and port from GlobalState in the top right corner
        string displayText = $"Host: {GlobalState.host}\nPort: {GlobalState.port}";
        
        // Create a style for the panel
        GUIStyle panelStyle = new GUIStyle(GUI.skin.box);
        panelStyle.padding = new RectOffset(10, 10, 10, 10);
        panelStyle.alignment = TextAnchor.UpperLeft;
        
        // Create a style for the text
        GUIStyle textStyle = new GUIStyle(GUI.skin.label);
        textStyle.fontSize = 14;
        textStyle.fontStyle = FontStyle.Bold;
        textStyle.normal.textColor = Color.white;
        
        // Calculate the size of the text
        GUIContent content = new GUIContent(displayText);
        Vector2 textSize = textStyle.CalcSize(content);
        
        // Position in the top right corner with some padding
        float padding = 10f;
        Rect panelRect = new Rect(
            Screen.width - textSize.x - 40,
            padding,
            textSize.x + 20,
            textSize.y + 20
        );
        
        // Draw the background panel
        GUI.Box(panelRect, "", panelStyle);
        
        // Draw the text inside the panel
        Rect textRect = new Rect(
            panelRect.x + 10,
            panelRect.y + 10,
            textSize.x,
            textSize.y
        );
        GUI.Label(textRect, displayText, textStyle);
    }
}
