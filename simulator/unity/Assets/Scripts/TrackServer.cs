using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using tk;
using System.Net;
using System.Net.Sockets;
using System;

/// <summary>
/// Track Server - Handles car spawning for track scenes.
/// Runs on port 9091 by default.
/// </summary>
[RequireComponent(typeof(tk.TcpServer))]
public class TrackServer : MonoBehaviour
{
    public string host;
    public int port;

    tk.TcpServer _server = null;

    public GameObject clientTemplateObj = null;
    public Transform spawn_pt;
    bool argHost = false;
    bool argPort = false;

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

        Debug.Log($"Track Server starting on {host}:{port}");
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
            Debug.LogError("Track Server: client template object was null.");
            return null;
        }

        if (_server.debug)
            Debug.Log("Track Server: creating client obj");

        GameObject go = GameObject.Instantiate(clientTemplateObj) as GameObject;

        go.transform.parent = this.transform;

        if (spawn_pt != null)
            go.transform.position = spawn_pt.position + UnityEngine.Random.insideUnitSphere * 2;

        tk.TcpClient client = go.GetComponent<tk.TcpClient>();

        InitClient(client);

        return client;
    }

    private void InitClient(tk.TcpClient client)
    {
        CarSpawner spawner = GameObject.FindFirstObjectByType<CarSpawner>();
        if (spawner != null)
        {
            if (_server.debug)
                Debug.Log("Track Server: spawning car.");

            spawner.Spawn(client.gameObject.GetComponent<tk.JsonTcpClient>(), false);
        }
        else
        {
            Debug.LogWarning("Track Server: CarSpawner not found in scene. This server should only be used in track scenes.");
        }
    }

    public void OnSceneLoaded(bool bFrontEnd)
    {
        // Reinitialize all clients when scene loads
        List<tk.TcpClient> clients = _server.GetClients();

        foreach (tk.TcpClient client in clients)
        {
            if (_server.debug)
                Debug.Log("Track Server: reinit network client.");

            InitClient(client);
        }

        if (GlobalState.paceCar && !bFrontEnd) // && clients.Count == 0
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
        CarSpawner spawner = GameObject.FindFirstObjectByType<CarSpawner>();
        if (spawner != null)
        {
            spawner.RemoveCar(client.gameObject.GetComponent<tk.JsonTcpClient>());
        }
        GameObject.Destroy(client.gameObject);
    }
}


