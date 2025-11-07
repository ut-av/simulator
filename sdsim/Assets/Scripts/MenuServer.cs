using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using tk;
using System.Net;
using System.Net.Sockets;
using System;

/// <summary>
/// Menu Server - Handles menu interactions.
/// Runs on port 9093 by default.
/// </summary>
[RequireComponent(typeof(tk.TcpServer))]
public class MenuServer : MonoBehaviour
{
    public string host;
    public int port;

    tk.TcpServer _server = null;

    public GameObject clientTemplateObj = null;
    public Transform spawn_pt;

    private void Awake()
    {
        _server = GetComponent<tk.TcpServer>();
    }

    // Start is called before the first frame update
    void Start()
    {
        // Use default Menu port
        if (string.IsNullOrEmpty(host)) { host = GlobalState.host; }
        if (port == 0) { port = GlobalState.portMenu; }

        Debug.Log($"Menu Server starting on {host}:{port}");
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
            Debug.LogError("Menu Server: client template object was null.");
            return null;
        }

        if (_server.debug)
            Debug.Log("Menu Server: creating client obj");

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
        tk.TcpMenuHandler handler = GameObject.FindFirstObjectByType<TcpMenuHandler>();
        if (handler != null)
        {
            if (_server.debug)
                Debug.Log("Menu Server: init menu handler.");

            handler.Init(client.gameObject.GetComponent<tk.JsonTcpClient>());
        }
        else
        {
            Debug.LogWarning("Menu Server: TcpMenuHandler not found in scene. This server should only be used in menu scenes.");
        }
    }

    public void OnClientDisconnected(tk.TcpClient client)
    {
        // Menu clients don't spawn cars, so just clean up
        GameObject.Destroy(client.gameObject);
    }
}


