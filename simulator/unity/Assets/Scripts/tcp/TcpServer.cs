using UnityEngine;
using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Threading;

namespace tk
{   
    public class TcpServer : MonoBehaviour
    {
        // register for OnClientConnected to handle the game specific creation of TcpClients with a MonoBehavior
        public delegate TcpClient OnClientConnected();
        public OnClientConnected onClientConntedCB;

        // register for OnClientDisconnected to have an opportunity to handle dropped clients
        public delegate void OnClientDisconnected(TcpClient client);
        public OnClientDisconnected onClientDisconntedCB;

        // Server listener socket
        Socket listener = null;

        // Accept thread
        Thread thread = null;

        // Thread signal.  
        public ManualResetEvent allDone = new ManualResetEvent(false);

        // All connected clients
        List<TcpClient> clients = new List<TcpClient>();

        // All new clients that need a onClientConntedCB callback
        List<Socket> new_clients = new List<Socket>();

        // Lock object to protect access to new_clients
        readonly object _locker = new object();

        // Verbose messages
        public bool debug = false;

        // Store the bound host and port to sync with GlobalState
        private string boundHost = "";
        private int boundPort = 0;

        // Public properties to access bound host and port
        public string BoundHost => boundHost;
        public int BoundPort => boundPort;

        // Call the Run method to start the server. The ip address is typically 127.0.0.1 to accept only local connections.
        // Or 0.0.0.0 to bind to all incoming connections for this NIC.
        public void Run(string ip, int port)
        {
            boundHost = ip;
            boundPort = port;
            
            // Update GlobalState with the requested host and port
            GlobalState.host = ip;
            GlobalState.port = port;
            
            Bind(ip, port);

            Debug.Log("Listening on " + ip + ":" + port.ToString());

            // Poll for new connections in the ListenLoop
            thread = new Thread(ListenLoop);
            thread.Start();
        }

        // Stop the server. Will disconnect all clients and shutdown networking.
        public void Stop()
        {
            foreach( TcpClient client in clients)
            {
                client.ReleaseServer();
                client.Disconnect();
            }

            clients.Clear();

            if (thread != null)
            {
                thread.Abort();
                thread = null;
            }

            if(listener != null)
            {
                listener.Close();
                listener = null;
                Debug.Log("Server stopped.");
            }
        }

        // When GameObject is deleted..
        void OnDestroy()
        {
            Stop();
        }

        // SendData will broadcast send to all peers
        public void SendData(byte[] data, TcpClient skip = null)
        {
            foreach (TcpClient client in clients)
            {
                if (client == skip)
                    continue;

                client.SendData(data);

                if(debug)
                {
                    Debug.Log("sent: " + System.Text.Encoding.Default.GetString(data));
                }
            }
        }

        // Remove reference to TcpClient
        public void RemoveClient(TcpClient client)
        {
            clients.Remove(client);
        }

        public List<TcpClient> GetClients()
        {
            return clients;
        }

        public void Update()
        {
            lock (_locker)
            {
                // Because we might be creating GameObjects we need this callback to happen in the main
                // thread context. So we queue new sockets and then create their TcpClients from here.
                if (new_clients.Count > 0)
                {
                    if (onClientConntedCB != null)
                    {
                        foreach (Socket handler in new_clients)
                        {
                            TcpClient client = onClientConntedCB.Invoke();

                            if (client != null)
                            {
                                if(client.OnServerAccept(handler, this))
                                {
                                    clients.Add(client);
                                    client.SetDebug(debug);
                                    client.ClientFinishedConnect();
                                }
                            }
                        }
                    }

                    new_clients.Clear();
                }
            }

            //Poll for dropped connection.
            foreach(TcpClient client in clients)
            {
                if(client.IsDropped())
                {
                    onClientDisconntedCB.Invoke(client);
                }
            }
        }

        // Start listening for connections
        private void Bind(string ip, int port)
        {
            IPAddress ipAddress = IPAddress.Parse(ip);
            int originalPort = port;
            int currentPort = port;
            bool bound = false;
            const int maxAttempts = 100; // Prevent infinite loop
            int attempts = 0;

            while (!bound && attempts < maxAttempts)
            {
                try
                {
                    IPEndPoint localEndPoint = new IPEndPoint(ipAddress, currentPort);

                    // Create a TCP/IP socket.  
                    listener = new Socket(ipAddress.AddressFamily,
                        SocketType.Stream, ProtocolType.Tcp);

                    //Bind to address
                    listener.Bind(localEndPoint);
                    listener.Listen(100);

                    bound = true;

                    if (currentPort != originalPort)
                    {
                        Debug.Log($"Port {originalPort} was in use");
                        // Update GlobalState with the actual bound port
                        GlobalState.port = currentPort;
                        boundPort = currentPort;
                    }
                    Debug.Log("Simulation Server Listening on: " + ip + ":" + GlobalState.port.ToString());
                }
                catch (SocketException e)
                {
                    // Check if the error is due to address already in use
                    if (e.SocketErrorCode == SocketError.AddressAlreadyInUse)
                    {
                        attempts++;
                        currentPort++;
                        if (listener != null)
                        {
                            listener.Close();
                            listener = null;
                        }
                        if (debug)
                        {
                            Debug.Log($"Port {currentPort - 1} is in use, trying {currentPort}...");
                        }
                    }
                    else
                    {
                        // Re-throw if it's a different socket error
                        throw;
                    }
                }
            }

            if (!bound)
            {
                throw new Exception($"Could not find an available port after {maxAttempts} attempts starting from port {originalPort}");
            }
        }

        // Thread loop to wait for new connections
        private void ListenLoop()
        {
            while(true)
            {
                // Set the event to non-signaled state.  
                allDone.Reset();

                // Start an asynchronous socket to listen for connections.  
                Debug.Log("Waiting for a connection...");
                listener.BeginAccept(
                    new AsyncCallback(AcceptCallback),
                    listener);

                // Wait until a connection is made before continuing.  
                allDone.WaitOne();
            }
        }

        // Callback to handle new connections
        private void AcceptCallback(IAsyncResult ar)
        {
            // Signal the main thread to continue.  
            allDone.Set();

            // Get the socket that handles the client request.  
            Socket listener = (Socket)ar.AsyncState;

            try
            {
                Socket handler = listener.EndAccept(ar);

                Debug.Log("client connected.");

                lock (_locker)
                {
                    // Add clients to this new_clients list.
                    // They will get a onClientConntedCB later on in the Update method.
                    new_clients.Add(handler);
                }
            }
            catch(SocketException e)
            {
                Debug.LogError(e.ToString());
            }
            
        }
    }

}