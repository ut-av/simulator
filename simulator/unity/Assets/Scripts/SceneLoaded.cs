using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SceneLoaded : MonoBehaviour
{
    public bool inFrontEnd = false;

    // Start is called before the first frame update
    void Start()
    {
        // Notify unified SandboxServer
        SandboxServer server = GameObject.FindFirstObjectByType<SandboxServer>();
        if (server != null)
        {
            server.OnSceneLoaded(inFrontEnd);
        }
    }    
}
