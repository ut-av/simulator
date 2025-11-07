using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using SimpleFileBrowser;

public class SceneLoader : MonoBehaviour
{

    public void LoadMenuScene()
    {
        ShutdownTcpServer();
        SceneManager.LoadSceneAsync("menu");
    }

    public void LoadScene(string scene_name)
    {
        Debug.Log(scene_name);
        ShutdownTcpServer();
        SceneManager.LoadSceneAsync(scene_name);
    }

    public void QuitApplication()
    {
        Application.Quit();
    }

    // Shutdown any existing TCP server and wait for the port to be freed
    private void ShutdownTcpServer()
    {
        // Shutdown all active TCP servers (both TrackServer and MenuServer may exist)
        tk.TcpServer tcpServer = GameObject.FindFirstObjectByType<tk.TcpServer>();
        if (tcpServer != null)
        {
            int boundPort = tcpServer.BoundPort;
            Debug.Log($"Shutting down existing TCP server on port {boundPort}...");
            tcpServer.Stop();
        }
        
        // Also shutdown any TrackServer components that may exist
        TrackServer trackServer = GameObject.FindFirstObjectByType<TrackServer>();
        if (trackServer != null)
        {
            Debug.Log("Shutting down TrackServer...");
            Destroy(trackServer.gameObject);
        }
        
        // Also shutdown any MenuServer components that may exist
        MenuServer menuServer = GameObject.FindFirstObjectByType<MenuServer>();
        if (menuServer != null)
        {
            Debug.Log("Shutting down MenuServer...");
            Destroy(menuServer.gameObject);
        }
    }

    public void SetLogDir()
    {
        // Show a select folder dialog 
        // onSuccess event: print the selected folder's path
        // onCancel event: print "Canceled"
        // Load file/folder: folder, Initial path: default (Documents), Title: "Select Folder", submit button text: "Select"
        FileBrowser.ShowLoadDialog((path) => { OnSetLogDir(path); },
                                        () => { Debug.Log("Canceled"); },
                                        true, null, "Select Log Folder", "Select");
    }

    public void OnSetLogDir(string path)
    {
        Debug.Log("Selected: " + path);
        GlobalState.log_path = path;
    }

    public string[] LoadScenePathsFromFile(string directoryPath)
    {
        List<string> scenePaths = new List<string>();

        // search in the directory
        string[] paths = System.IO.Directory.GetFiles(directoryPath, "*", System.IO.SearchOption.AllDirectories);
        foreach (string path in paths)
        {
            string extension = System.IO.Path.GetExtension(path);
            if (extension != "") { continue; } // get only bundle Assets (no file extension)

            AssetBundle bundle = AssetBundle.LoadFromFile(path);

            if (GlobalState.bundleScenes != null)
            {
                scenePaths.AddRange(bundle.GetAllScenePaths());
                GlobalState.bundleScenes.Add(bundle);
            }

        }
        return scenePaths.ToArray();
    }
}
