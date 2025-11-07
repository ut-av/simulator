using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using System;

/// <summary>
/// Command Line Argument Handler - Parses command line arguments and handles scene loading.
/// </summary>
public class CommandLineArgsHandler : MonoBehaviour
{
    private static bool sceneLoadedFromArgs = false;

    void Awake()
    {
        // Only process command line args once
        if (sceneLoadedFromArgs)
            return;

        string[] args = System.Environment.GetCommandLineArgs();
        string sceneToLoad = null;

        // Parse command line arguments
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--scene" || args[i] == "--environment")
            {
                if (i + 1 < args.Length)
                {
                    sceneToLoad = args[i + 1];
                    Debug.Log($"Command line argument: Loading scene '{sceneToLoad}'");
                }
                else
                {
                    Debug.LogWarning($"Command line argument '{args[i]}' provided but no scene name specified.");
                }
                break;
            }
        }

        // Load the scene if specified
        if (!string.IsNullOrEmpty(sceneToLoad))
        {
            sceneLoadedFromArgs = true;
            StartCoroutine(LoadSceneFromCommandLine(sceneToLoad));
        }
    }

    private IEnumerator LoadSceneFromCommandLine(string sceneName)
    {
        // Wait a frame to ensure everything is initialized
        yield return null;

        Debug.Log($"Loading scene from command line: {sceneName}");

        // Check if scene exists in build settings
        // Try exact match first, then partial match
        string sceneToLoad = null;
        string partialMatch = null;
        
        for (int i = 0; i < SceneManager.sceneCountInBuildSettings; i++)
        {
            string scenePath = SceneUtility.GetScenePathByBuildIndex(i);
            string sceneNameFromPath = System.IO.Path.GetFileNameWithoutExtension(scenePath);
            
            // Exact match (case-insensitive) - highest priority
            if (sceneNameFromPath.Equals(sceneName, StringComparison.OrdinalIgnoreCase))
            {
                sceneToLoad = sceneNameFromPath;
                break; // Found exact match, stop searching
            }
            
            // Partial match (case-insensitive) - store first match but continue searching for exact
            if (partialMatch == null && sceneNameFromPath.IndexOf(sceneName, StringComparison.OrdinalIgnoreCase) >= 0)
            {
                partialMatch = sceneNameFromPath;
            }
        }

        // Use partial match if no exact match found
        if (string.IsNullOrEmpty(sceneToLoad) && !string.IsNullOrEmpty(partialMatch))
        {
            sceneToLoad = partialMatch;
        }

        if (!string.IsNullOrEmpty(sceneToLoad))
        {
            // Set flags similar to what TcpMenuHandler does
            GlobalState.bAutoHideSceneMenu = true;
            GlobalState.bCreateCarWithoutNetworkClient = false;

            Debug.Log($"Loading scene: {sceneToLoad}");
            SceneManager.LoadSceneAsync(sceneToLoad);
        }
        else
        {
            Debug.LogError($"Scene '{sceneName}' not found in build settings. Available scenes:");
            for (int i = 0; i < SceneManager.sceneCountInBuildSettings; i++)
            {
                string scenePath = SceneUtility.GetScenePathByBuildIndex(i);
                string sceneNameFromPath = System.IO.Path.GetFileNameWithoutExtension(scenePath);
                Debug.LogError($"  - {sceneNameFromPath}");
            }
        }
    }
}

