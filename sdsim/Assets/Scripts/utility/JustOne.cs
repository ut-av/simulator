using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class JustOne : MonoBehaviour
{
    // Ensure One and only one JustOne Object that is not deleted when scene exits.
    public string label = "NameMe";

    void Awake()
    {
        JustOne[] all_objs = GameObject.FindObjectsByType<JustOne>(FindObjectsSortMode.None);

        List<JustOne> objs = new List<JustOne>();
        
        foreach (JustOne obj in all_objs)
        {
            if (obj.label == label)
            {
                objs.Add(obj);
            }
        }

        foreach (JustOne obj in objs)
        {
            if (obj.label == label && this == obj && objs.Count > 1)
            {
                Debug.Log("JustOne removing instance." + label);
                GameObject.DestroyImmediate(obj.gameObject);
                return;
            }
        }

        // DontDestroyOnLoad must be called on a root GameObject. If this component
        // is attached to a child, use the root GameObject to avoid Unity's warning.
        GameObject root = this.gameObject.transform.root.gameObject;
        DontDestroyOnLoad(root);
    }
}