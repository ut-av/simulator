using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class JustOne : MonoBehaviour
{
    // Ensure One and only one JustOne Object that is not deleted when scene exits.
    public string label = "NameMe";

    void Awake()
    {
        // Find any other JustOne with the same label. If one already exists, destroy this one.
        JustOne[] all_objs = GameObject.FindObjectsByType<JustOne>(FindObjectsSortMode.None);

        foreach (JustOne obj in all_objs)
        {
            if (obj == this) continue;
            if (obj.label == label)
            {
                // Another instance with the same label exists; destroy this duplicate and exit.
                Debug.LogWarning("JustOne found duplicate instance." + label);
                // Use Destroy so Unity can cleanly remove the GameObject during the lifecycle.
                //GameObject.Destroy(this.gameObject);
                return;
            }
        }

        // DontDestroyOnLoad must be called on a root GameObject. If this component
        // is attached to a child, use the root GameObject to avoid Unity's warning.
        GameObject root = this.gameObject.transform.root.gameObject;

        // Prevent persisting UI EventSystem across scenes which commonly causes the
        // "There can be only one active Event System" error when the next scene also
        // contains an EventSystem. If the root contains an EventSystem, we avoid
        // calling DontDestroyOnLoad so only the scene's EventSystem remains.
        var eventSystem = root.GetComponentInChildren<UnityEngine.EventSystems.EventSystem>(true);
        if (eventSystem != null)
        {
            Debug.Log("JustOne: root contains EventSystem; not calling DontDestroyOnLoad for '" + label + "' to avoid duplicate EventSystem.");
        }
        else
        {
            DontDestroyOnLoad(root);
        }
    }
}