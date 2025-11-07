using UnityEngine;
using UnityEngine.EventSystems;

[DefaultExecutionOrder(-1000)] // run very early
public class EventSystemDeduplicator : MonoBehaviour
{
    void Awake()
    {
        // Find all EventSystems in the loaded scenes
        EventSystem[] systems = GameObject.FindObjectsByType<EventSystem>(FindObjectsSortMode.None);
        if (systems == null || systems.Length <= 1)
            return;

        // Keep the first one we found and remove the rest
        // Prefer an EventSystem that is part of a DontDestroyOnLoad root if present
        EventSystem keep = systems[0];
        foreach (var s in systems)
        {
            if (s == null) continue;
            if (keep == null)
            {
                keep = s;
                continue;
            }
            // If the candidate is on a root marked DontDestroyOnLoad, prefer it
            bool sPersistent = s.gameObject.scene.rootCount == 0; // approximate check

            if (sPersistent && keep != s)
            {
                // prefer persistent
                GameObject toRemove = keep.gameObject;
                keep = s;
                if (toRemove != null)
                {
                    Debug.Log("EventSystemDeduplicator: Removing extra EventSystem on '" + toRemove.name + "'");
                    GameObject.Destroy(toRemove);
                }
                continue;
            }
            if (s != keep)
            {
                Debug.Log("EventSystemDeduplicator: Removing extra EventSystem on '" + s.gameObject.name + "'");
                GameObject.Destroy(s.gameObject);
            }
        }
    }
}
