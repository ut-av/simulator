using UnityEngine;

public class DontDestroy : MonoBehaviour
{
    void Awake()
    {
        // Ensure we call DontDestroyOnLoad on the root GameObject to avoid Unity warning
        GameObject root = this.gameObject.transform.root.gameObject;
        DontDestroyOnLoad(root);
    }
}