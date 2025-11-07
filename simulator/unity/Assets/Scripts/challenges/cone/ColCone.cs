using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ColCone : MonoBehaviour
{
    public int index = 0;
    string target = "body";
    public float penalty = 1;

    void OnTriggerEnter(Collider col)
    {
        if (col.gameObject.name != target) { return; }

        Transform parent = col.transform.parent.parent;
        if (parent == null) { return; }
        string carName = parent.name;

        Timer[] status = parent.gameObject.GetComponentsInChildren<Timer>();
        foreach (Timer t in status)
        {
            Debug.Log("Collision with Challenge");
            t.OnCollideChallenge(penalty);
        }
    }
}
