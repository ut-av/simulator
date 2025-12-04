using UnityEngine;
using System.Collections;
using System;
using System.Text;

public class Hokuyo10lx : MonoBehaviour
{
    public float minAngle = -135f;
    public float maxAngle = 135f;
    public float angleIncrement = 0.25f;
    public float maxRange = 30f;
    public float minRange = 0.1f;
    public float noise = 0.01f; // meters sigma

    private float[] ranges;
    private int numRays;

    void Start()
    {
        numRays = Mathf.RoundToInt((maxAngle - minAngle) / angleIncrement) + 1;
        ranges = new float[numRays];
    }

    public JSONObject GetOutputAsJson()
    {
        if (ranges == null || ranges.Length != numRays)
        {
            numRays = Mathf.RoundToInt((maxAngle - minAngle) / angleIncrement) + 1;
            ranges = new float[numRays];
        }

        // Perform scan
        Scan();

        JSONObject json = new JSONObject(JSONObject.Type.OBJECT);
        json.AddField("min_angle", minAngle * Mathf.Deg2Rad);
        json.AddField("max_angle", maxAngle * Mathf.Deg2Rad);
        json.AddField("angle_increment", angleIncrement * Mathf.Deg2Rad);
        json.AddField("range_min", minRange);
        json.AddField("range_max", maxRange);
        
        // Convert ranges to byte array
        byte[] byteArray = new byte[ranges.Length * 4];
        Buffer.BlockCopy(ranges, 0, byteArray, 0, byteArray.Length);
        string base64Ranges = Convert.ToBase64String(byteArray);
        
        json.AddField("ranges", base64Ranges);
        
        return json;
    }

    void Scan()
    {
        Vector3 pos = transform.position;
        
        for (int i = 0; i < numRays; i++)
        {
            float angle = minAngle + i * angleIncrement;
            
            // Calculate direction in local space then transform to world
            // Assumes sensor is mounted with Y up. 
            // Angle is around Y axis. 0 is forward (Z).
            // -135 is right-back? 
            // In Unity: Z is forward, X is right.
            // Rotation around Y: positive is clockwise? No, Unity is left-handed?
            // Quaternion.Euler(0, angle, 0) rotates around Y.
            
            Vector3 localDir = Quaternion.Euler(0, angle, 0) * Vector3.forward;
            Vector3 worldDir = transform.TransformDirection(localDir);
            
            RaycastHit hit;
            if (Physics.Raycast(pos, worldDir, out hit, maxRange))
            {
                float dist = hit.distance;
                // Add noise
                if (noise > 0)
                {
                    dist += UnityEngine.Random.Range(-noise, noise);
                }
                ranges[i] = Mathf.Clamp(dist, minRange, maxRange);
            }
            else
            {
                ranges[i] = float.PositiveInfinity;
            }
        }
    }
}
