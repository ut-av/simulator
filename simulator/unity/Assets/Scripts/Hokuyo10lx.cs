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
    public bool visualizeScan = false;
    public bool logScanSent = false;

    private float[] ranges;
    private int numRays;

    // Visualization
    private Material lineMaterial;
    private Vector3[] debugRayEnds;
    private bool[] debugRayHits;

    void Start()
    {
        numRays = Mathf.RoundToInt((maxAngle - minAngle) / angleIncrement) + 1;
        ranges = new float[numRays];
        debugRayEnds = new Vector3[numRays];
        debugRayHits = new bool[numRays];
    }

    public JSONObject GetOutputAsJson()
    {
        if (ranges == null || ranges.Length != numRays)
        {
            numRays = Mathf.RoundToInt((maxAngle - minAngle) / angleIncrement) + 1;
            ranges = new float[numRays];
            debugRayEnds = new Vector3[numRays];
            debugRayHits = new bool[numRays];
        }

        // Perform scan
        Scan();

        if (logScanSent)
        {
            float sum = 0;
            int count = 0;
            for (int i = 0; i < ranges.Length; i++)
            {
                if (!float.IsInfinity(ranges[i]))
                {
                    sum += ranges[i];
                    count++;
                }
            }
            float avg = count > 0 ? sum / count : 0;
            Debug.Log($"Scan sent at {Time.time} with numRays {numRays} with average range of {avg:F2}");
        }

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

                if (visualizeScan)
                {
                    debugRayEnds[i] = pos + worldDir * ranges[i];
                    debugRayHits[i] = true;
                }
            }
            else
            {
                ranges[i] = float.PositiveInfinity;
                if (visualizeScan)
                {
                    debugRayEnds[i] = pos + worldDir * maxRange;
                    debugRayHits[i] = false;
                }
            }
        }
    }

    void CreateLineMaterial()
    {
        if (!lineMaterial)
        {
            // Unity has a built-in shader that is useful for drawing
            // simple colored things.
            Shader shader = Shader.Find("Hidden/Internal-Colored");
            lineMaterial = new Material(shader);
            lineMaterial.hideFlags = HideFlags.HideAndDontSave;
            // Turn on alpha blending
            lineMaterial.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
            lineMaterial.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
            // Turn backface culling off
            lineMaterial.SetInt("_Cull", (int)UnityEngine.Rendering.CullMode.Off);
            // Turn off depth writes
            lineMaterial.SetInt("_ZWrite", 0);
        }
    }

    void OnRenderObject()
    {
        if (visualizeScan && debugRayEnds != null)
        {
            CreateLineMaterial();
            lineMaterial.SetPass(0);

            GL.PushMatrix();
            // Set transformation matrix for drawing to match our transform
            // We already calculated world positions, so we can use identity or handle it differently.
            // Since we stored world positions in debugRayEnds, we don't need to apply the object's transform again if we use MultMatrix(Identity).
            // However, GL.Begin(GL.LINES) draws in world space if we don't set a matrix? 
            // Actually, usually GL draws in immediate mode. 
            // Let's use world space coordinates we calculated.
            GL.MultMatrix(Matrix4x4.identity);

            GL.Begin(GL.LINES);
            
            Vector3 pos = transform.position;

            for (int i = 0; i < numRays; i++)
            {
                if (debugRayHits[i])
                {
                    GL.Color(Color.green);
                }
                else
                {
                    GL.Color(Color.red);
                }
                
                GL.Vertex(pos);
                GL.Vertex(debugRayEnds[i]);
            }
            
            GL.End();
            GL.PopMatrix();
        }
    }
}
