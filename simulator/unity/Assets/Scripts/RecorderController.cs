using UnityEngine;
using System.Collections;
using System.Diagnostics;
using System.IO;

public class RecorderController : MonoBehaviour
{
    private Process ffmpegProcess;
    private Texture2D screenTexture;
    private bool isRecording = false;
    private int width;
    private int height;
    private int frameRate = 30; // Target capture frame rate

    // Buffer for raw image data
    private byte[] rawImageBytes;

    public void StartRecording(string filePath)
    {
        if (isRecording)
        {
            UnityEngine.Debug.LogWarning("RecorderController: Already recording.");
            return;
        }

        width = Screen.width;
        height = Screen.height;
        
        // Ensure even dimensions for ffmpeg
        if (width % 2 != 0) width--;
        if (height % 2 != 0) height--;

        UnityEngine.Debug.Log($"RecorderController: Starting recording to {filePath} ({width}x{height} @ {frameRate}fps)");

        // Ensure directory exists
        string dir = Path.GetDirectoryName(filePath);
        if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
        {
            Directory.CreateDirectory(dir);
        }

        // Initialize texture
        screenTexture = new Texture2D(width, height, TextureFormat.RGB24, false);
        
        // Start FFmpeg process
        StartFFmpegProcess(filePath);

        if (ffmpegProcess != null && !ffmpegProcess.HasExited)
        {
            isRecording = true;
            StartCoroutine(CaptureFrames());
        }
    }

    private void StartFFmpegProcess(string outputPath)
    {
        try
        {
            string ffmpegPath = "ffmpeg"; // Assume in PATH
            
            // Arguments for raw video input from pipe
            // -f rawvideo: input format
            // -vcodec rawvideo: input codec
            // -s: resolution
            // -r: frame rate
            // -pix_fmt rgb24: pixel format from Unity ReadPixels
            // -i -: input from stdin
            // -c:v libx264: output codec
            // -preset ultrafast: encoding speed (sacrifice compression for speed)
            // -qp 0: lossless (or -crf 18 for high quality)
            // -pix_fmt yuv420p: output pixel format (compatible with most players)
            // -y: overwrite output
            string args = $"-f rawvideo -vcodec rawvideo -s {width}x{height} -r {frameRate} -pix_fmt rgb24 -i - -c:v libx264 -preset ultrafast -crf 23 -pix_fmt yuv420p -y \"{outputPath}\"";

            ProcessStartInfo startInfo = new ProcessStartInfo
            {
                FileName = ffmpegPath,
                Arguments = args,
                UseShellExecute = false,
                RedirectStandardInput = true,
                RedirectStandardError = true, // Capture stderr for debugging
                CreateNoWindow = true
            };

            ffmpegProcess = new Process();
            ffmpegProcess.StartInfo = startInfo;
            ffmpegProcess.ErrorDataReceived += (sender, e) => 
            {
                if (!string.IsNullOrEmpty(e.Data))
                {
                    // UnityEngine.Debug.Log($"FFmpeg: {e.Data}"); // Uncomment for verbose ffmpeg logs
                }
            };
            
            ffmpegProcess.Start();
            ffmpegProcess.BeginErrorReadLine();
            
            UnityEngine.Debug.Log("RecorderController: FFmpeg process started.");
        }
        catch (System.Exception e)
        {
            UnityEngine.Debug.LogError($"RecorderController: Failed to start FFmpeg: {e.Message}");
            isRecording = false;
        }
    }

    private IEnumerator CaptureFrames()
    {
        WaitForEndOfFrame wait = new WaitForEndOfFrame();
        float frameInterval = 1.0f / frameRate;
        float nextFrameTime = Time.time;

        while (isRecording && ffmpegProcess != null && !ffmpegProcess.HasExited)
        {
            yield return wait;

            if (Time.time >= nextFrameTime)
            {
                try
                {
                    // Capture screen
                    // ReadPixels reads from the active RenderTexture (screen buffer after rendering)
                    screenTexture.ReadPixels(new Rect(0, 0, width, height), 0, 0);
                    // screenTexture.Apply(); // Not strictly needed for GetRawTextureData

                    // Get raw bytes
                    rawImageBytes = screenTexture.GetRawTextureData();

                    // Write to ffmpeg stdin
                    ffmpegProcess.StandardInput.BaseStream.Write(rawImageBytes, 0, rawImageBytes.Length);
                    ffmpegProcess.StandardInput.BaseStream.Flush();
                    
                    nextFrameTime = Time.time + frameInterval;
                }
                catch (System.Exception e)
                {
                    UnityEngine.Debug.LogError($"RecorderController: Error capturing frame: {e.Message}");
                    StopRecording();
                }
            }
        }
    }

    public void StopRecording()
    {
        if (!isRecording) return;

        isRecording = false;
        StopAllCoroutines();

        if (ffmpegProcess != null)
        {
            try
            {
                UnityEngine.Debug.Log("RecorderController: Stopping FFmpeg...");
                ffmpegProcess.StandardInput.Close(); // Close stdin to signal EOF
                ffmpegProcess.WaitForExit(2000); // Wait up to 2 seconds
                
                if (!ffmpegProcess.HasExited)
                {
                    ffmpegProcess.Kill();
                }
                
                ffmpegProcess.Close();
                ffmpegProcess = null;
                UnityEngine.Debug.Log("RecorderController: Recording stopped.");
            }
            catch (System.Exception e)
            {
                UnityEngine.Debug.LogError($"RecorderController: Error stopping FFmpeg: {e.Message}");
            }
        }

        if (screenTexture != null)
        {
            Destroy(screenTexture);
            screenTexture = null;
        }
    }

    void OnDestroy()
    {
        StopRecording();
    }
}
