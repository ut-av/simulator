using System.Collections;
using System.Collections.Generic;
using Unity.Collections;
using UnityEngine;

public class CameraSensor : MonoBehaviour {

	public Camera sensorCam;
	public int width = 256;
	public int height = 256;
	public int depth = 3;
	public string img_enc = "JPG"; //accepts JPG, PNG, TGA
	Texture2D tex;
	RenderTexture ren;
	Rect ImageRect;

	public int compressionQuality = 75;
	public int superSampling = 4;
	public int antiAliasing = 1;

	public void SetConfig(float fov, float offset_x, float offset_y, float offset_z, float rot_x, float rot_y, float rot_z, int img_w, int img_h, int img_d, string _img_enc, int _compressionQuality, int _superSampling, int _antiAliasing)
	{
		Debug.Log($"[CameraSensor] SetConfig: quality={_compressionQuality}, w={img_w}, h={img_h}, super_sampling={_superSampling}, anti_aliasing={_antiAliasing}");
		compressionQuality = _compressionQuality;
		superSampling = _superSampling;
		antiAliasing = _antiAliasing;

		if (img_d != 0)
		{
			depth = img_d;
		}

		if (img_w != 0 && img_h != 0)
		{
			width = img_w;
			height = img_h;
			
			Awake();
		}

		if(_img_enc.Length == 3)
			img_enc = _img_enc;

		if(offset_x != 0.0f || offset_y != 0.0f || offset_z != 0.0f)
			transform.localPosition = new Vector3(offset_x, offset_y, offset_z);

		if (rot_x != 0.0f || rot_y != 0.0f || rot_z != 0.0f)
		{
			transform.localEulerAngles = new Vector3(rot_x, rot_y, rot_z);
		}

		if(fov != 0.0f)
			sensorCam.fieldOfView = fov;
	}

	void Awake()
	{
		if (tex != null)
			Destroy(tex);
		if (ren != null)
			ren.Release();

		Debug.Log($"[CameraSensor] Initialized: Res={width}x{height}, SuperSampling={superSampling}x, AA={antiAliasing}x, Compression={compressionQuality}");
		tex = new Texture2D(width, height, TextureFormat.RGB24, false);
		ren = new RenderTexture(width * superSampling, height * superSampling, 16, RenderTextureFormat.ARGB32);
		ren.antiAliasing = antiAliasing;
		sensorCam.targetTexture = ren;
	}

	Texture2D RTImage() 
	{
		var currentRT = RenderTexture.active;
		RenderTexture.active = sensorCam.targetTexture;

		sensorCam.Render();

		if (superSampling > 1)
		{
			RenderTexture finalRen = RenderTexture.GetTemporary(width, height, 0, RenderTextureFormat.ARGB32);
			Graphics.Blit(ren, finalRen);
			RenderTexture.active = finalRen;
			tex.ReadPixels(new Rect(0, 0, width, height), 0, 0);
			RenderTexture.ReleaseTemporary(finalRen);
		}
		else
		{
			tex.ReadPixels(new Rect(0, 0, width, height), 0, 0);
		}

		if(depth == 1)
		{
			//assumes TextureFormat.RGB24. Convert to grey scale image
			NativeArray<byte> bytes = tex.GetRawTextureData<byte>();
			for (int i=0; i<bytes.Length; i+=3)
			{
				byte gray = (byte)(0.2126f * bytes[i+0] + 0.7152f * bytes[i+1] + 0.0722f * bytes[i+2]);
				bytes[i+2] = bytes[i+1] = bytes[i+0] = gray;
			}
		}

		tex.Apply();
		RenderTexture.active = currentRT;

		return tex;
	}

	public Texture2D GetImage()
	{
		return RTImage();
	}

	public byte[] GetImageBytes()
	{
		if(img_enc == "PNG")
			return GetImage().EncodeToPNG();

		if(img_enc == "TGA")
			return GetImage().EncodeToTGA();
			
		return GetImage().EncodeToJPG(compressionQuality);
	}
}
