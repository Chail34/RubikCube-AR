using System.IO;
using System;
using System.Collections.Generic;
using UnityEngine;
using Vuforia;
using UnityEngine.UI;
using UnityEngine.InputSystem;

public enum CubeColor { White, Yellow, Red, Orange, Green, Blue, Unknown }

/// Détecte la couleur du sticker central de chaque face tournée vers la caméra.
/// Résultat : Detected[couleur] = axe local du ModelTarget portant cette couleur.
public class CubeFaceDetector : MonoBehaviour
{
    [Tooltip("Le GameObject du ModelTarget Vuforia (PAS le prefab overlay)")]
    public ObserverBehaviour modelTarget;
    public float halfSize = 0.0285f;      // 57 mm / 2 : mesure ton cube
    [Range(0f, 1f)] public float minFacing = 0.4f;
    public int patchRadius = 2;           // moyenne sur (2r+1)² pixels

    public readonly Dictionary<CubeColor, Vector3> Detected = new();

    static readonly Vector3[] axes = { Vector3.right, Vector3.left, Vector3.up,
                                       Vector3.down, Vector3.forward, Vector3.back };
    Texture2D camTex;

    public RawImage image ;

    long counter = 0 ;

    void Start()
    {
        VuforiaApplication.Instance.OnVuforiaStarted += () =>
            VuforiaBehaviour.Instance.CameraDevice.SetFrameFormat(PixelFormat.RGB888, true);
    }

    void Update()
    {
        counter++ ;
        Detected.Clear();
        if (modelTarget == null || modelTarget.TargetStatus.Status != Status.TRACKED) return;

        var img = VuforiaBehaviour.Instance.CameraDevice.GetCameraImage(PixelFormat.RGB888);
        if (img == null) return;
        if (camTex == null || camTex.width != img.Width || camTex.height != img.Height)
            camTex = new Texture2D(img.Width, img.Height, TextureFormat.RGB24, false);
        img.CopyToTexture(camTex, false);

        Transform cube = modelTarget.transform;
        Camera cam = Camera.main;

        foreach (var a in axes)
        {
            Vector3 center = cube.TransformPoint(a * halfSize);
            Vector3 toCam = (cam.transform.position - center).normalized;
            if (Vector3.Dot(cube.TransformDirection(a), toCam) < minFacing) continue;

            // ATTENTION : viewport -> pixels caméra suppose le même cadrage.
            // Vuforia recadre le flux et l'image peut être inversée en Y : à calibrer.
            Vector3 vp = cam.WorldToViewportPoint(center);
            int px = (int)(vp.x * camTex.width);
            int py = (int)(vp.y * camTex.height);

            CubeColor col = Classify(SamplePatch(px, py));
            if (col == CubeColor.Unknown) continue;
            if (Detected.ContainsKey(col)) { Detected.Clear(); return; } // même couleur 2 fois : frame rejetée
            Detected[col] = a;
        }

        // Pour faire une capture de la caméra
        // if (Keyboard.current != null && Keyboard.current.spaceKey.wasPressedThisFrame)
        // {
        //     var bytes = camTex.EncodeToPNG() ;
        //     File.WriteAllBytes("C:/Users/ruzik/Desktop/Cours/RubikCube-AR/RubikCubeAR/Assets/Images/out/"+counter+".png", bytes);
        //     print("===== Capture d'écran effectuée =====");
        // }
    }

    Color SamplePatch(int cx, int cy)
    {
        Color sum = Color.black; int n = 0;
        for (int dx = -patchRadius; dx <= patchRadius; dx++)
        for (int dy = -patchRadius; dy <= patchRadius; dy++)
        {
            int x = cx + dx, y = cy + dy;
            if (x < 0 || y < 0 || x >= camTex.width || y >= camTex.height) continue;
            sum += camTex.GetPixel(x, y); n++;
        }
        return n > 0 ? sum / n : Color.black;
    }

    static CubeColor Classify(Color c)
    {
        Texture2D white = Resources.Load<Texture2D>("Assets/Images/cubeColors/white.png");


        Color.RGBToHSV(c, out float h, out float s, out float v);
        // RGBtoHSL(c, out float h, out float s, out float l);
        if (v < 0.2f) return CubeColor.Unknown;
        if (s < 0.2f) return v > 0.6f ? CubeColor.White : CubeColor.Unknown;
        h *= 360f;
        if (h < 15f || h > 340f) return CubeColor.Red;
        if (h < 40f) return CubeColor.Orange;
        if (h < 70f) return CubeColor.Yellow;
        if (h < 170f) return CubeColor.Green;
        if (h < 260f) return CubeColor.Blue;
        return CubeColor.Unknown;
    }


    static void RGBtoHSL(Color c, out float h, out float s, out float l){

        Color.RGBToHSV(c, out h, out s, out float v);

        l = v*(1-(s/2)) ;

        if(l==1 || l==0){s = 0 ;}
        else{s = (v-l)/Math.Min(l, 1-l) ;}
    }
}