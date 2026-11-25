using System;
using System.Collections.Generic;
using System.IO;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.InputSystem;
using Vuforia;

/*
 *  YOLO (OBB) Inference Script
 *  ===========================
 *
 *  Adapted from the Unity "inference-engine-yolo" sample for the rubik-yolo model,
 *  which is a YOLOv8n-OBB (oriented bounding boxes) model.
 *
 *  Raw model output: (1, 4 + numClasses + 1, 8400)
 *      rows [0..4)              : cx, cy, w, h   (pixels, in the 640x640 letterboxed input)
 *      rows [4..4+numClasses)   : class scores (already sigmoid'ed)
 *      row  [4+numClasses]      : rotation angle (radians, clockwise in image space)
 *
 *  A standard (non-OBB) YOLO output is (1, 4 + numClasses, 8400): set "isOBB" to false for it.
 *
 *  Place this script on a camera in a test scene and fill the fields according to the tooltips.
 */

public class RunYOLO : MonoBehaviour
{
    [Tooltip("Drag a YOLO model .onnx file here")]
    public ModelAsset modelAsset;

    [Tooltip("Drag the classes.txt here (one class per line, same order as data.yaml)")]
    public TextAsset classesAsset;

    [Tooltip("Create a Raw Image in the scene and link it here")]
    public RawImage displayImage;

    [Tooltip("Drag a border box texture here")]
    public Texture2D borderTexture;

    [Tooltip("Select an appropriate font for the labels")]
    public Font font;

    [Tooltip("True for a YOLO-OBB model (extra angle channel at the end of the output)")]
    public bool isOBB = true;

    [Tooltip("Backend used to run the model")]
    public BackendType backend = BackendType.GPUCompute;

    [Tooltip("Log the number of detections and the best score once per second")]
    public bool logDiagnostics = true;

    private Transform displayLocation;
    private Worker worker;
    private string[] labels;
    private RenderTexture targetRT;   // 640x640 letterboxed model input
    private RenderTexture resizeRT;   // video frame resized to fit inside 640x640
    private Sprite borderSprite;

    //Image size for the model
    private const int imageWidth = 640;
    private const int imageHeight = 640;

    // Ultralytics pads with gray (114, 114, 114)
    static readonly Color letterboxColor = new Color(114f / 255f, 114f / 255f, 114f / 255f, 1f);

    // Pour utiliser la camera Vuforia
    Texture2D camTex;
    float lbScale;
    Vector2Int lbPad;

    List<GameObject> boxPool = new();

    [Tooltip("Intersection over union threshold used for non-maximum suppression")]
    [SerializeField, Range(0, 1)]
    float iouThreshold = 0.5f;

    [Tooltip("Confidence score threshold used for non-maximum suppression")]
    [SerializeField, Range(0, 1)]
    float scoreThreshold = 0.4f;

    Tensor<float> centersToCorners;
    float nextLogTime;

    //bounding box data
    public struct BoundingBox
    {
        public float centerX;
        public float centerY;
        public float width;
        public float height;
        public float angle;     // radians, clockwise in image space
        public float score;
        public int classID;
        public string label;
    }

    void Start()
    {
        Application.targetFrameRate = 60;
        Screen.orientation = ScreenOrientation.LandscapeLeft;

        //Parse neural net labels. Handles CRLF line endings and a trailing empty line.
        labels = classesAsset.text.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
        for (int i = 0; i < labels.Length; i++) labels[i] = labels[i].Trim();

        LoadModel();

        targetRT = new RenderTexture(imageWidth, imageHeight, 0);

        displayLocation = displayImage.transform;

        SetupInput();

        borderSprite = Sprite.Create(borderTexture, new Rect(0, 0, borderTexture.width, borderTexture.height), new Vector2(borderTexture.width / 2, borderTexture.height / 2));
    }

    void LoadModel()
    {
        var model1 = ModelLoader.Load(modelAsset);
        int numClasses = labels.Length;

        ValidateOutputShape(model1, numClasses);

        centersToCorners = new Tensor<float>(new TensorShape(4, 4),
        new float[]
        {
                    1,      0,      1,      0,
                    0,      1,      0,      1,
                    -0.5f,  0,      0.5f,   0,
                    0,      -0.5f,  0,      0.5f
        });

        // Post-processing added on top of the raw model:
        // split the output into boxes / class scores / angle, then run NMS.
        var graph = new FunctionalGraph();
        var inputs = graph.AddInputs(model1);
        var modelOutput = Functional.Forward(model1, inputs)[0];                         //shape=(1, 4+nc(+1), 8400)
        var boxCoords = modelOutput[0, 0..4, ..].Transpose(0, 1);                         //shape=(8400,4)
        // Only the class rows: the angle row must NOT be part of the score max/argmax.
        var allScores = modelOutput[0, 4..(4 + numClasses), ..];                          //shape=(nc,8400)
        var scores = Functional.ReduceMax(allScores, 0);                                  //shape=(8400)
        var classIDs = Functional.ArgMax(allScores, 0);                                   //shape=(8400)
        var boxCorners = Functional.MatMul(boxCoords, Functional.Constant(centersToCorners)); //shape=(8400,4)
        // NMS is axis-aligned: an approximation for oriented boxes, good enough for stickers/faces.
        var indices = Functional.NMS(boxCorners, scores, iouThreshold, scoreThreshold);  //shape=(N)
        var coords = Functional.IndexSelect(boxCoords, 0, indices);                       //shape=(N,4)
        var labelIDs = Functional.IndexSelect(classIDs, 0, indices);                      //shape=(N)
        var keptScores = Functional.IndexSelect(scores, 0, indices);                      //shape=(N)

        if (isOBB)
        {
            var angles = modelOutput[0, 4 + numClasses, ..];                              //shape=(8400)
            var keptAngles = Functional.IndexSelect(angles, 0, indices);                  //shape=(N)
            worker = new Worker(graph.Compile(coords, labelIDs, keptScores, keptAngles), backend);
        }
        else
        {
            worker = new Worker(graph.Compile(coords, labelIDs, keptScores), backend);
        }
    }

    // Runs the raw model once on a blank image to check that classes.txt / isOBB match the model output.
    void ValidateOutputShape(Model model, int numClasses)
    {
        using var probe = new Worker(model, backend);
        using var blank = new Tensor<float>(new TensorShape(1, 3, imageHeight, imageWidth));
        probe.Schedule(blank);
        var shape = probe.PeekOutput().shape;
        int expected = 4 + numClasses + (isOBB ? 1 : 0);
        if (shape.rank != 3 || shape[1] != expected)
            Debug.LogError($"[RunYOLO] Model output shape {shape} does not match {numClasses} classes " +
                           $"(isOBB={isOBB}, expected (1, {expected}, N)). Check classes.txt and the isOBB flag.");
        else
            Debug.Log($"[RunYOLO] Model output {shape}, {numClasses} classes, OBB={isOBB}.");
    }

    void SetupInput()
    {
        // // Version avec vidéo
        // video = gameObject.AddComponent<VideoPlayer>();
        // video.renderMode = VideoRenderMode.APIOnly;
        // video.source = VideoSource.Url;
        // video.url = Path.Join(Application.streamingAssetsPath, videoFilename);
        // video.isLooping = true;
        // video.Play();

        // Version avec caméra Vuforia
        VuforiaApplication.Instance.OnVuforiaStarted += () => VuforiaBehaviour.Instance.CameraDevice.SetFrameFormat(PixelFormat.RGB888, true);
    }

    private void Update()
    {
        ExecuteML();

        if (Keyboard.current != null && Keyboard.current.escapeKey.wasPressedThisFrame)
        {
            Application.Quit();
        }
    }

    // Letterbox like ultralytics: keep the aspect ratio, fit into 640x640, pad with gray.
    // Returns the scale and padding so boxes can be mapped back to the source image later.
    public void Letterbox(Texture source, out float scale, out Vector2Int pad)
    {
        scale = Mathf.Min((float)imageWidth / source.width, (float)imageHeight / source.height);
        int w = Mathf.Clamp(Mathf.RoundToInt(source.width * scale), 1, imageWidth);
        int h = Mathf.Clamp(Mathf.RoundToInt(source.height * scale), 1, imageHeight);
        pad = new Vector2Int((imageWidth - w) / 2, (imageHeight - h) / 2);

        if (resizeRT == null || resizeRT.width != w || resizeRT.height != h)
        {
            if (resizeRT != null) resizeRT.Release();
            resizeRT = new RenderTexture(w, h, 0, targetRT.format);
        }
        Graphics.Blit(source, resizeRT);

        var prev = RenderTexture.active;
        RenderTexture.active = targetRT;
        GL.Clear(false, true, letterboxColor);
        RenderTexture.active = prev;

        // The padding is symmetric, so the bottom-left origin of CopyTexture does not matter.
        Graphics.CopyTexture(resizeRT, 0, 0, 0, 0, w, h, targetRT, 0, 0, pad.x, pad.y);
    }

    public void ExecuteML()
    {
        ClearAnnotations();

        // // Version avec vidéo
        // if (video && video.texture)
        // {
        //     Letterbox(video.texture, out _, out _);
        //     displayImage.texture = targetRT;
        // }
        // else return;

        var img = VuforiaBehaviour.Instance.CameraDevice.GetCameraImage(PixelFormat.RGB888);
        if (img == null) return;   // Vuforia not started yet

        if (camTex == null || camTex.width != img.Width || camTex.height != img.Height)
            camTex = new Texture2D(img.Width, img.Height, TextureFormat.RGB24, false);
        img.CopyToTexture(camTex, false);

        Letterbox(camTex, out lbScale, out lbPad);   // keep these to map boxes back to camera pixels
        displayImage.texture = targetRT;

        using Tensor<float> inputTensor = new Tensor<float>(new TensorShape(1, 3, imageHeight, imageWidth));
        TextureConverter.ToTensor(targetRT, inputTensor, default);
        worker.Schedule(inputTensor);

        using var output = (worker.PeekOutput("output_0") as Tensor<float>).ReadbackAndClone();
        using var labelIDs = (worker.PeekOutput("output_1") as Tensor<int>).ReadbackAndClone();
        using var scores = (worker.PeekOutput("output_2") as Tensor<float>).ReadbackAndClone();
        using var angles = isOBB ? (worker.PeekOutput("output_3") as Tensor<float>).ReadbackAndClone() : null;

        float displayWidth = displayImage.rectTransform.rect.width;
        float displayHeight = displayImage.rectTransform.rect.height;

        float scaleX = displayWidth / imageWidth;
        float scaleY = displayHeight / imageHeight;

        int boxesFound = output.shape[0];
        float bestScore = 0;
        //Draw the bounding boxes
        for (int n = 0; n < Mathf.Min(boxesFound, 200); n++)
        {
            int id = labelIDs[n];
            var box = new BoundingBox
            {
                centerX = output[n, 0] * scaleX - displayWidth / 2,
                centerY = output[n, 1] * scaleY - displayHeight / 2,
                width = output[n, 2] * scaleX,
                height = output[n, 3] * scaleY,
                angle = angles != null ? angles[n] : 0f,
                score = scores[n],
                classID = id,
                label = id >= 0 && id < labels.Length ? labels[id] : $"class {id}",
            };
            bestScore = Mathf.Max(bestScore, box.score);
            DrawBox(box, n, displayHeight * 0.03f);
        }

        if (logDiagnostics && Time.time >= nextLogTime)
        {
            nextLogTime = Time.time + 1f;
            Debug.Log($"[RunYOLO] {boxesFound} detections, best score {bestScore:0.00}");
        }
    }

    public static Color ClassColor(string label) => label switch
    {
        "Blue" => new Color(0.2f, 0.4f, 1f),
        "Green" => Color.green,
        "Orange" => new Color(1f, 0.55f, 0f),
        "Red" => Color.red,
        "White" => Color.white,
        "Yellow" => Color.yellow,
        _ => Color.magenta,     // cube face / others
    };

    public void DrawBox(BoundingBox box, int id, float fontSize)
    {
        //Create the bounding box graphic or get from pool
        GameObject panel;
        if (id < boxPool.Count)
        {
            panel = boxPool[id];
            panel.SetActive(true);
        }
        else
        {
            panel = CreateNewBox(Color.yellow);
        }
        //Set box position (UI y axis points up, image y axis points down)
        panel.transform.localPosition = new Vector3(box.centerX, -box.centerY);
        // Angle is clockwise in image space -> negative Z rotation in UI space
        panel.transform.localRotation = Quaternion.Euler(0, 0, -box.angle * Mathf.Rad2Deg);

        //Set box size
        RectTransform rt = panel.GetComponent<RectTransform>();
        rt.sizeDelta = new Vector2(box.width, box.height);

        var color = ClassColor(box.label);
        panel.GetComponent<UnityEngine.UI.Image>().color = color;

        //Set label text
        var label = panel.GetComponentInChildren<Text>();
        label.text = $"{box.label} {box.score:0.00}";
        label.color = color;
        label.fontSize = Mathf.Max(8, (int)fontSize);
    }

    public GameObject CreateNewBox(Color color)
    {
        //Create the box and set image

        var panel = new GameObject("ObjectBox");
        panel.AddComponent<CanvasRenderer>();
        UnityEngine.UI.Image img = panel.AddComponent< UnityEngine.UI.Image>();
        img.color = color;
        img.sprite = borderSprite;
        img.type = UnityEngine.UI.Image.Type.Sliced;
        panel.transform.SetParent(displayLocation, false);

        //Create the label

        var text = new GameObject("ObjectLabel");
        text.AddComponent<CanvasRenderer>();
        text.transform.SetParent(panel.transform, false);
        Text txt = text.AddComponent<Text>();
        txt.font = font;
        txt.color = color;
        txt.fontSize = 40;
        txt.horizontalOverflow = HorizontalWrapMode.Overflow;

        RectTransform rt2 = text.GetComponent<RectTransform>();
        rt2.offsetMin = new Vector2(20, rt2.offsetMin.y);
        rt2.offsetMax = new Vector2(0, rt2.offsetMax.y);
        rt2.offsetMin = new Vector2(rt2.offsetMin.x, 0);
        rt2.offsetMax = new Vector2(rt2.offsetMax.x, 30);
        rt2.anchorMin = new Vector2(0, 0);
        rt2.anchorMax = new Vector2(1, 1);

        boxPool.Add(panel);
        return panel;
    }

    public void ClearAnnotations()
    {
        foreach (var box in boxPool)
        {
            box.SetActive(false);
        }
    }

    void OnDestroy()
    {
        centersToCorners?.Dispose();
        worker?.Dispose();
        if (resizeRT != null) resizeRT.Release();
        if (targetRT != null) targetRT.Release();
    }
}
