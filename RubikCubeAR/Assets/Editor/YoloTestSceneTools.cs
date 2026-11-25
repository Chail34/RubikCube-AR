using System.IO;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

// Menu "Tools/YOLO": builds the YOLO test scene (no Vuforia) and cleans RunYOLO out of the Vuforia scene.
public static class YoloTestSceneTools
{
    const string TestScenePath = "Assets/Scenes/YoloTest.unity";
    const string VuforiaScenePath = "Assets/Scenes/SampleScene.unity";
    const string ModelPath = "Assets/ONNX/best.onnx";
    const string ClassesPath = "Assets/ONNX/classes.txt";
    const string FontPath = "Assets/TextMesh Pro/Fonts/LiberationSans.ttf";

    [MenuItem("Tools/YOLO/Create Test Scene")]
    public static void CreateTestScene()
    {
        if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) return;
        if (File.Exists(TestScenePath) &&
            !EditorUtility.DisplayDialog("YOLO", $"{TestScenePath} already exists. Overwrite?", "Overwrite", "Cancel"))
            return;

        var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

        // Camera (RunYOLO lives here, as in the original sample)
        var camGo = new GameObject("Main Camera") { tag = "MainCamera" };
        var cam = camGo.AddComponent<Camera>();
        cam.clearFlags = CameraClearFlags.SolidColor;
        cam.backgroundColor = Color.black;
        camGo.AddComponent<AudioListener>();

        // Overlay canvas, scaled from a 1920x1080 reference
        var canvasGo = new GameObject("Canvas", typeof(Canvas), typeof(CanvasScaler), typeof(GraphicRaycaster));
        canvasGo.GetComponent<Canvas>().renderMode = RenderMode.ScreenSpaceOverlay;
        var scaler = canvasGo.GetComponent<CanvasScaler>();
        scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
        scaler.referenceResolution = new Vector2(1920, 1080);
        scaler.matchWidthOrHeight = 1f;

        // Square RawImage showing the 640x640 letterboxed model input; boxes are drawn as its children
        var imgGo = new GameObject("ModelInputImage", typeof(RectTransform), typeof(CanvasRenderer), typeof(RawImage));
        imgGo.transform.SetParent(canvasGo.transform, false);
        var rt = imgGo.GetComponent<RectTransform>();
        rt.anchorMin = rt.anchorMax = rt.pivot = new Vector2(0.5f, 0.5f);
        rt.sizeDelta = new Vector2(1000, 1000);
        rt.anchoredPosition = Vector2.zero;
        var rawImage = imgGo.GetComponent<RawImage>();
        rawImage.raycastTarget = false;

        var yolo = camGo.AddComponent<RunYOLO>();
        yolo.modelAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelPath);
        yolo.classesAsset = AssetDatabase.LoadAssetAtPath<TextAsset>(ClassesPath);
        yolo.displayImage = rawImage;
        yolo.borderTexture = null; // generated frame
        yolo.font = AssetDatabase.LoadAssetAtPath<Font>(FontPath) ?? Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        yolo.isOBB = true;
        yolo.backend = BackendType.GPUCompute;

        if (yolo.modelAsset == null) Debug.LogError($"[YOLO] Model not found at {ModelPath}");
        if (yolo.classesAsset == null) Debug.LogError($"[YOLO] classes.txt not found at {ClassesPath}");

        EditorSceneManager.SaveScene(scene, TestScenePath);
        Debug.Log($"[YOLO] Test scene created: {TestScenePath}");
    }

    [MenuItem("Tools/YOLO/Remove RunYOLO from Vuforia Scene")]
    public static void RemoveFromVuforiaScene()
    {
        if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) return;

        Scene scene = EditorSceneManager.OpenScene(VuforiaScenePath, OpenSceneMode.Single);
        int removed = 0;
        foreach (var root in scene.GetRootGameObjects())
            foreach (var comp in root.GetComponentsInChildren<RunYOLO>(true))
            {
                Debug.Log($"[YOLO] Removing RunYOLO from '{comp.gameObject.name}' in {VuforiaScenePath}");
                Object.DestroyImmediate(comp);
                removed++;
            }

        if (removed > 0)
        {
            EditorSceneManager.MarkSceneDirty(scene);
            EditorSceneManager.SaveScene(scene);
        }
        EditorUtility.DisplayDialog("YOLO", $"{removed} RunYOLO component(s) removed from {VuforiaScenePath}.", "OK");
    }
}
