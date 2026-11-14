using System.Collections.Generic;
using UnityEngine;

/// Réoriente le prefab overlay (enfant du ModelTarget) pour que ses étiquettes
/// correspondent aux couleurs réellement vues par CubeFaceDetector.
public class CubeOverlayCorrection : MonoBehaviour
{
    public CubeFaceDetector detector;
    [Tooltip("Le prefab overlay, enfant direct du ModelTarget, localPosition = 0")]
    public Transform overlay;
    [Tooltip("Nombre de frames consécutives d'accord avant d'appliquer une nouvelle orientation")]
    public int framesToConfirm = 5;

    static readonly Dictionary<CubeColor, Vector3> prefabAxes = new()
    {
        { CubeColor.White,  Vector3.up },
        { CubeColor.Yellow, Vector3.down },
        { CubeColor.Green,  Vector3.forward },
        { CubeColor.Blue,   Vector3.back },
        { CubeColor.Red,    Vector3.right },
        { CubeColor.Orange, Vector3.left },
    };

    Quaternion applied = Quaternion.identity;
    Quaternion candidate = Quaternion.identity;
    int streak;

    void LateUpdate() // après CubeFaceDetector.Update
    {
        if (TryComputeCorrection(detector.Detected, out Quaternion q))
        {
            if (Quaternion.Angle(q, candidate) < 1f) streak++;
            else { candidate = q; streak = 1; }

            if (streak >= framesToConfirm) applied = candidate;
        }
        // Une seule face visible ou frame incohérente : on garde la dernière correction.
        overlay.localRotation = applied;
    }

    /// Cherche R tel que R * axePrefab(couleur) = axeDétecté(couleur) pour toutes les couleurs vues.
    /// Il faut au moins deux faces adjacentes.
    static bool TryComputeCorrection(Dictionary<CubeColor, Vector3> det, out Quaternion q)
    {
        q = Quaternion.identity;
        if (det.Count < 2) return false;

        var keys = new List<CubeColor>(det.Keys);
        CubeColor c1 = keys[0], c2 = keys[1];
        Vector3 p1 = prefabAxes[c1], p2 = prefabAxes[c2];
        Vector3 d1 = det[c1],        d2 = det[c2];

        // Deux faces visibles en même temps sont toujours adjacentes.
        // Si le prefab dit qu'elles sont opposées (ex. blanc + jaune), c'est une erreur de couleur.
        if (Mathf.Abs(Vector3.Dot(p1, p2)) > 0.1f) return false;

        // LookRotation(f, u) envoie Z -> f et Y -> u.
        // Donc L_d * L_p^-1 envoie p1 -> d1 et p2 -> d2.
        q = Quaternion.LookRotation(d1, d2) * Quaternion.Inverse(Quaternion.LookRotation(p1, p2));

        // Vérification sur toutes les faces vues (une 3e face contrôle aussi la chiralité).
        foreach (var k in keys)
            if (Vector3.Dot(q * prefabAxes[k], det[k]) < 0.9f) return false;

        return true;
    }
}