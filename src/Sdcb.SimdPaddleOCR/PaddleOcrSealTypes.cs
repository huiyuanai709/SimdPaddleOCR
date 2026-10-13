namespace Sdcb.SimdPaddleOCR;

/// <summary>
/// How <see cref="PaddleOcrSealDetector"/> chooses the network input size.
/// Defaults follow the PaddleX <c>seal_recognition</c> pipeline (<see cref="Min"/>).
/// </summary>
public enum PaddleOcrSideLimit
{
    /// <summary>Scale down when the long side exceeds <c>LimitSideLength</c>, then round to a multiple of 32.</summary>
    Max = 0,
    /// <summary>Scale up when the short side is below <c>LimitSideLength</c>, then round to a multiple of 32.</summary>
    Min = 1,
    /// <summary>Scale the long side to <c>LimitSideLength</c>, then round up to a multiple of 128.</summary>
    ResizeLong = 2,
}

/// <summary>
/// Options for <see cref="PaddleOcrSealDetector"/>. Defaults match PaddleX
/// <c>seal_recognition</c>: short side 736, max side 4000, bitmap 0.2, box 0.6, unclip 0.5.
/// </summary>
public sealed class PaddleOcrSealDetectorOptions
{
    /// <summary>
    /// Upper bound on pooled reusable sessions. Sessions returned while the pool
    /// is full are disposed instead of pooled. The default is
    /// <see cref="Environment.ProcessorCount"/>; set to <see cref="int.MaxValue"/>
    /// for an unbounded pool.
    /// </summary>
    public int MaxPooledSessions { get; init; } = Environment.ProcessorCount;
    public int LimitSideLength { get; init; } = 736;
    public PaddleOcrSideLimit LimitType { get; init; } = PaddleOcrSideLimit.Min;
    /// <summary>Longest side allowed after the 32-pixel rounding used by <see cref="PaddleOcrSideLimit.Min"/> and <see cref="PaddleOcrSideLimit.Max"/>.</summary>
    public int MaxSideLimit { get; init; } = 4000;
    public int MaxCandidates { get; init; } = 1000;
    public bool UseDilation { get; init; }
    public float BitmapThreshold { get; init; } = 0.2f;
    public float BoxThreshold { get; init; } = 0.6f;
    public float UnclipRatio { get; init; } = 0.5f;
    public long MaxImagePixels { get; init; } = 40_000_000;
    /// <summary>Compute backend for the seal detection graph. Default <see cref="OcrBackend.Auto"/>.</summary>
    public OcrBackend Backend { get; init; } = OcrBackend.Auto;
}

/// <summary>Options for <see cref="PaddleOcrSealAll"/>.</summary>
public sealed class PaddleOcrSealOptions
{
    public PaddleOcrSealDetectorOptions Detector { get; init; } = new();
    public PaddleOcrRecognizerOptions Recognizer { get; init; } = new();
    /// <summary>
    /// Intra-op threads inside the seal detector. <c>0</c> uses the same budget
    /// as <see cref="PaddleOcrAll"/> (up to 8, or up to 16 when the channels-last
    /// planner is on).
    /// </summary>
    public int DetIntraOpThreads { get; init; }
    public long MaxCropPixels { get; init; } = 16_000_000;
}

/// <summary>One vertex of a seal-text polygon, in source-image pixels.</summary>
public readonly record struct PaddleOcrPoint(float X, float Y);

/// <summary>One seal-text region. <see cref="Points"/> is the expanded ring in source-image pixels.</summary>
public sealed class PaddleOcrPolygon
{
    public required PaddleOcrPoint[] Points { get; init; }
    /// <summary>Mean detector probability inside the polygon, before unclip.</summary>
    public required float Score { get; init; }
}

/// <summary>Polygons from one <see cref="PaddleOcrSealDetector.Detect"/> call.</summary>
public sealed class PaddleOcrSealDetectionResult
{
    public required PaddleOcrPolygon[] Polygons { get; init; }
    public required int ResizedWidth { get; init; }
    public required int ResizedHeight { get; init; }
}

/// <summary>One rectified seal-text line.</summary>
public sealed class PaddleOcrSealLine
{
    public required PaddleOcrPolygon Polygon { get; init; }
    public required string Text { get; init; }
    public required float RecognitionScore { get; init; }
    public required float DetectionScore { get; init; }
}

/// <summary>Seal OCR result. <see cref="Text"/> joins <see cref="Lines"/> with newlines.</summary>
public sealed class PaddleOcrSealResult
{
    public required PaddleOcrSealLine[] Lines { get; init; }
    public required int ResizedWidth { get; init; }
    public required int ResizedHeight { get; init; }

    public string Text => string.Join("\n", Lines.Select(line => line.Text));
}
