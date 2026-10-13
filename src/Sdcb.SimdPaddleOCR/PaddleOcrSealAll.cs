using Sdcb.SimdPaddleOCR.ModelProvider;

namespace Sdcb.SimdPaddleOCR;

/// <summary>
/// Seal OCR: polygon detection, curved-text rectification, then the existing CTC recognizer.
/// Direction classification stays off, matching PaddleX <c>seal_recognition</c>
/// (<c>use_textline_orientation: false</c>).
/// </summary>
public sealed class PaddleOcrSealAll : IDisposable
{
    private readonly PaddleOcrSealDetector _detector;
    private readonly PaddleOcrRecognizer _recognizer;
    private readonly PaddleOcrSealOptions _options;
    private bool _disposed;

    public PaddleOcrSealAll(Stream detectorModel, Stream recognizerModel, Stream dictionaryUtf8,
        PaddleOcrSealOptions? options = null)
    {
        if (detectorModel is null) throw new ArgumentNullException(nameof(detectorModel));
        if (recognizerModel is null) throw new ArgumentNullException(nameof(recognizerModel));
        if (dictionaryUtf8 is null) throw new ArgumentNullException(nameof(dictionaryUtf8));
        _options = options ?? new PaddleOcrSealOptions();
        if (_options.DetIntraOpThreads < 0 || _options.DetIntraOpThreads > 16)
            throw new ArgumentOutOfRangeException(nameof(options));
        if (_options.MaxCropPixels <= 0) throw new ArgumentOutOfRangeException(nameof(options));
        _detector = new PaddleOcrSealDetector(detectorModel, _options.Detector, ResolveDetectorThreads(_options));
        try
        {
            _recognizer = new PaddleOcrRecognizer(recognizerModel, dictionaryUtf8, _options.Recognizer);
        }
        catch
        {
            _detector.Dispose();
            throw;
        }
    }

    public static PaddleOcrSealAll Load(string detectionPath, string recognitionPath, string dictionaryPath,
        PaddleOcrSealOptions? options = null)
    {
        using FileStream detection = File.OpenRead(detectionPath);
        using FileStream recognition = File.OpenRead(recognitionPath);
        using FileStream dictionary = File.OpenRead(dictionaryPath);
        return new PaddleOcrSealAll(detection, recognition, dictionary, options);
    }

    public static async Task<PaddleOcrSealAll> LoadAsync(string detectionPath, string recognitionPath,
        string dictionaryPath, PaddleOcrSealOptions? options = null, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return await Task.Run(() => Load(detectionPath, recognitionPath, dictionaryPath, options), cancellationToken)
            .ConfigureAwait(false);
    }

    public static PaddleOcrSealAll Load(Stream detectionOnnx, Stream recognitionOnnx, Stream dictionaryUtf8,
        PaddleOcrSealOptions? options = null)
        => new(detectionOnnx, recognitionOnnx, dictionaryUtf8, options);

    public static PaddleOcrSealAll Load(IPaddleOcrModelProvider detection, IPaddleOcrModelProvider recognition,
        IPaddleOcrModelProvider dictionary, PaddleOcrSealOptions? options = null)
    {
        if (detection is null) throw new ArgumentNullException(nameof(detection));
        if (recognition is null) throw new ArgumentNullException(nameof(recognition));
        if (dictionary is null) throw new ArgumentNullException(nameof(dictionary));
        using Stream detectionStream = detection.OpenRead();
        using Stream recognitionStream = recognition.OpenRead();
        using Stream dictionaryStream = dictionary.OpenRead();
        return new PaddleOcrSealAll(detectionStream, recognitionStream, dictionaryStream, options);
    }

    public static async Task<PaddleOcrSealAll> LoadAsync(IPaddleOcrModelProvider detection,
        IPaddleOcrModelProvider recognition, IPaddleOcrModelProvider dictionary,
        PaddleOcrSealOptions? options = null, CancellationToken cancellationToken = default)
    {
        if (detection is null) throw new ArgumentNullException(nameof(detection));
        if (recognition is null) throw new ArgumentNullException(nameof(recognition));
        if (dictionary is null) throw new ArgumentNullException(nameof(dictionary));
        cancellationToken.ThrowIfCancellationRequested();
        Stream detectionStream = await detection.OpenReadAsync(cancellationToken).ConfigureAwait(false);
        Stream recognitionStream = await recognition.OpenReadAsync(cancellationToken).ConfigureAwait(false);
        Stream dictionaryStream = await dictionary.OpenReadAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            return new PaddleOcrSealAll(detectionStream, recognitionStream, dictionaryStream, options);
        }
        finally
        {
            detectionStream.Dispose();
            recognitionStream.Dispose();
            dictionaryStream.Dispose();
        }
    }

    /// <summary>Detects seal text, rectifies each ring, and recognizes the strip.</summary>
    public PaddleOcrSealResult Run(ReadOnlySpan<byte> source, int sourceWidth, int sourceHeight,
        int sourceStride = 0, ImagePixelFormat format = ImagePixelFormat.Bgr24)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(PaddleOcrSealAll));
        PaddleOcrSealDetectionResult detected = _detector.Detect(source, sourceWidth, sourceHeight, sourceStride, format);
        List<PaddleOcrSealLine> lines = new(detected.Polygons.Length);
        foreach (PaddleOcrPolygon polygon in detected.Polygons)
        {
            byte[] strip;
            int width, height;
            try
            {
                strip = PaddleOcrSealRectify.Rectify(source, sourceWidth, sourceHeight, polygon.Points,
                    out width, out height, sourceStride, format);
            }
            catch (InvalidDataException)
            {
                continue;
            }
            if ((long)width * height > _options.MaxCropPixels)
                throw new InvalidOperationException("OCR crop exceeds MaxCropPixels.");
            PaddleOcrRecognitionResult recognition = _recognizer.Recognize(strip, width, height,
                format: ImagePixelFormat.Bgr24);
            lines.Add(new PaddleOcrSealLine
            {
                Polygon = polygon,
                Text = recognition.Text,
                RecognitionScore = recognition.Score,
                DetectionScore = polygon.Score
            });
        }
        return new PaddleOcrSealResult
        {
            Lines = [.. lines],
            ResizedWidth = detected.ResizedWidth,
            ResizedHeight = detected.ResizedHeight
        };
    }

    private static int ResolveDetectorThreads(PaddleOcrSealOptions options)
    {
        if (options.DetIntraOpThreads > 0) return Math.Min(options.DetIntraOpThreads, 16);
        if (int.TryParse(Environment.GetEnvironmentVariable("PPOCR_DET_THREADS"), out int env) && env > 0)
            return Math.Min(env, 16);
        if (OnnxSharp.LayoutPlanner.IsEnabled) return Math.Min(16, Environment.ProcessorCount);
        return Math.Min(8, Environment.ProcessorCount);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _detector.Dispose();
        _recognizer.Dispose();
    }
}
