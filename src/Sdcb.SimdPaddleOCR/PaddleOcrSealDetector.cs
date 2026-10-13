using System.Collections.Concurrent;
using Sdcb.SimdPaddleOCR.OnnxSharp;

namespace Sdcb.SimdPaddleOCR;

/// <summary>
/// PP-OCRv4 seal text detector. Loads the same way as <see cref="PaddleOcrDetector"/>
/// and returns multi-point rings for curved stamp text. Pass these models here
/// (or to <see cref="PaddleOcrSealAll"/>); the quad detector keeps its own thresholds.
/// </summary>
public sealed class PaddleOcrSealDetector : IDisposable
{
    private readonly Model _model;
    private readonly PaddleOcrSealDetectorOptions _options;
    private readonly int _intraOpThreads;
    private readonly CompiledModel _compiled;
    private readonly ConcurrentBag<IOcrSession> _sessions = [];
    private readonly bool _ownsModel;
    private int _pooledCount;
    private bool _disposed;

    public PaddleOcrSealDetector(Model model, PaddleOcrSealDetectorOptions? options = null, int intraOpThreads = 1)
        : this(model, options, intraOpThreads, ownsModel: false)
    {
    }

    /// <summary>Loads a seal detector from a stream without retaining the serialized payload.</summary>
    public PaddleOcrSealDetector(Stream model, PaddleOcrSealDetectorOptions? options = null, int intraOpThreads = 1)
        : this(Model.Load(model ?? throw new ArgumentNullException(nameof(model))), options, intraOpThreads, ownsModel: true)
    {
    }

    private PaddleOcrSealDetector(Model model, PaddleOcrSealDetectorOptions? options, int intraOpThreads, bool ownsModel)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _ownsModel = ownsModel;
        try
        {
            _options = options ?? new PaddleOcrSealDetectorOptions();
            if (_options.MaxPooledSessions < 0) throw new ArgumentOutOfRangeException(nameof(options));
            if (_options.LimitSideLength is < 32 or > 4096) throw new ArgumentOutOfRangeException(nameof(options));
            if (_options.MaxSideLimit is < 32 or > 8192) throw new ArgumentOutOfRangeException(nameof(options));
            if (_options.MaxCandidates is <= 0 or > 10_000) throw new ArgumentOutOfRangeException(nameof(options));
            if (_options.BitmapThreshold is < 0 or > 1 || _options.BoxThreshold is < 0 or > 1 ||
                !MathCompat.IsFinite(_options.BitmapThreshold) || !MathCompat.IsFinite(_options.BoxThreshold) ||
                _options.UnclipRatio <= 0 || _options.UnclipRatio > 10 || !MathCompat.IsFinite(_options.UnclipRatio))
                throw new ArgumentOutOfRangeException(nameof(options));
            if (!Enum.IsDefined(typeof(PaddleOcrSideLimit), _options.LimitType))
                throw new ArgumentOutOfRangeException(nameof(options));
            _intraOpThreads = MathCompat.Clamp(intraOpThreads, 1, 16);
            _compiled = new CompiledModel(_model, _intraOpThreads);
        }
        catch
        {
            if (ownsModel) _model.Dispose();
            throw;
        }
    }

    public PaddleOcrSealDetectionResult Detect(ReadOnlySpan<byte> source, int sourceWidth, int sourceHeight,
        int sourceStride = 0, ImagePixelFormat format = ImagePixelFormat.Bgr24)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(PaddleOcrSealDetector));
        sourceStride = ImagePixels.ResolveStride(sourceWidth, sourceStride, format);
        int originalWidth = sourceWidth, originalHeight = sourceHeight;
        if ((long)sourceWidth * sourceHeight > _options.MaxImagePixels)
            throw new InvalidOperationException("Source image exceeds MaxImagePixels.");
        byte[]? paddedSource = null;
        if (sourceWidth + (long)sourceHeight < 64)
        {
            int bpp = ImagePixels.BytesPerPixel(format);
            int paddedWidth = Math.Max(32, sourceWidth);
            int paddedHeight = Math.Max(32, sourceHeight);
            int paddedStride = checked(paddedWidth * bpp);
            int rowBytes = checked(sourceWidth * bpp);
            paddedSource = PooledArrays.Rent<byte>(checked(paddedStride * paddedHeight));
            Span<byte> padded = paddedSource.AsSpan(0, checked(paddedStride * paddedHeight));
            padded.Clear();
            for (int y = 0; y < sourceHeight; y++)
            {
                source.Slice(y * sourceStride, rowBytes)
                    .CopyTo(padded.Slice(y * paddedStride, rowBytes));
            }
            source = padded;
            sourceWidth = paddedWidth;
            sourceHeight = paddedHeight;
            sourceStride = paddedStride;
        }

        try
        {
            (int Width, int Height) size = ComputeInputSize(sourceWidth, sourceHeight, _options);
            IOcrSession session = RentSession();
            session.Reshape([1, 3, size.Height, size.Width]);
            Span<float> inputSpan = session.InputData;
            try
            {
                PPOCRPreprocess.Det(source, sourceWidth, sourceHeight, sourceStride,
                    size.Width, size.Height, inputSpan, session.ResizeWorkspace, _intraOpThreads,
                    session.InputIsNhwc, format);
                ReadOnlySpan<float> probabilities = session.RunInternal(inputSpan);
                if (probabilities.Length != checked(size.Width * size.Height))
                    throw new InvalidDataException("Seal detector output is not a single probability map.");
                PaddleOcrPolygon[] polygons = DbPolyPostprocess.Run(probabilities, size.Width, size.Height, _options,
                    originalWidth, originalHeight);
                return new PaddleOcrSealDetectionResult
                {
                    Polygons = polygons,
                    ResizedWidth = size.Width,
                    ResizedHeight = size.Height
                };
            }
            finally
            {
                ReturnSession(session);
            }
        }
        finally
        {
            if (paddedSource is not null) PooledArrays.Return(paddedSource);
        }
    }

    /// <summary>
    /// PaddleOCR <c>DetResizeForTest</c> for <c>limit_type</c> max, min, and resize_long.
    /// <see cref="PPOCRPreprocess.ComputeDetSize"/> stays the quad-detector path (max only).
    /// </summary>
    internal static (int Width, int Height) ComputeInputSize(int sourceWidth, int sourceHeight,
        PaddleOcrSealDetectorOptions options)
    {
        if (sourceWidth <= 0) throw new ArgumentOutOfRangeException(nameof(sourceWidth));
        if (sourceHeight <= 0) throw new ArgumentOutOfRangeException(nameof(sourceHeight));
        int limit = options.LimitSideLength;
        if (options.LimitType == PaddleOcrSideLimit.ResizeLong)
        {
            double ratio = (double)limit / Math.Max(sourceWidth, sourceHeight);
            int width = CeilTo(Truncate(sourceWidth * ratio), 128);
            int height = CeilTo(Truncate(sourceHeight * ratio), 128);
            return (width, height);
        }

        int minSide = Math.Min(sourceWidth, sourceHeight);
        int maxSide = Math.Max(sourceWidth, sourceHeight);
        double scale = options.LimitType == PaddleOcrSideLimit.Min
            ? minSide < limit ? (double)limit / minSide : 1.0
            : maxSide > limit ? (double)limit / maxSide : 1.0;
        int scaledWidth = Truncate(sourceWidth * scale);
        int scaledHeight = Truncate(sourceHeight * scale);
        int roundedWidth = RoundTo32(scaledWidth);
        int roundedHeight = RoundTo32(scaledHeight);
        int longest = Math.Max(roundedWidth, roundedHeight);
        if (longest > options.MaxSideLimit)
        {
            double shrink = (double)options.MaxSideLimit / longest;
            roundedWidth = RoundTo32(Truncate(roundedWidth * shrink));
            roundedHeight = RoundTo32(Truncate(roundedHeight * shrink));
        }
        return (roundedWidth, roundedHeight);
    }

    private static int Truncate(double value)
    {
        if (!MathCompat.IsFinite(value) || value <= 0)
            throw new InvalidOperationException("Seal detector input is too large.");
        if (value >= int.MaxValue)
            throw new InvalidOperationException("Seal detector input is too large.");
        return (int)value;
    }

    private static int RoundTo32(int value)
    {
        int rounded = (int)Math.Round(value / 32.0, MidpointRounding.ToEven) * 32;
        return Math.Max(32, rounded);
    }

    private static int CeilTo(int value, int multiple)
    {
        if (value <= 0) return multiple;
        int rounded = (value + multiple - 1) / multiple * multiple;
        return Math.Max(multiple, rounded);
    }

    private IOcrSession RentSession()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(PaddleOcrSealDetector));
        if (_sessions.TryTake(out IOcrSession? session))
        {
            Interlocked.Decrement(ref _pooledCount);
            return session;
        }
        return OcrSessionFactory.Create(_compiled, _options.Backend);
    }

    private void ReturnSession(IOcrSession session)
    {
        if (_disposed)
        {
            session.Dispose();
            return;
        }
        int count = Interlocked.Increment(ref _pooledCount);
        if (count > _options.MaxPooledSessions)
        {
            Interlocked.Decrement(ref _pooledCount);
            session.Dispose();
            return;
        }
        _sessions.Add(session);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        while (_sessions.TryTake(out IOcrSession? session))
            session.Dispose();
        _compiled.Dispose();
        if (_ownsModel) _model.Dispose();
    }
}
