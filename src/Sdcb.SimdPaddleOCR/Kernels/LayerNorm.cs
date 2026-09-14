using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Sdcb.SimdPaddleOCR.Kernels;

/// <summary>
/// Fused LayerNorm over the trailing axis, replacing the exported
/// ReduceMean/Sub/Pow/ReduceMean/Add/Sqrt/Div/Mul/Add chain. Means keep the
/// sequential summation order of the generic ReduceMean so the statistics
/// match the unfused graph; the per-element steps are the same IEEE ops.
/// </summary>
internal static class LayerNorm
{
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public static void Rows(ReadOnlySpan<float> input, ReadOnlySpan<float> gamma, ReadOnlySpan<float> beta,
        float epsilon, int rows, int channels, Span<float> output)
    {
        if (channels <= 0 || rows < 0 || input.Length < rows * channels || output.Length < rows * channels ||
            gamma.Length < channels || beta.Length < channels)
            throw new ArgumentException("LayerNorm buffer too small.");
        int width = Vector<float>.Count;
        bool vectorize = Vector.IsHardwareAccelerated && channels >= width;
        for (int row = 0; row < rows; row++)
        {
            ReadOnlySpan<float> x = input.Slice(row * channels, channels);
            Span<float> y = output.Slice(row * channels, channels);
            float sum = 0f;
            for (int c = 0; c < channels; c++) sum += x[c];
            float mean = sum / channels;
            float variance = 0f;
            for (int c = 0; c < channels; c++)
            {
                float d = x[c] - mean;
                variance += d * d;
            }
            variance /= channels;
            float std = MathF.Sqrt(variance + epsilon);
            int c0 = 0;
            if (vectorize)
            {
                Vector<float> vMean = new(mean), vStd = new(std);
                for (; c0 <= channels - width; c0 += width)
                {
                    Vector<float> d = Load(x, c0) - vMean;
                    Vector<float> normalized = d / vStd;
                    Store(y, c0, normalized * Load(gamma, c0) + Load(beta, c0));
                }
            }
            for (; c0 < channels; c0++)
                y[c0] = (x[c0] - mean) / std * gamma[c0] + beta[c0];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<float> Load(ReadOnlySpan<float> source, int offset)
        => Unsafe.ReadUnaligned<Vector<float>>(ref Unsafe.As<float, byte>(ref MemoryMarshal.GetReference(source.Slice(offset))));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Store(Span<float> destination, int offset, Vector<float> value)
        => Unsafe.WriteUnaligned(ref Unsafe.As<float, byte>(ref MemoryMarshal.GetReference(destination.Slice(offset))), value);
}
