#if NETSTANDARD2_0
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;

using static Sdcb.SimdPaddleOCR.Kernels.SimdOps;

namespace Sdcb.SimdPaddleOCR.Kernels;

// Vector<float> (netstandard2.0) layout/pool/reduce/elementwise kernels; the
// same semantics as the AVX2 versions with Vector<float>.Count-wide lanes and
// a scalar tail. Transposes are scalar 8x8 block copies (bit-exact moves,
// memory-bound either way); pooling divides by the in-bounds tap count and
// ReduceMeanSpatial divides by the plane, never by a reciprocal.
internal static unsafe partial class Nhwc
{
    // ------------------------------------------------------ layout conversion

    /// <summary>[n][c][plane] -> [n][plane][c] using scalar 8x8 block transposes.</summary>
    internal static void NchwToNhwc(ReadOnlySpan<float> source, Span<float> destination, int batch, int channels, int plane, int threads)
    {
        long volume = (long)batch * channels * plane;
        if (source.Length < volume || destination.Length < volume) throw new ArgumentException("Layout conversion buffer too small.");
        if (volume == 0) return;
        fixed (float* srcPtr = source, dstPtr = destination)
        {
            nint srcA = (nint)srcPtr, dstA = (nint)dstPtr;
            int workers = threads > 1 && volume >= 262_144 ? Math.Min(threads, Math.Max(1, plane / 64)) : 1;
            void Worker(int worker)
            {
                int pBegin = (int)((long)plane * worker / workers) & ~7;
                int pEnd = worker == workers - 1 ? plane : (int)((long)plane * (worker + 1) / workers) & ~7;
                for (int b = 0; b < batch; b++)
                {
                    float* src = (float*)srcA + (long)b * channels * plane;
                    float* dst = (float*)dstA + (long)b * channels * plane;
                    // src[c * plane + p] -> dst[p * channels + c]
                    TransposeBlocksVec(src, plane, dst, channels, channels, pBegin, pEnd);
                }
            }
            if (workers > 1) Parallel.For(0, workers, Worker);
            else Worker(0);
        }
    }

    /// <summary>[n][plane][c] -> [n][c][plane].</summary>
    internal static void NhwcToNchw(ReadOnlySpan<float> source, Span<float> destination, int batch, int channels, int plane, int threads)
    {
        long volume = (long)batch * channels * plane;
        if (source.Length < volume || destination.Length < volume) throw new ArgumentException("Layout conversion buffer too small.");
        if (volume == 0) return;
        fixed (float* srcPtr = source, dstPtr = destination)
        {
            nint srcA = (nint)srcPtr, dstA = (nint)dstPtr;
            int workers = threads > 1 && volume >= 262_144 ? Math.Min(threads, Math.Max(1, channels / 8)) : 1;
            void Worker(int worker)
            {
                int cBegin = (int)((long)channels * worker / workers) & ~7;
                int cEnd = worker == workers - 1 ? channels : (int)((long)channels * (worker + 1) / workers) & ~7;
                for (int b = 0; b < batch; b++)
                {
                    float* src = (float*)srcA + (long)b * channels * plane;
                    float* dst = (float*)dstA + (long)b * channels * plane;
                    // src[p * channels + c] -> dst[c * plane + p]
                    TransposeBlocksVec(src, channels, dst, plane, plane, cBegin, cEnd);
                }
            }
            if (workers > 1) Parallel.For(0, workers, Worker);
            else Worker(0);
        }
    }

    // Transposes the matrix src[rows][cols] (row stride srcStride) into
    // dst[cols][rows] (row stride dstStride) for source columns [colBegin, colEnd).
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static void TransposeBlocksVec(float* src, int srcStride, float* dst, int dstStride, int rows, int colBegin, int colEnd)
    {
        int rows8 = rows & ~7;
        int col = colBegin;
        for (; col + 8 <= colEnd; col += 8)
        {
            int row = 0;
            for (; row < rows8; row += 8)
            {
                float* s = src + (long)row * srcStride + col;
                float* d = dst + (long)col * dstStride + row;
                for (int i = 0; i < 8; i++, s += srcStride, d++)
                {
                    d[0] = s[0];
                    d[(long)dstStride] = s[1];
                    d[(long)2 * dstStride] = s[2];
                    d[(long)3 * dstStride] = s[3];
                    d[(long)4 * dstStride] = s[4];
                    d[(long)5 * dstStride] = s[5];
                    d[(long)6 * dstStride] = s[6];
                    d[(long)7 * dstStride] = s[7];
                }
            }
            for (; row < rows; row++)
                for (int j = 0; j < 8; j++)
                    dst[(long)(col + j) * dstStride + row] = src[(long)row * srcStride + col + j];
        }
        for (; col < colEnd; col++)
            for (int row = 0; row < rows; row++)
                dst[(long)col * dstStride + row] = src[(long)row * srcStride + col];
    }

    // ------------------------------------------------------------------ pooling

    /// <summary>Max / average pooling (average divides by the number of in-bounds taps, as the NCHW path does).</summary>
    internal static void Pool(ReadOnlySpan<float> input, Span<float> output, int batch, int channels, int height, int width,
        int outputHeight, int outputWidth, int kernelH, int kernelW, int strideH, int strideW, int padTop, int padLeft, bool max,
        int threads = 1)
    {
        long outVolume = (long)batch * outputHeight * outputWidth * channels;
        if (input.Length < (long)batch * height * width * channels || output.Length < outVolume)
            throw new ArgumentException("NHWC pool buffer too small.");
        if (outVolume == 0) return;
        int vecWidth = Vector<float>.Count;
        int vectorEnd = channels / vecWidth * vecWidth;
        int rowsTotal = batch * outputHeight;
        int workers = threads > 1 && outVolume >= 1 << 18 ? Math.Min(threads, rowsTotal) : 1;
        fixed (float* inPtr = input, outPtr = output)
        {
            nint inA = (nint)inPtr, outA = (nint)outPtr;
            void Worker(int worker)
            {
                int rowBegin = (int)((long)rowsTotal * worker / workers), rowEnd = (int)((long)rowsTotal * (worker + 1) / workers);
                for (int row = rowBegin; row < rowEnd; row++)
                {
                    int b = row / outputHeight, y = row - b * outputHeight;
                    float* inBatch = (float*)inA + (long)b * height * width * channels;
                    int iy0 = y * strideH - padTop;
                    int kyBegin = Math.Max(0, -iy0), kyEnd = Math.Min(kernelH, height - iy0);
                    for (int x = 0; x < outputWidth; x++)
                    {
                        int ix0 = x * strideW - padLeft;
                        int kxBegin = Math.Max(0, -ix0), kxEnd = Math.Min(kernelW, width - ix0);
                        float* dst = (float*)outA + (((long)b * outputHeight + y) * outputWidth + x) * channels;
                        int count = Math.Max(0, kyEnd - kyBegin) * Math.Max(0, kxEnd - kxBegin);
                        if (count == 0)
                        {
                            float fill = max ? float.NegativeInfinity : float.NaN;
                            for (int ch = 0; ch < channels; ch++) dst[ch] = fill;
                            continue;
                        }
                        int c = 0;
                        for (; c < vectorEnd; c += vecWidth)
                        {
                            Vector<float> acc = max ? new Vector<float>(float.NegativeInfinity) : Vector<float>.Zero;
                            for (int ky = kyBegin; ky < kyEnd; ky++)
                            {
                                float* srcRow = inBatch + ((long)(iy0 + ky) * width + ix0) * channels + c;
                                for (int kx = kxBegin; kx < kxEnd; kx++)
                                {
                                    Vector<float> v = VectorLoad(srcRow + (long)kx * channels);
                                    acc = max ? Vector.Max(acc, v) : acc + v;
                                }
                            }
                            VectorStore(dst + c, max ? acc : acc / new Vector<float>((float)count));
                        }
                        for (; c < channels; c++)
                        {
                            float acc = max ? float.NegativeInfinity : 0f;
                            for (int ky = kyBegin; ky < kyEnd; ky++)
                                for (int kx = kxBegin; kx < kxEnd; kx++)
                                {
                                    float v = inBatch[((long)(iy0 + ky) * width + ix0 + kx) * channels + c];
                                    acc = max ? MathF.Max(acc, v) : acc + v;
                                }
                            dst[c] = max ? acc : acc / count;
                        }
                    }
                }
            }
            if (workers > 1) Parallel.For(0, workers, Worker);
            else Worker(0);
        }
    }

    // ------------------------------------------------------------------- resize

    /// <summary>Nearest-neighbour integer upsampling: each input pixel vector is repeated factorW times, each row factorH times.</summary>
    internal static void ResizeNearest(ReadOnlySpan<float> input, Span<float> output, int batch, int channels, int height, int width,
        int factorH, int factorW, int threads = 1)
    {
        int outputWidth = width * factorW, outputHeight = height * factorH;
        long outVolume = (long)batch * outputHeight * outputWidth * channels;
        if (input.Length < (long)batch * height * width * channels || output.Length < outVolume)
            throw new ArgumentException("NHWC resize buffer too small.");
        if (outVolume == 0) return;
        long rowFloats = (long)outputWidth * channels;
        int rowsTotal = batch * height;
        int workers = threads > 1 && outVolume >= 1 << 18 ? Math.Min(threads, rowsTotal) : 1;
        fixed (float* inPtr = input, outPtr = output)
        {
            nint inA = (nint)inPtr, outA = (nint)outPtr;
            void Worker(int worker)
            {
                int rowBegin = (int)((long)rowsTotal * worker / workers), rowEnd = (int)((long)rowsTotal * (worker + 1) / workers);
                for (int row = rowBegin; row < rowEnd; row++)
                {
                    int b = row / height, y = row - b * height;
                    float* src = (float*)inA + ((long)b * height + y) * width * channels;
                    float* firstRow = (float*)outA + ((long)b * outputHeight + (long)y * factorH) * rowFloats;
                    float* dst = firstRow;
                    for (int x = 0; x < width; x++, src += channels)
                        for (int repeat = 0; repeat < factorW; repeat++, dst += channels)
                            Buffer.MemoryCopy(src, dst, channels * sizeof(float), channels * sizeof(float));
                    for (int repeat = 1; repeat < factorH; repeat++)
                        Buffer.MemoryCopy(firstRow, firstRow + repeat * rowFloats, rowFloats * sizeof(float), rowFloats * sizeof(float));
                }
            }
            if (workers > 1) Parallel.For(0, workers, Worker);
            else Worker(0);
        }
    }

    // -------------------------------------------------------------- reductions

    /// <summary>Spatial mean per (batch, channel): output is [n][c] (== NCHW [n,c,1,1]).</summary>
    internal static void ReduceMeanSpatial(ReadOnlySpan<float> input, Span<float> output, int batch, int channels, int plane)
    {
        if (input.Length < (long)batch * channels * plane || output.Length < (long)batch * channels)
            throw new ArgumentException("NHWC reduce buffer too small.");
        if (plane <= 0) { output.Slice(0, batch * channels).Clear(); return; }
        int vecWidth = Vector<float>.Count;
        int vectorEnd = channels / vecWidth * vecWidth;
        fixed (float* inPtr = input, outPtr = output)
        {
            for (int b = 0; b < batch; b++)
            {
                float* src = inPtr + (long)b * plane * channels;
                float* dst = outPtr + (long)b * channels;
                new Span<float>(dst, channels).Clear();
                for (int p = 0; p < plane; p++, src += channels)
                {
                    int c = 0;
                    for (; c < vectorEnd; c += vecWidth)
                        VectorStore(dst + c, VectorLoad(dst + c) + VectorLoad(src + c));
                    for (; c < channels; c++) dst[c] += src[c];
                }
                float inv = plane;
                for (int c = 0; c < channels; c++) dst[c] /= inv;
            }
        }
    }

    // -------------------------------------------------------------- elementwise

    /// <summary>
    /// Binary op between an NHWC activation [n][plane][c] and a channel vector
    /// ([c] shared, or [n][c] when <paramref name="channelPerBatch"/>).
    /// </summary>
    internal static void BinaryChannel<TOp>(ReadOnlySpan<float> left, ReadOnlySpan<float> channel, Span<float> output,
        int batch, int channels, int plane, bool channelIsLeft, bool channelPerBatch) where TOp : struct, IBinaryOp
    {
        long volume = (long)batch * plane * channels;
        if (left.Length < volume || output.Length < volume || channel.Length < (channelPerBatch ? (long)batch * channels : channels))
            throw new ArgumentException("NHWC channel broadcast buffer too small.");
        TOp op = default;
        int vecWidth = Vector<float>.Count;
        int vectorEnd = channels / vecWidth * vecWidth;
        fixed (float* aPtr = left, cPtr = channel, oPtr = output)
        {
            for (int b = 0; b < batch; b++)
            {
                float* ch = cPtr + (channelPerBatch ? (long)b * channels : 0);
                float* a = aPtr + (long)b * plane * channels;
                float* o = oPtr + (long)b * plane * channels;
                for (int p = 0; p < plane; p++, a += channels, o += channels)
                {
                    int c = 0;
                    for (; c < vectorEnd; c += vecWidth)
                    {
                        Vector<float> x = VectorLoad(a + c), y = VectorLoad(ch + c);
                        VectorStore(o + c, channelIsLeft ? op.Apply(y, x) : op.Apply(x, y));
                    }
                    for (; c < channels; c++)
                        o[c] = channelIsLeft ? op.Apply(ch[c], a[c]) : op.Apply(a[c], ch[c]);
                }
            }
        }
    }

    /// <summary>Inference batch-norm: (x - mean) * scale / sqrt(var + eps) + bias, per channel.</summary>
    internal static void BatchNorm(ReadOnlySpan<float> input, Span<float> output, int pixels, int channels,
        ReadOnlySpan<float> scale, ReadOnlySpan<float> bias, ReadOnlySpan<float> mean, ReadOnlySpan<float> variance, float epsilon)
    {
        if (input.Length < (long)pixels * channels || output.Length < (long)pixels * channels ||
            scale.Length < channels || bias.Length < channels || mean.Length < channels || variance.Length < channels)
            throw new ArgumentException("NHWC batch-norm buffer too small.");
        float[] k = new float[channels];
        for (int c = 0; c < channels; c++) k[c] = scale[c] / MathF.Sqrt(variance[c] + epsilon);
        int vecWidth = Vector<float>.Count;
        int vectorEnd = channels / vecWidth * vecWidth;
        fixed (float* inPtr = input, outPtr = output, kPtr = k, meanPtr = mean, biasPtr = bias)
        {
            float* src = inPtr, dst = outPtr;
            for (int p = 0; p < pixels; p++, src += channels, dst += channels)
            {
                int c = 0;
                for (; c < vectorEnd; c += vecWidth)
                {
                    Vector<float> x = VectorLoad(src + c) - VectorLoad(meanPtr + c);
                    VectorStore(dst + c, x * VectorLoad(kPtr + c) + VectorLoad(biasPtr + c));
                }
                for (; c < channels; c++)
                    dst[c] = (src[c] - mean[c]) * k[c] + bias[c];
            }
        }
    }
}
#endif
