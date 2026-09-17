#if !NETSTANDARD2_0
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Threading.Tasks;

namespace Sdcb.SimdPaddleOCR.Kernels;

// ARM64 AdvSimd (NEON) versions of the NHWC misc operators: layout transposes,
// pooling, spatial reduction, channel-broadcast binary ops and inference
// batch-norm. Same loop nests and multithreading skeletons as the AVX2 file
// (Nhwc.Misc.Avx.cs), with 4-wide Vector128 blocks instead of 8-wide
// Vector256. ResizeNearest is a pure gather of whole pixel vectors
// (Buffer.MemoryCopy), so it forwards to the Vector implementation.
internal static unsafe partial class Nhwc
{
    // ------------------------------------------------------ layout conversion

    /// <summary>AdvSimd driver of <see cref="NchwToNhwc"/>; 4x4 register transposes.</summary>
    internal static void NchwToNhwcAdvSimd(ReadOnlySpan<float> source, Span<float> destination, int batch, int channels, int plane, int threads)
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
                int pBegin = (int)((long)plane * worker / workers) & ~3;
                int pEnd = worker == workers - 1 ? plane : (int)((long)plane * (worker + 1) / workers) & ~3;
                for (int b = 0; b < batch; b++)
                {
                    float* src = (float*)srcA + (long)b * channels * plane;
                    float* dst = (float*)dstA + (long)b * channels * plane;
                    // src[c * plane + p] -> dst[p * channels + c]
                    TransposeBlocksAdvSimd(src, plane, dst, channels, channels, pBegin, pEnd);
                }
            }
            if (workers > 1) Parallel.For(0, workers, Worker);
            else Worker(0);
        }
    }

    /// <summary>AdvSimd driver of <see cref="NhwcToNchw"/>.</summary>
    internal static void NhwcToNchwAdvSimd(ReadOnlySpan<float> source, Span<float> destination, int batch, int channels, int plane, int threads)
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
                int cBegin = (int)((long)channels * worker / workers) & ~3;
                int cEnd = worker == workers - 1 ? channels : (int)((long)channels * (worker + 1) / workers) & ~3;
                for (int b = 0; b < batch; b++)
                {
                    float* src = (float*)srcA + (long)b * channels * plane;
                    float* dst = (float*)dstA + (long)b * channels * plane;
                    // src[p * channels + c] -> dst[c * plane + p]
                    TransposeBlocksAdvSimd(src, channels, dst, plane, plane, cBegin, cEnd);
                }
            }
            if (workers > 1) Parallel.For(0, workers, Worker);
            else Worker(0);
        }
    }

    // Transposes the matrix src[rows][cols] (row stride srcStride) into
    // dst[cols][rows] (row stride dstStride) for source columns [colBegin, colEnd).
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static void TransposeBlocksAdvSimd(float* src, int srcStride, float* dst, int dstStride, int rows, int colBegin, int colEnd)
    {
        int rows4 = rows & ~3;
        int col = colBegin;
        for (; col + 4 <= colEnd; col += 4)
        {
            int row = 0;
            for (; row < rows4; row += 4)
                Transpose4x4AdvSimd(src + (long)row * srcStride + col, srcStride, dst + (long)col * dstStride + row, dstStride);
            for (; row < rows; row++)
                for (int j = 0; j < 4; j++)
                    dst[(long)(col + j) * dstStride + row] = src[(long)row * srcStride + col + j];
        }
        for (; col < colEnd; col++)
            for (int row = 0; row < rows; row++)
                dst[(long)col * dstStride + row] = src[(long)row * srcStride + col];
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Transpose4x4AdvSimd(float* src, int srcStride, float* dst, int dstStride)
    {
        Vector128<float> r0 = AdvSimd.LoadVector128(src);
        Vector128<float> r1 = AdvSimd.LoadVector128(src + srcStride);
        Vector128<float> r2 = AdvSimd.LoadVector128(src + 2L * srcStride);
        Vector128<float> r3 = AdvSimd.LoadVector128(src + 3L * srcStride);
        Vector128<float> t0 = AdvSimd.Arm64.ZipLow(r0, r1), t1 = AdvSimd.Arm64.ZipHigh(r0, r1);
        Vector128<float> t2 = AdvSimd.Arm64.ZipLow(r2, r3), t3 = AdvSimd.Arm64.ZipHigh(r2, r3);
        AdvSimd.Store(dst, AdvSimd.Arm64.ZipLow(t0.AsDouble(), t2.AsDouble()).AsSingle());
        AdvSimd.Store(dst + dstStride, AdvSimd.Arm64.ZipHigh(t0.AsDouble(), t2.AsDouble()).AsSingle());
        AdvSimd.Store(dst + 2L * dstStride, AdvSimd.Arm64.ZipLow(t1.AsDouble(), t3.AsDouble()).AsSingle());
        AdvSimd.Store(dst + 3L * dstStride, AdvSimd.Arm64.ZipHigh(t1.AsDouble(), t3.AsDouble()).AsSingle());
    }

    // ------------------------------------------------------------------ pooling

    /// <summary>AdvSimd driver of <see cref="Pool"/>; same semantics (average divides by the in-bounds tap count).</summary>
    internal static void PoolAdvSimd(ReadOnlySpan<float> input, Span<float> output, int batch, int channels, int height, int width,
        int outputHeight, int outputWidth, int kernelH, int kernelW, int strideH, int strideW, int padTop, int padLeft, bool max,
        int threads = 1)
    {
        long outVolume = (long)batch * outputHeight * outputWidth * channels;
        if (input.Length < (long)batch * height * width * channels || output.Length < outVolume)
            throw new ArgumentException("NHWC pool buffer too small.");
        if (outVolume == 0) return;
        int vectorEnd = channels & ~3;
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
                        for (; c < vectorEnd; c += 4)
                        {
                            Vector128<float> acc = max ? Vector128.Create(float.NegativeInfinity) : Vector128<float>.Zero;
                            for (int ky = kyBegin; ky < kyEnd; ky++)
                            {
                                float* srcRow = inBatch + ((long)(iy0 + ky) * width + ix0) * channels + c;
                                for (int kx = kxBegin; kx < kxEnd; kx++)
                                {
                                    Vector128<float> v = AdvSimd.LoadVector128(srcRow + (long)kx * channels);
                                    acc = max ? AdvSimd.Max(acc, v) : AdvSimd.Add(acc, v);
                                }
                            }
                            AdvSimd.Store(dst + c, max ? acc : Vector128.Divide(acc, Vector128.Create((float)count)));
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

    /// <summary>Pure gather of whole pixel vectors; forwards to the Vector implementation.</summary>
    internal static void ResizeNearestAdvSimd(ReadOnlySpan<float> input, Span<float> output, int batch, int channels, int height, int width,
        int factorH, int factorW, int threads = 1)
        => ResizeNearestVec(input, output, batch, channels, height, width, factorH, factorW, threads);

    // -------------------------------------------------------------- reductions

    /// <summary>AdvSimd driver of <see cref="ReduceMeanSpatial"/>.</summary>
    internal static void ReduceMeanSpatialAdvSimd(ReadOnlySpan<float> input, Span<float> output, int batch, int channels, int plane)
    {
        if (input.Length < (long)batch * channels * plane || output.Length < (long)batch * channels)
            throw new ArgumentException("NHWC reduce buffer too small.");
        if (plane <= 0) { output.Slice(0, batch * channels).Clear(); return; }
        int vectorEnd = channels & ~3;
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
                    for (; c < vectorEnd; c += 4)
                        AdvSimd.Store(dst + c, AdvSimd.Add(AdvSimd.LoadVector128(dst + c), AdvSimd.LoadVector128(src + c)));
                    for (; c < channels; c++) dst[c] += src[c];
                }
                float inv = plane;
                for (int c = 0; c < channels; c++) dst[c] /= inv;
            }
        }
    }

    // -------------------------------------------------------------- elementwise

    /// <summary>AdvSimd driver of <see cref="BinaryChannel{TOp}"/>.</summary>
    internal static void BinaryChannelAdvSimd<TOp>(ReadOnlySpan<float> left, ReadOnlySpan<float> channel, Span<float> output,
        int batch, int channels, int plane, bool channelIsLeft, bool channelPerBatch) where TOp : struct, IBinaryOp
    {
        long volume = (long)batch * plane * channels;
        if (left.Length < volume || output.Length < volume || channel.Length < (channelPerBatch ? (long)batch * channels : channels))
            throw new ArgumentException("NHWC channel broadcast buffer too small.");
        TOp op = default;
        int vectorEnd = channels & ~3;
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
                    for (; c < vectorEnd; c += 4)
                    {
                        Vector128<float> x = AdvSimd.LoadVector128(a + c), y = AdvSimd.LoadVector128(ch + c);
                        AdvSimd.Store(o + c, channelIsLeft ? op.Apply(y, x) : op.Apply(x, y));
                    }
                    for (; c < channels; c++)
                        o[c] = channelIsLeft ? op.Apply(ch[c], a[c]) : op.Apply(a[c], ch[c]);
                }
            }
        }
    }

    /// <summary>AdvSimd driver of <see cref="BatchNorm"/>; same (x - mean) * k + bias operation order.</summary>
    internal static void BatchNormAdvSimd(ReadOnlySpan<float> input, Span<float> output, int pixels, int channels,
        ReadOnlySpan<float> scale, ReadOnlySpan<float> bias, ReadOnlySpan<float> mean, ReadOnlySpan<float> variance, float epsilon)
    {
        if (input.Length < (long)pixels * channels || output.Length < (long)pixels * channels ||
            scale.Length < channels || bias.Length < channels || mean.Length < channels || variance.Length < channels)
            throw new ArgumentException("NHWC batch-norm buffer too small.");
        float[] k = new float[channels];
        for (int c = 0; c < channels; c++) k[c] = scale[c] / MathF.Sqrt(variance[c] + epsilon);
        int vectorEnd = channels & ~3;
        fixed (float* inPtr = input, outPtr = output, kPtr = k, meanPtr = mean, biasPtr = bias)
        {
            float* src = inPtr, dst = outPtr;
            for (int p = 0; p < pixels; p++, src += channels, dst += channels)
            {
                int c = 0;
                for (; c < vectorEnd; c += 4)
                {
                    Vector128<float> x = AdvSimd.Subtract(AdvSimd.LoadVector128(src + c), AdvSimd.LoadVector128(meanPtr + c));
                    AdvSimd.Store(dst + c, AdvSimd.Add(AdvSimd.Multiply(x, AdvSimd.LoadVector128(kPtr + c)), AdvSimd.LoadVector128(biasPtr + c)));
                }
                for (; c < channels; c++)
                    dst[c] = (src[c] - mean[c]) * k[c] + bias[c];
            }
        }
    }
}
#endif
