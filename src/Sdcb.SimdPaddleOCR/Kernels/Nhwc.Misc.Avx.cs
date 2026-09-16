using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;

namespace Sdcb.SimdPaddleOCR.Kernels;

internal static unsafe partial class Nhwc
{
    // ------------------------------------------------------ layout conversion

    /// <summary>[n][c][plane] -> [n][plane][c] using 8x8 register transposes.</summary>
    private static void NchwToNhwcAvx(ReadOnlySpan<float> source, Span<float> destination, int batch, int channels, int plane, int threads)
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
                    TransposeBlocks(src, plane, dst, channels, channels, pBegin, pEnd);
                }
            }
            if (workers > 1) Parallel.For(0, workers, Worker);
            else Worker(0);
        }
    }

    /// <summary>[n][plane][c] -> [n][c][plane].</summary>
    private static void NhwcToNchwAvx(ReadOnlySpan<float> source, Span<float> destination, int batch, int channels, int plane, int threads)
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
                    TransposeBlocks(src, channels, dst, plane, plane, cBegin, cEnd);
                }
            }
            if (workers > 1) Parallel.For(0, workers, Worker);
            else Worker(0);
        }
    }

    // Transposes the matrix src[rows][cols] (row stride srcStride) into
    // dst[cols][rows] (row stride dstStride) for source columns [colBegin, colEnd).
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static void TransposeBlocks(float* src, int srcStride, float* dst, int dstStride, int rows, int colBegin, int colEnd)
    {
        int rows8 = rows & ~7;
        int col = colBegin;
        for (; col + 8 <= colEnd; col += 8)
        {
            int row = 0;
            for (; row < rows8; row += 8)
                Transpose8x8(src + (long)row * srcStride + col, srcStride, dst + (long)col * dstStride + row, dstStride);
            for (; row < rows; row++)
                for (int j = 0; j < 8; j++)
                    dst[(long)(col + j) * dstStride + row] = src[(long)row * srcStride + col + j];
        }
        for (; col < colEnd; col++)
            for (int row = 0; row < rows; row++)
                dst[(long)col * dstStride + row] = src[(long)row * srcStride + col];
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Transpose8x8(float* src, int srcStride, float* dst, int dstStride)
    {
        Vector256<float> r0 = Avx.LoadVector256(src);
        Vector256<float> r1 = Avx.LoadVector256(src + srcStride);
        Vector256<float> r2 = Avx.LoadVector256(src + 2L * srcStride);
        Vector256<float> r3 = Avx.LoadVector256(src + 3L * srcStride);
        Vector256<float> r4 = Avx.LoadVector256(src + 4L * srcStride);
        Vector256<float> r5 = Avx.LoadVector256(src + 5L * srcStride);
        Vector256<float> r6 = Avx.LoadVector256(src + 6L * srcStride);
        Vector256<float> r7 = Avx.LoadVector256(src + 7L * srcStride);
        Vector256<float> t0 = Avx.UnpackLow(r0, r1), t1 = Avx.UnpackHigh(r0, r1);
        Vector256<float> t2 = Avx.UnpackLow(r2, r3), t3 = Avx.UnpackHigh(r2, r3);
        Vector256<float> t4 = Avx.UnpackLow(r4, r5), t5 = Avx.UnpackHigh(r4, r5);
        Vector256<float> t6 = Avx.UnpackLow(r6, r7), t7 = Avx.UnpackHigh(r6, r7);
        Vector256<float> s0 = Avx.Shuffle(t0, t2, 0x44), s1 = Avx.Shuffle(t0, t2, 0xEE);
        Vector256<float> s2 = Avx.Shuffle(t1, t3, 0x44), s3 = Avx.Shuffle(t1, t3, 0xEE);
        Vector256<float> s4 = Avx.Shuffle(t4, t6, 0x44), s5 = Avx.Shuffle(t4, t6, 0xEE);
        Vector256<float> s6 = Avx.Shuffle(t5, t7, 0x44), s7 = Avx.Shuffle(t5, t7, 0xEE);
        Avx.Store(dst, Avx.Permute2x128(s0, s4, 0x20));
        Avx.Store(dst + dstStride, Avx.Permute2x128(s1, s5, 0x20));
        Avx.Store(dst + 2L * dstStride, Avx.Permute2x128(s2, s6, 0x20));
        Avx.Store(dst + 3L * dstStride, Avx.Permute2x128(s3, s7, 0x20));
        Avx.Store(dst + 4L * dstStride, Avx.Permute2x128(s0, s4, 0x31));
        Avx.Store(dst + 5L * dstStride, Avx.Permute2x128(s1, s5, 0x31));
        Avx.Store(dst + 6L * dstStride, Avx.Permute2x128(s2, s6, 0x31));
        Avx.Store(dst + 7L * dstStride, Avx.Permute2x128(s3, s7, 0x31));
    }

    // ------------------------------------------------------------------ pooling

    /// <summary>Max / average pooling (average divides by the number of in-bounds taps, as the NCHW path does).</summary>
    private static void PoolAvx(ReadOnlySpan<float> input, Span<float> output, int batch, int channels, int height, int width,
        int outputHeight, int outputWidth, int kernelH, int kernelW, int strideH, int strideW, int padTop, int padLeft, bool max,
        int threads = 1)
    {
        long outVolume = (long)batch * outputHeight * outputWidth * channels;
        if (input.Length < (long)batch * height * width * channels || output.Length < outVolume)
            throw new ArgumentException("NHWC pool buffer too small.");
        if (outVolume == 0) return;
        int vectorEnd = channels & ~7;
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
                        for (; c < vectorEnd; c += 8)
                        {
                            Vector256<float> acc = max ? Vector256.Create(float.NegativeInfinity) : Vector256<float>.Zero;
                            for (int ky = kyBegin; ky < kyEnd; ky++)
                            {
                                float* srcRow = inBatch + ((long)(iy0 + ky) * width + ix0) * channels + c;
                                for (int kx = kxBegin; kx < kxEnd; kx++)
                                {
                                    Vector256<float> v = Avx.LoadVector256(srcRow + (long)kx * channels);
                                    acc = max ? Avx.Max(acc, v) : Avx.Add(acc, v);
                                }
                            }
                            Avx.Store(dst + c, max ? acc : Avx.Divide(acc, Vector256.Create((float)count)));
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
    private static void ResizeNearestAvx(ReadOnlySpan<float> input, Span<float> output, int batch, int channels, int height, int width,
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
    private static void ReduceMeanSpatialAvx(ReadOnlySpan<float> input, Span<float> output, int batch, int channels, int plane)
    {
        if (input.Length < (long)batch * channels * plane || output.Length < (long)batch * channels)
            throw new ArgumentException("NHWC reduce buffer too small.");
        if (plane <= 0) { output.Slice(0, batch * channels).Clear(); return; }
        int vectorEnd = channels & ~7;
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
                    for (; c < vectorEnd; c += 8)
                        Avx.Store(dst + c, Avx.Add(Avx.LoadVector256(dst + c), Avx.LoadVector256(src + c)));
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
    private static void BinaryChannelAvx<TOp>(ReadOnlySpan<float> left, ReadOnlySpan<float> channel, Span<float> output,
        int batch, int channels, int plane, bool channelIsLeft, bool channelPerBatch) where TOp : struct, IBinaryOp
    {
        long volume = (long)batch * plane * channels;
        if (left.Length < volume || output.Length < volume || channel.Length < (channelPerBatch ? (long)batch * channels : channels))
            throw new ArgumentException("NHWC channel broadcast buffer too small.");
        TOp op = default;
        int vectorEnd = channels & ~7;
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
                    for (; c < vectorEnd; c += 8)
                    {
                        Vector256<float> x = Avx.LoadVector256(a + c), y = Avx.LoadVector256(ch + c);
                        Avx.Store(o + c, channelIsLeft ? op.Apply(y, x) : op.Apply(x, y));
                    }
                    for (; c < channels; c++)
                        o[c] = channelIsLeft ? op.Apply(ch[c], a[c]) : op.Apply(a[c], ch[c]);
                }
            }
        }
    }

    /// <summary>Inference batch-norm: (x - mean) * scale / sqrt(var + eps) + bias, per channel.</summary>
    private static void BatchNormAvx(ReadOnlySpan<float> input, Span<float> output, int pixels, int channels,
        ReadOnlySpan<float> scale, ReadOnlySpan<float> bias, ReadOnlySpan<float> mean, ReadOnlySpan<float> variance, float epsilon)
    {
        if (input.Length < (long)pixels * channels || output.Length < (long)pixels * channels ||
            scale.Length < channels || bias.Length < channels || mean.Length < channels || variance.Length < channels)
            throw new ArgumentException("NHWC batch-norm buffer too small.");
        float[] k = new float[channels];
        for (int c = 0; c < channels; c++) k[c] = scale[c] / MathF.Sqrt(variance[c] + epsilon);
        int vectorEnd = channels & ~7;
        fixed (float* inPtr = input, outPtr = output, kPtr = k, meanPtr = mean, biasPtr = bias)
        {
            float* src = inPtr, dst = outPtr;
            for (int p = 0; p < pixels; p++, src += channels, dst += channels)
            {
                int c = 0;
                for (; c < vectorEnd; c += 8)
                {
                    Vector256<float> x = Avx.Subtract(Avx.LoadVector256(src + c), Avx.LoadVector256(meanPtr + c));
                    Avx.Store(dst + c, Avx.Add(Avx.Multiply(x, Avx.LoadVector256(kPtr + c)), Avx.LoadVector256(biasPtr + c)));
                }
                for (; c < channels; c++)
                    dst[c] = (src[c] - mean[c]) * k[c] + bias[c];
            }
        }
    }
}
