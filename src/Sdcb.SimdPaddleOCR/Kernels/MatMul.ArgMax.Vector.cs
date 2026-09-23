using System.Numerics;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;

using static Sdcb.SimdPaddleOCR.Kernels.SimdOps;

namespace Sdcb.SimdPaddleOCR.Kernels;

internal static partial class MatMul
{
    private static readonly Vector<float> ArgMaxLanesVector = CreateArgMaxLanes();

    private static Vector<float> CreateArgMaxLanes()
    {
        int width = Vector<float>.Count;
        float[] lanes = new float[width];
        for (int i = 0; i < width; i++)
            lanes[i] = i;
        return new Vector<float>(lanes);
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void MatMulArgMaxPackedVector(ReadOnlySpan<float> input,
        ReadOnlySpan<float> weights, ReadOnlySpan<float> packedWeights,
        ReadOnlySpan<float> bias, Span<int> indices, Span<float> scores,
        int batch, int rows, int inner, int columns, int threads)
    {
        int full4 = rows / 4, tail = rows - full4 * 4;
        int rowJobs = full4 + tail;
        int jobs = checked(batch * rowJobs);
        int workers = Math.Min(ArgMaxWorkers(batch, rows, inner, columns, threads), jobs);
        bool hasBias = !bias.IsEmpty;
        fixed (float* inputPin = input, weightsPin = weights,
            packedPin = packedWeights, biasPin = bias)
        fixed (int* indicesPin = indices)
        fixed (float* scoresPin = scores)
        {
            float* inputPtr = inputPin, weightsPtr = weightsPin,
                packedPtr = packedPin, biasPtr = biasPin;
            int* indicesPtr = indicesPin;
            float* scoresPtr = scoresPin;
            void Worker(int w)
            {
                for (int job = w; job < jobs; job += workers)
                {
                    int b = job / rowJobs, j = job - b * rowJobs;
                    if (j < full4)
                        MatMulArgMaxPackedVectorRows4(inputPtr, weightsPtr, packedPtr,
                            biasPtr, hasBias, indicesPtr, scoresPtr, b, j * 4,
                            rows, inner, columns);
                    else
                        MatMulArgMaxPackedVectorRows1(inputPtr, weightsPtr, packedPtr,
                            biasPtr, hasBias, indicesPtr, scoresPtr, b,
                            full4 * 4 + j - full4, rows, inner, columns);
                }
            }
            if (workers > 1) Parallel.For(0, workers, Worker);
            else Worker(0);
        }
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void MatMulArgMaxPackedVectorRows4(float* input,
        float* weights, float* packed, float* bias, bool hasBias,
        int* indices, float* scores, int batch, int row,
        int rows, int inner, int columns)
    {
        int width = Vector<float>.Count;
        int parts = 16 / width;
        Span<Vector<float>> vectorMaxima = stackalloc Vector<float>[4];
        Span<Vector<float>> vectorIndices = stackalloc Vector<float>[4];
        float* maxima = stackalloc float[4];
        int* best = stackalloc int[4];
        vectorMaxima[0] = vectorMaxima[1] = vectorMaxima[2] = vectorMaxima[3] =
            new Vector<float>(float.NegativeInfinity);
        int inputBase = (batch * rows + row) * inner;
        int col = 0;
        Span<Vector<float>> acc = stackalloc Vector<float>[16];
        for (; col <= columns - 16; col += 16)
        {
            for (int i = 0; i < 4 * parts; i++)
                acc[i] = Vector<float>.Zero;
            float* tile = packed + (col / 16) * inner * 16;
            for (int k = 0; k < inner; k++)
            {
                Vector<float> v0 = new(input[inputBase + k]);
                Vector<float> v1 = new(input[inputBase + inner + k]);
                Vector<float> v2 = new(input[inputBase + inner * 2 + k]);
                Vector<float> v3 = new(input[inputBase + inner * 3 + k]);
                for (int p = 0; p < parts; p++)
                {
                    Vector<float> w = VectorLoad(tile + k * 16 + p * width);
                    acc[p] = VectorAddMul(acc[p], w, v0);
                    acc[parts + p] = VectorAddMul(acc[parts + p], w, v1);
                    acc[parts * 2 + p] = VectorAddMul(acc[parts * 2 + p], w, v2);
                    acc[parts * 3 + p] = VectorAddMul(acc[parts * 3 + p], w, v3);
                }
            }
            if (hasBias)
            {
                for (int p = 0; p < parts; p++)
                {
                    Vector<float> b = VectorLoad(bias + col + p * width);
                    acc[p] += b;
                    acc[parts + p] += b;
                    acc[parts * 2 + p] += b;
                    acc[parts * 3 + p] += b;
                }
            }
            for (int p = 0; p < parts; p++)
            {
                int column = col + p * width;
                UpdateVector(acc[p], column, 0, vectorMaxima, vectorIndices);
                UpdateVector(acc[parts + p], column, 1, vectorMaxima, vectorIndices);
                UpdateVector(acc[parts * 2 + p], column, 2, vectorMaxima, vectorIndices);
                UpdateVector(acc[parts * 3 + p], column, 3, vectorMaxima, vectorIndices);
            }
        }
        ReduceVector(vectorMaxima, vectorIndices, maxima, best);
        FinishScalarTail(input, weights, bias, hasBias, indices, scores,
            batch, row, 4, rows, inner, columns, col, maxima, best);
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void MatMulArgMaxPackedVectorRows1(float* input,
        float* weights, float* packed, float* bias, bool hasBias,
        int* indices, float* scores, int batch, int row,
        int rows, int inner, int columns)
    {
        int width = Vector<float>.Count;
        int parts = 16 / width;
        Span<Vector<float>> vectorMaxima = stackalloc Vector<float>[1];
        Span<Vector<float>> vectorIndices = stackalloc Vector<float>[1];
        float* maxima = stackalloc float[1];
        int* best = stackalloc int[1];
        vectorMaxima[0] = new Vector<float>(float.NegativeInfinity);
        int inputBase = (batch * rows + row) * inner;
        int col = 0;
        Span<Vector<float>> acc = stackalloc Vector<float>[4];
        for (; col <= columns - 16; col += 16)
        {
            for (int p = 0; p < parts; p++)
                acc[p] = Vector<float>.Zero;
            float* tile = packed + (col / 16) * inner * 16;
            for (int k = 0; k < inner; k++)
            {
                Vector<float> v = new(input[inputBase + k]);
                for (int p = 0; p < parts; p++)
                    acc[p] = VectorAddMul(acc[p], VectorLoad(tile + k * 16 + p * width), v);
            }
            if (hasBias)
            {
                for (int p = 0; p < parts; p++)
                    acc[p] += VectorLoad(bias + col + p * width);
            }
            for (int p = 0; p < parts; p++)
                UpdateVector(acc[p], col + p * width, 0, vectorMaxima, vectorIndices);
        }
        ReduceVector(vectorMaxima, vectorIndices, maxima, best);
        FinishScalarTail(input, weights, bias, hasBias, indices, scores,
            batch, row, 1, rows, inner, columns, col, maxima, best);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void UpdateVector(Vector<float> values, int column, int row,
        Span<Vector<float>> maxima, Span<Vector<float>> indices)
    {
        Vector<float> previous = maxima[row];
        Vector<int> replace = Vector.GreaterThan(values, previous);
        maxima[row] = Vector.Max(previous, values);
        Vector<float> candidate = new Vector<float>(column) + ArgMaxLanesVector;
        indices[row] = Vector.ConditionalSelect(replace, candidate, indices[row]);
    }

    private static unsafe void ReduceVector(ReadOnlySpan<Vector<float>> vectorMaxima,
        ReadOnlySpan<Vector<float>> vectorIndices, float* maxima, int* best)
    {
        int width = Vector<float>.Count;
        for (int row = 0; row < vectorMaxima.Length; row++)
        {
            Vector<float> values = vectorMaxima[row];
            Vector<float> positions = vectorIndices[row];
            float maximum = values.GetElement(0);
            int index = (int)positions.GetElement(0);
            for (int lane = 1; lane < width; lane++)
            {
                float value = values.GetElement(lane);
                int candidate = (int)positions.GetElement(lane);
                if (value > maximum || value == maximum && candidate < index)
                {
                    maximum = value;
                    index = candidate;
                }
            }
            if (!MathCompat.IsFinite(maximum))
                throw new InvalidDataException("Recognizer output is invalid.");
            maxima[row] = maximum;
            best[row] = index;
        }
    }
}
