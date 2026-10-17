#if !NETSTANDARD2_0
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Wasm;

namespace Sdcb.SimdPaddleOCR.Kernels;

// WebAssembly SIMD128 (PackedSimd) element-wise helpers. WASM SIMD has no fused
// multiply-add, so every polynomial step is a separate multiply then add in the
// same order as the Vector<float> kernels (ErfExactVector / GeluVectorExact),
// keeping results bit-identical to the Vector fallback.
internal static partial class SimdKernels
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> PackedMulAdd(Vector128<float> x, Vector128<float> y, Vector128<float> z) =>
        PackedSimd.Add(PackedSimd.Multiply(x, y), z);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool PackedAllTrue(Vector128<float> mask) => PackedSimd.AllTrue(mask.AsInt32());

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    internal static Vector128<float> ErfVectorPackedSimd(Vector128<float> value)
    {
        Vector128<float> one = Vector128.Create(1f), two = Vector128.Create(2f), four = Vector128.Create(4f);
        Vector128<float> abs = PackedSimd.Abs(value);
        Vector128<float> sq = PackedSimd.Multiply(abs, abs);
        Vector128<float> lessOne = PackedSimd.CompareLessThan(abs, one);
        Vector128<float> lessTwo = PackedSimd.CompareLessThan(abs, two);
        Vector128<float> atLeastFour = PackedSimd.CompareGreaterThanOrEqual(abs, four);
        // Feature-map values are spatially correlated, so many vectors stay
        // entirely inside one approximation interval (same split as the
        // Vector<float> kernel). Mixed vectors evaluate all three polynomials.
        Vector128<float> result;
        if (PackedAllTrue(lessOne))
            result = ErfPolySmallPackedSimd(abs, sq);
        else if (PackedAllTrue(atLeastFour))
            result = one;
        else if (PackedAllTrue(PackedSimd.AndNot(lessTwo, lessOne)))
            result = ErfPolyMiddlePackedSimd(abs);
        else if (PackedAllTrue(PackedSimd.Not(PackedSimd.Or(lessTwo, atLeastFour))))
            result = ErfPolyLargePackedSimd(abs);
        else
        {
            Vector128<float> small = ErfPolySmallPackedSimd(abs, sq);
            Vector128<float> middle = ErfPolyMiddlePackedSimd(abs);
            Vector128<float> large = ErfPolyLargePackedSimd(abs);
            result = Vector128.ConditionalSelect(lessTwo, middle, large);
            result = Vector128.ConditionalSelect(lessOne, small, result);
            result = Vector128.ConditionalSelect(atLeastFour, one, result);
        }
        return Vector128.ConditionalSelect(PackedSimd.CompareLessThan(value, Vector128<float>.Zero),
            PackedSimd.Negate(result), result);
    }

    /// <summary>One 128-bit vector of <see cref="Gelu"/>: same operation order as <see cref="GeluVectorExact"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector128<float> GeluVectorPackedSimd(Vector128<float> value)
    {
        Vector128<float> activated = PackedSimd.Add(
            ErfVectorPackedSimd(PackedSimd.Multiply(value, Vector128.Create(0.70710678118654752f))), Vector128.Create(1f));
        return PackedSimd.Multiply(PackedSimd.Multiply(value, activated), Vector128.Create(0.5f));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> ErfPolySmallPackedSimd(Vector128<float> a, Vector128<float> s)
    {
        Vector128<float> p = Vector128.Create(1.0590875083315439e-6f);
        p = PackedMulAdd(s, p, Vector128.Create(-1.3906452410711274e-5f));
        p = PackedMulAdd(s, p, Vector128.Create(1.1955437252428243e-4f));
        p = PackedMulAdd(s, p, Vector128.Create(-8.542475960079766e-4f));
        p = PackedMulAdd(s, p, Vector128.Create(5.223771899427153e-3f));
        p = PackedMulAdd(s, p, Vector128.Create(-2.686612888828867e-2f));
        p = PackedMulAdd(s, p, Vector128.Create(1.128379122662546e-1f));
        p = PackedMulAdd(s, p, Vector128.Create(-3.761263888304388e-1f));
        p = PackedMulAdd(s, p, Vector128.Create(1.1283791670929921f));
        return PackedSimd.Multiply(a, p);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> ErfPolyMiddlePackedSimd(Vector128<float> a)
    {
        Vector128<float> z = PackedSimd.Subtract(a, Vector128.Create(1.5f));
        Vector128<float> p = Vector128.Create(-2.400667527836574e-3f);
        p = PackedMulAdd(z, p, Vector128.Create(-3.8855162788028288e-3f));
        p = PackedMulAdd(z, p, Vector128.Create(1.9332860298601401e-2f));
        p = PackedMulAdd(z, p, Vector128.Create(-1.501360723487143e-2f));
        p = PackedMulAdd(z, p, Vector128.Create(-4.4599369472787913e-2f));
        p = PackedMulAdd(z, p, Vector128.Create(1.3876136033191724e-1f));
        p = PackedMulAdd(z, p, Vector128.Create(-1.7839541988688759e-1f));
        p = PackedMulAdd(z, p, Vector128.Create(1.1893013063163335e-1f));
        return PackedMulAdd(z, p, Vector128.Create(9.661051464140682e-1f));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> ErfPolyLargePackedSimd(Vector128<float> a)
    {
        Vector128<float> z = PackedSimd.Subtract(a, Vector128.Create(3f));
        Vector128<float> p = Vector128.Create(-8.875076532056503e-5f);
        p = PackedMulAdd(z, p, Vector128.Create(3.880528563007222e-4f));
        p = PackedMulAdd(z, p, Vector128.Create(-7.781201071142843e-4f));
        p = PackedMulAdd(z, p, Vector128.Create(1.0255980996254569e-3f));
        p = PackedMulAdd(z, p, Vector128.Create(-1.0307062838910627e-3f));
        p = PackedMulAdd(z, p, Vector128.Create(7.858608012972956e-4f));
        p = PackedMulAdd(z, p, Vector128.Create(-4.193605385522123e-4f));
        p = PackedMulAdd(z, p, Vector128.Create(1.3951109720745927e-4f));
        return PackedMulAdd(z, p, Vector128.Create(9.999779388683872e-1f));
    }
}
#endif
