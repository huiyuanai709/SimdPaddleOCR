using Sdcb.SimdPaddleOCR.Kernels;
using Sdcb.SimdPaddleOCR.Models.TextLineOrientation;
using Sdcb.SimdPaddleOCR.OnnxSharp;
using static Sdcb.SimdPaddleOCR.UnitTests.KernelCorrectnessTests;

namespace Sdcb.SimdPaddleOCR.UnitTests;

/// <summary>
/// Public <see cref="InferenceSession.Run"/> still takes logical NCHW even when
/// the planner stored the graph input channels-last. Internal writers that
/// already wrote NHWC into <see cref="InferenceSession.InputData"/> use
/// <see cref="InferenceSession.RunInternal"/> and skip the convert.
/// </summary>
public class InferenceSessionRunContractTests
{
    [Fact]
    public void PublicRun_AcceptsLogicalNchw_WhenGraphInputIsNhwc()
    {
        using Stream stream = TextLineOrientationModel.OpenRead();
        using Model model = Model.Load(stream);
        CompiledModel compiled = new(model, [1, 3, 80, 160], 1);
        try
        {
            using InferenceSession session = compiled.CreateRequest();
            if (!session.InputIsNhwc)
                return;

            float[] nchw = new float[1 * 3 * 80 * 160];
            Random rng = new(1);
            for (int i = 0; i < nchw.Length; i++)
                nchw[i] = (float)(rng.NextDouble() * 2 - 1);

            float[] fromPublic = session.Run(nchw);

            float[] nhwc = new float[nchw.Length];
            Nhwc.NchwToNhwc(nchw, nhwc, 1, 3, 80 * 160, 1);
            nhwc.AsSpan().CopyTo(session.InputData);
            ReadOnlySpan<float> fromInternal = session.RunInternal(session.InputData);
            AssertClose(fromPublic, fromInternal, 1e-5f, 1e-5f);

            nchw.AsSpan().CopyTo(session.InputData);
            float[] fromOverlap = session.Run(session.InputData);
            AssertClose(fromPublic, fromOverlap, 1e-5f, 1e-5f);
        }
        finally
        {
            compiled.Dispose();
        }
    }
}
