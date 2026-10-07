namespace Sdcb.SimdPaddleOCR.UnitTests;

public class ParallelismTests
{
    [Theory]
    [InlineData(4, 2, 2)]
    [InlineData(0, 2, 2)]
    [InlineData(0, 16, 4)]
    [InlineData(0, 1, 1)]
    [InlineData(0, 3, 3)]
    [InlineData(1, 2, 1)]
    [InlineData(8, 4, 4)]
    [InlineData(16, 32, 16)]
    public void ResolveLineWorkers(int requested, int processorCount, int expected) =>
        Assert.Equal(expected, Parallelism.ResolveLineWorkers(requested, processorCount));

    [Theory]
    [InlineData(8, 8, 2, 4)]
    [InlineData(8, 8, 1, 8)]
    [InlineData(8, 8, 4, 2)]
    [InlineData(20, 8, 2, 8)]
    [InlineData(5, 8, 2, 3)]
    [InlineData(1, 8, 4, 1)]
    [InlineData(3, 8, 2, 2)]
    public void RecognizeBatchSize_SpreadsAcrossWorkers(int group, int maxBatch, int workers, int expected) =>
        Assert.Equal(expected, Parallelism.RecognizeBatchSize(group, maxBatch, workers));

    [Theory]
    [InlineData(2, 2, 1)]
    [InlineData(1, 2, 2)]
    [InlineData(4, 16, 4)]
    [InlineData(4, 8, 2)]
    [InlineData(1, 8, 8)]
    [InlineData(1, 32, 8)]
    public void ResolveRecognizerIntraOp(int lineWorkers, int processorCount, int expected) =>
        Assert.Equal(expected, Parallelism.ResolveRecognizerIntraOp(lineWorkers, processorCount));

    [Theory]
    [InlineData(2, 12, 4)]
    [InlineData(2, 4, 2)]
    [InlineData(1, 2, 2)]
    [InlineData(1, 1, 1)]
    public void ResolveGpuCropWorkers_StaysAtMostFour(int lineWorkers, int processorCount, int expected) =>
        Assert.Equal(expected, Parallelism.ResolveGpuCropWorkers(lineWorkers, processorCount));
}
