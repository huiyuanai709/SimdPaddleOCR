namespace Sdcb.SimdPaddleOCR;

/// <summary>
/// Line-stage parallelism: workers are CLS/REC sessions, intra-op is leftover
/// cores inside each REC session. Never oversubscribe ProcessorCount.
/// </summary>
internal static class Parallelism
{
    public const int MaxLineWorkers = 16;
    public const int MaxAutoLineWorkers = 4;
    // Raised from 4: the channels-last GEMM path scales linearly to 8 cores,
    // so a lone REC worker should not idle half the machine.
    public const int MaxRecognizerIntraOpThreads = 8;

    public static int ResolveLineWorkers(int lineWorkerCount) =>
        ResolveLineWorkers(lineWorkerCount, Environment.ProcessorCount);

    // Positive LineWorkerCount is a maximum, clamped to the CPU. 0 (auto) fills
    // one worker per core up to MaxAutoLineWorkers so 2-core machines use both
    // cores as sessions instead of ProcessorCount/4 → 1.
    public static int ResolveLineWorkers(int lineWorkerCount, int processorCount)
    {
        int cpu = Math.Max(1, processorCount);
        int requested = lineWorkerCount > 0
            ? lineWorkerCount
            : Math.Min(cpu, MaxAutoLineWorkers);
        return MathCompat.Clamp(Math.Min(requested, cpu), 1, MaxLineWorkers);
    }

    public const int MaxAutoCropWorkers = 16;

    // The perspective crop runs in an exclusive window between DET and the
    // line stage, so it can use cores the line workers will claim later —
    // the same reasoning behind ResolveDetectorIntraThreads. It still never
    // takes fewer than the line workers, so a machine that asked for more
    // parallelism is not silently capped here.
    public static int ResolveCropWorkers(int lineWorkerCount) =>
        ResolveCropWorkers(lineWorkerCount, Environment.ProcessorCount);

    public static int ResolveCropWorkers(int lineWorkerCount, int processorCount)
    {
        int cpu = Math.Max(1, processorCount);
        int lines = ResolveLineWorkers(lineWorkerCount, cpu);
        return MathCompat.Clamp(Math.Max(lines, Math.Min(cpu, MaxAutoCropWorkers)), 1, MaxAutoCropWorkers);
    }

    public static int ResolveRecognizerIntraOp(int lineWorkers) =>
        ResolveRecognizerIntraOp(lineWorkers, Environment.ProcessorCount);

    // Leftover cores after placing line workers, capped at 8. Two workers on a
    // 2-core machine stay intra-op 1; a single worker gets the second core.
    public static int ResolveRecognizerIntraOp(int lineWorkers, int processorCount)
    {
        int cpu = Math.Max(1, processorCount);
        int workers = Math.Max(1, lineWorkers);
        return MathCompat.Clamp(cpu / workers, 1, MaxRecognizerIntraOpThreads);
    }
}
