namespace Sdcb.SimdPaddleOCR.OnnxSharp;

/// <summary>
/// Per-request scratch storage for resize implementations. A workspace is
/// intentionally not thread-safe; the owning inference request must be used
/// by one inference at a time.
/// </summary>
internal sealed class ResizeWorkspace
{
    public int[] XOffsets = [];
    public short[] XCoefficients = [];
    public int[] Row0 = [];
    public int[] Row1 = [];

    // Row scratch for the row-parallel preprocessing path. Two slots per
    // worker (Row0/Row1); worker 0 reuses the fields above so the serial
    // path allocates nothing extra.
    private int[][] _parallelRows = [];

    /// <summary>
    /// Allocates the row scratch every worker will use, plus the shared
    /// horizontal coefficient buffers. Must be called on one thread before
    /// entering the row-parallel region.
    /// </summary>
    /// <remarks>
    /// Growth assigns a fresh array to a shared field, so doing it lazily
    /// inside the parallel body races: two workers growing at once lose one
    /// resize and leave null slots behind (observed as a NullReferenceException
    /// in DetRow on medium). Preparing here keeps the parallel body read-only.
    /// </remarks>
    public void PrepareRows(int workers, int width)
    {
        Ensure(width);
        int rowCount = checked(width * 3);
        int need = checked(workers * 2);
        if (_parallelRows.Length < need) Array.Resize(ref _parallelRows, need);
        for (int worker = 1; worker < workers; worker++)
        {
            int slot = worker * 2;
            if (_parallelRows[slot] is null || _parallelRows[slot].Length < rowCount)
            {
                _parallelRows[slot] = new int[rowCount];
                _parallelRows[slot + 1] = new int[rowCount];
            }
        }
    }

    /// <summary>
    /// Row scratch for one worker. Read-only after <see cref="PrepareRows"/>;
    /// the parallel body must not allocate through this.
    /// </summary>
    public (int[] Row0, int[] Row1) RowsFor(int worker)
    {
        if (worker == 0) return (Row0, Row1);
        int slot = worker * 2;
        return (_parallelRows[slot], _parallelRows[slot + 1]);
    }
    public void Ensure(int width)
    {
        if (XOffsets.Length < width) Array.Resize(ref XOffsets, width);
        int coefficientCount = checked(width * 2);
        if (XCoefficients.Length < coefficientCount) Array.Resize(ref XCoefficients, coefficientCount);
        int rowCount = checked(width * 3);
        if (Row0.Length < rowCount) Array.Resize(ref Row0, rowCount);
        if (Row1.Length < rowCount) Array.Resize(ref Row1, rowCount);
    }
}
