namespace Sdcb.SimdPaddleOCR.UnitTests;

public class CharacterBoxTests
{
    [Fact]
    public void MissingAlignmentThrows()
    {
        PaddleOcrLine line = Line(Axis(200, 40), null, 8, 8, 8);
        InvalidOperationException ex = Assert.Throws<InvalidOperationException>(() => line.EstimateCharacterBoxes());
        Assert.Contains("returnCtcAlignment", ex.Message);
    }

    [Fact]
    public void BlankGapIsSplitAtMidpointAndBoxesMeet()
    {
        // Fired runs [0, 2) and [4, 6). The blank gap [2, 4) is cut at column 3.
        // Outer edges stay on the triggers: x = 0 and x = 60 of an 80px line.
        PaddleOcrCtcSpan[] spans =
        [
            new("姓", 0.9f, 0, 2),
            new("名", 0.8f, 4, 6)
        ];
        PaddleOcrCharacterBox[] boxes = Estimate(Axis(80, 20), spans, 8, 8, 8, 0);
        Assert.Equal(2, boxes.Length);
        Equal(boxes[0], "姓", 0.9f, 0, 0, 30, 0, 30, 20, 0, 20);
        Equal(boxes[1], "名", 0.8f, 30, 0, 60, 0, 60, 20, 30, 20);
        Assert.Equal(boxes[0].X2, boxes[1].X1);
        Assert.Equal(boxes[0].Y2, boxes[1].Y1);
        Assert.Equal(boxes[0].X3, boxes[1].X4);
        Assert.Equal(boxes[0].Y3, boxes[1].Y4);
    }

    [Fact]
    public void OddBlankGapSplitsOnAHalfColumn()
    {
        // Gap [1, 4) midpoint is column 2.5 → 25px on an 80px line.
        PaddleOcrCtcSpan[] spans =
        [
            new("A", 1f, 0, 1),
            new("B", 1f, 4, 5)
        ];
        PaddleOcrCharacterBox[] boxes = Estimate(Axis(80, 10), spans, 8, 8, 8, 0);
        Equal(boxes[0], "A", 1f, 0, 0, 25, 0, 25, 10, 0, 10);
        Equal(boxes[1], "B", 1f, 25, 0, 50, 0, 50, 10, 25, 10);
        Assert.Equal(boxes[0].X2, boxes[1].X1);
    }

    [Fact]
    public void RightPaddingUsesContentWidth()
    {
        // Column 0 is 1/10 of the tensor and 1/5 of the content, so 20px of a 100px line.
        PaddleOcrCtcSpan[] spans = [new("A", 1f, 0, 1)];
        PaddleOcrCharacterBox[] boxes = Estimate(Axis(100, 10), spans, 10, 5, 10, 0);
        Equal(boxes[0], "A", 1f, 0, 0, 20, 0, 20, 10, 0, 10);
    }

    [Fact]
    public void Rotation180FlipsTheCrop()
    {
        PaddleOcrCtcSpan[] spans = [new("A", 1f, 0, 2)];
        PaddleOcrCharacterBox[] boxes = Estimate(Axis(100, 10), spans, 10, 10, 10, 180);
        Equal(boxes[0], "A", 1f, 80, 0, 100, 0, 100, 10, 80, 10);
    }

    [Fact]
    public void SlantedQuadInterpolatesBothEdges()
    {
        PaddleOcrDetectionBox box = new(0, 0, 80, 16, 80, 36, 0, 20, 1f);
        PaddleOcrCtcSpan[] spans = [new("中", 1f, 2, 4)];
        PaddleOcrCharacterBox[] boxes = Estimate(box, spans, 8, 8, 8, 0);
        Equal(boxes[0], "中", 1f, 20, 4, 40, 8, 40, 28, 20, 24);
    }

    [Fact]
    public void FullSpanRecoversPerspectiveQuad()
    {
        PaddleOcrDetectionBox box = new(0, 0, 100, 0, 80, 40, 10, 40, 1f);
        PaddleOcrCtcSpan[] spans = [new("文", 1f, 0, 4)];
        PaddleOcrCharacterBox[] boxes = Estimate(box, spans, 4, 4, 4, 0);
        Equal(boxes[0], "文", 1f, box.X1, box.Y1, box.X2, box.Y2, box.X3, box.Y3, box.X4, box.Y4);
    }

    [Fact]
    public void TallQuadReadsFromTheBottomEdge()
    {
        PaddleOcrDetectionBox box = new(10, 10, 30, 10, 30, 90, 10, 90, 1f);
        PaddleOcrCtcSpan[] spans = [new("直", 1f, 0, 1)];
        PaddleOcrCharacterBox[] boxes = Estimate(box, spans, 4, 4, 4, 0);
        Equal(boxes[0], "直", 1f, 10, 90, 10, 70, 30, 70, 30, 90);
    }

    [Fact]
    public void TallQuadUndoesRotation180()
    {
        PaddleOcrDetectionBox box = new(10, 10, 30, 10, 30, 90, 10, 90, 1f);
        PaddleOcrCtcSpan[] spans = [new("直", 1f, 0, 1)];
        PaddleOcrCharacterBox[] boxes = Estimate(box, spans, 4, 4, 4, 180);
        Equal(boxes[0], "直", 1f, 10, 30, 10, 10, 30, 10, 30, 30);
    }

    [Fact]
    public void EmptyAlignmentReturnsNoBoxes()
    {
        PaddleOcrCharacterBox[] boxes = Estimate(Axis(40, 10), [], 4, 4, 4, 0);
        Assert.Empty(boxes);
    }

    [Fact]
    public void RejectsASpanPastTheTimeAxis()
    {
        PaddleOcrCtcSpan[] spans = [new("A", 1f, 0, 9)];
        Assert.Throws<ArgumentOutOfRangeException>(() => Estimate(Axis(40, 10), spans, 4, 4, 4, 0));
    }

    private static PaddleOcrCharacterBox[] Estimate(PaddleOcrDetectionBox box, PaddleOcrCtcSpan[]? spans,
        int timeSteps, int contentWidth, int tensorWidth, int rotation)
        => Line(box, spans, timeSteps, contentWidth, tensorWidth, rotation).EstimateCharacterBoxes();

    private static PaddleOcrLine Line(PaddleOcrDetectionBox box, PaddleOcrCtcSpan[]? spans,
        int timeSteps, int contentWidth, int tensorWidth, int rotation = 0) => new PaddleOcrLine
    {
        Box = box,
        Text = "",
        RecognitionScore = 1,
        ClassificationScore = 0,
        ClassificationLabel = 0,
        AppliedRotationDegrees = rotation,
        EmittedCount = (uint)(spans?.Length ?? 0),
        CtcSpans = spans,
        RecognitionContentWidth = contentWidth,
        RecognitionTensorWidth = tensorWidth,
        RecognitionTimeSteps = timeSteps
    };

    private static PaddleOcrDetectionBox Axis(float width, float height) =>
        new(0, 0, width, 0, width, height, 0, height, 1f);

    private static void Equal(PaddleOcrCharacterBox box, string text, float score,
        float x1, float y1, float x2, float y2, float x3, float y3, float x4, float y4)
    {
        Assert.Equal(text, box.Text);
        Assert.Equal(score, box.Score);
        Near(x1, box.X1);
        Near(y1, box.Y1);
        Near(x2, box.X2);
        Near(y2, box.Y2);
        Near(x3, box.X3);
        Near(y3, box.Y3);
        Near(x4, box.X4);
        Near(y4, box.Y4);
    }

    private static void Near(float expected, float actual) =>
        Assert.InRange(actual, expected - 0.02f, expected + 0.02f);
}
