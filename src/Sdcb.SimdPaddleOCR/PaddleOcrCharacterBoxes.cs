namespace Sdcb.SimdPaddleOCR;

/// <summary>
/// Estimates one source-image quad per CTC token. The recognizer never sees
/// the detection quad, so this is a pure function of the alignment captured
/// while decoding and of the line geometry already on <see cref="PaddleOcrLine"/>.
/// Each span stays the fired run (repeated frames in, blank frames out). The
/// blank gap between two spans is split at its midpoint, so neighboring boxes
/// share an edge; an outer edge with no neighbor stays on the trigger.
/// </summary>
public static class PaddleOcrCharacterBoxes
{
    public static PaddleOcrCharacterBox[] Estimate(PaddleOcrLine line)
    {
        if (line is null) throw new ArgumentNullException(nameof(line));
        return Estimate(line.Box, line.CtcSpans, line.RecognitionTimeSteps,
            line.RecognitionContentWidth, line.RecognitionTensorWidth, line.AppliedRotationDegrees);
    }

    public static PaddleOcrCharacterBox[] Estimate(in PaddleOcrDetectionBox lineBox,
        in PaddleOcrRecognitionResult recognition, int appliedRotationDegrees = 0)
        => Estimate(lineBox, recognition.CtcSpans, recognition.TimeSteps,
            recognition.ResizedWidth, recognition.TensorWidth, appliedRotationDegrees);

    public static PaddleOcrCharacterBox[] Estimate(in PaddleOcrDetectionBox lineBox,
        PaddleOcrCtcSpan[]? spans, int timeSteps, int contentWidth, int tensorWidth,
        int appliedRotationDegrees = 0)
    {
        if (spans is null)
            throw new InvalidOperationException(
                "CTC alignment was not captured. Pass returnCtcAlignment: true to Run or Recognize.");
        return PPOCRCrop.EstimateCharacterBoxes(lineBox, spans, timeSteps, contentWidth, tensorWidth,
            appliedRotationDegrees);
    }
}
