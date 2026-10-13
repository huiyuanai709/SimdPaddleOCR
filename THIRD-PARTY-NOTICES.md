# Third-party notices

The source code written for this repository is released under the Apache
License, Version 2.0. The following files and assets are supplied under their
respective upstream terms.

## PP-OCRv6 and text-line orientation model assets

The ONNX model files and dictionaries in `models/` are derived from the
PaddleOCR/PP-OCR model distribution. The upstream model metadata identifies
these assets as Apache-2.0. The Apache License 2.0 text is included in the
repository root as [`LICENSE`](LICENSE).

Upstream attribution: Copyright (c) 2016 PaddlePaddle Authors. All Rights
Reserved.

Sources:

- <https://github.com/PaddlePaddle/PaddleOCR>
- <https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_det_onnx>
- <https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_rec_onnx>
- <https://huggingface.co/PaddlePaddle/PP-OCRv6_small_det_onnx>
- <https://huggingface.co/PaddlePaddle/PP-OCRv6_small_rec_onnx>
- <https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_det_onnx>
- <https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec_onnx>
- <https://huggingface.co/PaddlePaddle/PP-LCNet_x0_25_textline_ori_onnx>
- <https://huggingface.co/PaddlePaddle/PP-OCRv4_mobile_seal_det>
- <https://huggingface.co/PaddlePaddle/PP-OCRv4_server_seal_det>

The seal detection weights ship as Paddle inference models (`inference.json` +
`inference.pdiparams`). The ONNX files in this repository
(`models/seal_v4_mobile_det.onnx`, `models/seal_v4_server_det.onnx`) were
exported with paddle2onnx 2.1.0 at opset 11 and simplified with onnxsim.
`seal_v4_server_det.onnx` stores FLOAT initializers as FLOAT16 so the blob
stays within GitHub's file-size limit; `Model.Load` widens those weights back
to float32 before the graph runs. They remain Apache-2.0 assets from PaddlePaddle.

The model files are distributed for use with this project and remain subject
to the upstream model terms. PaddleOCR and PP-OCR are trademarks of their
respective owners; this project is not endorsed by or affiliated with them.

## Example dependencies

The example projects use third-party NuGet packages. They retain their own
licenses and notices, including Avalonia (MIT), SkiaSharp (MIT), OpenCvSharp5
(Apache-2.0), and Six Labors ImageSharp (Six Labors Split License). Consult
the corresponding package distributions before redistributing an example
application.
