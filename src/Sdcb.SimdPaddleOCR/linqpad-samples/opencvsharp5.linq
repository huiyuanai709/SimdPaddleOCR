<Query Kind="Program">
  <AllowUnsafe>true</AllowUnsafe>
  <NuGetReference>OpenCvSharp5</NuGetReference>
  <NuGetReference>OpenCvSharp5.runtime.win</NuGetReference>
  <NuGetReference>Sdcb.SimdPaddleOCR</NuGetReference>
  <NuGetReference>Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny</NuGetReference>
  <Namespace>OpenCvSharp</Namespace>
  <Namespace>Sdcb.SimdPaddleOCR</Namespace>
  <Namespace>Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny</Namespace>
  <Namespace>System.Net.Http</Namespace>
</Query>

async Task Main()
{
	using HttpClient http = new();
	byte[] jpeg = await http.GetByteArrayAsync("https://cv-public.sdcb.ai/2026/sample.jpg");

	using PaddleOcrAll ocr = await PaddleOcrAll.LoadAsync(ChineseV6TinyModels.Default);
	using Mat image = Cv2.ImDecode(jpeg, ImreadModes.Color);
	if (image.Empty()) throw new InvalidDataException("无法读取图片");

	int stride = (int)image.Step();
	unsafe
	{
		PaddleOcrResult result = ocr.Run(
			new ReadOnlySpan<byte>((byte*)image.Data, stride * image.Height),
			image.Width, image.Height, stride, ImagePixelFormat.Bgr24);
		Console.WriteLine(result.Text);
	}
}
