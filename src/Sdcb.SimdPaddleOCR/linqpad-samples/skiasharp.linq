<Query Kind="Program">
  <AllowUnsafe>true</AllowUnsafe>
  <NuGetReference>Sdcb.SimdPaddleOCR</NuGetReference>
  <NuGetReference>Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny</NuGetReference>
  <NuGetReference Version="4.151.1">SkiaSharp</NuGetReference>
  <Namespace>Sdcb.SimdPaddleOCR</Namespace>
  <Namespace>Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny</Namespace>
  <Namespace>SkiaSharp</Namespace>
  <Namespace>System.Net.Http</Namespace>
  <Namespace>System.Threading.Tasks</Namespace>
</Query>

async Task Main()
{
	using HttpClient http = new();
	byte[] jpeg = await http.GetByteArrayAsync("https://cv-public.sdcb.ai/2026/sample.jpg");

	using PaddleOcrAll ocr = await PaddleOcrAll.LoadAsync(ChineseV6TinyModels.Default);
	using SKBitmap bitmap = DecodeBgra(jpeg);
	int stride = bitmap.RowBytes;
	unsafe
	{
		PaddleOcrResult result = ocr.Run(
			new ReadOnlySpan<byte>((byte*)bitmap.GetPixels(), stride * bitmap.Height),
			bitmap.Width, bitmap.Height, stride, ImagePixelFormat.Bgra32);
		Console.WriteLine(result.Text);
	}
}

static SKBitmap DecodeBgra(byte[] jpeg)
{
	SKBitmap decoded = SKBitmap.Decode(jpeg) ?? throw new InvalidDataException("Failed to decode image");
	if (decoded.ColorType == SKColorType.Bgra8888) return decoded;
	SKBitmap? bgra = decoded.Copy(SKColorType.Bgra8888);
	decoded.Dispose();
	return bgra ?? throw new InvalidDataException("Failed to convert to BGRA");
}
