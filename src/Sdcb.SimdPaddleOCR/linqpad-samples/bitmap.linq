<Query Kind="Program">
  <AllowUnsafe>true</AllowUnsafe>
  <NuGetReference>Sdcb.SimdPaddleOCR</NuGetReference>
  <NuGetReference>Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny</NuGetReference>
  <NuGetReference>System.Drawing.Common</NuGetReference>
  <Namespace>Sdcb.SimdPaddleOCR</Namespace>
  <Namespace>Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny</Namespace>
  <Namespace>System.Drawing</Namespace>
  <Namespace>System.Drawing.Imaging</Namespace>
  <Namespace>System.Net.Http</Namespace>
</Query>

async Task Main()
{
	using HttpClient http = new();
	byte[] jpeg = await http.GetByteArrayAsync("https://cv-public.sdcb.ai/2026/sample.jpg");

	using PaddleOcrAll ocr = await PaddleOcrAll.LoadAsync(ChineseV6TinyModels.Default);
	using MemoryStream stream = new(jpeg);
	using Bitmap bitmap = new(stream);
	Rectangle rectangle = new(0, 0, bitmap.Width, bitmap.Height);
	BitmapData data = bitmap.LockBits(rectangle, ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);
	try
	{
		unsafe
		{
			PaddleOcrResult result = ocr.Run(
				new ReadOnlySpan<byte>((byte*)data.Scan0, data.Stride * bitmap.Height),
				bitmap.Width, bitmap.Height, data.Stride, ImagePixelFormat.Bgra32);
			Console.WriteLine(result.Text);
		}
	}
	finally
	{
		bitmap.UnlockBits(data);
	}
}
