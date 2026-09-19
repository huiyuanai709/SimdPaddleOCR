<Query Kind="Program">
  <NuGetReference>Sdcb.SimdPaddleOCR</NuGetReference>
  <NuGetReference>Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny</NuGetReference>
  <NuGetReference Version="3.1.11">SixLabors.ImageSharp</NuGetReference>
  <Namespace>Sdcb.SimdPaddleOCR</Namespace>
  <Namespace>Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny</Namespace>
  <Namespace>SixLabors.ImageSharp</Namespace>
  <Namespace>SixLabors.ImageSharp.PixelFormats</Namespace>
  <Namespace>System.Net.Http</Namespace>
  <Namespace>System.Runtime.InteropServices</Namespace>
</Query>

async Task Main()
{
	using HttpClient http = new();
	byte[] jpeg = await http.GetByteArrayAsync("https://cv-public.sdcb.ai/2026/sample.jpg");

	using PaddleOcrAll ocr = await PaddleOcrAll.LoadAsync(ChineseV6TinyModels.Default);
	using Image<Rgba32> image = Image.Load<Rgba32>(jpeg);
	if (!image.DangerousTryGetSinglePixelMemory(out Memory<Rgba32> memory))
		throw new InvalidDataException("图片像素不是连续内存");

	PaddleOcrResult result = ocr.Run(MemoryMarshal.AsBytes(memory.Span), image.Width, image.Height,
		format: ImagePixelFormat.Rgba32);
	Console.WriteLine(result.Text);
}
