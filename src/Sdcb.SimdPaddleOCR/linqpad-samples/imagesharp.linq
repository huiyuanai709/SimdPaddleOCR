<Query Kind="Program">
  <NuGetReference>Sdcb.SimdPaddleOCR</NuGetReference>
  <NuGetReference>Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny</NuGetReference>
  <NuGetReference Version="3.1.11">SixLabors.ImageSharp</NuGetReference>
  <Namespace>Sdcb.SimdPaddleOCR</Namespace>
  <Namespace>Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny</Namespace>
  <Namespace>SixLabors.ImageSharp</Namespace>
  <Namespace>SixLabors.ImageSharp.Formats</Namespace>
  <Namespace>SixLabors.ImageSharp.PixelFormats</Namespace>
  <Namespace>System.Net.Http</Namespace>
  <Namespace>System.Runtime.InteropServices</Namespace>
  <Namespace>System.Threading.Tasks</Namespace>
</Query>

async Task Main()
{
	using HttpClient http = new();
	byte[] jpeg = await http.GetByteArrayAsync("https://cv-public.sdcb.ai/2026/sample.jpg");

	using PaddleOcrAll ocr = await PaddleOcrAll.LoadAsync(ChineseV6TinyModels.Default);
	using Image<Rgba32> image = Image.Load<Rgba32>(CreateDecoderOptions(), jpeg);
	Rgba32[]? packedCopy = null;
	if (!image.DangerousTryGetSinglePixelMemory(out Memory<Rgba32> memory))
	{
		packedCopy = new Rgba32[checked(image.Width * image.Height)];
		image.CopyPixelDataTo(packedCopy);
		memory = packedCopy;
	}

	PaddleOcrResult result = ocr.Run(MemoryMarshal.AsBytes(memory.Span), image.Width, image.Height,
		format: ImagePixelFormat.Rgba32);
	Console.WriteLine(result.Text);
}

// Clone Default so PNG/JPEG decoders stay registered. Do not set this on Configuration.Default.
// Default allocator splits pixels into 4MB chunks; large images then fail DangerousTryGetSinglePixelMemory.
static DecoderOptions CreateDecoderOptions()
{
	Configuration configuration = Configuration.Default.Clone();
	configuration.PreferContiguousImageBuffers = true;
	return new DecoderOptions { Configuration = configuration };
}
