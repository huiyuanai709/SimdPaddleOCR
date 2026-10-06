using Sdcb.SimdPaddleOCR.Backends.Metal;

namespace Sdcb.SimdPaddleOCR.UnitTests;

public class VulkanBudgetTests
{
    [Fact]
    public void EmptySelectorIsDefault()
    {
        OcrVulkan.VulkanSelector parsed = OcrVulkan.ParseSelector(null);
        Assert.True(parsed.IsDefault);
        Assert.Equal(-1, parsed.Index);
        Assert.Null(parsed.Name);
    }

    [Theory]
    [InlineData("0", 0)]
    [InlineData("1", 1)]
    [InlineData(" 2 ", 2)]
    public void NumericSelectorIsEnumerateIndex(string text, int index)
    {
        OcrVulkan.VulkanSelector parsed = OcrVulkan.ParseSelector(text);
        Assert.False(parsed.IsDefault);
        Assert.Equal(index, parsed.Index);
        Assert.Null(parsed.Name);
    }

    [Theory]
    [InlineData("MX450")]
    [InlineData("llvmpipe")]
    [InlineData("UHD 630")]
    public void TextSelectorIsNameSubstring(string text)
    {
        OcrVulkan.VulkanSelector parsed = OcrVulkan.ParseSelector(text);
        Assert.False(parsed.IsDefault);
        Assert.Equal(-1, parsed.Index);
        Assert.Equal(text, parsed.Name);
    }

    [Theory]
    [InlineData(0UL, 256UL << 20)]
    [InlineData(2UL << 30, 256UL << 20)]
    [InlineData(3UL << 30, 256UL << 20)]
    [InlineData(4UL << 30, 512UL << 20)]
    [InlineData(8UL << 30, 512UL << 20)]
    [InlineData(128UL << 20, 64UL << 20)]
    public void BufferCapTracksHeap(ulong deviceLocal, ulong expected)
    {
        ulong previous = OcrVulkan.BufferByteCapOverride ?? 0;
        OcrVulkan.BufferByteCapOverride = null;
        try
        {
            Assert.Equal(expected, OcrVulkan.BufferByteCap(deviceLocal));
        }
        finally
        {
            OcrVulkan.BufferByteCapOverride = previous == 0 ? null : previous;
        }
    }

    [Fact]
    public void OverrideReplacesHeapCap()
    {
        OcrVulkan.BufferByteCapOverride = 1024;
        try
        {
            Assert.Equal(1024UL, OcrVulkan.BufferByteCap(8UL << 30));
        }
        finally
        {
            OcrVulkan.BufferByteCapOverride = null;
        }
    }

    [Theory]
    [InlineData(2UL << 30, 2, 1)]
    [InlineData((4UL << 30) - 1, 6, 1)]
    [InlineData(4UL << 30, 2, 2)]
    [InlineData(8UL << 30, 6, 6)]
    [InlineData(0UL, 2, 2)]
    public void SmallHeapsRecommendOneEngine(ulong deviceLocal, int requested, int expected) =>
        Assert.Equal(expected, OcrVulkan.RecommendedEngineCount(deviceLocal, requested));
}

public class MetalShaderTests
{
    [Fact]
    public void EmbeddedShadersMatchTheAotPreserveList()
    {
        var assembly = typeof(OcrBackend).Assembly;
        string[] embedded = assembly.GetManifestResourceNames()
            .Where(name => name.StartsWith("Sdcb.SimdPaddleOCR.Backends.Metal.Shaders.", StringComparison.Ordinal)
                           && name.EndsWith(".metal", StringComparison.Ordinal))
            .OrderBy(name => name, StringComparer.Ordinal)
            .ToArray();
        string[] listed = MetalShaders.Names.OrderBy(name => name, StringComparer.Ordinal).ToArray();
        Assert.Equal(listed, embedded);
        string source = MetalShaders.LoadSource(assembly);
        Assert.Contains("kernel", source, StringComparison.Ordinal);
    }

    [Fact]
    public void ProbeIsNullOffMacOs()
    {
        if (OperatingSystem.IsMacOS())
            return;
        Assert.Null(OcrMetal.TryProbe());
    }
}
