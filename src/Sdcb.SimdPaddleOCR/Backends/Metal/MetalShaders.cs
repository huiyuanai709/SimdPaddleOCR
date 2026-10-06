using System.Reflection;
using System.Text;

namespace Sdcb.SimdPaddleOCR.Backends.Metal;

/// <summary>
/// Metal shader sources embedded in this assembly. <see cref="LoadSource"/>
/// is loaded by name. Native AOT keeps those resources because
/// <c>ILLink.Descriptors.xml</c> marks each one preserve. Adding a
/// <c>.metal</c> file requires both lists; the unit test checks
/// <see cref="Names"/> against the manifest.
/// </summary>
internal static class MetalShaders
{
    internal static readonly string[] Names =
    [
        "Sdcb.SimdPaddleOCR.Backends.Metal.Shaders.a_common.metal",
        "Sdcb.SimdPaddleOCR.Backends.Metal.Shaders.conv.metal",
        "Sdcb.SimdPaddleOCR.Backends.Metal.Shaders.dot.metal",
        "Sdcb.SimdPaddleOCR.Backends.Metal.Shaders.elem.metal",
        "Sdcb.SimdPaddleOCR.Backends.Metal.Shaders.gemm.metal",
        "Sdcb.SimdPaddleOCR.Backends.Metal.Shaders.space.metal",
        "Sdcb.SimdPaddleOCR.Backends.Metal.Shaders.xform.metal",
    ];

    internal static string LoadSource(Assembly assembly)
    {
        var sb = new StringBuilder();
        Read(assembly, sb, "Sdcb.SimdPaddleOCR.Backends.Metal.Shaders.a_common.metal");
        Read(assembly, sb, "Sdcb.SimdPaddleOCR.Backends.Metal.Shaders.conv.metal");
        Read(assembly, sb, "Sdcb.SimdPaddleOCR.Backends.Metal.Shaders.dot.metal");
        Read(assembly, sb, "Sdcb.SimdPaddleOCR.Backends.Metal.Shaders.elem.metal");
        Read(assembly, sb, "Sdcb.SimdPaddleOCR.Backends.Metal.Shaders.gemm.metal");
        Read(assembly, sb, "Sdcb.SimdPaddleOCR.Backends.Metal.Shaders.space.metal");
        Read(assembly, sb, "Sdcb.SimdPaddleOCR.Backends.Metal.Shaders.xform.metal");
        return sb.ToString();
    }

    private static void Read(Assembly assembly, StringBuilder sb, string name)
    {
        // The seven call sites above pass literals. A local copy keeps the
        // stream read in one place; the descriptor file is what the trimmer
        // honors if this parameter is not folded.
        using Stream stream = assembly.GetManifestResourceStream(name)
            ?? throw new FileNotFoundException("Metal shader missing: " + name);
        using var reader = new StreamReader(stream);
        sb.AppendLine(reader.ReadToEnd());
    }
}
