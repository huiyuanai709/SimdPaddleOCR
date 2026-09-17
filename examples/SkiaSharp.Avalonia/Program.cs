using System;
using System.IO;
using Avalonia;

namespace SkiaSharp.Avalonia;

internal static class Program
{
    public static string StartupImagePath { get; private set; } = "";

    [STAThread]
    private static void Main(string[] args)
    {
        string repositoryRoot = FindRepositoryRoot();
        StartupImagePath = args.Length > 0
            ? Path.GetFullPath(args[0])
            : Path.Combine(repositoryRoot, "examples", "sample.jpg");

        BuildAvaloniaApp().StartWithClassicDesktopLifetime(args);
    }

    public static AppBuilder BuildAvaloniaApp() =>
        AppBuilder.Configure<App>()
            .UsePlatformDetect()
            .LogToTrace();

    private static string FindRepositoryRoot()
    {
        DirectoryInfo? directory = new(AppContext.BaseDirectory);
        while (directory is not null)
        {
            if (File.Exists(Path.Combine(directory.FullName, "Sdcb.SimdPaddleOCR.slnx")))
                return directory.FullName;
            directory = directory.Parent;
        }

        return Directory.GetCurrentDirectory();
    }
}
