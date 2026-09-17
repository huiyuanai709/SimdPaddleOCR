using System;
using System.IO;
using System.Windows;

namespace OpenCvSharp5.Wpf;

public partial class App : Application
{
    private void OnStartup(object sender, StartupEventArgs e)
    {
        MainWindow window = new(ResolveImagePath(e.Args));
        window.Show();
    }

    private static string ResolveImagePath(string[] args)
    {
        if (args.Length > 0)
            return Path.GetFullPath(args[0]);

        return Path.Combine(FindRepositoryRoot(), "examples", "sample.jpg");
    }

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
