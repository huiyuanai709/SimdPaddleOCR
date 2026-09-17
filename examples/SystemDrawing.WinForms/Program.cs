using System;
using System.Windows.Forms;

namespace SystemDrawing.WinForms;

internal static class Program
{
    [STAThread]
    private static void Main()
    {
#if NET6_0_OR_GREATER
        // Applies ApplicationHighDpiMode / ApplicationDefaultFont from the csproj so
        // .NET 10 uses the same YaHei UI 9pt metrics as the Designer Font on net48.
        ApplicationConfiguration.Initialize();
#else
        Application.EnableVisualStyles();
        Application.SetCompatibleTextRenderingDefault(false);
#endif
        Application.Run(new MainForm());
    }
}
