using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Forms;
using Sdcb.SimdPaddleOCR;

namespace SystemDrawing.WinForms;

public partial class MainForm : Form
{
    private Bitmap? _sourceBitmap;
    private Bitmap? _overlayBitmap;
    private PaddleOcrAll? _ocr;

    public MainForm()
    {
        InitializeComponent();
        if (LicenseManager.UsageMode == LicenseUsageMode.Designtime)
            return;

        Text = $"Sdcb.SimdPaddleOCR - System.Drawing WinForms [{TargetFrameworkDisplayName}]";
        string root = FindRepositoryRoot();
        imagePath.Text = Path.Combine(root, "examples", "sample.jpg");
        detPath.Text = Path.Combine(root, "models", "det.onnx");
        clsPath.Text = Path.Combine(root, "models", "cls.onnx");
        recPath.Text = Path.Combine(root, "models", "rec.onnx");
        dictPath.Text = Path.Combine(root, "models", "ppocr_keys.txt");
        Shown += async (_, _) =>
        {
            LoadPreview();
            await InitializeModelAsync();
        };
    }

    private void Split_SizeChanged(object? sender, EventArgs e)
    {
        int target = split.ClientSize.Width / 2;
        if (target > 0 && split.SplitterDistance != target)
            split.SplitterDistance = target;
    }

    private void BrowseImage_Click(object? sender, EventArgs e) =>
        SelectFile(imagePath, "图片|*.jpg;*.jpeg;*.png;*.bmp;*.webp|所有文件|*.*", image: true);

    private void BrowseDet_Click(object? sender, EventArgs e) =>
        SelectFile(detPath, "模型|*.onnx|所有文件|*.*", image: false);

    private void BrowseCls_Click(object? sender, EventArgs e) =>
        SelectFile(clsPath, "模型|*.onnx|所有文件|*.*", image: false);

    private void BrowseRec_Click(object? sender, EventArgs e) =>
        SelectFile(recPath, "模型|*.onnx|所有文件|*.*", image: false);

    private void BrowseDict_Click(object? sender, EventArgs e) =>
        SelectFile(dictPath, "字典|*.txt|所有文件|*.*", image: false);

    private async void Run_Click(object? sender, EventArgs e) => await RunOcrAsync();

    private void ShowOriginal_CheckedChanged(object? sender, EventArgs e) => RefreshPreview();

    private void Result_TextChanged(object? sender, EventArgs e) =>
        resultPlaceholder.Visible = string.IsNullOrWhiteSpace(result.Text);

    private void SelectFile(TextBox target, string filter, bool image)
    {
        using OpenFileDialog dialog = new()
        {
            Filter = filter,
            FileName = target.Text
        };
        if (dialog.ShowDialog(this) != DialogResult.OK)
            return;

        target.Text = dialog.FileName;
        if (image)
        {
            result.Clear();
            LoadPreview();
        }
        else
        {
            _ = InitializeModelAsync();
        }
    }

    private async Task InitializeModelAsync()
    {
        run.Enabled = false;
        paths.Enabled = false;
        status.Text = "正在加载模型…";
        string det = detPath.Text;
        string cls = clsPath.Text;
        string rec = recPath.Text;
        string dict = dictPath.Text;
        try
        {
            PaddleOcrAll loaded = await Task.Run(() => PaddleOcrAll.LoadAsync(det, cls, rec, dict));
            _ocr?.Dispose();
            _ocr = loaded;
            status.Text = $"模型已加载    图片：{imagePath.Text}    配置：{BuildConfiguration}";
            run.Enabled = true;
        }
        catch (Exception ex)
        {
            status.Text = $"模型加载失败：{ex.Message}";
        }
        finally
        {
            paths.Enabled = true;
        }
    }

    private void LoadPreview()
    {
        try
        {
            using Bitmap loaded = new(imagePath.Text);
            ReplaceImages(new Bitmap(loaded), overlay: null);
            status.Text = $"图片：{imagePath.Text}    配置：{BuildConfiguration}";
        }
        catch (Exception ex)
        {
            status.Text = $"图片加载失败：{ex.Message}";
        }
    }

    private async Task RunOcrAsync()
    {
        if (_ocr is null || _sourceBitmap is null)
            return;

        run.Enabled = false;
        paths.Enabled = false;
        Stopwatch stopwatch = Stopwatch.StartNew();
        try
        {
            string path = imagePath.Text;
            OcrResult ocrResult = await Task.Run(() => RunPipeline(path));
            stopwatch.Stop();
            result.Text = ocrResult.Text;
            showOriginal.Checked = false;
            ReplaceImages(ocrResult.Source, ocrResult.Overlay);
            status.Text =
                $"完成：{ocrResult.Count} 行，总耗时 {stopwatch.Elapsed.TotalMilliseconds:F1} ms。" +
                $"当前为 {BuildConfiguration}，可再次点击“运行 OCR”。";
        }
        catch (Exception ex)
        {
            status.Text = $"OCR 执行失败：{ex.Message}";
        }
        finally
        {
            run.Enabled = _ocr is not null;
            paths.Enabled = true;
        }
    }

    private OcrResult RunPipeline(string path)
    {
        Bitmap source = new(path);
        try
        {
            PaddleOcrResult ocrResult = RunLocked(source);
            Bitmap overlay = OcrOverlay.Draw(source, ocrResult.Lines);
            return new OcrResult(
                string.Join(Environment.NewLine, ocrResult.Lines.Select(line => line.Text)),
                ocrResult.DetectedCount,
                source,
                overlay);
        }
        catch
        {
            source.Dispose();
            throw;
        }
    }

    private void ReplaceImages(Bitmap source, Bitmap? overlay)
    {
        preview.Image = null;
        Bitmap? oldSource = _sourceBitmap;
        Bitmap? oldOverlay = _overlayBitmap;
        _sourceBitmap = source;
        _overlayBitmap = overlay;
        RefreshPreview();
        if (oldSource is not null && oldSource != source)
            oldSource.Dispose();
        if (oldOverlay is not null && oldOverlay != overlay)
            oldOverlay.Dispose();
    }

    private void RefreshPreview()
    {
        if (showOriginal.Checked || _overlayBitmap is null)
            preview.Image = _sourceBitmap;
        else
            preview.Image = _overlayBitmap;
    }

    private PaddleOcrResult RunLocked(Bitmap bitmap)
    {
        Rectangle rectangle = new(0, 0, bitmap.Width, bitmap.Height);
        BitmapData data = bitmap.LockBits(rectangle, ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);
        try
        {
            unsafe
            {
                return _ocr!.Run(
                    new ReadOnlySpan<byte>((byte*)data.Scan0, checked(data.Stride * bitmap.Height)),
                    bitmap.Width,
                    bitmap.Height,
                    data.Stride,
                    ImagePixelFormat.Bgra32);
            }
        }
        finally
        {
            bitmap.UnlockBits(data);
        }
    }

    private static string BuildConfiguration
    {
        get
        {
#if DEBUG
            return "Debug";
#else
            return "Release";
#endif
        }
    }

    private static string TargetFrameworkDisplayName
    {
#if NET48
        get => ".NET Framework 4.8";
#elif NET10_0_OR_GREATER
        get => ".NET 10 Windows";
#else
        get => ".NET";
#endif
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

    private sealed class OcrResult
    {
        public OcrResult(string text, int count, Bitmap source, Bitmap overlay)
        {
            Text = text;
            Count = count;
            Source = source;
            Overlay = overlay;
        }

        public string Text { get; }
        public int Count { get; }
        public Bitmap Source { get; }
        public Bitmap Overlay { get; }
    }
}
