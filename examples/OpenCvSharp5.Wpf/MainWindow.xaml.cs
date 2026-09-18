using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media.Imaging;
using Microsoft.Win32;
using OpenCvSharp;
using Sdcb.SimdPaddleOCR;
using DrawingBitmap = System.Drawing.Bitmap;

namespace OpenCvSharp5.Wpf;

public partial class MainWindow : System.Windows.Window
{
    private PaddleOcrAll? _ocr;
    private string _imagePath;
    private BitmapSource? _sourcePreview;
    private BitmapSource? _overlayPreview;

    public MainWindow() : this(Path.Combine("examples", "sample.jpg"))
    {
    }

    public MainWindow(string imagePath)
    {
        InitializeComponent();
        _imagePath = imagePath;
        Title = $"Sdcb.SimdPaddleOCR - OpenCvSharp5 WPF [{BuildConfiguration}]";
        ImagePathText.Text = imagePath;
        Status.Text = $"正在加载模型…    配置：{BuildConfiguration}";
        if (System.ComponentModel.DesignerProperties.GetIsInDesignMode(this))
            return;

        Loaded += async (_, _) =>
        {
            LoadPreview();
            await InitializeModelAsync();
        };
        Closed += (_, _) => _ocr?.Dispose();
    }

    private void SelectImage_Click(object sender, RoutedEventArgs e)
    {
        OpenFileDialog dialog = new()
        {
            Title = "选择 OCR 图片",
            Filter = "图片文件|*.jpg;*.jpeg;*.png;*.bmp;*.webp|所有文件|*.*",
            FileName = _imagePath
        };
        if (dialog.ShowDialog(this) != true)
            return;

        _imagePath = dialog.FileName;
        ImagePathText.Text = _imagePath;
        Output.Clear();
        LoadPreview();
    }

    private async void Model_Checked(object sender, RoutedEventArgs e)
    {
        if (IsLoaded)
            await InitializeModelAsync();
    }

    private async void RunOcr_Click(object sender, RoutedEventArgs e) => await RunOcrAsync();

    private void ShowOriginal_Changed(object sender, RoutedEventArgs e) => RefreshPreview();

    private void Output_TextChanged(object sender, TextChangedEventArgs e) =>
        OutputPlaceholder.Visibility = string.IsNullOrWhiteSpace(Output.Text)
            ? Visibility.Visible
            : Visibility.Collapsed;

    private string SelectedModelName =>
        TinyRadio.IsChecked == true ? "tiny" :
        MediumRadio.IsChecked == true ? "medium" :
        "small";

    private static System.Threading.Tasks.Task<PaddleOcrAll> LoadModelAsync(string name) => name switch
    {
        "tiny" => PaddleOcrAll.LoadAsync(Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny.ChineseV6TinyModels.Default),
        "medium" => PaddleOcrAll.LoadAsync(Sdcb.SimdPaddleOCR.Models.ChineseV6Medium.ChineseV6MediumModels.Default),
        _ => PaddleOcrAll.LoadAsync(Sdcb.SimdPaddleOCR.Models.ChineseV6Small.ChineseV6SmallModels.Default)
    };

    private async Task InitializeModelAsync()
    {
        RunButton.IsEnabled = false;
        ModelPanel.IsEnabled = false;
        SelectImageButton.IsEnabled = false;
        string modelName = SelectedModelName;
        try
        {
            Status.Text = $"正在加载 {modelName} 模型…";
            PaddleOcrAll loaded = await Task.Run(() => LoadModelAsync(modelName));
            PaddleOcrAll? old = _ocr;
            _ocr = loaded;
            old?.Dispose();
            Status.Text = $"模型已加载（{modelName}）    图片：{_imagePath}    配置：{BuildConfiguration}";
            RunButton.IsEnabled = true;
        }
        catch (Exception ex)
        {
            Status.Text = $"模型加载失败：{ex.Message}";
        }
        finally
        {
            ModelPanel.IsEnabled = true;
            SelectImageButton.IsEnabled = true;
        }
    }

    private void LoadPreview()
    {
        try
        {
            using Mat image = Cv2.ImRead(_imagePath, ImreadModes.Color);
            if (image.Empty())
                throw new InvalidDataException($"无法读取图片：{_imagePath}");
            ReplacePreviews(ToBitmapSource(image), overlay: null);
            Status.Text = $"图片：{_imagePath}    配置：{BuildConfiguration}";
        }
        catch (Exception ex)
        {
            Status.Text = $"图片加载失败：{ex.Message}";
        }
    }

    private async Task RunOcrAsync()
    {
        PaddleOcrAll? ocr = _ocr;
        string path = _imagePath;
        if (ocr is null || string.IsNullOrWhiteSpace(path))
            return;

        RunButton.IsEnabled = false;
        ModelPanel.IsEnabled = false;
        SelectImageButton.IsEnabled = false;
        Stopwatch total = Stopwatch.StartNew();
        try
        {
            OcrRunResult run = await Task.Run(() => RunPipeline(path, ocr));
            total.Stop();
            try
            {
                ShowOriginalCheck.IsChecked = false;
                ReplacePreviews(ToBitmapSource(run.Source), ToBitmapSource(run.Overlay));
                Output.Text = string.Join(Environment.NewLine, run.Result.Lines.Select(line => line.Text));
                Status.Text =
                    $"完成：{run.Result.DetectedCount} 行，总耗时 {total.Elapsed.TotalMilliseconds:F1} ms。" +
                    $"当前为 {BuildConfiguration}，可再次点击“运行 OCR”。";
            }
            finally
            {
                run.Source.Dispose();
                run.Overlay.Dispose();
            }
        }
        catch (Exception ex)
        {
            Output.Text = ex.ToString();
            Status.Text = "OCR 执行失败";
        }
        finally
        {
            RunButton.IsEnabled = _ocr is not null;
            ModelPanel.IsEnabled = true;
            SelectImageButton.IsEnabled = true;
        }
    }

    private static unsafe OcrRunResult RunPipeline(string path, PaddleOcrAll ocr)
    {
        using Mat image = Cv2.ImRead(path, ImreadModes.Color);
        if (image.Empty())
            throw new InvalidDataException($"无法读取图片：{path}");

        int stride = checked((int)image.Step());
        PaddleOcrResult result = ocr.Run(
            new ReadOnlySpan<byte>((byte*)image.Data, checked(stride * image.Height)),
            image.Width,
            image.Height,
            stride,
            ImagePixelFormat.Bgr24);
        DrawingBitmap source = MatToBitmap(image);
        DrawingBitmap overlay = OcrOverlay.Draw(source, result.Lines);
        return new OcrRunResult(result, source, overlay);
    }

    private void ReplacePreviews(BitmapSource source, BitmapSource? overlay)
    {
        _sourcePreview = source;
        _overlayPreview = overlay;
        RefreshPreview();
    }

    private void RefreshPreview()
    {
        if (ShowOriginalCheck.IsChecked == true || _overlayPreview is null)
            Preview.Source = _sourcePreview;
        else
            Preview.Source = _overlayPreview;
    }

    private static BitmapSource ToBitmapSource(Mat image)
    {
        Cv2.ImEncode(".png", image, out byte[] png);
        return ToBitmapSource(png);
    }

    private static BitmapSource ToBitmapSource(DrawingBitmap image)
    {
        using MemoryStream stream = new();
        image.Save(stream, System.Drawing.Imaging.ImageFormat.Png);
        return ToBitmapSource(stream.ToArray());
    }

    private static BitmapSource ToBitmapSource(byte[] png)
    {
        using MemoryStream stream = new(png, writable: false);
        BitmapImage bitmap = new();
        bitmap.BeginInit();
        bitmap.CacheOption = BitmapCacheOption.OnLoad;
        bitmap.StreamSource = stream;
        bitmap.EndInit();
        bitmap.Freeze();
        return bitmap;
    }

    private static DrawingBitmap MatToBitmap(Mat image)
    {
        Cv2.ImEncode(".png", image, out byte[] png);
        using MemoryStream stream = new(png, writable: false);
        using System.Drawing.Image source = System.Drawing.Image.FromStream(stream);
        return new DrawingBitmap(source);
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

    private sealed record OcrRunResult(PaddleOcrResult Result, DrawingBitmap Source, DrawingBitmap Overlay);
}
