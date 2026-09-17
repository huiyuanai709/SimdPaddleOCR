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
using DrawingBrush = System.Drawing.Brush;
using DrawingColor = System.Drawing.Color;
using DrawingGraphics = System.Drawing.Graphics;
using DrawingPen = System.Drawing.Pen;

namespace OpenCvSharp5.Wpf;

public partial class MainWindow : System.Windows.Window
{
    private PaddleOcrAll? _ocr;
    private string _imagePath;

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
            Preview.Source = ToBitmapSource(image);
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
            Preview.Source = ToBitmapSource(run.Annotated);
            Output.Text = string.Join(Environment.NewLine, run.Result.Lines.Select(line => line.Text));
            Status.Text =
                $"完成：{run.Result.DetectedCount} 行，总耗时 {total.Elapsed.TotalMilliseconds:F1} ms。" +
                $"当前为 {BuildConfiguration}，可再次点击“运行 OCR”。";
            run.Annotated.Dispose();
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
        return new OcrRunResult(result, Annotate(image, result));
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

    private static DrawingBitmap Annotate(Mat image, PaddleOcrResult result)
    {
        Cv2.ImEncode(".png", image, out byte[] png);
        using MemoryStream stream = new(png, writable: false);
        using System.Drawing.Image source = System.Drawing.Image.FromStream(stream);
        DrawingBitmap bitmap = new(source);
        using DrawingGraphics graphics = DrawingGraphics.FromImage(bitmap);
        using DrawingPen pen = new(DrawingColor.LimeGreen, 3);
        using DrawingBrush brush = new System.Drawing.SolidBrush(DrawingColor.Red);
        foreach (PaddleOcrLine line in result.Lines)
        {
            System.Drawing.PointF[] points =
            [
                new(line.Box.X1, line.Box.Y1),
                new(line.Box.X2, line.Box.Y2),
                new(line.Box.X3, line.Box.Y3),
                new(line.Box.X4, line.Box.Y4)
            ];
            graphics.DrawPolygon(pen, points);
            graphics.DrawString(line.Text, System.Drawing.SystemFonts.DefaultFont, brush, points[0]);
        }

        return bitmap;
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

    private sealed record OcrRunResult(PaddleOcrResult Result, DrawingBitmap Annotated);
}
