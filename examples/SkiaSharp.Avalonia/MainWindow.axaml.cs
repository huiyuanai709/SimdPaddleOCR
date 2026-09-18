using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Platform.Storage;
using Avalonia.Threading;
using Sdcb.SimdPaddleOCR;
using SkiaSharp;
using AvaloniaBitmap = Avalonia.Media.Imaging.Bitmap;

namespace SkiaSharp.Avalonia;

public partial class MainWindow : Window
{
    private AvaloniaBitmap? _sourcePreview;
    private AvaloniaBitmap? _overlayPreview;
    private PaddleOcrAll? _ocr;
    private string _imagePath;

    public MainWindow() : this(Program.StartupImagePath)
    {
    }

    public MainWindow(string imagePath)
    {
        InitializeComponent();
        _imagePath = imagePath;
        Title = $"Sdcb.SimdPaddleOCR - SkiaSharp/Avalonia [{BuildConfiguration}]";
        ImagePathText.Text = imagePath;
        Status.Text = $"请选择图片后运行 OCR    配置：{BuildConfiguration}";
        if (Design.IsDesignMode)
            return;

        Opened += async (_, _) =>
        {
            LoadPreview();
            await InitializeModelAsync();
        };
        Closed += (_, _) => DisposeResources();
    }

    private async void SelectImage_Click(object? sender, RoutedEventArgs e) => await SelectImageAsync();

    private async void RunOcr_Click(object? sender, RoutedEventArgs e) => await RunOcrAsync();

    private void ShowOriginal_IsCheckedChanged(object? sender, RoutedEventArgs e) => RefreshPreview();

    private void Output_TextChanged(object? sender, TextChangedEventArgs e) =>
        OutputPlaceholder.IsVisible = string.IsNullOrWhiteSpace(Output.Text);

    private async void Model_IsCheckedChanged(object? sender, RoutedEventArgs e)
    {
        if (IsVisible)
            await InitializeModelAsync();
    }

    private async Task SelectImageAsync()
    {
        IStorageProvider? provider = GetTopLevel(this)?.StorageProvider;
        if (provider is null || !provider.CanOpen)
            return;

        IReadOnlyList<IStorageFile> files = await provider.OpenFilePickerAsync(
            new FilePickerOpenOptions
            {
                AllowMultiple = false,
                Title = "选择 OCR 图片",
                FileTypeFilter =
                [
                    new FilePickerFileType("图片")
                    {
                        Patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
                    }
                ]
            });

        if (files.Count == 0)
            return;

        string? path = files[0].TryGetLocalPath();
        if (string.IsNullOrWhiteSpace(path))
        {
            Status.Text = "当前平台无法取得所选文件的本地路径";
            return;
        }

        _imagePath = path;
        ImagePathText.Text = path;
        Output.Text = string.Empty;
        LoadPreview();
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

    private void LoadPreview()
    {
        try
        {
            if (!File.Exists(_imagePath))
                throw new FileNotFoundException("图片不存在", _imagePath);

            ReplacePreviews(new AvaloniaBitmap(_imagePath), overlay: null);
            Status.Text = $"图片：{_imagePath}    配置：{BuildConfiguration}";
        }
        catch (Exception ex)
        {
            Status.Text = $"图片加载失败：{ex.Message}";
        }
    }

    private async Task InitializeModelAsync()
    {
        RunButton.IsEnabled = false;
        SelectImageButton.IsEnabled = false;
        try
        {
            string modelName = SelectedModelName;
            Status.Text = $"正在加载 {modelName} 模型…";
            PaddleOcrAll loaded = await Task.Run(() => LoadModelAsync(modelName));
            _ocr?.Dispose();
            _ocr = loaded;
            Status.Text = $"模型已加载（{modelName}）    图片：{_imagePath}    配置：{BuildConfiguration}";
            RunButton.IsEnabled = true;
        }
        catch (Exception ex)
        {
            Status.Text = $"模型加载失败：{ex.Message}";
        }
        finally
        {
            SelectImageButton.IsEnabled = true;
        }
    }

    private string SelectedModelName =>
        TinyRadio.IsChecked == true ? "tiny" :
        SmallRadio.IsChecked == true ? "small" :
        "medium";

    private static Task<PaddleOcrAll> LoadModelAsync(string name) => name switch
    {
        "tiny" => PaddleOcrAll.LoadAsync(Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny.ChineseV6TinyModels.Default),
        "small" => PaddleOcrAll.LoadAsync(Sdcb.SimdPaddleOCR.Models.ChineseV6Small.ChineseV6SmallModels.Default),
        _ => PaddleOcrAll.LoadAsync(Sdcb.SimdPaddleOCR.Models.ChineseV6Medium.ChineseV6MediumModels.Default)
    };

    private async Task RunOcrAsync()
    {
        if (string.IsNullOrWhiteSpace(_imagePath))
            return;

        Stopwatch total = Stopwatch.StartNew();
        try
        {
            RunButton.IsEnabled = false;
            SelectImageButton.IsEnabled = false;
            Status.Text = "正在解码图片…";

            Stopwatch stage = Stopwatch.StartNew();
            SKBitmap bitmap = await Task.Run(() => DecodeBitmap(_imagePath));
            try
            {
                if (_ocr is null)
                    throw new InvalidOperationException("模型尚未加载");

                Status.Text = "正在运行 OCR…";
                stage.Restart();
                PaddleOcrResult result = await Task.Run(() =>
                {
                    int stride = bitmap.RowBytes;
                    unsafe
                    {
                        return _ocr.Run(
                            new ReadOnlySpan<byte>((byte*)bitmap.GetPixels(), checked(stride * bitmap.Height)),
                            bitmap.Width,
                            bitmap.Height,
                            stride,
                            ImagePixelFormat.Bgra32);
                    }
                });

                Status.Text = "正在绘制就地贴字…";
                stage.Restart();
                byte[] sourcePng = EncodePng(bitmap);
                byte[] overlayPng;
                using (SKBitmap overlay = OcrOverlay.Draw(bitmap, result.Lines))
                    overlayPng = EncodePng(overlay);

                total.Stop();
                await Dispatcher.UIThread.InvokeAsync(() =>
                {
                    ShowOriginalCheck.IsChecked = false;
                    ReplacePreviews(LoadBitmap(sourcePng), LoadBitmap(overlayPng));
                    Output.Text = string.Join(Environment.NewLine, result.Lines.Select(line => line.Text));
                    Status.Text =
                        $"完成：{result.DetectedCount} 行，总耗时 {total.Elapsed.TotalMilliseconds:F1} ms。" +
                        $"当前为 {BuildConfiguration}，再次点击“运行 OCR”可重复运行。";
                });
            }
            finally
            {
                bitmap.Dispose();
            }
        }
        catch (Exception ex)
        {
            total.Stop();
            await Dispatcher.UIThread.InvokeAsync(() =>
            {
                Output.Text = ex.ToString();
                Status.Text = "OCR 执行失败";
            });
        }
        finally
        {
            await Dispatcher.UIThread.InvokeAsync(() =>
            {
                RunButton.IsEnabled = true;
                SelectImageButton.IsEnabled = true;
            });
        }
    }

    private void ReplacePreviews(AvaloniaBitmap source, AvaloniaBitmap? overlay)
    {
        Preview.Source = null;
        AvaloniaBitmap? oldSource = _sourcePreview;
        AvaloniaBitmap? oldOverlay = _overlayPreview;
        _sourcePreview = source;
        _overlayPreview = overlay;
        RefreshPreview();
        if (oldSource is not null && oldSource != source)
            oldSource.Dispose();
        if (oldOverlay is not null && oldOverlay != overlay)
            oldOverlay.Dispose();
    }

    private void RefreshPreview()
    {
        if (ShowOriginalCheck.IsChecked == true || _overlayPreview is null)
            Preview.Source = _sourcePreview;
        else
            Preview.Source = _overlayPreview;
    }

    private static AvaloniaBitmap LoadBitmap(byte[] png)
    {
        using MemoryStream stream = new(png, writable: false);
        return new AvaloniaBitmap(stream);
    }

    private static SKBitmap DecodeBitmap(string path)
    {
        SKBitmap bitmap = SKBitmap.Decode(path)
            ?? throw new InvalidDataException($"无法读取图片：{path}");
        if (bitmap.ColorType == SKColorType.Bgra8888)
            return bitmap;

        SKBitmap? converted = bitmap.Copy(SKColorType.Bgra8888);
        bitmap.Dispose();
        return converted ?? throw new InvalidDataException("无法转换到 BGRA。");
    }

    private static byte[] EncodePng(SKBitmap bitmap)
    {
        using SKImage image = SKImage.FromBitmap(bitmap);
        using SKData data = image.Encode(SKEncodedImageFormat.Png, 100)
            ?? throw new InvalidOperationException("无法编码图片");
        return data.ToArray();
    }

    private void DisposeResources()
    {
        _ocr?.Dispose();
        _ocr = null;
        Preview.Source = null;
        _sourcePreview?.Dispose();
        _sourcePreview = null;
        _overlayPreview?.Dispose();
        _overlayPreview = null;
    }
}
