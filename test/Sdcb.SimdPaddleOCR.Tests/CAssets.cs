using Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny;

namespace Sdcb.SimdPaddleOCR.Tests;

static class CAssets
{
    // 正式索引：https://cv-public.sdcb.ai/2026/lwm.manifest.json
    // 现在还是按下面常量拉文件；以后可以考虑改成读这份 manifest。
    public const string BaseUrl = "https://cv-public.sdcb.ai/2026/";
    public const string DllName = "lw_ppocr_c.dll";
    public const string RemoteDllName = "lw_ppocr_c.20260914.20d0de6.dll";
    public const string DetName = "det.lwm";
    public const string ClsName = "cls.lwm";
    public const string RecName = "rec.lwm";
    public const string DictName = "ppocr_keys.txt";

    public static readonly (string Remote, string Local)[] Files =
    [
        (RemoteDllName, DllName),
        (DetName, DetName),
        (ClsName, ClsName),
        (RecName, RecName),
    ];

    public static CAssetSet Resolve(string dir, string modelType)
    {
        string modelDir = Path.Combine(dir, modelType);
        if (TryLocal(modelDir, out CAssetSet? nested))
            return nested!;
        if (TryLocal(dir, out nested) && File.Exists(Path.Combine(dir, RemoteDllName)))
            return nested!;

        if (modelType != "tiny")
            throw new FileNotFoundException(
                $"local C assets for {modelType} were not found under {dir} or {modelDir}");

        EnsureAsync(dir).GetAwaiter().GetResult();
        string dictPath = WriteDictionary(dir);
        return new CAssetSet(dir, Path.Combine(dir, DllName), DetPath(dir), ClsPath(dir),
            RecPath(dir), dictPath, "download");
    }

    public static async Task EnsureAsync(string dir, CancellationToken cancellationToken = default)
    {
        Directory.CreateDirectory(dir);
        using var http = new HttpClient { Timeout = TimeSpan.FromMinutes(10) };
        await Task.WhenAll(Files.Select(file =>
            DownloadIfMissing(http, dir, file.Remote, file.Local, cancellationToken)));
    }

    public static string WriteDictionary(string dir)
    {
        Directory.CreateDirectory(dir);
        string path = Path.Combine(dir, DictName);
        using Stream src = ChineseV6TinyModel.Dictionary.OpenRead();
        using FileStream dst = File.Create(path);
        src.CopyTo(dst);
        return path;
    }

    public static void CopyDll(string dllPath)
    {
        if (!File.Exists(dllPath))
            throw new FileNotFoundException("lw_ppocr_c.dll was not found", dllPath);
        File.Copy(dllPath, Path.Combine(AppContext.BaseDirectory, DllName), overwrite: true);
    }

    static bool TryLocal(string dir, out CAssetSet? assets)
    {
        assets = null;
        string dll = Path.Combine(dir, DllName);
        if (!File.Exists(dll))
        {
            string parentDll = Path.Combine(Directory.GetParent(dir)?.FullName ?? dir, DllName);
            dll = File.Exists(parentDll) ? parentDll : dll;
        }
        string det = Path.Combine(dir, DetName);
        string cls = Path.Combine(dir, ClsName);
        string rec = Path.Combine(dir, RecName);
        string dict = Path.Combine(dir, DictName);
        if (!File.Exists(det) || !File.Exists(cls) || !File.Exists(rec) || !File.Exists(dict) ||
            !File.Exists(dll))
            return false;
        assets = new CAssetSet(dir, dll, det, cls, rec, dict, "local");
        return true;
    }

    public static string DetPath(string dir) => Path.Combine(dir, DetName);
    public static string ClsPath(string dir) => Path.Combine(dir, ClsName);
    public static string RecPath(string dir) => Path.Combine(dir, RecName);
    public static string DictPath(string dir) => Path.Combine(dir, DictName);

    static async Task DownloadIfMissing(HttpClient http, string dir, string remoteName, string localName,
        CancellationToken cancellationToken)
    {
        string remotePath = Path.Combine(dir, remoteName);
        string localPath = Path.Combine(dir, localName);
        if (File.Exists(remotePath) && new FileInfo(remotePath).Length > 0)
        {
            if (!string.Equals(remotePath, localPath, StringComparison.OrdinalIgnoreCase))
                File.Copy(remotePath, localPath, overwrite: true);
            Console.WriteLine($"skip {remoteName} (cached, {new FileInfo(remotePath).Length} bytes)");
            return;
        }

        string url = BaseUrl + remoteName;
        Console.WriteLine($"download {url}");
        using HttpResponseMessage response = await http.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
        response.EnsureSuccessStatusCode();
        string tmp = remotePath + ".tmp";
        await using (FileStream fs = File.Create(tmp))
            await response.Content.CopyToAsync(fs, cancellationToken);
        File.Move(tmp, remotePath, overwrite: true);
        if (!string.Equals(remotePath, localPath, StringComparison.OrdinalIgnoreCase))
            File.Copy(remotePath, localPath, overwrite: true);
        Console.WriteLine($"saved {localName} from {remoteName} ({new FileInfo(localPath).Length} bytes)");
    }
}

sealed record CAssetSet(
    string Directory,
    string DllPath,
    string DetPath,
    string ClsPath,
    string RecPath,
    string DictPath,
    string Source);
