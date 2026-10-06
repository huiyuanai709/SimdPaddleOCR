using System.Diagnostics;
using System.Runtime.Intrinsics.X86;
using Sdcb.SimdPaddleOCR;
using Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny;
using Sdcb.SimdPaddleOCR.OnnxSharp;
using SkiaSharp;

namespace Sdcb.SimdPaddleOCR.Tests;

/// <summary>
/// End-to-end profile of one scanned-document page (default 1105×1430, the
/// miniocr 130 DPI raster). Prints a per-stage millisecond breakdown and can
/// run several engines at once so thread oversubscription is visible.
/// <para>
/// <c>--page-profile [--engines N] [--line-workers N] [--det-threads N]
/// [--rec-batch N] [--rec-intra N] [--repeats N] [--warmup N]
/// [--format bgra|bgr|gray]</c>
/// </para>
/// </summary>
static class PageProfile
{
    public static int Run(string[] args)
    {
        int engines = 1, lineWorkers = 2, detThreads = 1, recBatch = 8, recIntra = 0, repeats = 4, warmup = 1;
        int width = 1105, height = 1430;
        string formatName = "bgra";
        for (int i = 1; i < args.Length; i++)
        {
            string Next() => i + 1 < args.Length ? args[++i] : throw new ArgumentException($"missing value for {args[i]}");
            switch (args[i])
            {
                case "--engines": engines = int.Parse(Next()); break;
                case "--line-workers": lineWorkers = int.Parse(Next()); break;
                case "--det-threads": detThreads = int.Parse(Next()); break;
                case "--rec-batch": recBatch = int.Parse(Next()); break;
                case "--rec-intra": recIntra = int.Parse(Next()); break;
                case "--repeats": repeats = int.Parse(Next()); break;
                case "--warmup": warmup = int.Parse(Next()); break;
                case "--width": width = int.Parse(Next()); break;
                case "--height": height = int.Parse(Next()); break;
                case "--format": formatName = Next().ToLowerInvariant(); break;
                default: throw new ArgumentException($"unknown argument: {args[i]}");
            }
        }
        if (engines is < 1 or > 16) throw new ArgumentOutOfRangeException(nameof(engines));
        if (formatName is not ("bgra" or "bgr" or "gray"))
            throw new ArgumentException("--format must be bgra, bgr, or gray");

        ImagePixelFormat format = formatName switch
        {
            "bgr" => ImagePixelFormat.Bgr24,
            "gray" => ImagePixelFormat.Gray8,
            _ => ImagePixelFormat.Bgra32,
        };
        int bpp = formatName switch { "bgr" => 3, "gray" => 1, _ => 4 };
        byte[] pixels = RenderPage(width, height, format);
        Console.WriteLine(
            $"page={width}x{height} format={formatName} bytes={pixels.Length} " +
            $"cpu={Environment.ProcessorCount} avx2={Avx2.IsSupported} avx512={Avx512F.IsSupported} " +
            $"engines={engines} lineWorkers={lineWorkers} detThreads={detThreads} recBatch={recBatch} recIntra={recIntra}");

        PaddleOcrOptions options = new()
        {
            UseDirectionClassification = false,
            LineWorkerCount = lineWorkers,
            DetIntraOpThreads = detThreads,
            RecBatchLines = recBatch,
            RecIntraOpThreads = recIntra,
            Detector = new PaddleOcrDetectorOptions { LimitSideLength = 960, Backend = OcrBackend.Cpu },
            Recognizer = new PaddleOcrRecognizerOptions { Backend = OcrBackend.Cpu },
            Classifier = new PaddleOcrClassifierOptions { Backend = OcrBackend.Cpu },
        };

        using var detStream = ChineseV6TinyModel.Detection.OpenRead();
        using var recStream = ChineseV6TinyModel.Recognition.OpenRead();
        using var dictStream = ChineseV6TinyModel.Dictionary.OpenRead();
        Model det = Model.Load(detStream);
        Model rec = Model.Load(recStream);
        byte[] dict = ReadAll(dictStream);
        PaddleOcrAll[] pool = new PaddleOcrAll[engines];
        try
        {
            for (int i = 0; i < engines; i++)
                pool[i] = new PaddleOcrAll(det, null, rec, dict, options);

            PipelineProfiler.Enable(true);
            InferenceSession.EnableProfiling(true);
            string? text = null;
            int lines = 0;
            for (int i = 0; i < warmup; i++)
            {
                PaddleOcrResult warm = pool[0].Run(pixels, width, height, width * bpp, format);
                text = warm.Text;
                lines = warm.Lines.Length;
            }
            PipelineProfiler.Enable(true);
            InferenceSession.EnableProfiling(true);
            long allocBefore = GC.GetTotalAllocatedBytes(precise: true);
            int gen0 = GC.CollectionCount(0), gen1 = GC.CollectionCount(1), gen2 = GC.CollectionCount(2);
            var samples = new double[repeats];
            var sw = Stopwatch.StartNew();
            for (int r = 0; r < repeats; r++)
            {
                long t0 = Stopwatch.GetTimestamp();
                if (engines == 1)
                {
                    PaddleOcrResult result = pool[0].Run(pixels, width, height, width * bpp, format);
                    if (result.Text != text) throw new InvalidOperationException("text changed between repeats");
                }
                else
                {
                    Parallel.For(0, engines, new ParallelOptions { MaxDegreeOfParallelism = engines }, e =>
                    {
                        PaddleOcrResult result = pool[e].Run(pixels, width, height, width * bpp, format);
                        if (result.Text != text) throw new InvalidOperationException("text diverged across engines");
                    });
                }
                samples[r] = (Stopwatch.GetTimestamp() - t0) * 1000.0 / Stopwatch.Frequency;
            }
            sw.Stop();
            long allocAfter = GC.GetTotalAllocatedBytes(precise: true);
            Array.Sort(samples);
            double median = samples[samples.Length / 2];
            double mean = samples.Average();
            var stages = PipelineProfiler.Snapshot();
            double perPage = engines == 1 ? 1 : engines;
            double pages = repeats * (engines == 1 ? 1 : engines);
            Console.WriteLine($"lines={lines} chars={text!.Length} hash={Fnv(text):x16}");
            Console.WriteLine($"wall median={median:F2} ms  mean={mean:F2} ms  (engines={(engines == 1 ? "serial" : "parallel")})");
            Console.WriteLine(
                $"alloc={((allocAfter - allocBefore) / pages) / (1024d * 1024d):F2} MB/page  " +
                $"gc Δg0={GC.CollectionCount(0) - gen0} Δg1={GC.CollectionCount(1) - gen1} Δg2={GC.CollectionCount(2) - gen2}");
            Console.WriteLine("stages (sum across engines and repeats, ms/page):");
            for (int s = 0; s < PipelineProfiler.StageCount; s++)
            {
                if (stages[s].Calls == 0) continue;
                Console.WriteLine(
                    $"  {PipelineProfiler.StageNames[s],-16} {stages[s].Milliseconds / pages,8:F2} ms   calls/page={stages[s].Calls / pages:F1}");
            }
            var ops = InferenceSession.ProfileSnapshot();
            string[] opNames = Enum.GetNames<OperatorId>();
            Console.WriteLine("operators (ms/page, calls>0):");
            var ranked = new List<(string Name, double Ms, long Calls)>();
            for (int op = 1; op < opNames.Length; op++)
            {
                if (ops[op].Calls == 0) continue;
                double ms = ops[op].Ticks * 1000.0 / Stopwatch.Frequency / pages;
                ranked.Add((opNames[op], ms, ops[op].Calls));
            }
            foreach (var row in ranked.OrderByDescending(x => x.Ms).Take(12))
                Console.WriteLine($"  {row.Name,-16} {row.Ms,8:F2} ms   calls/page={row.Calls / pages:F1}");
            Console.WriteLine($"effectiveLineWorkers={pool[0].EffectiveLineWorkerCount}");
            _ = perPage;
            _ = sw;
            return 0;
        }
        finally
        {
            foreach (PaddleOcrAll engine in pool)
                engine?.Dispose();
            det.Dispose();
            rec.Dispose();
        }
    }

    private static byte[] ReadAll(Stream source)
    {
        using var buffer = new MemoryStream();
        source.CopyTo(buffer);
        return buffer.ToArray();
    }

    private static ulong Fnv(string text)
    {
        ulong hash = 14695981039346656037UL;
        foreach (byte value in System.Text.Encoding.UTF8.GetBytes(text))
        {
            hash ^= value;
            hash *= 1099511628211UL;
        }
        return hash;
    }

    /// <summary>
    /// Dense two-column Chinese form at the requested pixel size. Ink is
    /// grayscale so B, G and R match; that is what a Gray8 scan expanded to
    /// BGRA looks like.
    /// </summary>
    internal static byte[] RenderPage(int width, int height, ImagePixelFormat format)
    {
        string fontPath = new[]
        {
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
            @"C:\Windows\Fonts\msyh.ttc",
            @"C:\Windows\Fonts\simhei.ttf",
        }.FirstOrDefault(File.Exists) ?? throw new InvalidOperationException("No CJK font found.");
        using SKTypeface typeface = SKTypeface.FromFile(fontPath)
            ?? throw new InvalidOperationException($"failed to load {fontPath}");
        using SKBitmap bitmap = new(new SKImageInfo(width, height, SKColorType.Bgra8888, SKAlphaType.Opaque));
        using SKCanvas canvas = new(bitmap);
        canvas.Clear(new SKColor(248, 248, 246));
        using SKPaint ink = new()
        {
            Color = new SKColor(28, 28, 28),
            IsAntialias = true,
        };
        string[] left =
        [
            "单位名称：杭州汇远智能科技有限公司",
            "纳税人识别号：91330106MA2XXXXX1A",
            "地址：浙江省杭州市西湖区文三路 188 号",
            "开户行：中国工商银行杭州分行",
            "账号：1202020209000123456",
            "联系人：张伟明",
            "电话：0571-88886666",
            "经办人：李秀兰",
            "复核：王建国",
            "财务负责人：陈晓燕",
            "法定代表人：赵德海",
            "销售方：宁波海曙商贸有限公司",
            "购买方：苏州工业园区启明电子有限公司",
            "货物名称：激光打印机硒鼓 黑色",
            "规格型号：Q2612A 兼容",
            "数量：24  单价：86.50  金额：2076.00",
            "税额：269.88  价税合计：2345.88",
            "备注：合同编号 HT-2026-0418",
            "收款人：周敏",
            "开票人：孙丽华",
            "项目负责人：吴志强",
            "供应商：上海浦东纸业股份有限公司",
            "承运单位：顺丰速运有限公司",
            "收货人：郑小军",
            "验收人：冯国平",
        ];
        string[] right =
        [
            "员工姓名 部门 岗位 入职日期",
            "张伟明 研发部 算法工程师 2019-03-12",
            "李秀兰 财务部 会计主管 2016-07-01",
            "王建国 销售部 区域经理 2018-11-20",
            "陈晓燕 财务部 财务总监 2014-05-08",
            "赵德海 总经办 总经理 2012-01-15",
            "周敏 财务部 出纳 2021-09-03",
            "孙丽华 行政部 人事专员 2020-04-22",
            "吴志强 研发部 项目经理 2017-08-30",
            "郑小军 仓储部 仓管员 2022-02-14",
            "冯国平 质量部 质检员 2019-12-09",
            "何美玲 市场部 品牌主管 2018-06-18",
            "马超 研发部 后端工程师 2023-01-06",
            "林雪 法务部 法务专员 2021-03-27",
            "黄磊 销售部 客户经理 2020-10-11",
            "徐静 客服部 客服主管 2017-04-19",
            "高远 研发部 测试工程师 2022-08-02",
            "罗娟 行政部 行政经理 2015-09-25",
            "谢军 生产部 车间主任 2013-12-03",
            "邓丽 采购部 采购专员 2024-05-16",
            "曹阳 研发部 嵌入式工程师 2019-07-29",
            "彭燕 财务部 成本会计 2020-11-08",
            "蒋涛 销售部 大客户经理 2016-02-21",
            "蔡敏 人力资源 招聘主管 2018-03-14",
            "潘磊 信息技术 系统管理员 2021-06-30",
        ];
        float margin = 36;
        float colGap = 28;
        float colWidth = (width - margin * 2 - colGap) / 2f;
        float top = 48;
        using var titleFont = new SKFont(typeface, 22) { Edging = SKFontEdging.Antialias };
        using var bodyFont = new SKFont(typeface, 16) { Edging = SKFontEdging.Antialias };
        canvas.DrawText("增值税电子普通发票  扫描件 130 DPI", margin, 32, SKTextAlign.Left, titleFont, ink);
        float y = top + 8;
        for (int i = 0; i < left.Length; i++)
        {
            y += 28;
            canvas.DrawText(left[i], margin, y, SKTextAlign.Left, bodyFont, ink);
            if (i < right.Length)
                canvas.DrawText(right[i], margin + colWidth + colGap, y, SKTextAlign.Left, bodyFont, ink);
        }
        y += 36;
        canvas.DrawText("附：本合同由甲乙双方于二零二六年四月十八日在杭州市西湖区签订，", margin, y, SKTextAlign.Left, bodyFont, ink);
        y += 26;
        canvas.DrawText("甲方杭州汇远智能科技有限公司，乙方宁波海曙商贸有限公司。", margin, y, SKTextAlign.Left, bodyFont, ink);
        y += 26;
        canvas.DrawText("货物应于二零二六年五月三十日前送达苏州工业园区启明电子有限公司仓库。", margin, y, SKTextAlign.Left, bodyFont, ink);
        // Light scan noise so resize is not a flat field, still grayscale.
        Random rng = new(20261006);
        for (int row = 0; row < height; row += 3)
        {
            int jitter = rng.Next(-2, 3);
            if (jitter == 0) continue;
            for (int col = rng.Next(0, 5); col < width; col += 7)
            {
                SKColor c = bitmap.GetPixel(col, row);
                byte v = (byte)Math.Clamp(c.Red + jitter, 0, 255);
                bitmap.SetPixel(col, row, new SKColor(v, v, v));
            }
        }
        int bpp = format switch
        {
            ImagePixelFormat.Gray8 => 1,
            ImagePixelFormat.Bgr24 => 3,
            _ => 4,
        };
        byte[] packed = new byte[checked(width * height * bpp)];
        for (int row = 0; row < height; row++)
        {
            for (int col = 0; col < width; col++)
            {
                SKColor c = bitmap.GetPixel(col, row);
                int o = (row * width + col) * bpp;
                if (format == ImagePixelFormat.Gray8)
                    packed[o] = c.Red;
                else
                {
                    packed[o] = c.Blue;
                    packed[o + 1] = c.Green;
                    packed[o + 2] = c.Red;
                    if (bpp == 4) packed[o + 3] = 255;
                }
            }
        }
        return packed;
    }
}
