# 示例：就地贴识别结果

给之后做 WinForms / Avalonia / WPF 的人（或 AI）看。**参考实现是** [`examples/ImageSharp.AspNetCore/wwwroot/overlay.js`](../examples/ImageSharp.AspNetCore/wwwroot/overlay.js)（`OcrOverlay.draw`），和页面逻辑无关。[`app.js`](../examples/ImageSharp.AspNetCore/wwwroot/app.js) 只负责上传和调用。那三个桌面示例还停留在「绿框 + 红字堆在 `box[0]`」，不要抄它们的画法。

目标不是翻译，是：**盖掉原字，按检测框的方向和高度把识别结果写回去**，字色尽量跟原字。不描检测多边形。

```javascript
OcrOverlay.draw(ctx, image, lines, { showOriginal: false });
// lines: [{ text, box: [[x,y],[x,y],[x,y],[x,y]] }]
```

## 先不要做的

- 不要沿框法向去采邻域、不要做 Telea / inpaint。密表（论文数字表）行距小，法向 ±3px 就会采到上一行数字，去墨会花成残影。已经试过，放弃。
- 不要用整图一个字号，红字画在框角上方。
- 不要用 `PaddleOcrLine.AppliedRotationDegrees` 当倾角。那是 CLS 的 0/180。
- 不要竖着一个字母一个字母排 `HELLO`。90° 是整行旋转，字母躺着。
- 不要加填底/对照图开关。UI：**跑完默认就地替换**；可选「显示原图」。
- 不要把算法 sunk 进 `Sdcb.SimdPaddleOCR` 核心。

## 参考实现在哪

| 文件 | 作用 |
| --- | --- |
| [`overlay.js`](../examples/ImageSharp.AspNetCore/wwwroot/overlay.js) | `OcrOverlay.draw`：几何、框内主色、字心取色、旋转贴字 |
| [`app.js`](../examples/ImageSharp.AspNetCore/wwwroot/app.js) | 上传 / API / `#showOriginal` |
| [`Program.cs`](../examples/ImageSharp.AspNetCore/Program.cs) `ToLineDto` | API 已给四点 `box` 和 `text`，桌面侧直接用 `PaddleOcrLine` |

算法请 **移植 `overlay.js`**。下面是要点。

## 每条 `PaddleOcrLine` 做什么

```
p0,p1,p2,p3 = Box 四点（通常 p0→p1 是长边）
geom = 长边方向、中心、长、高、倾角
框内像素：量化众数 → 纯色背景；与底差得大的像素 → 字色
整框铺该纯色（FillPolygon）
字号二分，落入 0.94*长 × 0.9*高
save → 中心 → rotate → 字色居中写 text → restore
```

### 几何

- `len = |p1-p0|`，`ht = |p3-p0|`。若 `ht > len`，长边改 `p0→p3`。
- `angle = atan2(along.y, along.x)`。中心四点平均。
- `AppliedRotationDegrees` 贴字不用。

### 纯色背景（框内众数）

扫四边形内部像素，RGB 各右移 4 位做桶，**计数最多的桶的均值**当背景。白纸黑字、绿高亮黑字、深底白字都走这一条。

密字框里黑墨可能赢过众数：若第一名占比 < 55% 且很暗、第二名明显是浅底（亮度 > 140），改用第二名。这是 best-effort，不要再叠聚类。

代价：横向渐变会被铺成一条单色。接受。换来的是密表不再把邻行数字拖进来。

### 字色

内部像素里，和背景 RGB 距离 ≥ 16 的当墨水；再取距离 ≥ 中位数的那一半，RGB 各取中位数。膨胀边不要采。采不到就按底色亮度退回近黑/近白。一行多色会糊成一种色。

### 写字

- `"Microsoft YaHei UI"`，粗体。
- `textAlign=center`，`textBaseline=middle`，画在变换后的 `(0,0)`。
- 不要描绿框。

## 三个桌面示例怎么改

只换 annotated 画法。加「显示原图」。

### WinForms

[`MainForm.cs`](../examples/SystemDrawing.WinForms/MainForm.cs) `RunPipeline`：现在 `DrawPolygon` + `DrawString(..., points[0])`。

1. `Bitmap.Clone` 再画，留住原图。
2. 框内扫像素算众数背景和字色（`LockBits` 只为取样，或 `GetPixel` 也行，示例无所谓）。
3. `FillPolygon` 铺背景。
4. `TranslateTransform` / `RotateTransform(角度°)` / `DrawString` 居中。`MeasureString` 二分字号。

旋转是角度制，JS `atan2` 是弧度。

### Avalonia（Skia）

[`AnnotateAndEncode`](../examples/SkiaSharp.Avalonia/MainWindow.axaml.cs)：绿线红字换成同样的取样 + `DrawFilledPath` + `Save/Translate/RotateDegrees/DrawText`。

### WPF

[`Annotate`](../examples/OpenCvSharp5.Wpf/MainWindow.xaml.cs)：继续 `System.Drawing`，和 WinForms 同一套 `FillPolygon`。不要用 `Cv2.PutText` 画中文。

## 验收

1. 微信绿气泡：原字被绿底盖住，识别字在框里，字色接近原黑字。
2. 密表（论文 R/P/F 那种）：格子是干净色块，**不能**把上一行数字拖成残影。
3. 斜字、整行 90°：字顺着框；90° 拉丁词字母躺着。
4. 「显示原图」勾上只见原图。
5. 不要为可视化改推理结果。

## 已知边界

- 渐变底会变成一条单色，这是有意的。
- 识别比原句长很多时字号会变小。不要自动换行。
- ImageSharp 示例用独立 `Configuration.PreferContiguousImageBuffers` 尽量零拷贝；拿不到连续缓冲再 `CopyPixelDataTo`。桌面 `LockBits` / `Mat` 不受影响。
