using System.Drawing;
using System.Windows.Forms;

namespace SystemDrawing.WinForms;

partial class MainForm
{
    private System.ComponentModel.IContainer components = null;

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            components?.Dispose();
            _ocr?.Dispose();
            _bitmap?.Dispose();
        }

        base.Dispose(disposing);
    }

    private void InitializeComponent()
    {
        preview = new PictureBox();
        imagePath = new TextBox();
        detPath = new TextBox();
        clsPath = new TextBox();
        recPath = new TextBox();
        dictPath = new TextBox();
        result = new TextBox();
        resultPlaceholder = new Label();
        resultHost = new Panel();
        previewHost = new Panel();
        statusBar = new StatusStrip();
        status = new ToolStripStatusLabel();
        run = new Button();
        paths = new TableLayoutPanel();
        imageLabel = new Label();
        detLabel = new Label();
        clsLabel = new Label();
        recLabel = new Label();
        dictLabel = new Label();
        browseImage = new Button();
        browseDet = new Button();
        browseCls = new Button();
        browseRec = new Button();
        browseDict = new Button();
        root = new TableLayoutPanel();
        split = new SplitContainer();
        ((System.ComponentModel.ISupportInitialize)preview).BeginInit();
        statusBar.SuspendLayout();
        paths.SuspendLayout();
        resultHost.SuspendLayout();
        previewHost.SuspendLayout();
        root.SuspendLayout();
        ((System.ComponentModel.ISupportInitialize)split).BeginInit();
        split.Panel1.SuspendLayout();
        split.Panel2.SuspendLayout();
        split.SuspendLayout();
        SuspendLayout();

        Font = new Font("Microsoft YaHei UI", 9F);
        AutoScaleMode = AutoScaleMode.Font;
        AutoScaleDimensions = new SizeF(7F, 17F);

        imageLabel.Anchor = AnchorStyles.Left;
        imageLabel.AutoSize = true;
        imageLabel.Margin = new Padding(0, 7, 8, 7);
        imageLabel.Text = "图片";
        imageLabel.TextAlign = ContentAlignment.MiddleLeft;

        detLabel.Anchor = AnchorStyles.Left;
        detLabel.AutoSize = true;
        detLabel.Margin = new Padding(0, 7, 8, 7);
        detLabel.Text = "DET 模型";
        detLabel.TextAlign = ContentAlignment.MiddleLeft;

        clsLabel.Anchor = AnchorStyles.Left;
        clsLabel.AutoSize = true;
        clsLabel.Margin = new Padding(0, 7, 8, 7);
        clsLabel.Text = "CLS 模型";
        clsLabel.TextAlign = ContentAlignment.MiddleLeft;

        recLabel.Anchor = AnchorStyles.Left;
        recLabel.AutoSize = true;
        recLabel.Margin = new Padding(0, 7, 8, 7);
        recLabel.Text = "REC 模型";
        recLabel.TextAlign = ContentAlignment.MiddleLeft;

        dictLabel.Anchor = AnchorStyles.Left;
        dictLabel.AutoSize = true;
        dictLabel.Margin = new Padding(0, 7, 8, 7);
        dictLabel.Text = "字典";
        dictLabel.TextAlign = ContentAlignment.MiddleLeft;

        imagePath.Dock = DockStyle.Fill;
        imagePath.Margin = new Padding(0, 3, 8, 3);
        imagePath.ReadOnly = true;

        detPath.Dock = DockStyle.Fill;
        detPath.Margin = new Padding(0, 3, 8, 3);
        detPath.ReadOnly = true;

        clsPath.Dock = DockStyle.Fill;
        clsPath.Margin = new Padding(0, 3, 8, 3);
        clsPath.ReadOnly = true;

        recPath.Dock = DockStyle.Fill;
        recPath.Margin = new Padding(0, 3, 8, 3);
        recPath.ReadOnly = true;

        dictPath.Dock = DockStyle.Fill;
        dictPath.Margin = new Padding(0, 3, 8, 3);
        dictPath.ReadOnly = true;

        browseImage.Dock = DockStyle.Fill;
        browseImage.Margin = new Padding(0, 2, 0, 2);
        browseImage.Text = "选择";
        browseImage.UseVisualStyleBackColor = true;
        browseImage.Click += BrowseImage_Click;

        browseDet.Dock = DockStyle.Fill;
        browseDet.Margin = new Padding(0, 2, 0, 2);
        browseDet.Text = "选择";
        browseDet.UseVisualStyleBackColor = true;
        browseDet.Click += BrowseDet_Click;

        browseCls.Dock = DockStyle.Fill;
        browseCls.Margin = new Padding(0, 2, 0, 2);
        browseCls.Text = "选择";
        browseCls.UseVisualStyleBackColor = true;
        browseCls.Click += BrowseCls_Click;

        browseRec.Dock = DockStyle.Fill;
        browseRec.Margin = new Padding(0, 2, 0, 2);
        browseRec.Text = "选择";
        browseRec.UseVisualStyleBackColor = true;
        browseRec.Click += BrowseRec_Click;

        browseDict.Dock = DockStyle.Fill;
        browseDict.Margin = new Padding(0, 2, 0, 2);
        browseDict.Text = "选择";
        browseDict.UseVisualStyleBackColor = true;
        browseDict.Click += BrowseDict_Click;

        run.AutoSize = true;
        run.Dock = DockStyle.Fill;
        run.Enabled = false;
        run.Margin = new Padding(8, 2, 0, 2);
        run.Text = "运行 OCR";
        run.UseVisualStyleBackColor = true;
        run.Click += Run_Click;

        paths.AutoSize = true;
        paths.AutoSizeMode = AutoSizeMode.GrowAndShrink;
        paths.ColumnCount = 4;
        paths.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));
        paths.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100F));
        paths.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 80F));
        paths.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));
        paths.Controls.Add(imageLabel, 0, 0);
        paths.Controls.Add(imagePath, 1, 0);
        paths.Controls.Add(browseImage, 2, 0);
        paths.Controls.Add(run, 3, 0);
        paths.Controls.Add(detLabel, 0, 1);
        paths.Controls.Add(detPath, 1, 1);
        paths.Controls.Add(browseDet, 2, 1);
        paths.Controls.Add(clsLabel, 0, 2);
        paths.Controls.Add(clsPath, 1, 2);
        paths.Controls.Add(browseCls, 2, 2);
        paths.Controls.Add(recLabel, 0, 3);
        paths.Controls.Add(recPath, 1, 3);
        paths.Controls.Add(browseRec, 2, 3);
        paths.Controls.Add(dictLabel, 0, 4);
        paths.Controls.Add(dictPath, 1, 4);
        paths.Controls.Add(browseDict, 2, 4);
        paths.Dock = DockStyle.Fill;
        paths.Padding = new Padding(0, 0, 0, 8);
        paths.RowCount = 5;
        paths.RowStyles.Add(new RowStyle(SizeType.AutoSize));
        paths.RowStyles.Add(new RowStyle(SizeType.AutoSize));
        paths.RowStyles.Add(new RowStyle(SizeType.AutoSize));
        paths.RowStyles.Add(new RowStyle(SizeType.AutoSize));
        paths.RowStyles.Add(new RowStyle(SizeType.AutoSize));
        paths.SetColumnSpan(browseDet, 2);
        paths.SetColumnSpan(browseCls, 2);
        paths.SetColumnSpan(browseRec, 2);
        paths.SetColumnSpan(browseDict, 2);

        preview.BackColor = Color.White;
        preview.Dock = DockStyle.Fill;
        preview.SizeMode = PictureBoxSizeMode.Zoom;
        preview.TabStop = false;

        previewHost.BackColor = Color.White;
        previewHost.BorderStyle = BorderStyle.FixedSingle;
        previewHost.Dock = DockStyle.Fill;
        previewHost.Controls.Add(preview);

        result.Dock = DockStyle.Fill;
        result.Multiline = true;
        result.ReadOnly = true;
        result.ScrollBars = ScrollBars.Vertical;
        result.BorderStyle = BorderStyle.None;
        result.TextChanged += Result_TextChanged;

        resultPlaceholder.BackColor = SystemColors.Window;
        resultPlaceholder.Dock = DockStyle.Fill;
        resultPlaceholder.ForeColor = Color.Gray;
        resultPlaceholder.Text = "识别出来的文本将显示在这";
        resultPlaceholder.TextAlign = ContentAlignment.MiddleCenter;

        resultHost.BackColor = SystemColors.Window;
        resultHost.BorderStyle = BorderStyle.FixedSingle;
        resultHost.Dock = DockStyle.Fill;
        resultHost.Controls.Add(result);
        resultHost.Controls.Add(resultPlaceholder);
        resultPlaceholder.BringToFront();

        split.Dock = DockStyle.Fill;
        split.Orientation = Orientation.Vertical;
        split.Panel1.BackColor = Color.White;
        split.Panel1.Padding = new Padding(0, 0, 8, 0);
        split.Panel1.Controls.Add(previewHost);
        split.Panel2.BackColor = Color.White;
        split.Panel2.Controls.Add(resultHost);
        split.SizeChanged += Split_SizeChanged;

        status.Spring = true;
        status.TextAlign = ContentAlignment.MiddleLeft;

        statusBar.AutoSize = false;
        statusBar.BackColor = SystemColors.Control;
        statusBar.Dock = DockStyle.Fill;
        statusBar.ForeColor = Color.Black;
        statusBar.GripStyle = ToolStripGripStyle.Hidden;
        statusBar.Items.Add(status);
        statusBar.LayoutStyle = ToolStripLayoutStyle.HorizontalStackWithOverflow;
        statusBar.SizingGrip = true;

        root.ColumnCount = 1;
        root.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100F));
        root.Controls.Add(paths, 0, 0);
        root.Controls.Add(split, 0, 1);
        root.Controls.Add(statusBar, 0, 2);
        root.Dock = DockStyle.Fill;
        root.Padding = new Padding(12);
        root.RowCount = 3;
        root.RowStyles.Add(new RowStyle(SizeType.AutoSize));
        root.RowStyles.Add(new RowStyle(SizeType.Percent, 100F));
        root.RowStyles.Add(new RowStyle(SizeType.Absolute, 28F));

        BackColor = SystemColors.Control;
        ClientSize = new Size(1200, 760);
        Controls.Add(root);
        MinimumSize = new Size(900, 600);
        Name = "MainForm";
        Text = "Sdcb.SimdPaddleOCR - System.Drawing WinForms";

        ((System.ComponentModel.ISupportInitialize)preview).EndInit();
        statusBar.ResumeLayout(false);
        statusBar.PerformLayout();
        paths.ResumeLayout(false);
        paths.PerformLayout();
        resultHost.ResumeLayout(false);
        previewHost.ResumeLayout(false);
        split.Panel1.ResumeLayout(false);
        split.Panel2.ResumeLayout(false);
        ((System.ComponentModel.ISupportInitialize)split).EndInit();
        split.ResumeLayout(false);
        root.ResumeLayout(false);
        root.PerformLayout();
        ResumeLayout(false);
    }

    private PictureBox preview;
    private TextBox imagePath;
    private TextBox detPath;
    private TextBox clsPath;
    private TextBox recPath;
    private TextBox dictPath;
    private TextBox result;
    private Label resultPlaceholder;
    private Panel resultHost;
    private Panel previewHost;
    private StatusStrip statusBar;
    private ToolStripStatusLabel status;
    private Button run;
    private TableLayoutPanel paths;
    private Label imageLabel;
    private Label detLabel;
    private Label clsLabel;
    private Label recLabel;
    private Label dictLabel;
    private Button browseImage;
    private Button browseDet;
    private Button browseCls;
    private Button browseRec;
    private Button browseDict;
    private TableLayoutPanel root;
    private SplitContainer split;
}
