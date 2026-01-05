# Text Mosaic - 文字打码工具

基于 PaddleOCR 的字符级精确文字打码工具，只打码指定字符，不影响其他内容。

## 功能

- 字符级精确打码（不是整行）
- 支持长图（聊天截图等）
- WebUI 支持粘贴/拖拽图片
- 命令行批量处理

## 安装

```bash
pip install -r requirements.txt
```

## 使用

### WebUI（推荐）

```bash
python webui.py
```

打开 http://127.0.0.1:7861 ，粘贴图片，输入要打码的文字。

### 命令行

```bash
# 基本用法
python mosaic.py image.jpg -t 要打码的文字

# 多个文字
python mosaic.py image.jpg -t 文字1 文字2

# 从剪贴板
python mosaic.py -c -t 文字

# 调整马赛克大小
python mosaic.py image.jpg -t 文字 -b 15
```

## 参数

| 参数 | 说明 |
|------|------|
| `-t, --text` | 要打码的文字（必填） |
| `-o, --output` | 输出路径 |
| `-b, --block-size` | 马赛克块大小，默认 10 |
| `-c, --clipboard` | 从剪贴板获取图片 |
| `--accurate` | 使用高精度模型 |

## License

MIT
