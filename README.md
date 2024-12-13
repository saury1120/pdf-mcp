# PDF MCP Service

一个强大的 PDF 处理 MCP（Model Context Protocol）服务，提供全面的 PDF 文档分析功能。

## 功能特点

1. 基础功能
   - 文本提取：精确提取 PDF 文档中的文本内容
   - 图片提取：提取 PDF 中的所有图片
   - 表格提取：识别和提取 PDF 中的表格数据

2. 高级分析
   - 文档分类：自动对文档进行主题分类
   - 文档相似度：计算文档间的语义相似度
   - 多语言支持：检测和分析文档中使用的语言
   - 高级文本分析：包括复杂度分析、词性分析等

## 安装要求

- Python >= 3.10
- 依赖包：见 pyproject.toml

## 快速开始

1. 克隆仓库：
```bash
git clone https://github.com/saury1120/pdf-mcp.git
cd pdf-mcp
```

2. 创建虚拟环境并安装依赖：
```bash
uv venv
source .venv/bin/activate  # Linux/MacOS
.venv\Scripts\activate     # Windows
uv pip install -e .
```

3. 配置 Claude Desktop：
编辑 `~/Library/Application Support/Claude/claude_desktop_config.json`：
```json
{
    "mcpServers": {
        "pdf_reader": {
            "command": "uv",
            "args": [
                "--directory",
                "/path/to/pdf-mcp",
                "run",
                "pdf_reader"
            ]
        }
    }
}
```

## 使用示例

1. 提取文本：
```
请读取 [PDF文件路径] 的内容
```

2. 提取图片：
```
请提取 [PDF文件路径] 中的图片
```

3. 文档分类：
```
请对 [PDF文件路径] 进行分类，可能的类别包括：技术文档、学术论文、新闻报道、商业报告
```

4. 相似度比较：
```
请比较 [PDF文件1路径] 和 [PDF文件2路径] 的相似度
```

5. 语言检测：
```
请检测 [PDF文件路径] 中使用的语言
```

6. 高级分析：
```
请对 [PDF文件路径] 进行高级文本分析
```

## 注意事项

- 首次运行时会下载必要的模型文件
- 处理大型PDF文件可能需要较长时间
- 建议使用完整路径指定PDF文件

## 许可证

MIT License

## 贡献指南

欢迎提交 Issue 和 Pull Request！
