# PDF MCP 服务

一个基于 MCP（Model Context Protocol）的高性能 PDF 文档处理服务，提供全面的文档分析和处理能力。

## ✨ 主要特性

### 🔍 基础功能
- **文本提取**
  - 精确提取 PDF 文档中的文本内容
  - 支持多语言文本识别
  - 保留原始文档格式和布局
  
- **图片处理**
  - 自动提取 PDF 中的所有图片
  - 支持图片格式转换和优化
  - 智能图片压缩和质量控制
  
- **表格识别**
  - 精准识别和提取表格数据
  - 支持复杂表格结构
  - 自动转换为结构化数据

### 🚀 高级分析
- **智能分类**
  - 自动文档主题分类
  - 支持自定义分类类别
  - 基于深度学习的分类模型

- **相似度分析**
  - 文档语义相似度计算
  - 支持跨语言相似度比较
  - 基于先进的向量模型

- **多语言支持**
  - 自动语言检测
  - 支持 100+ 种语言
  - 多语言并行处理

- **深度分析**
  - 文本复杂度分析
  - 关键信息提取
  - 文档结构解析

### 🔋 性能优化
- **智能模型管理**
  - 延迟加载机制，按需加载模型
  - 自动内存管理，防止内存溢出
  - 智能模型卸载，优化资源使用

- **高效缓存**
  - LRU 缓存机制
  - 分布式缓存支持
  - 智能缓存清理

- **并行处理**
  - 异步 IO 操作
  - 多线程文档处理
  - 批处理优化

- **资源优化**
  - 自适应设备选择（CPU/GPU）
  - 智能量化配置
  - 动态线程管理

## 💻 系统要求

### 硬件要求
- CPU: 2核心及以上
- 内存: 4GB 及以上 (推荐 8GB)
- 硬盘: 2GB 可用空间
- GPU: 可选（支持 CUDA 加速）

### 软件要求
- Python >= 3.10
- 操作系统:
  - Windows 10/11
  - macOS 10.15+
  - Linux (Ubuntu 20.04+/CentOS 7+)
- CUDA Toolkit >= 11.8 (可选，用于 GPU 加速)

## 🚀 快速开始

### 方法一：使用 pip（推荐）

```bash
# 1. 克隆代码仓库
git clone https://github.com/saury1120/pdf-mcp.git
cd pdf-mcp

# 2. 创建并激活虚拟环境
## 使用 uv（推荐，更快）
uv venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# 3. 安装依赖
uv pip install -r requirements.txt
uv pip install -e .

# 4. 启动服务
uv run pdf_reader
```

### 方法二：从源码安装

```bash
git clone https://github.com/saury1120/pdf-mcp.git
cd pdf-mcp
pip install build
python -m build
pip install dist/*.whl
```

## ⚙️ 配置说明

### 环境变量配置
```bash
# GPU 配置
CUDA_VISIBLE_DEVICES=0  # 指定使用的 GPU
TORCH_THREADS=4         # PyTorch 线程数

# 内存管理
MEMORY_THRESHOLD=0.8    # 内存使用阈值（0-1）
MAX_IDLE_TIME=300      # 模型最大空闲时间（秒）

# 缓存配置
CACHE_SIZE=1000        # LRU 缓存大小
CACHE_TTL=3600        # 缓存过期时间（秒）
```

### Claude Desktop 配置
1. 找到配置文件：
   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - Windows: `%AppData%/Claude/claude_desktop_config.json`

2. 添加以下配置：
```json
{
    "mcpServers": {
        "pdf_reader": {
            "command": "uv",
            "args": [
                "--directory",
                "/path/to/pdf-mcp",  # 替换为实际路径
                "run",
                "pdf_reader"
            ]
        }
    }
}
```

## 📚 API 文档

### 基础功能
```python
# 文本提取
extract_text(file_path: str, page_number: Optional[int] = None) -> str

# 图片提取
extract_images(file_path: str, page_number: Optional[int] = None) -> List[str]

# 表格提取
extract_tables(file_path: str, page_number: Optional[int] = None) -> List[pd.DataFrame]
```

### 高级分析
```python
# 文档分类
classify_document(file_path: str, categories: List[str]) -> Dict[str, float]

# 相似度计算
calculate_similarity(file_path1: str, file_path2: str) -> float

# 语言检测
detect_languages(file_path: str) -> Dict[str, float]

# 高级分析
analyze_content(file_path: str, analysis_type: str) -> Dict[str, Any]
```

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 开源协议

本项目采用 MIT 协议 - 详见 [LICENSE](LICENSE) 文件