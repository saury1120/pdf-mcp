from typing import Any, List, Dict
import asyncio
import base64
import fitz  # PyMuPDF
from PIL import Image
import io
import nltk
import spacy
import pandas as pd
from tabula import read_pdf
from pdfminer.high_level import extract_text as pdfminer_extract_text
from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
import mcp.server.stdio
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from langdetect import detect
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from collections import defaultdict
import threading
from functools import lru_cache
import concurrent.futures
import time
import psutil

class ModelManager:
    _instance = None
    _models = {}
    _lock = threading.Lock()
    _last_used = {}
    _memory_threshold = 0.8  # 内存使用率阈值
    _max_idle_time = 300  # 模型最大空闲时间（秒）

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        self._model_configs = {
            'spacy': {
                'name': 'en_core_web_sm',
                'loader': self._load_spacy,
                'quantize': False
            },
            'classifier': {
                'name': 'facebook/bart-large-mnli',
                'loader': self._load_classifier,
                'quantize': {
                    'enabled': True,
                    'method': 'dynamic',
                    'dtype': torch.qint8,
                    'layers': [torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d],
                    'calibration_size': 100
                }
            },
            'sentence_transformer': {
                'name': 'paraphrase-MiniLM-L6-v2',
                'loader': self._load_sentence_transformer,
                'quantize': {
                    'enabled': True,
                    'method': 'dynamic',
                    'dtype': torch.qint8,
                    'layers': [torch.nn.Linear],
                    'calibration_size': 100
                }
            }
        }
        self.device = self._get_optimal_device()
        self._setup_quantization_backend()
        self._cleanup_thread = threading.Thread(target=self._cleanup_models, daemon=True)
        self._cleanup_thread.start()

    def _get_optimal_device(self):
        """根据系统配置选择最优设备"""
        if not torch.cuda.is_available():
            return 'cpu'
        
        # 检查GPU内存
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        if gpu_memory < 4 * 1024 * 1024 * 1024:  # 小于4GB
            return 'cpu'
        
        return 'cuda'

    def _get_optimal_quantization_config(self, model_type):
        """根据设备和模型类型选择最优量化配置"""
        base_config = self._model_configs[model_type]['quantize']
        if not base_config:
            return base_config

        if self.device == 'cpu':
            return {
                'enabled': True,
                'method': 'dynamic',
                'dtype': torch.qint8,
                'layers': base_config['layers'],
                'calibration_size': 50  # CPU下使用更小的校准集
            }
        else:
            return {
                'enabled': True,
                'method': 'static',
                'dtype': torch.float16,  # GPU下使用半精度
                'layers': base_config['layers'],
                'calibration_size': base_config['calibration_size']
            }

    def get_model(self, model_type: str):
        """延迟加载模型"""
        with self._lock:
            if model_type not in self._models:
                config = self._model_configs.get(model_type)
                if not config:
                    raise ModelLoadError(f"Unknown model type: {model_type}")
                
                # 检查内存使用情况
                self._check_memory_usage()
                
                # 加载模型
                try:
                    model = config['loader'](config['name'])
                    quant_config = self._get_optimal_quantization_config(model_type)
                    if quant_config and quant_config['enabled']:
                        model = self._prepare_model_for_quantization(model, quant_config)
                    self._models[model_type] = model
                except Exception as e:
                    raise ModelLoadError(f"Failed to load model {model_type}: {str(e)}")
            
            # 更新最后使用时间
            self._last_used[model_type] = time.time()
            return self._models[model_type]

    def _check_memory_usage(self):
        """检查内存使用情况并在必要时卸载模型"""
        memory_percent = psutil.virtual_memory().percent / 100

        if memory_percent > self._memory_threshold:
            self._unload_least_used_model()

    def _unload_least_used_model(self):
        """卸载最少使用的模型"""
        if not self._last_used:
            return

        current_time = time.time()
        least_used_model = min(self._last_used.items(), key=lambda x: x[1])[0]
        if current_time - self._last_used[least_used_model] > self._max_idle_time:
            self._unload_model(least_used_model)

    def _unload_model(self, model_type: str):
        """卸载指定模型"""
        if model_type in self._models:
            del self._models[model_type]
            del self._last_used[model_type]
            # 强制进行垃圾回收
            import gc
            gc.collect()
            if self.device == 'cuda':
                torch.cuda.empty_cache()

    def _cleanup_models(self):
        """定期清理未使用的模型"""
        while True:
            time.sleep(60)  # 每分钟检查一次
            with self._lock:
                current_time = time.time()
                models_to_unload = [
                    model_type for model_type, last_used in self._last_used.items()
                    if current_time - last_used > self._max_idle_time
                ]
                for model_type in models_to_unload:
                    self._unload_model(model_type)

    def _setup_quantization_backend(self):
        """设置量化后端"""
        if self.device == 'cuda':
            # 在GPU上使用CUDA量化后端
            torch.backends.quantized.engine = 'fbgemm'
        else:
            # 在CPU上使用fbgemm (Windows compatible)
            torch.backends.quantized.engine = 'fbgemm'

    def _prepare_model_for_quantization(self, model, config):
        """准备模型进行量化"""
        if not config['enabled']:
            return model

        if config['method'] == 'dynamic':
            return self._apply_dynamic_quantization(model, config)
        elif config['method'] == 'static':
            return self._apply_static_quantization(model, config)
        return model

    def _apply_dynamic_quantization(self, model, config):
        """应用动态量化"""
        try:
            print(f"Applying dynamic quantization with dtype {config['dtype']}")
            model = torch.quantization.quantize_dynamic(
                model,
                qconfig_spec={
                    layer: torch.quantization.default_dynamic_qconfig
                    for layer in config['layers']
                },
                dtype=config['dtype']
            )
            print("Dynamic quantization applied successfully")
            return model
        except Exception as e:
            print(f"Dynamic quantization failed: {str(e)}")
            return model

    def _apply_static_quantization(self, model, config):
        """应用静态量化"""
        try:
            print(f"Applying static quantization")
            # 准备量化配置
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm' if self.device == 'cuda' else 'fbgemm')
            
            # 融合操作
            model = torch.quantization.fuse_modules(model, [['conv', 'bn', 'relu']])
            
            # 准备量化
            model = torch.quantization.prepare(model)
            
            # 校准（这里需要实际的校准数据）
            # self._calibrate_model(model, config['calibration_size'])
            
            # 转换为量化模型
            model = torch.quantization.convert(model)
            
            print("Static quantization applied successfully")
            return model
        except Exception as e:
            print(f"Static quantization failed: {str(e)}")
            return model

    def _calibrate_model(self, model, calibration_size):
        """使用校准数据集校准模型（用于静态量化）"""
        # 这里应该使用实际的校准数据
        # 为了示例，我们使用随机数据
        with torch.no_grad():
            for _ in range(calibration_size):
                dummy_input = torch.randn(1, 3, 224, 224)
                model(dummy_input)

    def _load_spacy(self):
        try:
            return spacy.load('en_core_web_sm')
        except OSError:
            spacy.cli.download('en_core_web_sm')
            return spacy.load('en_core_web_sm')

    def _load_classifier(self):
        print("Loading classifier model...")
        model = pipeline("zero-shot-classification",
                        model='facebook/bart-large-mnli',
                        device=self.device)
        
        if self._model_configs['classifier']['quantize']['enabled']:
            print("Applying quantization to classifier model")
            model.model = self._prepare_model_for_quantization(
                model.model, 
                self._model_configs['classifier']['quantize']
            )
        return model

    def _load_sentence_transformer(self):
        print("Loading sentence transformer model...")
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        model = model.to(self.device)
        
        if self._model_configs['sentence_transformer']['quantize']['enabled']:
            print("Applying quantization to sentence transformer model")
            model.encoder = self._prepare_model_for_quantization(
                model.encoder,
                self._model_configs['sentence_transformer']['quantize']
            )
        return model

    def clear_cache(self, model_type: str = None):
        with self._lock:
            if model_type:
                if model_type in self._models:
                    del self._models[model_type]
            else:
                self._models.clear()

    def get_model_memory_usage(self, model_type: str = None):
        """获取模型内存使用情况"""
        if model_type:
            if model_type in self._models:
                model = self._models[model_type]
                return self._get_model_size(model)
            return None
        
        memory_usage = {}
        for model_type, model in self._models.items():
            memory_usage[model_type] = self._get_model_size(model)
        return memory_usage

    def _get_model_size(self, model):
        """计算模型大小（以MB为单位）"""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
            
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return round(size_mb, 2)

class ModelContext:
    """模型使用的上下文管理器"""
    def __init__(self, model_type: str, manager: 'ModelManager'):
        self.model_type = model_type
        self.manager = manager
        self.model = None
        self.error = None

    def __enter__(self):
        try:
            self.model = self.manager.get_model(self.model_type)
            return self.model
        except Exception as e:
            self.error = e
            raise ModelError(f"Error loading model {self.model_type}: {str(e)}")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # 记录错误但不处理
            print(f"Error using model {self.model_type}: {str(exc_val)}")
        return False  # 让异常继续传播

class ModelError(Exception):
    """模型相关错误的基类"""
    pass

class ModelLoadError(ModelError):
    """模型加载错误"""
    pass

class ModelInferenceError(ModelError):
    """模型推理错误"""
    pass

# 创建全局ModelManager实例
model_manager = ModelManager()

# 服务器初始化
server = Server("pdf_reader")

# 下载必要的NLTK数据
nltk_resources = ['punkt', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words', 'stopwords']
for resource in nltk_resources:
    try:
        nltk.data.find(f'tokenizers/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """列出可用的工具"""
    return [
        types.Tool(
            name="extract-text",
            description="从PDF文件中提取文本内容",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "PDF文件的路径",
                    },
                    "page_number": {
                        "type": "integer",
                        "description": "要提取的页码（从0开始）",
                    },
                },
                "required": ["file_path"],
            },
        ),
        types.Tool(
            name="extract-images",
            description="从PDF文件中提取图片",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "PDF文件的路径",
                    },
                    "page_number": {
                        "type": "integer",
                        "description": "要提取的页码（从0开始）",
                    },
                },
                "required": ["file_path"],
            },
        ),
        types.Tool(
            name="extract-tables",
            description="从PDF文件中提取表格",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "PDF文件的路径",
                    },
                    "page_number": {
                        "type": "integer",
                        "description": "要提取的页码（从0开始）",
                    },
                },
                "required": ["file_path"],
            },
        ),
        types.Tool(
            name="analyze-content",
            description="分析PDF文件内容，提取关键信息",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "PDF文件的路径",
                    },
                    "analysis_type": {
                        "type": "string",
                        "description": "分析类型：entities（实体）, summary（摘要）, keywords（关键词）",
                        "enum": ["entities", "summary", "keywords"],
                    },
                },
                "required": ["file_path", "analysis_type"],
            },
        ),
        types.Tool(
            name="get-metadata",
            description="获取PDF文件的元数据信息",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "PDF文件的路径",
                    },
                },
                "required": ["file_path"],
            },
        ),
        types.Tool(
            name="classify-document",
            description="对PDF文档进行分类",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "PDF文件的路径",
                    },
                    "categories": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "可能的分类类别列表",
                    },
                },
                "required": ["file_path", "categories"],
            },
        ),
        types.Tool(
            name="calculate-similarity",
            description="计算两个PDF文档的相似度",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path1": {
                        "type": "string",
                        "description": "第一个PDF文件的路径",
                    },
                    "file_path2": {
                        "type": "string",
                        "description": "第二个PDF文件的路径",
                    },
                },
                "required": ["file_path1", "file_path2"],
            },
        ),
        types.Tool(
            name="detect-languages",
            description="检测PDF文档中使用的语言",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "PDF文件的路径",
                    },
                },
                "required": ["file_path"],
            },
        ),
        types.Tool(
            name="advanced-analysis",
            description="执行高级文本分析",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "PDF文件的路径",
                    },
                },
                "required": ["file_path"],
            },
        ),
    ]

async def extract_text_from_pdf(file_path: str, page_number: int = None) -> str:
    """从PDF中提取文本"""
    try:
        doc = fitz.open(file_path)
        if page_number is not None:
            if 0 <= page_number < len(doc):
                text = doc[page_number].get_text()
                doc.close()
                return text
            else:
                doc.close()
                return f"页码 {page_number} 超出范围。PDF共有 {len(doc)} 页。"
        
        # 如果没有指定页码，提取所有页面的文本
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
        return text
    except Exception as e:
        return f"提取文本时出错: {str(e)}"

async def extract_images_from_pdf(file_path: str, page_number: int = None):
    """从PDF中提取图片，返回base64编码的图片列表"""
    try:
        doc = fitz.open(file_path)
        images = []
        pages = [page_number] if page_number is not None else range(len(doc))
        
        for page_num in pages:
            page = doc[page_num]
            image_list = page.get_images()
            
            # 并行处理图片
            def process_image(img_index):
                try:
                    xref = image_list[img_index][0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # 转换和优化图片
                    image = Image.open(io.BytesIO(image_bytes))
                    image = optimize_image(image)
                    
                    # 转换为base64
                    buffered = io.BytesIO()
                    image.save(buffered, format="PNG", optimize=True)
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    return img_str
                except Exception as e:
                    print(f"处理图片时出错: {str(e)}")
                    return None
            
            # 使用线程池并行处理图片
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(process_image, i) for i in range(len(image_list))]
                for future in concurrent.futures.as_completed(futures):
                    if future.result():
                        images.append(future.result())
        
        doc.close()
        return images
    except Exception as e:
        print(f"提取图片时出错: {str(e)}")
        return []

async def extract_tables_from_pdf(file_path: str, page_number: int = None) -> List[str]:
    """从PDF中提取表格"""
    try:
        if page_number is not None:
            tables = read_pdf(file_path, pages=page_number + 1)  # tabula使用1-based页码
        else:
            tables = read_pdf(file_path, pages='all')
        
        if not tables:
            return ["未找到表格"]
        
        result = []
        for i, table in enumerate(tables):
            result.append(f"表格 {i+1}:\n{table.to_string()}\n---")
        return result
    except Exception as e:
        return [f"提取表格时出错: {str(e)}"]

async def analyze_pdf_content(file_path: str, analysis_type: str) -> Dict[str, Any]:
    """分析PDF内容"""
    try:
        text = extract_text_from_pdf(file_path)
        
        if analysis_type == "entities":
            with ModelContext('spacy', model_manager) as nlp:
                doc = nlp(text)
                entities = [(ent.text, ent.label_) for ent in doc.ents]
                return {"entities": entities}
            
        elif analysis_type == "summary":
            with ModelContext('classifier', model_manager) as classifier:
                sentences = nltk.sent_tokenize(text)
                results = classifier(sentences, 
                                  candidate_labels=["important", "not important"],
                                  multi_label=False)
                important_sentences = [sent for sent, score in zip(sentences, results['scores']) 
                                    if score > 0.7]
                return {"summary": " ".join(important_sentences[:5])}
            
        elif analysis_type == "keywords":
            with ModelContext('spacy', model_manager) as nlp:
                doc = nlp(text)
                keywords = [token.text for token in doc if not token.is_stop and token.is_alpha]
                return {"keywords": list(set(keywords[:20]))}
            
    except ModelError as e:
        return {"error": f"Model error: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

async def get_pdf_metadata(file_path: str) -> Dict[str, Any]:
    """获取PDF元数据"""
    try:
        doc = fitz.open(file_path)
        metadata = doc.metadata
        doc.close()
        return {
            "title": metadata.get("title", "未知"),
            "author": metadata.get("author", "未知"),
            "subject": metadata.get("subject", "未知"),
            "keywords": metadata.get("keywords", "未知"),
            "creator": metadata.get("creator", "未知"),
            "producer": metadata.get("producer", "未知"),
            "creation_date": metadata.get("creationDate", "未知"),
            "modification_date": metadata.get("modDate", "未知"),
            "page_count": doc.page_count
        }
    except Exception as e:
        return {"error": str(e)}

async def classify_document(file_path: str, categories: List[str]) -> Dict[str, Any]:
    """对文档进行分类"""
    try:
        text = pdfminer_extract_text(file_path)
        with ModelContext('classifier', model_manager) as classifier:
            result = classifier(text, categories)
            return {
                "labels": result["labels"],
                "scores": [float(score) for score in result["scores"]]
            }
    except ModelError as e:
        return {"error": f"Model error: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

async def calculate_similarity(file_path1: str, file_path2: str) -> Dict[str, float]:
    """计算两个文档的相似度"""
    try:
        text1 = pdfminer_extract_text(file_path1)
        text2 = pdfminer_extract_text(file_path2)
        
        with ModelContext('sentence_transformer', model_manager) as model:
            # 将文本分成较小的块进行处理
            def chunk_text(text, chunk_size=1000):
                return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            
            # 计算文本块的嵌入向量
            def get_embeddings(text):
                chunks = chunk_text(text)
                embeddings = model.encode(chunks)
                return np.mean(embeddings, axis=0)
            
            # 计算两个文档的相似度
            embedding1 = get_embeddings(text1)
            embedding2 = get_embeddings(text2)
            similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
            
            return {"similarity_score": float(similarity)}
            
    except ModelError as e:
        return {"error": f"Model error: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

async def detect_languages(file_path: str) -> Dict[str, Any]:
    """检测文档中的语言"""
    try:
        text = pdfminer_extract_text(file_path)
        with ModelContext('spacy', model_manager) as nlp:
            # 将文本分成段落
            paragraphs = text.split('\n\n')
            language_info = []
            
            for para in paragraphs:
                if not para.strip():
                    continue
                    
                try:
                    lang = detect(para)
                    doc = nlp(para)
                    # 获取段落的语言特征
                    features = {
                        'text': para[:100] + '...' if len(para) > 100 else para,
                        'language': lang,
                        'tokens': len(doc),
                        'sentences': len(list(doc.sents))
                    }
                    language_info.append(features)
                except Exception as e:
                    print(f"Error processing paragraph: {str(e)}")
                    continue
            
            return {
                "language_analysis": language_info,
                "document_stats": {
                    "total_paragraphs": len(paragraphs),
                    "processed_paragraphs": len(language_info)
                }
            }
            
    except ModelError as e:
        return {"error": f"Model error: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

async def advanced_text_analysis(file_path: str) -> Dict[str, Any]:
    """执行高级文本分析"""
    try:
        text = pdfminer_extract_text(file_path)
        
        with ModelContext('spacy', model_manager) as nlp:
            doc = nlp(text)
            
            # 1. 复杂度分析
            sentences = list(doc.sents)
            avg_sentence_length = sum(len(sent) for sent in sentences) / len(sentences)
            
            # 2. 词性分布
            pos_dist = defaultdict(int)
            for token in doc:
                pos_dist[token.pos_] += 1
            
            # 3. 依存关系分析
            dep_dist = defaultdict(int)
            for token in doc:
                dep_dist[token.dep_] += 1
            
            # 4. 主题建模（使用TF-IDF找出最重要的词组）
            vectorizer = TfidfVectorizer(max_features=10)
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            important_phrases = [
                {"phrase": phrase, "importance": float(score)} 
                for phrase, score in zip(feature_names, scores)
            ]
            
            return {
                "complexity_metrics": {
                    "avg_sentence_length": float(avg_sentence_length),
                    "vocabulary_size": len(set(token.text.lower() for token in doc)),
                    "readability_score": float(avg_sentence_length * 0.39 + 11.8)
                },
                "pos_distribution": dict(pos_dist),
                "dependency_patterns": dict(dep_dist),
                "important_phrases": sorted(important_phrases, 
                                         key=lambda x: x["importance"], 
                                         reverse=True)[:10]
            }
            
    except ModelError as e:
        return {"error": f"Model error: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

@lru_cache(maxsize=100)
def process_text(text: str) -> str:
    """处理文本并缓存结果"""
    try:
        with ModelContext('spacy', model_manager) as nlp:
            doc = nlp(text)
            return " ".join([token.text for token in doc])
    except ModelError as e:
        print(f"Error processing text: {str(e)}")
        return text  # 返回原始文本作为后备方案
    except Exception as e:
        print(f"Unexpected error processing text: {str(e)}")
        return text

async def main():
    """运行服务器"""
    try:
        print("PDF Reader MCP 服务启动中...")
        
        # 在后台线程中初始化依赖
        init_thread = threading.Thread(target=initialize_dependencies)
        init_thread.start()
        
        # 启动服务器
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="pdf_reader",
                    server_version="0.1.0",
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )
    except Exception as e:
        print(f"服务器运行错误: {str(e)}")
        raise

def initialize_dependencies():
    """异步初始化所需的依赖"""
    try:
        # NLTK数据 - 使用异步方式下载
        nltk_resources = ['punkt', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words', 'stopwords']
        for resource in nltk_resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                nltk.download(resource, quiet=True)
        
        print("NLTK resources loaded")
        print(f"GPU acceleration: {'available' if torch.cuda.is_available() else 'not available'}")
        
        return True
    except Exception as e:
        print(f"Initialization failed: {str(e)}")
        return False

# 优化图片处理
def optimize_image(image: Image.Image, max_size: int = 1024) -> Image.Image:
    """优化图片大小和质量"""
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = tuple(int(dim * ratio) for dim in image.size)
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    return image

if __name__ == "__main__":
    # 设置torch使用的线程数
    torch.set_num_threads(4)
    
    # 确保在主模块中运行
    import sys
    if 'src.pdf_reader.server' in sys.modules:
        del sys.modules['src.pdf_reader.server']
    
    # 初始化并运行服务器
    asyncio.run(main())

@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """处理工具调用请求"""
    if not arguments:
        raise ValueError("缺少参数")

    file_path = arguments.get("file_path")
    if not file_path:
        raise ValueError("缺少文件路径")

    if name == "extract-text":
        page_number = arguments.get("page_number")
        text = await extract_text_from_pdf(file_path, page_number)
        return [types.TextContent(type="text", text=text)]
    
    elif name == "extract-images":
        page_number = arguments.get("page_number")
        images = await extract_images_from_pdf(file_path, page_number)
        result = []
        for i, img_base64 in enumerate(images):
            if img_base64.startswith("提取图片时出错"):
                result.append(types.TextContent(type="text", text=img_base64))
            else:
                result.append(types.ImageContent(
                    type="image",
                    format="image/png",
                    data=img_base64
                ))
        return result if result else [types.TextContent(type="text", text="未找到图片")]
    
    elif name == "extract-tables":
        page_number = arguments.get("page_number")
        tables = await extract_tables_from_pdf(file_path, page_number)
        return [types.TextContent(type="text", text="\n".join(tables))]
    
    elif name == "analyze-content":
        analysis_type = arguments.get("analysis_type")
        if not analysis_type:
            raise ValueError("缺少分析类型")
        
        result = await analyze_pdf_content(file_path, analysis_type)
        if "error" in result:
            return [types.TextContent(type="text", text=f"分析出错: {result['error']}")]
        
        if analysis_type == "entities":
            text = "识别到的实体:\n"
            for entity_type, entities in result["entities"].items():
                text += f"\n{entity_type}:\n- " + "\n- ".join(entities)
        elif analysis_type == "summary":
            text = f"文档摘要:\n{result['summary']}"
        elif analysis_type == "keywords":
            text = "关键词:\n- " + "\n- ".join(result["keywords"])
        
        return [types.TextContent(type="text", text=text)]
    
    elif name == "get-metadata":
        metadata = await get_pdf_metadata(file_path)
        if "error" in metadata:
            return [types.TextContent(type="text", text=f"获取元数据出错: {metadata['error']}")]
        
        text = "PDF元数据:\n"
        for key, value in metadata.items():
            text += f"{key}: {value}\n"
        
        return [types.TextContent(type="text", text=text)]
    
    elif name == "classify-document":
        categories = arguments.get("categories")
        if not categories:
            raise ValueError("缺少分类类别")
        
        result = await classify_document(file_path, categories)
        if "error" in result:
            return [types.TextContent(type="text", text=f"分类出错: {result['error']}")]
        
        text = "文档分类结果:\n"
        for label, score in zip(result["labels"], result["scores"]):
            text += f"{label}: {score:.2%}\n"
        
        return [types.TextContent(type="text", text=text)]
    
    elif name == "calculate-similarity":
        file_path2 = arguments.get("file_path2")
        if not file_path2:
            raise ValueError("缺少第二个文件路径")
        
        result = await calculate_similarity(file_path, file_path2)
        if "error" in result:
            return [types.TextContent(type="text", text=f"计算相似度出错: {result['error']}")]
        
        text = f"文档相似度: {result['similarity_score']:.2%}\n"
        text += result["interpretation"]
        
        return [types.TextContent(type="text", text=text)]
    
    elif name == "detect-languages":
        result = await detect_languages(file_path)
        if "error" in result:
            return [types.TextContent(type="text", text=f"语言检测出错: {result['error']}")]
        
        text = f"主要语言: {result['primary_language']}\n\n"
        text += "语言分布:\n"
        for lang, ratio in result["language_distribution"].items():
            text += f"{lang}: {ratio:.1%}\n"
        
        return [types.TextContent(type="text", text=text)]
    
    elif name == "advanced-analysis":
        result = await advanced_text_analysis(file_path)
        if "error" in result:
            return [types.TextContent(type="text", text=f"分析出错: {result['error']}")]
        
        text = "高级文本分析结果:\n\n"
        
        # 复杂度指标
        text += "1. 复杂度指标:\n"
        metrics = result["complexity_metrics"]
        text += f"- 平均句子长度: {metrics['avg_sentence_length']:.1f}\n"
        text += f"- 词汇量: {metrics['vocabulary_size']}\n"
        text += f"- 可读性评分: {metrics['readability_score']:.1f}\n\n"
        
        # 词性分布
        text += "2. 词性分布:\n"
        for pos, count in result["pos_distribution"].items():
            text += f"- {pos}: {count}\n"
        text += "\n"
        
        # 重要短语
        text += "3. 重要短语:\n"
        for item in result["important_phrases"]:
            text += f"- {item['phrase']}: {item['importance']:.3f}\n"
        
        return [types.TextContent(type="text", text=text)]
    
    else:
        raise ValueError(f"未知的工具: {name}")