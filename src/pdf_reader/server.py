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

# 全局变量
nlp = None
classifier = None
sentence_model = None
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

def load_model_in_thread(model_name: str):
    """在单独的线程中加载模型"""
    global nlp, classifier, sentence_model
    
    if model_name == 'spacy':
        try:
            return spacy.load('en_core_web_sm')
        except OSError:
            spacy.cli.download('en_core_web_sm')
            return spacy.load('en_core_web_sm')
    elif model_name == 'classifier':
        return pipeline("zero-shot-classification", 
                      device='cuda' if torch.cuda.is_available() else 'cpu',
                      model='facebook/bart-large-mnli')  # 使用更小的模型
    elif model_name == 'sentence_transformer':
        return SentenceTransformer('paraphrase-MiniLM-L6-v2')

def initialize_dependencies():
    """异步初始化所需的依赖和模型"""
    global nlp, classifier, sentence_model
    
    try:
        # NLTK数据 - 使用异步方式下载
        nltk_resources = ['punkt', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words', 'stopwords']
        for resource in nltk_resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                nltk.download(resource, quiet=True)
        
        # 使用线程池并行加载模型
        future_spacy = executor.submit(load_model_in_thread, 'spacy')
        future_classifier = executor.submit(load_model_in_thread, 'classifier')
        future_sentence = executor.submit(load_model_in_thread, 'sentence_transformer')
        
        # 等待所有模型加载完成
        nlp = future_spacy.result()
        classifier = future_classifier.result()
        sentence_model = future_sentence.result()
        
        print("模型加载完成")
        print(f"GPU加速: {'可用' if torch.cuda.is_available() else '不可用'}")
        
        return True
    except Exception as e:
        print(f"初始化失败: {str(e)}")
        return False

# 使用缓存装饰器优化文本处理
@lru_cache(maxsize=100)
def process_text(text: str) -> str:
    """处理文本并缓存结果"""
    doc = nlp(text)
    return " ".join([token.text for token in doc])

# 优化图片处理
def optimize_image(image: Image.Image, max_size: int = 1024) -> Image.Image:
    """优化图片大小和质量"""
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = tuple(int(dim * ratio) for dim in image.size)
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    return image

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
        # 使用pdfminer提取文本，可以处理更复杂的PDF布局
        text = pdfminer_extract_text(file_path)
        
        if analysis_type == "entities":
            # 使用spaCy进行命名实体识别
            doc = nlp(text)
            entities = {}
            for ent in doc.ents:
                if ent.label_ not in entities:
                    entities[ent.label_] = []
                entities[ent.label_].append(ent.text)
            return {"entities": entities}
            
        elif analysis_type == "summary":
            # 使用spaCy生成简单摘要（选择重要句子）
            doc = nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents]
            # 简单选择前3个句子作为摘要
            summary = " ".join(sentences[:3])
            return {"summary": summary}
            
        elif analysis_type == "keywords":
            # 使用NLTK提取关键词
            tokens = nltk.word_tokenize(text)
            pos_tags = nltk.pos_tag(tokens)
            # 选择名词作为关键词
            keywords = [word for word, pos in pos_tags if pos.startswith('NN')]
            # 去重并限制数量
            keywords = list(set(keywords))[:10]
            return {"keywords": keywords}
            
    except Exception as e:
        return {"error": str(e)}

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

async def classify_document(file_path: str, categories: List[str]) -> Dict[str, float]:
    """对文档进行分类"""
    try:
        text = pdfminer_extract_text(file_path)
        result = classifier(text, categories)
        return {
            "labels": result["labels"],
            "scores": result["scores"]
        }
    except Exception as e:
        return {"error": str(e)}

async def calculate_similarity(file_path1: str, file_path2: str) -> Dict[str, Any]:
    """计算两个文档的相似度"""
    try:
        text1 = pdfminer_extract_text(file_path1)
        text2 = pdfminer_extract_text(file_path2)
        
        # 使用sentence-transformers计算语义相似度
        embeddings1 = sentence_model.encode(text1, convert_to_tensor=True)
        embeddings2 = sentence_model.encode(text2, convert_to_tensor=True)
        
        similarity = torch.nn.functional.cosine_similarity(embeddings1.unsqueeze(0), 
                                                         embeddings2.unsqueeze(0))
        
        return {
            "similarity_score": float(similarity),
            "interpretation": "相似度范围从0（完全不同）到1（完全相同）"
        }
    except Exception as e:
        return {"error": str(e)}

async def detect_languages(file_path: str) -> Dict[str, Any]:
    """检测文档中的语言"""
    try:
        text = pdfminer_extract_text(file_path)
        
        # 将文本分成段落
        paragraphs = text.split('\n\n')
        
        # 对每个段落进行语言检测
        language_stats = defaultdict(int)
        total_paragraphs = len(paragraphs)
        
        for para in paragraphs:
            if para.strip():
                try:
                    lang = detect(para)
                    language_stats[lang] += 1
                except:
                    continue
        
        # 计算每种语言的比例
        language_distribution = {
            lang: count/total_paragraphs 
            for lang, count in language_stats.items()
        }
        
        return {
            "primary_language": max(language_stats.items(), key=lambda x: x[1])[0],
            "language_distribution": language_distribution
        }
    except Exception as e:
        return {"error": str(e)}

async def advanced_text_analysis(file_path: str) -> Dict[str, Any]:
    """执行高级文本分析"""
    try:
        text = pdfminer_extract_text(file_path)
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
            (phrase, score) 
            for phrase, score in zip(feature_names, scores)
        ]
        important_phrases.sort(key=lambda x: x[1], reverse=True)
        
        return {
            "complexity_metrics": {
                "avg_sentence_length": avg_sentence_length,
                "vocabulary_size": len(set(token.text.lower() for token in doc)),
                "readability_score": avg_sentence_length * 0.39 + 11.8  # 简化的可读性评分
            },
            "pos_distribution": dict(pos_dist),
            "dependency_patterns": dict(dep_dist),
            "important_phrases": [
                {"phrase": phrase, "importance": float(score)} 
                for phrase, score in important_phrases[:10]
            ]
        }
    except Exception as e:
        return {"error": str(e)}

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

if __name__ == "__main__":
    # 设置torch使用的线程数
    torch.set_num_threads(4)