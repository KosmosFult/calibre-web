# -*- coding: utf-8 -*-
"""
书籍内容提取器
用于从各种格式的电子书中提取文本内容，供 AI Agent 使用
"""

import os
import zipfile
import re
from lxml import etree, html
from io import BytesIO

from . import config, logger
from .epub_helper import get_content_opf, default_ns

log = logger.create()


class BookContentExtractor:
    """书籍内容提取器基类"""
    
    @staticmethod
    def extract(file_path, format_type):
        """
        根据格式提取书籍内容
        
        Args:
            file_path: 书籍文件路径
            format_type: 格式类型 (epub, pdf, txt 等)
            
        Returns:
            dict: {
                "chapters": [
                    {"title": "Chapter 1", "content": "...", "index": 0},
                    ...
                ],
                "total_chapters": 10,
                "title": "Book Title"
            }
        """
        format_type = format_type.lower()
        
        if format_type in ['epub', 'kepub']:
            return EPUBExtractor.extract(file_path)
        elif format_type == 'txt':
            return TXTExtractor.extract(file_path)
        # elif format_type == 'pdf':
        #     return PDFExtractor.extract(file_path)
        else:
            raise ValueError(f"Unsupported format: {format_type}")


class EPUBExtractor:
    """EPUB 格式提取器"""
    
    @staticmethod
    def extract(file_path):
        """
        提取 EPUB 内容
        
        Returns:
            dict: 包含章节列表的字典
        """
        try:
            epub_zip = zipfile.ZipFile(file_path)
            
            # 1. 获取 content.opf
            tree, cf_name = get_content_opf(file_path, default_ns)
            cover_path = os.path.dirname(cf_name)
            
            # 2. 解析书名
            title_nodes = tree.xpath('/pkg:package/pkg:metadata/dc:title/text()', 
                                    namespaces={
                                        'pkg': 'http://www.idpf.org/2007/opf',
                                        'dc': 'http://purl.org/dc/elements/1.1/'
                                    })
            book_title = title_nodes[0] if title_nodes else "Unknown"
            
            # 3. 获取章节顺序 (spine)
            spine_items = tree.xpath('/pkg:package/pkg:spine/pkg:itemref/@idref', 
                                    namespaces=default_ns)
            
            # 4. 获取所有资源的映射 (manifest)
            manifest = {}
            for item in tree.xpath('/pkg:package/pkg:manifest/pkg:item', 
                                  namespaces=default_ns):
                item_id = item.get('id')
                item_href = item.get('href')
                manifest[item_id] = item_href
            
            # 5. 按顺序读取章节内容
            chapters = []
            for idx, spine_id in enumerate(spine_items):
                if spine_id not in manifest:
                    continue
                    
                chapter_href = manifest[spine_id]
                chapter_path = os.path.join(cover_path, chapter_href).replace('\\', '/')
                
                try:
                    # 读取 XHTML 内容
                    chapter_data = epub_zip.read(chapter_path)
                    chapter_tree = etree.fromstring(chapter_data)
                    
                    # 提取章节标题（尝试从 h1, h2, title 等标签）
                    chapter_title = EPUBExtractor._extract_chapter_title(chapter_tree, idx)
                    
                    # 提取纯文本内容
                    chapter_text = EPUBExtractor._extract_text_from_html(chapter_tree)
                    
                    # 过滤掉太短的章节（可能是封面、版权页等）
                    if len(chapter_text.strip()) < 50:
                        continue
                    
                    chapters.append({
                        "index": len(chapters),  # 重新编号
                        "title": chapter_title,
                        "content": chapter_text,
                        "word_count": len(chapter_text)
                    })
                    
                except Exception as e:
                    log.warning(f"Failed to extract chapter {spine_id}: {e}")
                    continue
            
            return {
                "title": book_title,
                "total_chapters": len(chapters),
                "chapters": chapters
            }
            
        except Exception as e:
            log.error(f"Failed to extract EPUB content: {e}")
            raise
    
    @staticmethod
    def _extract_chapter_title(tree, index):
        """从 XHTML 中提取章节标题"""
        # 尝试各种可能的标题标签
        for tag in ['h1', 'h2', 'h3', 'title']:
            titles = tree.xpath(f'//*[local-name()="{tag}"]/text()')
            if titles:
                title = titles[0].strip()
                if title and len(title) < 100:  # 标题不应该太长
                    return title
        
        # 如果没找到，使用默认名称
        return f"Chapter {index + 1}"
    
    @staticmethod
    def _extract_text_from_html(tree):
        """从 XHTML 中提取纯文本，保留段落结构"""
        # 移除 script 和 style 标签
        for element in tree.xpath('.//script | .//style'):
            element.getparent().remove(element)
        
        # 提取 body 中的文本
        body = tree.xpath('//*[local-name()="body"]')
        if not body:
            # 如果没有 body，尝试从根节点提取
            text = tree.xpath('string()')
        else:
            text = body[0].xpath('string()')
        
        # 清理文本：
        # 1. 替换多个空白字符为单个空格
        text = re.sub(r'\s+', ' ', text)
        # 2. 移除首尾空白
        text = text.strip()
        
        return text


class TXTExtractor:
    """TXT 格式提取器"""
    
    @staticmethod
    def extract(file_path):
        """
        提取 TXT 内容
        
        TXT 没有章节概念，我们按段落或固定字数分块
        """
        try:
            with open(file_path, 'rb') as f:
                rawdata = f.read()
            
            # 检测编码
            import chardet
            result = chardet.detect(rawdata)
            encoding = result['encoding'] or 'utf-8'
            
            try:
                text = rawdata.decode(encoding)
            except:
                text = rawdata.decode('utf-8', errors='ignore')
            
            # 尝试智能分章节（基于常见的章节标记）
            chapter_pattern = r'(第[一二三四五六七八九十百千\d]+章|Chapter\s+\d+|CHAPTER\s+\d+)'
            parts = re.split(f'({chapter_pattern})', text, flags=re.IGNORECASE)
            
            chapters = []
            
            if len(parts) > 1:
                # 找到了章节标记
                i = 0
                while i < len(parts):
                    if re.match(chapter_pattern, parts[i], re.IGNORECASE):
                        title = parts[i].strip()
                        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
                        
                        if len(content) > 50:
                            chapters.append({
                                "index": len(chapters),
                                "title": title,
                                "content": content,
                                "word_count": len(content)
                            })
                        i += 2
                    else:
                        i += 1
            else:
                # 没有找到章节标记，按固定字数分块（每块 5000 字）
                chunk_size = 5000
                for i in range(0, len(text), chunk_size):
                    chunk = text[i:i + chunk_size].strip()
                    if len(chunk) > 50:
                        chapters.append({
                            "index": len(chapters),
                            "title": f"Part {len(chapters) + 1}",
                            "content": chunk,
                            "word_count": len(chunk)
                        })
            
            return {
                "title": os.path.basename(file_path),
                "total_chapters": len(chapters),
                "chapters": chapters
            }
            
        except Exception as e:
            log.error(f"Failed to extract TXT content: {e}")
            raise


# 可扩展：PDF 提取器（需要额外的库，如 pdfplumber 或 PyPDF2）
# class PDFExtractor:
#     @staticmethod
#     def extract(file_path):
#         try:
#             import pdfplumber
#             # ... PDF 提取逻辑
#         except ImportError:
#             raise ImportError("pdfplumber is required for PDF extraction")

