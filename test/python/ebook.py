import os
from http.client import responses

import ebooklib
import lancedb
from click import prompt

from ebooklib import epub
from bs4 import BeautifulSoup
from google import genai
from langchain_text_splitters import RecursiveCharacterTextSplitter

from cps.ai_search.context_fuse import ContextFuser

from cps.ai_search.chunking import BookParser

GEMINI_API_KEY = os.environ['GEMINI_API_KEY']

def read_epub(path):
    book = epub.read_epub(path)
    chapters = []

    # 获取书名等元数据
    title = book.get_metadata('DC', 'title')[0][0]

    for item in book.get_items():
        # 只要文档类型的章节
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            text = soup.get_text()
            # 简单的过滤，去掉太短的章节（如目录页）
            if len(text) > 500:
                chapters.append(text)
    return title, chapters


title, chapters = read_epub('/Users/kosmosfult/Documents/calibre/[Ri ] Xie Zhen Zhong Xian/E Nu De Gao Bai (17)/E Nu De Gao Bai - [Ri ] Xie Zhen Zhong Xian.epub')


# db = lancedb.connect("./ebook_test")

prompt = "".join(chapters)

client = genai.Client()

res = client.models.generate_content(
    model="gemini-flash-latest",
    contents="我正在为电子书平台撰写书籍介绍摘要，你阅读全文，来帮我写一下，要求能够吸引读者，并且不要剧透" + "<book>" + prompt + "</book>",
)

# fuser = ContextFuser()
#
# book_parser = BookParser()

# chunks = book_parser.chunk_book(17, "/Users/kosmosfult/Documents/calibre/[Ri ] Xie Zhen Zhong Xian/E Nu De Gao Bai (17)/E Nu De Gao Bai - [Ri ] Xie Zhen Zhong Xian.epub")


print(res.text)
