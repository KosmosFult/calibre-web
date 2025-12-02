import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import os
from google import genai
import lancedb

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


db = lancedb.connect("./ebook_test")

client = genai.Client()



# result = client.models.embed_content(
#         model="gemini-embedding-001",
#         contents= chapters[0:4])

total_tokens = client.models.count_tokens(
    model="gemini-embedding-001", contents=chapters[0]
)

print(total_tokens)

model_info = client.models.get(model="gemini-embedding-001")

print(f"{model_info.input_token_limit=}")

# for embedding in result.embeddings:
#     print(embedding)
