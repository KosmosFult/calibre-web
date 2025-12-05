import os
from google import genai
from typing import Any


class ContextFuser:
    def __init__(self):
        self.client = genai.Client()

    def fuse(self, doc_and_chunks : list[list[Any]]):
        fused_contexts = []
        for doc, chunks in doc_and_chunks:
            for chunk in chunks:
                context = self._assemble_context(doc, chunk)
                response = self.client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[context]
                )

                fused_contexts.append(response.text)

        return fused_contexts

    def batch_fuse(self, doc_and_chunks : list[list[Any]]):
        pass


    def _assemble_context(self, doc, chunk):
        prompt = f"""# Role
                你是一个专精于长篇小说 RAG（检索增强生成）系统的语境处理专家。

                # Task
                我将提供给你一份**<document>**（来自小说的某一章节或长片段）和一个从中截取的**<chunk>**。
                你的任务是编写一段简练的**语境补全说明（Context String）**。这段说明将被添加到该切片的开头，以便在该切片被单独检索时，读者（或模型）能立刻明白其背景。

                # Input Data
                <document>
                {doc}
                </document>

                <chunk>
                {chunk}
                </chunk>

                # Requirements
                请根据<document>，为<chunk>生成一段 1-3 句话的背景描述，必须满足以下要求：

                1. **指代消解**：如果切片中出现“他”、“她”、“那个人”等代词，或者只有对话没有名字，必须在语境说明中明确指出这些人具体是谁（使用全名）。
                2. **场景定位**：说明当前情节发生的地点、时间或场景（如：“在林黛玉的房间里”、“决战前夕”）。
                3. **前情提要**：简要概括导致当前切片中事件或对话的直接原因（上文刚刚发生了什么）。
                4. **格式规范**：
                - 直接陈述事实，**不要**使用“这段文字出自……”、“在这个片段中……”或“主要讲了……”这类元语言。
                - **不要**重复切片中已有的详细内容，只做铺垫。
                - **不要**输出任何解释性文字，仅输出这段补全说明。

                # Example
                *假设全文语境是《红楼梦》中宝玉和黛玉闹别扭的章节。*

                **输入目标切片**：
                “我不听！你只管去见你的好姐姐便是，何苦来哄我？”说完便扭过头去，拿着手帕拭泪。

                **（错误的输出）**：
                这段话写的是林黛玉在哭，她很生气因为贾宝玉去找薛宝钗。

                **（正确的输出 - 你应该生成的）**：
                在贾母处吃完饭后，林黛玉因误会贾宝玉偏袒薛宝钗而感到委屈生气，独自在房内哭泣，拒绝贾宝玉的解释。

                # Output
                请为<chunk>生成语境补全说明："""
        
        return prompt
