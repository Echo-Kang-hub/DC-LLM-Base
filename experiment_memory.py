"""
记忆机制实验：Buffer Memory vs Summary Memory
对比两种不同的对话记忆方式，观察它们在多轮对话中的表现
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from config import Config
import time


def count_tokens(text):
    """简单的token计数估算（粗略）"""
    # 英文：约4个字符 = 1 token
    # 中文：约1.5个字符 = 1 token
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    other_chars = len(text) - chinese_chars
    return int(chinese_chars / 1.5 + other_chars / 4)


def summarize_history(llm, messages):
    """将对话历史压缩为摘要"""
    if not messages:
        return ""
    
    # 构建对话历史文本
    history_text = "\n".join([
        f"{'用户' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}"
        for msg in messages
    ])
    
    # 生成摘要
    summary_prompt = f"""请将以下对话历史压缩为简洁的摘要，保留关键信息：

{history_text}

摘要："""
    
    response = llm.invoke([HumanMessage(content=summary_prompt)])
    return response.content


class BufferMemoryChat:
    """Buffer Memory：存储完整对话历史"""
    
    def __init__(self, llm):
        self.llm = llm
        self.messages = []
    
    def chat(self, user_input, verbose=False):
        # 添加用户消息
        self.messages.append(HumanMessage(content=user_input))
        
        if verbose:
            print(f"\n历史消息数: {len(self.messages)}")
            print(f"Context: {[m.content[:50] + '...' if len(m.content) > 50 else m.content for m in self.messages]}")
        
        # 调用LLM
        response = self.llm.invoke(self.messages)
        
        # 添加AI响应
        self.messages.append(AIMessage(content=response.content))
        
        return response.content
    
    def get_memory_content(self):
        """获取内存内容"""
        return "\n".join([
            f"{'Human' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}"
            for msg in self.messages
        ])


class SummaryMemoryChat:
    """Summary Memory：将历史压缩为摘要"""
    
    def __init__(self, llm, summary_threshold=4):
        self.llm = llm
        self.messages = []
        self.summary = ""
        self.summary_threshold = summary_threshold  # 超过这个数量就压缩
    
    def chat(self, user_input, verbose=False):
        # 如果历史消息过多，进行压缩
        if len(self.messages) >= self.summary_threshold:
            if verbose:
                print(f"\n触发摘要压缩（消息数: {len(self.messages)}）")
            self.summary = summarize_history(self.llm, self.messages)
            self.messages = []
            if verbose:
                print(f"摘要: {self.summary[:100]}...")
        
        # 构建当前上下文：摘要 + 最近的消息
        context_messages = []
        if self.summary:
            context_messages.append(HumanMessage(content=f"之前对话的摘要：{self.summary}"))
        context_messages.extend(self.messages)
        context_messages.append(HumanMessage(content=user_input))
        
        if verbose:
            print(f"\n当前消息数: {len(self.messages)}, 是否有摘要: {bool(self.summary)}")
        
        # 调用LLM
        response = self.llm.invoke(context_messages)
        
        # 只保存最近的消息到历史
        self.messages.append(HumanMessage(content=user_input))
        self.messages.append(AIMessage(content=response.content))
        
        return response.content
    
    def get_memory_content(self):
        """获取内存内容"""
        content = ""
        if self.summary:
            content += f"=== 摘要 ===\n{self.summary}\n\n"
        if self.messages:
            content += "=== 最近消息 ===\n"
            content += "\n".join([
                f"{'Human' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}"
                for msg in self.messages
            ])
        return content


# 初始化配置
Config.validate()

# 初始化LLM
llm = ChatOpenAI(
    model=Config.OPENAI_MODEL,
    temperature=Config.TEMPERATURE,
    api_key=Config.OPENAI_API_KEY,
    base_url=Config.OPENAI_API_BASE
)

# 创建两种聊天实例
print("初始化 Buffer Memory...")
buffer_chat = BufferMemoryChat(llm)

print("初始化 Summary Memory...")
summary_chat = SummaryMemoryChat(llm, summary_threshold=6)  # 6条消息后开始摘要

# 测试消息（至少10轮）
test_messages = [
    "你好，我叫小明，我是一名大三学生",
    "我的专业是计算机科学",
    "我最近在学习机器学习",
    "我对NLP方向特别感兴趣",
    "你能推荐一些NLP的学习资源吗？",
    "我之前学过Python和Java",
    "我想做一个文本分类的项目",
    "你还记得我叫什么名字吗？",
    "你还记得我的专业是什么吗？",
    "根据我之前告诉你的信息，你觉得我适合做什么方向的研究？"
]

print("\n" + "="*80)
print("=== Buffer Memory 测试 ===")
print("="*80 + "\n")

buffer_responses = []
buffer_start_time = time.time()

for i, msg in enumerate(test_messages, 1):
    print(f"\n【第 {i} 轮对话】")
    print(f"用户: {msg}")
    response = buffer_chat.chat(msg, verbose=True)
    buffer_responses.append(response)
    print(f"AI: {response}")
    print("-" * 80)

buffer_end_time = time.time()

print("\n" + "="*80)
print("=== Summary Memory 测试 ===")
print("="*80 + "\n")

summary_responses = []
summary_start_time = time.time()

for i, msg in enumerate(test_messages, 1):
    print(f"\n【第 {i} 轮对话】")
    print(f"用户: {msg}")
    response = summary_chat.chat(msg, verbose=True)
    summary_responses.append(response)
    print(f"AI: {response}")
    print("-" * 80)

summary_end_time = time.time()

# 对比分析
print("\n" + "="*80)
print("=== 对比分析 ===")
print("="*80 + "\n")

print("1. 关键问题回答对比（第8-10轮）：")
print("-" * 80)
for i in range(7, 10):
    print(f"\n问题 {i+1}: {test_messages[i]}")
    print(f"\nBuffer Memory 回答：\n{buffer_responses[i]}")
    print(f"\nSummary Memory 回答：\n{summary_responses[i]}")
    print("-" * 80)

print("\n2. 记忆内容对比：")
print("-" * 80)
print("\nBuffer Memory 存储内容：")
buffer_content = buffer_chat.get_memory_content()
print(buffer_content)
print(f"\nBuffer Memory Token 估算: {count_tokens(buffer_content)} tokens")

print("\n" + "-" * 80)
print("\nSummary Memory 存储内容：")
summary_content = summary_chat.get_memory_content()
print(summary_content)
print(f"\nSummary Memory Token 估算: {count_tokens(summary_content)} tokens")

print("\n3. 性能对比：")
print("-" * 80)
print(f"Buffer Memory 总耗时: {buffer_end_time - buffer_start_time:.2f} 秒")
print(f"Summary Memory 总耗时: {summary_end_time - summary_start_time:.2f} 秒")

print("\n4. 总结：")
print("-" * 80)
buffer_tokens = count_tokens(buffer_content)
summary_tokens = count_tokens(summary_content)
token_reduction = (1 - summary_tokens / buffer_tokens) * 100 if buffer_tokens > 0 else 0

print(f"""
Buffer Memory:
  - 优点: 保留完整对话历史，信息无损失
  - 缺点: Token消耗大（{buffer_tokens} tokens），成本高
  - 适用场景: 短期对话，需要精确记忆每个细节

Summary Memory:
  - 优点: Token消耗小（{summary_tokens} tokens），节省约 {token_reduction:.1f}% 的tokens
  - 缺点: 可能丢失某些细节信息
  - 适用场景: 长期对话，注重整体上下文而非每个细节

建议:
  - 对于需要记住精确信息（如姓名、数字等），Buffer Memory更可靠
  - 对于长对话或成本敏感场景，Summary Memory更经济
  - 可以考虑混合使用：重要信息用Buffer，一般对话用Summary
""")

print("\n实验完成！")
