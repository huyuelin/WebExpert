from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ================= 本地模型加载（仅初始化一次） =================
LOCAL_MODEL_PATH = '/workspace2/linwen/models/Qwen3-32B'

# 加载 tokenizer & model，并保持为模块级全局变量，保证整个进程只加载一次
_tokenizer = AutoTokenizer.from_pretrained(
    LOCAL_MODEL_PATH,
    trust_remote_code=True
)
_model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL_PATH,
    device_map='auto',           # 根据硬件环境自动放置到 GPU/CPU
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
_model.eval()

# -------------------- 内部封装 --------------------

def _run_local_model(prompt: str, max_new_tokens: int = 2048) -> str:
    """使用已加载的本地模型生成回复。"""
    inputs = _tokenizer(prompt, return_tensors='pt')
    # 将输入张量移动到模型所在设备
    inputs = {k: v.to(_model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # 关闭采样，保持 deterministic，可按需修改
            eos_token_id=_tokenizer.eos_token_id,
        )

    # 取生成部分（去掉 prompt 本身）
    generated_ids = output_ids[0][inputs['input_ids'].shape[1]:]
    return _tokenizer.decode(generated_ids, skip_special_tokens=True)


# -------------------- 对外主接口 --------------------

def get_llm_response(
    prompt: str,
    stop: List[str],
    model: str = 'qwen3-32b',  # 为了向后兼容，保留无用形参
    api_key: str | None = None,
    base_url: str = '',
    timeout: int = 3600,
    echo_stream: bool = True,
) -> str:
    """兼容旧签名的本地模型调用接口。

    参数中的 model、api_key、base_url、timeout 均被忽略，仅用于保持向后兼容，
    以避免大规模修改调用方代码。
    """
    # 直接调用本地模型生成完整回复
    output = _run_local_model(prompt)

    # 如果给定 stop 标记，则在首个标记位置截断（保留标记本身）
    if stop:
        min_idx: int | None = None
        for tag in stop:
            idx = output.find(tag)
            if idx != -1:
                end_pos = idx + len(tag)
                if min_idx is None or end_pos < min_idx:
                    min_idx = end_pos
        if min_idx is not None:
            output = output[:min_idx]

    # 按需模拟流式打印
    if echo_stream:
        print(output, end='', flush=True)

    return output


# ------------------ 用    例 ------------------
if __name__ == "__main__":
    END_SEARCH_QUERY = "<|end_search_query|>"
    my_prompt = """You are a reasoning assistant with the ability to perform web searches to help you answer the user\'s question accurately. You have special tools:\n\n- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\nThen, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\nYou can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to 20.\n\nOnce you have all the information you need, continue your reasoning.\n\nExample:\nQuestion: "Alice David is the voice of Lara Croft in a video game developed by which company?"\nAssistant thinking steps:\n- I need to find out who voices Lara Croft in the video game.\n- Then, I need to determine which company developed that video game.\n\nAssistant:\n<|begin_search_query|>Alice David Lara Croft voice<|end_search_query|>\n\n(System returns processed information from relevant web pages)\n\nAssistant thinks: The search results indicate that Alice David is the voice of Lara Croft in a specific video game. Now, I need to find out which company developed that game.\n\nAssistant:\n<|begin_search_query|>video game developed by Alice David Lara Croft<|end_search_query|>\n\n(System returns processed information from relevant web pages)\n\nAssistant continues reasoning with the new information...\n\nRemember:\n- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n- When done searching, continue your reasoning.\n\nPlease answer the following question. You should think step by step to solve it.\n\nProvide your final answer in the format \\boxed{YOUR_ANSWER}.\n\nQuestion:\nWhat is OpenAI Deep Research?\n\n"""
    result = get_llm_response(my_prompt, stop=[END_SEARCH_QUERY])
    print("\n\n=== 最终 buffer ===\n", result)
