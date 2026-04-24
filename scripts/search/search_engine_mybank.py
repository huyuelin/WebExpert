import json
import socket
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.connection import HTTPConnection

class KeepAliveHTTPAdapter(HTTPAdapter):
    """支持 TCP SO_KEEPALIVE 的 HTTP 适配器"""
    def __init__(self, idle=60, interval=60, count=5, **kwargs):
        self.idle, self.interval, self.count = idle, interval, count
        super().__init__(**kwargs)

    def init_poolmanager(self, *args, **kwargs):
        opts = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]
        if hasattr(socket, "TCP_KEEPIDLE"):     # Linux
            opts.append((socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, self.idle))
        elif hasattr(socket, "TCP_KEEPALIVE"):  # macOS
            opts.append((socket.IPPROTO_TCP, socket.TCP_KEEPALIVE, self.idle))
        if hasattr(socket, "TCP_KEEPINTVL"):
            opts.append((socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, self.interval))
        if hasattr(socket, "TCP_KEEPCNT"):
            opts.append((socket.IPPROTO_TCP, socket.TCP_KEEPCNT, self.count))

        kwargs["socket_options"] = HTTPConnection.default_socket_options + opts
        return super().init_poolmanager(*args, **kwargs)


def rag_server_maya(query: str, env: str = "PROD", pretty: bool = False):
    """向 Maya 检索，返回解析后的结果；pretty=True 时返回漂亮字符串"""
    url = "https://persona.mybank.cn/api/v4/knowledge/indSchemaProduction"
    payload = {"query": query, "env": env}

    sess = requests.Session()
    sess.mount("http://",  KeepAliveHTTPAdapter(max_retries=5))
    sess.mount("https://", KeepAliveHTTPAdapter(max_retries=5))

    try:
        r = sess.post(url, json=payload, timeout=(300, 300))
        r.raise_for_status()
        res = r.json()

        # 关键：把 data 里的 JSON 字符串再解一层
        if isinstance(res.get("data"), str):
            try:
                res["data"] = json.loads(res["data"])
            except json.JSONDecodeError:
                pass   # 保留原字符串

        return True, (
            json.dumps(res, indent=2, ensure_ascii=False) if pretty else res
        )

    except requests.exceptions.RequestException as e:
        return False, f"Request error: {e}"
    

def search_engine_with_rag(query: str, env: str = "PROD", pretty: bool = False):
    """向 Maya 检索，返回解析后的结果；pretty=True 时返回漂亮字符串"""
    url = "https://persona.mybank.cn/api/v4/knowledge/indSchemaProduction"
    payload = {"query": query, "env": env}

    sess = requests.Session()
    sess.mount("http://",  KeepAliveHTTPAdapter(max_retries=5))
    sess.mount("https://", KeepAliveHTTPAdapter(max_retries=5))

    try:
        r = sess.post(url, json=payload, timeout=(300, 300))
        r.raise_for_status()
        res = r.json()

        # 关键：把 data 里的 JSON 字符串再解一层
        if isinstance(res.get("data"), str):
            try:
                res["data"] = json.loads(res["data"])
            except json.JSONDecodeError:
                pass   # 保留原字符串
        return res

    except requests.exceptions.RequestException as e:
        return False, f"Request error: {e}"



# ------------- 示例 -------------
if __name__ == "__main__":
    # ok, content = rag_server_maya(
    #     "信贷审批是什么",

    #     pretty=True
    # )
    # print(content if ok else f"调用失败：{content}")

    res = search_engine_with_rag(
        #"信贷审批是什么",
        #"杭州余杭区五常街道餐饮行业景气度及竞争情况",
        #"中国餐饮行业2024年景气度 PMI 数据",
        #"中国五金零售行业2023年景气度分析",
        "苏州五金零售市场竞争格局及电商渗透率",
        pretty=True
    )
    print(res["data"]["candidated_texts"][0]["webUrl"])

