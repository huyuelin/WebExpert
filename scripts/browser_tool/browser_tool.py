#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完整流程：
1. 取 token
2. 创建浏览器实例 → 得到 browser_id
3. 跳转到 qcc.com
4. 截图二维码，人工扫码登录
5. 登录成功后用同一 browser_id 查询企业信息
"""

import base64
import json
import os
import time
import pprint
import requests
from requests_toolbelt import MultipartEncoder

BASE_URL = "http://47.110.132.231:8000"
USERNAME  = "407106"
PASSWORD  = "123456"

# ============ 通用工具 ============ #
def post(url, token, *, json_body=None, data_body=None, headers=None, timeout=15):
    """带 Token 的 POST 请求（自动附加 Authorization）"""
    h = headers or {}
    if token:
        h["Authorization"] = f"Bearer {token}"
    resp = requests.post(url, json=json_body, data=data_body, headers=h, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


# ============ 具体流程 ============ #
def get_token() -> str:
    """① 获取 Bearer Token（表单/x-www-form-urlencoded 即可）"""
    r = requests.post(f"{BASE_URL}/token",
                      data={"username": USERNAME, "password": PASSWORD}, timeout=10)
    r.raise_for_status()
    token = r.json().get("access_token")
    if not token:
        raise RuntimeError("未取到 access_token，请检查账号或返回格式")
    return token


def create_browser(token: str, url: str | None = None) -> str:
    """② 创建浏览器实例，返回 browser_id"""
    payload = {"url": url} if url else {}
    res = post(f"{BASE_URL}/browser/create", token, json_body=payload)
    browser_id = res.get("browser_id")
    if not browser_id:
        raise RuntimeError("create_browser 未返回 browser_id")
    return browser_id


def navigate(token: str, browser_id: str, url: str):
    """③ 跳转到指定网址"""
    post(f"{BASE_URL}/browser/navigate",
         token,
         json_body={"url": url},
         headers={"Content-Type": "application/json"},
         timeout=20,
    )  # 按文档为 POST；browser_id 放在 querystring
    # 有的后端要求 browser_id 作为 query：/browser/navigate?browser_id=xxx
    # 若接口无法识别，请将 browser_id 追加到 URL：
    # post(f"{BASE_URL}/browser/navigate?browser_id={browser_id}", ...)


def screenshot(token: str, browser_id: str, save_path: str = "qcc_qr.png"):
    """④ 截图并保存到本地，供扫码登录"""
    res = post(f"{BASE_URL}/browser/screenshot",
               token,
               json_body={},  # 若不需 body，可省略
               headers={"Content-Type": "application/json"},
               timeout=20,
    )  # 同样注意 browser_id 可能要放到 querystring
    b64_data = res.get("image_base64") or res.get("data") or res.get("body")
    if not b64_data:
        raise RuntimeError("screenshot 返回未包含 base64 图像数据字段")
    img_bytes = base64.b64decode(b64_data)
    with open(save_path, "wb") as f:
        f.write(img_bytes)
    print(f"[提示] 二维码已保存至 {os.path.abspath(save_path)}，请扫码登录企查查后回车继续…")
    input()  # 等待用户扫码完毕


def qcc_search(token: str, browser_id: str, keyword: str):
    """⑤ 使用已登录的 browser_id 查询企查查"""
    m = MultipartEncoder(
        fields={
            "browser_id": browser_id,                       # ⭐ 必填，指定实例
            "url": "https://www.qcc.com/web/search",
            "method": "GET",
            "data": json.dumps({"key": keyword})
        }
    )
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": m.content_type,
    }
    resp = requests.post(f"{BASE_URL}/browser/request",
                         headers=headers, data=m, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ============ 主执行入口 ============ #
if __name__ == "__main__":
    token = get_token()
    print("[+] Token 获取成功")

    # 1. 创建浏览器实例
    browser_id = create_browser(token)
    print(f"[+] 浏览器实例创建成功 browser_id = {browser_id}")

    # 2. 打开企查查首页
    navigate(token, browser_id, "https://www.qcc.com")
    print("[+] 已跳转至 https://www.qcc.com")

    # 3. 截图二维码，等待人工扫码
    screenshot(token, browser_id, save_path="qcc_login_qr.png")

    # 4. 登录后执行搜索
    company = "大连万达集团股份有限公司"
    result = qcc_search(token, browser_id, company)

    # 5. 打印结果
    print("\n=== 搜索结果概要 ===")
    print("状态码:", result.get("status"))
    print("部分响应头:")
    pprint.pp(result.get("headers"))
    print("\n正文前 500 字：")
    print((result.get("body") or "")[:500])
