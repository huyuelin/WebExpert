import asyncio
from crawl4ai import AsyncWebCrawler

async def _fetch_markdown_async(url: str) -> str:
    """内部协程：抓取网页并返回 Markdown 字符串"""
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)
        return result.markdown

def click_url_and_return_md(url: str) -> str:
    """
    抓取指定 URL 并以 Markdown 形式返回网页内容。

    Parameters
    ----------
    url : str
        目标网页地址

    Returns
    -------
    str
        解析后的 Markdown 内容
    """
    return asyncio.run(_fetch_markdown_async(url))


# ------------------ 用    例 ------------------
if __name__ == "__main__":
    #md_text = click_url_and_return_md("https://blog.csdn.net/zzbzlw1218/article/details/140873033")
    #md_text = click_url_and_return_md("https://www.jobui.com/rank/company/view/suzhou/wujinlingpeijian/")
    #md_text = click_url_and_return_md("https://www.sohu.com/a/712250200_120928700")
    md_text = click_url_and_return_md("https://www.sohu.com/a/863035381_120939916")
    print(md_text)  # 只打印前 1000 字查看效果
