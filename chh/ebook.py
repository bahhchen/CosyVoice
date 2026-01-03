
import os
import re
import mobi
import shutil
from ebooklib import epub
from bs4 import BeautifulSoup


# 定义垃圾文本的正则模式（可以根据需要扩展）
BAD_PATTERNS = [
    r"本书由.*整理", 
    r"加.*微信", 
    r"QQ：?\d+", 
    r"微信公众号", 
    r"下载网站", 
    r"www\..*", 
    r"http[s]?://.*", 
    # r"小编", 
]

def is_bad_text(text: str) -> bool:
    """判断一段文字是否属于垃圾广告"""
    for pat in BAD_PATTERNS:
        if re.search(pat, text):
            return True
    return False


# 中文标点 -> 英文标点映射
punct_map = {
    '，': ',',
    '。': '.',
    '！': '!',
    '？': '?',
    '：': ':',
    '；': ';',
    '（': '(',
    '）': ')',
    '【': '[',
    '】': ']',
    '“': '"',
    '”': '"',
    '‘': "'",
    '’': "'",
    '、': ',',
    '《': '<',
    '》': '>',
}

# 特殊情况: "——" 和 "…" 需要单独处理
def normalize_punctuation(text: str) -> str:
    # 先替换多字符标点
    text = text.replace("——", "--").replace("…", "...")
    
    # 单字符映射用 translate
    trans_table = str.maketrans(punct_map)
    text = text.translate(trans_table)

    # 删除非中英文和数字的字符，替换为空格
    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9\s\.,!?;:()\-—\"'<>]", " ", text)

    return text


def read_epub(file_path):
    """解析 EPUB，按章节输出，返回 chapters 列表，每个元素为 (title, content)"""
    book = epub.read_epub(file_path)
    print(f"=== {os.path.basename(file_path)} ===")

    chapters = []
    current_chapter_title = None
    current_chapter_content = []

    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_body_content(), "html.parser")

        # 删除分页标记或空标签
        for tag in soup.find_all(["hr", "mbp:pagebreak"]):
            tag.decompose()

        for tag in soup.find_all(["h1", "h2", "h3", "p"]):
            text = tag.get_text(strip=True)
            if not text:
                continue
            # 过滤目录或版权信息
            if text.startswith("本书由") or text.startswith("Table of Contents"):
                continue

            if tag.name in ["h1", "h2"]:
                # 遇到新章节时，先保存之前章节
                if current_chapter_title or current_chapter_content:
                    chapters.append((current_chapter_title, "\n".join(current_chapter_content)))
                current_chapter_title = text
                current_chapter_content = [text]  # 标题也放到内容里
            else:
                current_chapter_content.append(text)

    # 保存最后一章
    if current_chapter_title or current_chapter_content:
        chapters.append((current_chapter_title, "\n".join(current_chapter_content)))

    # 输出示例
    for i, (title, content) in enumerate(chapters, start=1):
        print(f"\n--- 第 {i} 章: {title} ---\n")
        print(content[:800])
        print("\n=========================\n")

    return chapters


def read_mobi(file_path):
    """解析 MOBI，按目录锚点输出章节"""
    tempdir, filepath = mobi.extract(file_path)
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()

    print(f"源文件：{file_path}，解析后目录：{tempdir}, 解析后文件：{filepath}")

    soup = BeautifulSoup(html, "html.parser")

    for pagebreak in soup.find_all("mbp:pagebreak"):
        pagebreak.decompose()  # 从文档中删除

    # 1. 找目录区（Table of Contents）
    toc_tag = soup.find(string=lambda x: x and "Table of Contents" in x)
    if not toc_tag:
        print("❌ 没找到 TOC，回退到 <h1>/<h2> 方式")
        return

    toc_section = toc_tag.find_parent("p")
    toc_links = toc_section.find_all_next("a", href=True)

    # 2. 收集章节链接
    chapters_info = []
    seen_hrefs = set()  # 用来记录已添加过的 href

    for a in toc_links:
        href = a["href"]
        if not href.startswith("#filepos"):
            print(f" 异常的链接：{href} ")
            break # 异常退出
        if href not in seen_hrefs:
            chapters_info.append((a.get_text(strip=True), href[1:]))
            seen_hrefs.add(href)

    print(f"📑 找到 {len(chapters_info)} 个章节")

    # 3. 按锚点切正文
    chapters = []
    toc_texts = set([a.get_text(strip=True) for a in toc_links])  # TOC 的所有文字，方便过滤
    for i, (title, anchor) in enumerate(chapters_info):
        start = soup.find("a", {"id": anchor})
        if not start:
            continue

        # 找下一个章节的起点
        end_anchor = chapters_info[i+1][1] if i+1 < len(chapters_info) else None
        texts = []
        node = start
        while node:
            node = node.find_next()
            if not node:
                break
            if end_anchor and node.name == "a" and node.get("id") == end_anchor:
                break
            if node.name in ["p", "h1", "h2", "h3"]:
                txt = node.get_text(" ", strip=True)
                # 🔥 过滤广告 过滤掉目录的文字
                if txt and (not is_bad_text(txt)) and (txt not in toc_texts and "Table of Contents" not in txt):  
                    texts.append(txt)

        chapters.append((title, "\n".join(texts)))

    # 4. 输出结果
    # print(f"=== {os.path.basename(file_path)} ===")
    # for i, (title, content) in enumerate(chapters, start=1):
    #     print(f"\n--- 第 {i} 章 {title} ---\n")
    #     print(content[:100])  # 显示前 800 字
    #     print("\n=========================\n")

    shutil.rmtree(tempdir)
    return chapters


def read_book(file_path):
    """自动判断文件类型并解析"""
    if file_path.endswith(".epub"):
        return read_epub(file_path)
    elif file_path.endswith(".mobi"):
        return read_mobi(file_path)
    elif file_path.endswith(".txt"):
        return
    else:
        print("❌ 不支持的文件格式:", file_path)

def split_text_by_length(text, max_len=7200):
    """
    将文本按 max_len 分段，每段尽量在靠后的换行符处分割。
    """
    segments = []
    start = 0
    txt_len = len(text)
    parts = int(txt_len / max_len) + 1
    part_len  = int(txt_len / parts) + 1
    while start < txt_len:
        # 如果剩余长度小于最大长度，直接加入
        if txt_len - start <= part_len:
            segments.append(text[start:].strip())
            break
        
        # 默认切点
        split_pos = start + part_len

        newline_pos = text.find("\n", split_pos)
        if newline_pos != -1 and newline_pos > start:
            split_pos = newline_pos + 1  # 包含换行符
        else:
            # 尝试往后找到第一个换行符
            newline_pos = text.rfind("\n", 0, split_pos)
            if newline_pos != -1:
                split_pos = newline_pos + 1  # 包含换行符
        
        # 切分
        segment = text[start:split_pos].strip()
        segments.append(segment)
        
        # 更新起点
        start = split_pos

    return segments

def safe_filename(name: str) -> str:
    # 替换所有非法文件名字符
    for ch in ['\\', '/', ':', '*', '?', '"', '<', '>', '|']:
        name = name.replace(ch, '-')
    return name.strip()


def to_txt(file_path, txt_dir):
    
    chapters = read_book(file_path)
    if not chapters:
        return
    
    filename = os.path.splitext(os.path.basename(file_path))[0]
    # 创建输出目录
    output_dir = os.path.join(txt_dir, filename)
    os.makedirs(output_dir, exist_ok=True)

    for i, (title, content) in enumerate(chapters, start=1):
        safe_title = safe_filename(title)
        segments = split_text_by_length(content)
        if len(segments) > 1:
            for j, p in enumerate(segments, 1):
                txt_content = f"Speaker 1: 第 {i} 章 第 {j} 节 \n{p}"
                txt_filename = f"{i:04d}.{j:02d}.{filename}.{safe_title}"

                output_path = os.path.join(output_dir, f"{txt_filename}.txt")
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(txt_content)

        else:
            txt_content = f"Speaker 1: 第 {i} 章 {safe_title} \n{content}"
            txt_filename = f"{i:04d}.{filename}.{safe_title}"

            output_path = os.path.join(output_dir, f"{txt_filename}.txt")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(txt_content)

# 示例调用
if __name__ == "__main__":
    to_txt("./demo/text_examples/时间回旋三部曲.mobi", "./demo/text_examples/")
