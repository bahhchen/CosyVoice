

# 卸载方式之前的python
* 控制面板 / 设置 → 应用 → Python → 卸载
## 会发生什么
* ✅ **会删除 Python 安装目录**
  例如：
  ```
  C:\Users\xxx\AppData\Local\Programs\Python\Python310
  ```
* ⚠️ **不一定删除用户级缓存和配置**
  * `pip` 下载缓存：

    ```
    C:\Users\xxx\AppData\Local\pip\Cache
    ```
  * 有时残留：

    ```
    C:\Users\xxx\AppData\Roaming\Python
    ```
## 包会不会被删？
* **不会删**：如果你所有包都装在这个 Python 目录的 `Lib\site-packages`

# Miniconda3
## 一、下载安装包：
**这是你要记住的地址：**

> [https://repo.anaconda.com/miniconda/](https://repo.anaconda.com/miniconda/)

下载 Miniconda3-py310_25.11.1-1-Windows-x86_64

## 二、Conda 新版的“服务条款（Terms of Service, ToS）”未接受
在 Anaconda Prompt 中执行下面三条命令：

```bat
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2
```

执行后，conda 会记录在本地，之后就不会再提示。

# 在 VSCode 里添加 Anaconda Prompt 终端

1. 点击底部 terminal窗口 右上角 + 号 旁边的 'v' 
2. 选中 Configure Terminal Settings，在Settings界面找到 Terminal>Integrated>Profiles:Windows 并点击 Edit in Settings.json 
3. 增加 "Anaconda Prompt" 配置（args 从桌面图标 Anaconda Prompt 属性中的 '目标(T)' 复制）：
```json
"terminal.integrated.profiles.windows": {
        "Anaconda Prompt": {
            "path": [
                "${env:windir}\\Sysnative\\cmd.exe",
                "${env:windir}\\System32\\cmd.exe"
            ],
            "args": ["/K", "C:\\Users\\bahhc\\miniconda3\\Scripts\\activate.bat C:\\Users\\bahhc\\miniconda3"],
            "icon": "terminal-cmd"
        }
    },
```
```bash
# 如果设置为默认窗口
"terminal.integrated.defaultProfile.windows": "Anaconda Prompt"
```

# 在 VSCode 选择 conda 环境
让 VSCode 识别 conda 解释器（一次设置，永久生效）

1. `Ctrl + Shift + P`
2. 输入：`Python: Select Interpreter`
3. 选择：
```
Python 3.10.x ('cosyvoice': conda)
C:\Users\bahhc\miniconda3\envs\cosyvoice\python.exe
```

# whisper 找不到
## 1、先确认：whisper 目录真的不存在（验证一步）

请直接执行：

```bat
dir C:\Users\bahhc\miniconda3\envs\cosyvoice\Lib\site-packages | findstr whisper
```

你大概率只会看到类似：

```
openai_whisper-20231117.dist-info
```

而 **不会有：**

```
whisper\
```

👉 这就解释了一切。

---

## 2、✅ 正确修复方式（强烈推荐）

### ✅ 方案 1（最稳）：强制重装 + 禁用缓存

请 **完整复制执行**：

```bat
python -m pip uninstall -y openai-whisper
python -m pip cache purge
python -m pip install --no-cache-dir openai-whisper
```

然后立刻测试：

```bat
python -c "import whisper; print(whisper.__file__)"
```

👉 **只要能打印路径，就 100% 成功**

# 降级 ruamel.yaml
这个报错**非常典型，而且不是你代码写错**，而是 **依赖版本不匹配** 导致的：
> ❌ `AttributeError: 'Loader' object has no attribute 'max_depth'`

## 一句话结论（先看）
👉 **`hyperpyyaml` / `ruamel.yaml` 版本不兼容**
CosyVoice 里用的 `load_hyperpyyaml`，在你当前环境下，底层 `yaml Loader` 没有 `max_depth` 这个属性。

在你的 `cosyvoice` conda 环境里执行：
```bash
pip uninstall -y ruamel.yaml ruamel.yaml.clib
pip install ruamel.yaml==0.17.21
```
然后再次运行

# 在 conda 里删除一个环境
## 一、先看看有哪些环境（可选）
```bash
conda env list
```

## 二、删除指定环境（标准做法 ✅）
```bash
conda remove -n 环境名 --all
```

# 查看你当前系统支持的 CUDA 版本
```bash
# 查看对应关系
# https://pytorch.org/get-started/previous-versions/

# 打开 CMD / PowerShell：
nvidia-smi

# 1️⃣ 创建新环境（名字可自定义，这里用 cosyvoice_env）
conda create -n cosyvoice_env python=3.10 -y

# 2️⃣ 激活环境
conda activate cosyvoice_env

# 3️⃣ 安装 PyTorch 2.9.1 + torchaudio 2.9.1 + torch-complex + CUDA 11.7   优先于项目安装 避免 libiomp5md.dll 报错
# pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121 -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia

# 4️⃣ 安装 torch-complex（纯 Python 包，不依赖 CUDA）
# pip install torch-complex==0.4.4

# CUDA 版本不完整：从安装文件中找到 nvrtc-builtins64_121.dll 放到 system32中
# https://developer.nvidia.com/cuda-12-1-0-download-archive

# 报错 Initializing libiomp5md.dll, but found libiomp5md.dll already initialized
```

```bash
以上文本，不改动剧情、不润色，只做结构化与机械替换：
1.请整理出角色，区分男女，统一替换为 "Speaker 性别编号"（如 Speaker 男1）
2.只替换文中的对话 角色xx说 角色xx答，TA xx说，TAxx问 等类似替换 "角色代号:"，提为一个段落，且放在句首
3.其他文本原样输出，并 修正 数字和英文：半全角，空格等
4.输出角色对应表
```