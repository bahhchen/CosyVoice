

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

---