下面我**系统地、工程视角**把你列出的 **5 种 inference 接口**一次讲清楚，
并且给你一张 **“什么时候用哪个” 的决策表**，方便你在项目里直接选。

---

# 一、整体先给你一张“全景图”

> **这些接口的本质区别在于：**
>
> * **声音从哪来**
> * **风格怎么控制**
> * **有没有训练过这个说话人**

```
SFT        → 已训练说话人（最稳）
Zero-shot  → 用一段语音克隆声音
Cross-ling → 用一种语言的声音，说另一种语言
VC         → 语音 → 语音（换声音）
Instruct   → 预训练说话人 + 指令控制风格
```

---

# 二、逐个接口详细说明

---

## 1️⃣ `inference_sft`

```python
def inference_sft(
    self,
    tts_text,
    spk_id,
    stream=False,
    speed=1.0,
    text_frontend=True
)
```

### 🧠 本质

**SFT = Supervised Fine-Tuning**

> 用**训练集中已有的说话人**直接合成

---

### 参数说明

| 参数              | 含义             |
| --------------- | -------------- |
| `tts_text`      | 要合成的文本         |
| `spk_id`        | **模型内置说话人 ID** |
| `stream`        | 是否流式           |
| `speed`         | 语速             |
| `text_frontend` | 是否文本前处理        |

---

### 特点

✅ 音质最好
✅ 稳定性最高
❌ 只能用内置说话人
❌ 风格固定

---

### 适合场景

* 产品默认音色
* 客服 / 播报
* 批量生成

---

## 2️⃣ `inference_zero_shot`

```python
def inference_zero_shot(
    self,
    tts_text,
    prompt_text,
    prompt_wav,
    zero_shot_spk_id='',
    stream=False,
    speed=1.0,
    text_frontend=True
)
```

### 🧠 本质

**Zero-shot TTS**

> 用 **一段语音 + 对应文本** 克隆说话人

---

### 参数重点

| 参数                 | 作用                   |
| ------------------ | -------------------- |
| `prompt_text`      | prompt_wav 的**准确转写** |
| `prompt_wav`       | 参考音色                 |
| `zero_shot_spk_id` | speaker cache        |

---

### 特点

✅ 不需要训练
✅ 音色克隆能力强
❌ 必须文本–音频对齐
❌ 使用成本高

---

### 适合场景

* 声音复刻
* 个性化 TTS
* Demo / 实验

---

## 3️⃣ `inference_cross_lingual`

```python
def inference_cross_lingual(
    self,
    tts_text,
    prompt_wav,
    zero_shot_spk_id='',
    stream=False,
    speed=1.0,
    text_frontend=True
)
```

### 🧠 本质

**跨语言 Zero-shot**

> 用 **A 语言的声音** → 说 **B 语言的文本**

---

### 参数说明

| 参数           | 含义         |
| ------------ | ---------- |
| `prompt_wav` | 参考音色（任意语言） |
| `tts_text`   | 目标语言文本     |

🚫 不需要 `prompt_text`

---

### 特点

✅ 语言与音色解耦
✅ 国际化很强
❌ 发音可能带口音

---

### 适合场景

* 外语配音
* 多语言虚拟人
* 国际客服

---

## 4️⃣ `inference_vc`

```python
def inference_vc(
    self,
    source_wav,
    prompt_wav,
    stream=False,
    speed=1.0
)
```

### 🧠 本质

**VC = Voice Conversion**

> **语音 → 语音**（不经过文本）

---

### 参数说明

| 参数           | 含义   |
| ------------ | ---- |
| `source_wav` | 原始语音 |
| `prompt_wav` | 目标音色 |

---

### 特点

✅ 保留原语调 / 情绪
✅ 不依赖 ASR
❌ 不可改文本内容

---

### 适合场景

* 变声
* 配音迁移
* 情绪保留

---

## 5️⃣ `inference_instruct`

```python
def inference_instruct(
    self,
    tts_text,
    spk_id,
    instruct_text,
    stream=False,
    speed=1.0,
    text_frontend=True
)
```

### 🧠 本质

**Instruct-TTS（指令控制）**

> **已训练说话人 + 文本指令控制风格**

---

### 参数重点

| 参数              | 作用           |
| --------------- | ------------ |
| `spk_id`        | 训练好的说话人      |
| `instruct_text` | 风格 / 情绪 / 语气 |

---

### 特点

✅ 风格控制强
✅ 稳定性高
❌ 不能换音色

---

### 适合场景

* 情绪 TTS
* 剧情对白
* 智能助理

---

# 三、五种接口对比表（建议收藏）

| 模式            | 换音色 | 换语言 | 风格控制 | 稳定性   | 成本  |
| ------------- | --- | --- | ---- | ----- | --- |
| SFT           | ❌   | ❌   | ❌    | ⭐⭐⭐⭐⭐ | ⭐   |
| Zero-shot     | ✅   | ❌   | ⚠️   | ⭐⭐⭐   | ⭐⭐⭐ |
| Cross-lingual | ✅   | ✅   | ⚠️   | ⭐⭐    | ⭐⭐⭐ |
| VC            | ✅   | ❌   | 原样保留 | ⭐⭐⭐   | ⭐⭐  |
| Instruct      | ❌   | ❌   | ✅    | ⭐⭐⭐⭐  | ⭐   |

---

# 四、怎么选（直接结论）

* **要稳、要量产** → `inference_sft`
* **要克隆声音** → `inference_zero_shot`
* **要跨语言** → `inference_cross_lingual`
* **要保留情绪换声音** → `inference_vc`
* **要情绪 / 风格可控** → `inference_instruct`

---

## 五、一句话终极总结

> **SFT：模型认识你**
> **Zero-shot：模型模仿你**
> **Cross-lingual：模型用你的声音说外语**
> **VC：模型替你换个嗓子说同一句话**
> **Instruct：模型按你要求的方式说**

---

如果你愿意，我可以帮你：

* 给你 **一个自动路由函数**：根据需求选 inference
* 写一套 **统一 TTS API 封装**
* 或直接评估 **你现在项目最合理的模式组合**

你只要说一句：
👉 **“我现在的真实业务场景是 XXX”**
