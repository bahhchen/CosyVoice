import sys
sys.path.append('./third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import AutoModel
import torchaudio
import torch
import re
import os

# from chh.ebook import split_text_by_length
from chh.prompts import prompts as g_prompts
from chh.book_reader import read_book, read_txt_speaker_paragraphs

import json

def save_waves(filename, waves, sample_rate):
    
    # 获取上级目录
    dir_path = os.path.dirname(filename)
    # 判断是否存在，不存在就创建
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # 在时间维拼接
    merged = torch.cat(waves, dim=1)

    torchaudio.save(
        filename,
        merged,
        sample_rate
    )

def inference_zero_shot(cosyvoice, txt_contents, prompts, dstwav, isprompt = False):
    waves = []

    prompts_key0 = next(iter(prompts))
    def_speaker = prompts[prompts_key0]

    # "Speaker namexxx: 别整那些没用的了!"
    pattern = r"^Speaker\s+(\w+):\s*(.+)$"    

    for txt in txt_contents:

        m = re.match(pattern, txt)
        if m:
            speaker_name = m.group(1)   # namexxx
            text = m.group(2)           # 别整那些没用的了!
            if speaker_name not in prompts:
            #     cur_speaker = prompts[speaker_name]
            # else:
                # cur_speaker = def_speaker
                speaker_name = prompts_key0
        else:
            text = txt
            # cur_speaker = def_speaker
            speaker_name = prompts_key0

        # print("speaker_name:", speaker_name)
        # print("text:", text)
        if isprompt:
            cur_speaker = prompts[speaker_name]
        else : 
            cur_speaker = prompts['l_' + speaker_name]#prompts['s_' + speaker_name] if len(text) <= 6 else prompts['l_' + speaker_name]
            if len(text) <= 3:
                text = "……" + text + "……" 

        # instruct_list + cur_speaker['prompt_text']
        for i, j in enumerate(cosyvoice.inference_zero_shot(text, 
                                                            'You are a helpful assistant.<|endofprompt|>' + cur_speaker['prompt_text'],
                                                            cur_speaker['prompt_wav'], zero_shot_spk_id=speaker_name, stream=False)):
            # torchaudio.save('./chh/output/test{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
            wav = j['tts_speech']
            # 保证 shape 是 [1, T]
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)

            waves.append(wav)

    save_waves(dstwav, waves, cosyvoice.sample_rate)   

def initCosyVoice3():
    
    cosyvoice = AutoModel(model_dir='chh/pretrained_models/Fun-CosyVoice3-0.5B')
    # 增加缓存
    for key, value in g_prompts.items():
        cosyvoice.add_zero_shot_spk('You are a helpful assistant.<|endofprompt|>' + value['prompt_text'], value['prompt_wav'], key)

    return cosyvoice

# 合成小说 
def inference_book(txt_path, wav_path, chapter_idx = 0, end = '---end---'):
  
    chapters = read_book(txt_path, wav_path, chapter_idx, end)
    # 目录方式
    if len(chapters) < 1:
        return

    model = initCosyVoice3()

    # 合成
    for chapter in chapters:
        inference_zero_shot(model, chapter['contents'], g_prompts, chapter['wav'])

# 生成提示音
def inference_prompt(model, name, lcontent, scontent):
    res = {}
    if lcontent != None:
        txt_contents =[
            # 'Speaker '+name+':于是，我说道：'+lcontent,
            'Speaker '+name+':'+lcontent,
        ]
        lwav = './chh/assets2/zh-l_' + name +'.wav'
        inference_zero_shot(model, txt_contents, g_prompts, lwav, isprompt = True)

        res['l_' + name] = {
            'prompt_wav' : lwav,
            'prompt_text' : lcontent,
        }
     
    if scontent != None:
        txt_contents =[
            # 'Speaker '+name+':于是，我说道：'+scontent,
            'Speaker '+name+':'+scontent,
        ]
        swav = './chh/assets2/zh-s_' + name +'.wav'
        inference_zero_shot(model, txt_contents, g_prompts, swav, isprompt = True)

        res['s_' + name] = {
            'prompt_wav' : swav,
            'prompt_text' : scontent,
        }
    print("res:", json.dumps(res, ensure_ascii=False, indent=4))


def cosyvoice3_example():
    """ CosyVoice3 Usage, check https://funaudiollm.github.io/cosyvoice3/ for more details
    """
    cosyvoice = initCosyVoice3()
    # prompt_wav = './chh/assets/zh-ZL2_woman.wav'
    # prompt_text = '希望你以后能够做的比我还好呦。'
    # prompt_text = '也许你相亲过很多次，可是你有真心付出过吗？你有努力过吗？你一定以为物质就可以换来爱情是吧！你不如去买彩票好了！也许隔天中了奖就可以得到你想要的一切。'
    # zero_shot usage
    # for i, j in enumerate(cosyvoice.inference_zero_shot('八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。', 'You are a helpful assistant.<|endofprompt|>' + prompt_text,
    #                                                     prompt_wav, stream=False)):
    #     torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

    # # fine grained control, for supported control, check cosyvoice/tokenizer/tokenizer.py#L280
    # for i, j in enumerate(cosyvoice.inference_cross_lingual('You are a helpful assistant.<|endofprompt|>[breath]因为他们那一辈人[breath]在乡里面住的要习惯一点，[breath]邻居都很活络，[breath]嗯，都很熟悉。[breath]',
    #                                                         prompt_wav, stream=False)):
    #     torchaudio.save('fine_grained_control_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

    # # instruct usage, for supported control, check cosyvoice/utils/common.py#L28
    # for i, j in enumerate(cosyvoice.inference_instruct2('好少咯，一般系放嗰啲国庆啊，中秋嗰啲可能会咯。', 'You are a helpful assistant. 请用广东话表达。<|endofprompt|>',
    #                                                     prompt_wav, stream=False)):
    #     torchaudio.save('instruct_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
    # for i, j in enumerate(cosyvoice.inference_instruct2('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', 'You are a helpful assistant. 请用尽可能快地语速说一句话。<|endofprompt|>',
    #                                                     prompt_wav, stream=False)):
    #     torchaudio.save('instruct_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

    # # hotfix usage
    # for i, j in enumerate(cosyvoice.inference_zero_shot('高管也通过电话、短信、微信等方式对报道[j][ǐ]予好评。', 'You are a helpful assistant.<|endofprompt|>' + prompt_text,
    #                                                     prompt_wav, stream=False)):
    #     torchaudio.save('hotfix_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
    

    # with open('C:/projects/ai/vibeTTS/demo/text_examples/时间回旋三部曲/0029.时间回旋三部曲.极北.txt', "r", encoding="utf-8") as f:
    #     txt_content = f.read()
    #     waves = []
    #     for i, j in enumerate(cosyvoice.inference_zero_shot(txt_content, 
    #                                                         'You are a helpful assistant.<|endofprompt|>' + prompt_text,
    #                                                         prompt_wav, stream=False)):
    #         # torchaudio.save('./chh/output/时间回旋三部曲_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
    #         wav = j['tts_speech']
    #         # 保证 shape 是 [1, T]
    #         if wav.dim() == 1:
    #             wav = wav.unsqueeze(0)

    #         waves.append(wav)

    #     save_waves('./chh/output/时间回旋三部曲.wav', waves, cosyvoice.sample_rate)

    if 1:
        txt_contents = [
            # 'Speaker wl_man: 于是，我说道：卧槽!',
            # 'Speaker ldh_man: 于是，我说道：滚!',
            # 'Speaker hb_man: 于是，我说道：是你',
            # 'Speaker zjl_man: 于是，我说道：什么',
            'Speaker bl_woman: ……走',
            # 'Speaker zta_woman: 于是，我说道：什么!',
            # 'Speaker zyq_woman: 于是，我说道：走!',
            # 'Speaker fbb_woman: 卧槽! 撩了的吉他小妹儿!',
            # 'Speaker zly_woman: 卧槽! 这谁啊?',
            # 'Speaker YM_woman: 波奇酱你搁这儿呢啊! 虽然不知道你咋整的, 我还是买了一裤兜子甜水呢! 卧槽! 撩了的吉他小妹儿! 喜多, 你怎么搁这儿呢?',
            # 'Speaker dlrb_woman: 卧槽！别整那些没用的了!',
            # 'Speaker st_man: 收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。',
            # 'Speaker xz_man: 很快，她找到了底特律亨利·福特医院进行的长达三年有关心搏停止病人的研究。插入他们血流的导管探测到，四分之一的被诊断为没有心跳的人实际上有心跳。',
        ]
        inference_zero_shot(cosyvoice, txt_contents, g_prompts, './chh/output/test.wav')
    else:
        print("inference_prompt")
        # inference_prompt(cosyvoice, 'ZL2_woman', '河边的柳树轻轻摇曳，水面倒映出蓝天和白云，景色如画令人心旷神怡。', '小猫在草地玩耍。')
        # inference_prompt(cosyvoice, 'YM_woman', '在图书馆里，阳光洒在书页上，空气中弥漫着淡淡的书香，令人沉浸其中。', '雨滴落在屋檐。')
        # inference_prompt(cosyvoice, 'zlc_woman', '海边的浪花拍打着岩石，海风带着咸味吹拂脸庞，天空和海面交相辉映。', '河水轻轻流动。')
        # inference_prompt(cosyvoice, 'dlrb_woman', '古老的庭院中，石阶上落满枯叶，微风吹动，远处传来悠扬的鸟鸣声。', '微风吹拂脸庞。')
        # inference_prompt(cosyvoice, 'zly_woman', '公车缓缓驶过街道，窗外景物像电影镜头般掠过，乘客安静地坐在座位上。', '我站在天桥上。')
        # inference_prompt(cosyvoice, 'fbb_woman', '秋天的树林里，落叶铺满小径，踩上去发出沙沙声，空气中带着淡淡的泥土香。', '切换到夜间模式。')
        # inference_prompt(cosyvoice, 'zyq_woman', '夜晚的城市灯火辉煌，街道上偶尔传来车辆低沉的引擎声，微风拂过脸庞。', '鸟儿飞过蓝天。')
        # inference_prompt(cosyvoice, 'zta_woman', '清晨的公园里，跑步的人们穿梭在绿荫小道上，鸟鸣声和笑声交织在空气中。', '风铃随风作响。')
        # inference_prompt(cosyvoice, 'bl_woman', '清晨的市场熙熙攘攘，叫卖声、讨价声此起彼伏，生活气息充满街头巷尾。', '给我讲个故事。')

        # inference_prompt(cosyvoice, 'wyb_man', "山间小路蜿蜒曲折，溪水潺潺流淌，林间充满清新的泥土气息和鸟鸣声。", '花园里开满鲜花。')
        # inference_prompt(cosyvoice, 'zlc_man', "在海滩上，孩子们堆沙堡、追逐嬉戏，海浪拍打岸边发出悦耳的节奏声。", '孩子们在院子跑。')
        # inference_prompt(cosyvoice, 'xz_man', "清晨阳光透过薄雾洒在湖面上，水波荡漾，倒映出远山和飞鸟的剪影。", '公园里人影稀疏。')
        # inference_prompt(cosyvoice, 'st_man', "河边垂钓的人静静坐着，偶尔抛出钓竿，水面泛起微微涟漪，宁静而悠闲。", '小河潺潺流淌。')
        # inference_prompt(cosyvoice, 'lcw_man', "夜晚的咖啡馆里，轻柔的音乐伴随咖啡香，窗外路灯闪烁，行人匆匆而过。", '太阳落山了。')
        # inference_prompt(cosyvoice, 'ldh_man', "小镇的街道安静而宁和，远处传来钟声，孩子们在院子里嬉笑玩耍。", '阳光洒在窗台。')
        # inference_prompt(cosyvoice, 'zjl_man', "小河弯弯曲曲流向远方，水草随波摇曳，鱼儿偶尔跃出水面，微风吹动水面泛光。", '雪花飘落街道。')
        # inference_prompt(cosyvoice, 'hb_man', "天空中飘着几片白云，微风轻轻吹过河面，波光闪烁着温柔的光。", '风吹过湖面。')

def main():
    # cosyvoice_example()
    # cosyvoice2_example()
    # cosyvoice3_example()
    inference_book('./chh/books/终极实验/', './chh/output/终极实验/')


if __name__ == '__main__':
    main()