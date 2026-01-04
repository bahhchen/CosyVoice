import sys
sys.path.append('./third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import AutoModel
import torchaudio
import torch
import re
import os

g_prompts = {
    'ZL2_woman' :{
        'prompt_wav' : './chh/assets/zh-ZL2_woman.wav',
        'prompt_text' : '也许你相亲过很多次，可是你有真心付出过吗？你有努力过吗？你一定以为物质就可以换来爱情是吧！你不如去买彩票好了！也许隔天中了奖就可以得到你想要的一切。'
    },
    'Bowen_man' :{
        'prompt_wav' : './chh/assets/zh-Bowen_man.wav',
        'prompt_text' : '简单来说，人工智能是一门致力于让计算机和机器像人一样思考和行动的科学领域。'\
                        '它的目标，是模拟、延伸和扩展人的智能，让机器能够胜任通常需要人类智慧才能完成的任务。'\
                        '比如：解决问题、感知环境、理解语言，甚至进行创造。'\
                        'AI是一非常广泛的领域，它包含了计算机科学、语言学、神经科学，甚至哲学和心理学等多个学科的知识。'
    },
    'YM_woman' :{
        'prompt_wav' : './chh/assets/zh-YM_woman.wav',
        'prompt_text' : '我醒来时，看见了那个男孩子，他看着我，我想起了和他一起，在养老院晒洗床单的时光。'
    }
}

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

def inference_zero_shot(cosyvoice, txt_contents, prompts, dstwav):
    waves = []

    prompts_key0 = next(iter(prompts))
    def_speaker = prompts[prompts_key0]
    prev_speaker = def_speaker

    # "Speaker namexxx: 别整那些没用的了!"
    pattern = r"^Speaker\s+(\w+):\s*(.+)$"    

    for txt in txt_contents:

        m = re.match(pattern, txt)
        if m:
            speaker_name = m.group(1)   # namexxx
            text = m.group(2)           # 别整那些没用的了!
            if speaker_name in prompts:
                cur_speaker = prompts[speaker_name]
                prev_speaker = cur_speaker
            else:
                cur_speaker = prev_speaker
        else:
            text = txt
            cur_speaker = prev_speaker

        print("speaker_name:", speaker_name)
        # print("text:", text)

        # instruct_list + cur_speaker['prompt_text']
        for i, j in enumerate(cosyvoice.inference_zero_shot(text, 
                                                            'You are a helpful assistant.<|endofprompt|>' + cur_speaker['prompt_text'],
                                                            cur_speaker['prompt_wav'], stream=False)):
            # torchaudio.save('./chh/output/时间回旋三部曲_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
            wav = j['tts_speech']
            # 保证 shape 是 [1, T]
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)

            waves.append(wav)

    save_waves(dstwav, waves, cosyvoice.sample_rate)   


def cosyvoice3_example():
    """ CosyVoice3 Usage, check https://funaudiollm.github.io/cosyvoice3/ for more details
    """
    cosyvoice = AutoModel(model_dir='chh/pretrained_models/Fun-CosyVoice3-0.5B')
    prompt_wav = './chh/assets/zh-ZL2_woman.wav'
    # prompt_text = '希望你以后能够做的比我还好呦。'
    prompt_text = '也许你相亲过很多次，可是你有真心付出过吗？你有努力过吗？你一定以为物质就可以换来爱情是吧！你不如去买彩票好了！也许隔天中了奖就可以得到你想要的一切。'
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

    txt_contents = [
        'Speaker YM_woman: 波奇酱你搁这儿呢啊! 虽然不知道你咋整的, 我还是买了一裤兜子甜水呢! 卧槽! 撩了的吉他小妹儿! 喜多, 你怎么搁这儿呢?',
        'Speaker YM_woman: 卧槽! 这谁啊?',
        'Speaker ZL2_woman: 别整那些没用的了!',
    ]
    inference_zero_shot(cosyvoice, txt_contents, g_prompts, './chh/output/2p_yayi.wav')

def main():
    # cosyvoice_example()
    # cosyvoice2_example()
    cosyvoice3_example()


if __name__ == '__main__':
    main()