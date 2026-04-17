# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Optional
import yaml

from cirrus.configs.paths import JUDGE_PROMPT_PATH, CONFIGS_DIR
from cirrus.llm import call_llm

JUDGE_CONFIG_PATH = CONFIGS_DIR / "judge_config.yaml"

def _load_judge_config():
    with open(JUDGE_CONFIG_PATH, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg.get("judge", {})

def get_judge_prompt():
    judge_fp  = JUDGE_PROMPT_PATH
    with open(judge_fp,'r') as f:
        judge_prompt = f.read()
    return judge_prompt

def scoring_content(contentA,
                    contentB,
                    history,
                    model=None,
                    llm_args=None):
    judge_cfg = _load_judge_config()
    if model is None:
        model = judge_cfg.get("model", "deepseek-chat")
    provider = judge_cfg.get("provider")
    if llm_args is None:
        llm_args = {"temperature": judge_cfg.get("temperature", 0.0)}
    system_prompt = get_judge_prompt()
    messages = [{"role":"system","content":system_prompt}]
    user_prompt = '<recent_messages>\n'+history+'\n</recent_messages>\n'
    user_prompt += '<content_a>\n'+contentA+'\n</content_a>\n'
    user_prompt += '<content_b>\n'+contentB+'\n</content_b>\n'
    messages.append({"role":"user","content":user_prompt})
    try:
        result = call_llm(messages=messages, model=model, provider=provider, **llm_args)
        if 'Not Included' in result.content:
            return 0
        elif 'Included' in result.content:
            return 1
        else:
            return -1
    except:
        return -1

def main():
    contentA = '''
        "您好，目前已进入在线人工服务，售后工程师将尽快为您处理，请稍候。"
        '''
    contentB = '''
    "您好，针对您咨询的“.mji.cv”域名在阿里云账号间的转移问题，目前阿里云**不支持**该类域名（.cv为佛得角国家顶级域，.mji.cv为其子域）的账号间线上或线下转移操作。\n\n根据阿里云域名转移规则，仅支持已在阿里云正式开放注册并可管理的通用顶级域（如.com、.cn等）进行账号间转移。而“.mji.cv”属于非标准二级/三级国家域名，通常由特定国家注册局管理，阿里云暂未提供对此类域名的账号间转移服务。\n\n建议您：\n- 如该域名是在阿里云注册的，请确认是否能通过“域名转出”功能获取转移密码，再转入到目标阿里云账号（需目标账号重新以“域名转入”方式购入）；\n- 若域名并非在阿里云注册，而是通过其他注册商注册后解析托管至阿里云，则无法直接在阿里云内部完成账号转移。\n\n如有进一步疑问，可提供域名注册商信息或域名当前所在阿里云账号情况，我们可协助判断可行方案。",

    '''
    history = ''

    result = scoring_content(contentA,contentB,history)
    print(result)

    contentC = '''
        "您好，目前已进入在线人工服务，售后工程师将尽快为您处理，请稍候。"
        '''
    
    result = scoring_content(contentA,contentC,history)
    print(result)

if __name__ == "__main__":
    main()