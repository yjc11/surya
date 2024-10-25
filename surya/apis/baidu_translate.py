import json
import random
from hashlib import md5

import requests


def baidu_translate(query, from_lang, to_lang):
    # 请替换为您的APP ID和密钥
    appid = ''
    appkey = ''

    endpoint = 'http://api.fanyi.baidu.com/api/trans/vip/translate'

    salt = random.randint(32768, 65536)
    sign = md5((appid + query + str(salt) + appkey).encode('utf-8')).hexdigest()

    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    payload = {
        'appid': appid,
        'q': query,
        'from': from_lang,
        'to': to_lang,
        'salt': salt,
        'sign': sign,
    }

    try:
        response = requests.post(endpoint, params=payload, headers=headers)
        result = response.json()

        if 'error_code' in result:
            return f"翻译错误: {result['error_msg']}"
        else:
            return result['trans_result'][0]['dst']
    except Exception as e:
        return f"发生错误: {str(e)}"


# 使用示例
if __name__ == "__main__":
    text_to_translate = "你好的英文是Hello!"
    result = baidu_translate(text_to_translate, 'en', 'zh')
    print(f"原文: {text_to_translate}")
    print(f"翻译结果: {result}")
