import argparse
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path


def merge_lora(args):
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, device_map='cpu')
    print(model.generation_config)
    # ✅ 修复 generation config 的矛盾参数
    # model.generation_config.do_sample = True  #推理时随机采样
    # 或者反过来把 temperature/top_p 清掉：llava-1/5采用greedy decoding（do_sample=False），方便和文章对比
    # 设为None/del仍会报错
    # validate() 要求：do_sample=False 时 temperature 必须是 1.0，top_p 必须是 1.0
    model.generation_config.temperature = 1.0
    model.generation_config.top_p = 1.0

    print(model.generation_config)  # 确认一下再保存
    
    model.save_pretrained(args.save_model_path)
    tokenizer.save_pretrained(args.save_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, required=True)
    parser.add_argument("--save-model-path", type=str, required=True)

    args = parser.parse_args()

    merge_lora(args)
