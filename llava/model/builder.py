#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from llava.model import *
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", use_flash_attn=False, **kwargs):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    if 'llava' in model_name.lower():
        # Load LLaVA model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        if 'lora' in model_name.lower() and model_base is not None:
            from llava.model.language_model.llava_llama import LlavaConfig
            lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print('Loading LLaVA from base model...')
            model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            print('Loading additional LLaVA weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            
            # 在 strip 之后，load_state_dict 之前加
            print("strip后第一个compressor key:", [k for k in non_lora_trainables if 'compressor' in k][0])
            print("strip后第一个projector key:", [k for k in non_lora_trainables if 'mm_projector' in k][0])
            ### XIAOYA: 验证 compressor 和 projector 加载 ###
            # 取 compressor 第一个 key 的前5个值作为参考
            comp_ref_key = 'model.compressor.compress_layers.0.attn.q_proj.weight'
            proj_ref_key = 'model.mm_projector.0.weight'

            comp_ref = non_lora_trainables[comp_ref_key].flatten()[:5]
            proj_ref = non_lora_trainables[proj_ref_key].flatten()[:5]
            print("bin compressor q_proj[:5]:", comp_ref)
            print("bin mm_projector[:5]:", proj_ref)
            ### XIAOYA ###

            model.load_state_dict(non_lora_trainables, strict=False)
            ### XIAOYA: 验证 compressor 和 projector 加载 ###
            comp_w = model.model.compressor.compress_layers[0].attn.q_proj.weight.detach().flatten()[:5]
            proj_w = model.model.mm_projector[0].weight.detach().flatten()[:5]
            print("model compressor q_proj[:5]:", comp_w)
            print("model mm_projector[:5]:", proj_w)

            print("compressor match:", torch.allclose(comp_w.float(), comp_ref.float(), atol=1e-3))
            print("mm_projector match:", torch.allclose(proj_w.float(), proj_ref.float(), atol=1e-3))
            ### XIAOYA ###

            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            ### XIAOYA: 验证 LoRA 加载/直接对比 adapter 文件和模型中的 lora_B ###
            from safetensors import safe_open
            lora_B_model = model.model.model.layers[0].self_attn.q_proj.lora_B.default.weight.detach()
            lora_A_model = model.model.model.layers[0].self_attn.q_proj.lora_A.default.weight.detach()
            q_base       = model.model.model.layers[0].self_attn.q_proj.base_layer.weight.detach()
            proj_before  = model.model.model.mm_projector[0].weight.detach().clone()

            with safe_open(os.path.join(model_path, 'adapter_model.safetensors'), framework="pt") as f:
                lora_B_file = f.get_tensor("base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight")

            print("model_path:", model_path)
            print("lora_B from file [:5]:", lora_B_file.flatten()[:5])
            print("lora_B in model  [:5]:", lora_B_model.flatten()[:5])
            print("lora_B match:", torch.allclose(lora_B_model.float(), lora_B_file.float(), atol=1e-4))
            print("lora_B max diff:", (lora_B_model.float() - lora_B_file.float()).abs().max().item())
            ### XIAOYA ###
            
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            ### XIAOYA: 验证 merge 结果 ###
            ### XIAOYA: 验证 merge 结果 ###
            q_after  = model.model.layers[0].self_attn.q_proj.weight.detach()
            proj_after = model.model.mm_projector[0].weight.detach()

            # 手动计算期望的 merge 结果：W_base + lora_B @ lora_A * (alpha/r)
            # scaling 默认是 lora_alpha/r，从 adapter_config.json 读
            import json
            with open(os.path.join(model_path, 'adapter_config.json')) as f:
                adapter_cfg = json.load(f)
            scaling = adapter_cfg['lora_alpha'] / adapter_cfg['r']
            q_expected = (q_base.float() + (lora_B_model.float() @ lora_A_model.float()) * scaling).to(torch.float16)

            print("q_proj after merge [:5]:",   q_after.flatten()[:5])
            print("q_proj expected    [:5]:",   q_expected.flatten()[:5])
            print("merge correct:", torch.allclose(q_after.float(), q_expected.float(), atol=1e-2))
            print("projector unchanged:", torch.allclose(proj_after.float(), proj_before.float(), atol=1e-4))
            ### XIAOYA ###

            print('Model is loaded...')

            ### XIAOYA: 验证 vicuna base weights 从 HF 加载 ###
            w = model.model.layers[0].self_attn.q_proj.weight.detach()
            print("q_proj mean:", w.mean().item())
            print("q_proj std:", w.std().item())
            print("q_proj first row[:5]:", w[0, :5])
            ### XIAOYA ###
            
        elif model_base is not None:
            # this may be mm projector only
            print('Loading LLaVA from base model...')
            if 'mpt' in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                    shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                model = LlavaMptForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = LlavaMptForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            elif 'mistral' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = LlavaMistralForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None

    if 'llava' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model(device_map=device_map)
        if device_map != 'auto':
            vision_tower.to(device=device_map, dtype=torch.float16)
        image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
