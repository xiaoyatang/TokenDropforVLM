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


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.config=config  # ### FOR VISUAL TOKEN PRUNING BY XIAOYA

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        keep_ratios=None,        # ### FOR VISUAL TOKEN PRUNING BY XIAOYA
        current_epoch=None,      # ### FOR VISUAL TOKEN PRUNING BY XIAOYA
        skip_recon_loss=False,   # ### FOR VISUAL TOKEN PRUNING BY XIAOYA
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        # ### FOR VISUAL TOKEN PRUNING BY XIAOYA
        cfg = getattr(self.config, "compressor_cfg", None)
        if keep_ratios is None and cfg is not None:
            keep_ratios = cfg.get("compressive", {}).get("keep_ratio", None)
            
        if inputs_embeds is None:
            # print('DEBUG BY XIAOYA: enter inputs_embeds is None',inputs_embeds)
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes,
                keep_ratios=keep_ratios,           # ### FOR VISUAL TOKEN PRUNING BY XIAOYA
                current_epoch=current_epoch,       # ### FOR VISUAL TOKEN PRUNING BY XIAOYA
                skip_recon_loss=skip_recon_loss,   # ### FOR VISUAL TOKEN PRUNING BY XIAOYA
            )

        # return super().forward(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     position_ids=position_ids,
        #     past_key_values=past_key_values,
        #     inputs_embeds=inputs_embeds,
        #     labels=labels,
        #     use_cache=use_cache,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict
        # )
        ### FOR VISUAL TOKEN PRUNING BY XIAOYA
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        recon_loss = None
        compress_out = getattr(self, "_last_compress_out", None)

        if compress_out is not None:
            recon_losses = compress_out.get("recon_losses", [])
            if len(recon_losses) > 0:
                recon_loss = sum(recon_losses) / len(recon_losses) # use list(recon_loss).mean

        lambda_recon = 0.1
        if cfg is not None:
            lambda_recon = cfg.get("compressive", {}).get("lambda_recon", 0.1)

        if recon_loss is not None and outputs.loss is not None:  # add recon_loss with c.e. loss
            recon_loss = recon_loss.to(outputs.loss.dtype)       # recon_loss: bfloat16; c.e. loss: float32
            outputs.loss = outputs.loss + lambda_recon * recon_loss
            outputs.recon_loss = recon_loss                      # save recon_loss into the output

        return outputs
        ### FOR VISUAL TOKEN PRUNING BY XIAOYA
        

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        ### FOR VISUAL TOKEN PRUNING BY XIAOYA
        cfg = getattr(self.config, "compressor_cfg", None)
        keep_ratios = kwargs.pop("keep_ratios", None)
        if keep_ratios is None and cfg is not None:
            keep_ratios = cfg.get("compressive", {}).get("keep_ratio", None)
        ### FOR VISUAL TOKEN PRUNING BY XIAOYA

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes,
                keep_ratios=keep_ratios,  ### FOR VISUAL TOKEN PRUNING BY XIAOYA
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
