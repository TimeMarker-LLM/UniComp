import os
# from .llava_arch import LlavaMetaForCausalLM_holitom
# from .modeling_qwen2 import Qwen2Model_FastVid
from .llava_qwen import LlavaQwenForCausalLM_FastVid
from .llava_arch import LlavaMetaForCausalLM_FastVid

def fastvid(model):
    
    print("################################")
    print("############ FastVid ###########")
    print("################################")

    # from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
    # LlavaQwenForCausalLM.generate = LlavaQwenForCausalLM_FastVid.generate

    from llava.model.llava_arch import LlavaMetaForCausalLM
    LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal = LlavaMetaForCausalLM_FastVid.prepare_inputs_labels_for_multimodal
    LlavaMetaForCausalLM.encode_images = LlavaMetaForCausalLM_FastVid.encode_images
    LlavaMetaForCausalLM.get_attn_2dPool = LlavaMetaForCausalLM_FastVid.get_attn_2dPool


 
    # from transformers.models.qwen2.modeling_qwen2 import Qwen2Model
    # # Qwen2Model.__init__ = Qwen2Model_FastVid.__init__
    # Qwen2Model.forward = Qwen2Model_FastVid.forward
    # Qwen2Model.set_my_kwargs = Qwen2Model_FastVid.set_my_kwargs

    

    

    return model
