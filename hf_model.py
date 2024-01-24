# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava/modeling_llava.py
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

# model config dataclass
class LlavaConfig(PretrainedConfig):

  model_type = "llava"
  is_composition = False

  def __init__(
    self,
    vision_config=None,
    text_config=None,
    ignore_index=-100,
    image_token_index=32000,
    projector_hidden_act="gelu",
    vision_feature_select_strategy="default",
    vision_feature_layer=-2,
    vocab_size=32000,
    **kwargs,
  ):
    self.ignore_index = ignore_index
    self.image_token_index = image_token_index
    self.projector_hidden_act = projector_hidden_act
    self.vision_feature_select_strategy = vision_feature_select_strategy
    self.vision_feature_layer = vision_feature_layer
    self.vocab_size = vocab_size

    self.vision_config = CONFIG_MAPPING["clip_vision_model"](
        intermediate_size=4096,
        hidden_size=1024,
        patch_size=14,
        image_size=336,
        num_hidden_layers=24,
        num_attention_heads=16,
        vocab_size=32000,
        projection_dim=768,
      )
    self.vocab_size = self.vocab_size

    self.text_config = text_config

    if isinstance(self.text_config, dict):
      text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "llama"
      self.text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
      self.vocab_size = self.text_config.vocab_size
    elif text_config is None:
      self.text_config = CONFIG_MAPPING["llama"]()

    super().__init__(**kwargs)

# output dataclass
class LlavaCausalLMOutputWithPast(ModelOutput):
  loss: Optional[torch.FloatTensor] = None
  logits: torch.FloatTensor = None
  past_key_values: Optional[List[torch.FloatTensor]] = None
  hidden_states: Optional[Tuple[torch.FloatTensor]] = None
  attentions: Optional[Tuple[torch.FloatTensor]] = None
  image_hidden_states: Optional[Tuple[torch.FloatTensor]] = None

# input preprocessing
class LlavaProcessor(ProcessorMixin):

  attributes = ["image_processor", "tokenizer"]
  image_processor_class = "CLIPImageProcessor"
  tokenizer_class = ("LlamaTokenizer", "LlamaTokenizerFast")

  def __init__(self, image_processor=None, tokenizer=None):
    super().__init__(image_processor, tokenizer)

  def __call__(
    self,
    text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
    images: ImageInput = None,
    padding: Union[bool, str, PaddingStrategy] = False,
    truncation: Union[bool, str, TruncationStrategy] = None,
    max_length=None,
    return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
  ) -> BatchFeature:

    if images is not None:
      pixel_values = self.image_processor(images, return_tensors=return_tensors)["pixel_values"]
    else:
      pixel_values = None
    text_inputs = self.tokenizer(
      text, return_tensors=return_tensors, padding=padding, truncation=truncation, max_length=max_length
    )

    return BatchFeature(data={**text_inputs, "pixel_values": pixel_values})

  def batch_decode(self, *args, **kwargs):
    return self.tokenizer.batch_decode(*args, **kwargs)

  def decode(self, *args, **kwargs):
    return self.tokenizer.decode(*args, **kwargs)

  def model_input_names(self):
    tokenizer_input_names = self.tokenizer.model_input_names
    image_processor_input_names = self.image_processor.model_input_names
    return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


# project image features to text features
class LlavaMultiModalProjector(nn.Module):
  def __init__(self, config: LlavaConfig):
    super().__init__()

    self.linear_1 = nn.Linear(config.vision_config.hidden_size, config.text_config.hidden_size, bias=True)
    self.act = ACT2FN[config.projector_hidden_act]
    self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=True)

  def forward(self, image_features):
    hidden_states = self.linear_1(image_features)
    hidden_states = self.act(hidden_states)
    hidden_states = self.linear_2(hidden_states)
    return hidden_states


# base model outputs raw h_state
class LlavaPreTrainedModel(PreTrainedModel): 
  config_class = LlavaConfig
  base_model_prefix = "model"
  supports_gradient_checkpointing = True
  _no_split_modules = ["LlavaVisionAttention"]
  _skip_keys_device_placement = "past_key_values"
  _supports_flash_attn_2 = True

  def _init_weights(self, module):
    std = (
      self.config.initializer_range
      if hasattr(self.config, "initializer_range")
      else self.config.text_config.initializer_range
    )

    if hasattr(module, "class_embedding"):
      module.class_embedding.data.normal_(mean=0.0, std=std)

    if isinstance(module, (nn.Linear, nn.Conv2d)):
      module.weight.data.normal_(mean=0.0, std=std)
      if module.bias is not None:
        module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
      module.weight.data.normal_(mean=0.0, std=std)
      if module.padding_idx is not None:
        module.weight.data[module.padding_idx].zero_()


class LlavaForConditionalGeneration(LlavaPreTrainedModel):
  def __init__(self, config: LlavaConfig):
    super().__init__(config)
    self.vision_tower = AutoModel.from_config(config.vision_config)

    self.multi_modal_projector = LlavaMultiModalProjector(config)
    self.vocab_size = config.vocab_size
    self.language_model = AutoModelForCausalLM.from_config(
      config.text_config, attn_implementation=config._attn_implementation
    )
    self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
    self.post_init()
  
  def _merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids, attention_mask, labels):
    
    num_images, num_image_patches, embed_dim = image_features.shape
    batch_size, sequence_length = input_ids.shape
    left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(self.pad_token_id))
    
    # 1. Create a mask to know where special image tokens are
    special_image_token_mask = input_ids == self.config.image_token_index
    num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
    max_embed_dim = (num_special_image_tokens.max() * (num_image_patches - 1)) + sequence_length
    batch_indices, non_image_indices = torch.where(input_ids != self.config.image_token_index)

    # 2. Calculate new positions for text tokens in merged image-text sequence.
    # special_image_token_mask - identifies image tokens. Each image token will be replaced by `nb_text_tokens_per_images - 1` text tokens.
    # torch.cumsum - computes how each image token shifts subsequent text token positions. - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
    new_token_positions = torch.cumsum((special_image_token_mask * (num_image_patches - 1) + 1), -1) - 1
    nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
    if left_padding:
      new_token_positions += nb_image_pad[:, None]  # offset for left padding
    text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

    # 3. Create the full embedding, already padded to the maximum position
    final_embedding = torch.zeros(batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
    final_attention_mask = torch.zeros(batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device)
    if labels is not None:
      final_labels = torch.full(
        (batch_size, max_embed_dim), self.config.ignore_index, dtype=input_ids.dtype, device=input_ids.device
      )
    # If Vision/Language model offload to CPU, then set tensors to correct device
    target_device = inputs_embeds.device
    batch_indices, non_image_indices, text_to_overwrite = (
      batch_indices.to(target_device),
      non_image_indices.to(target_device),
      text_to_overwrite.to(target_device),
    )
    attention_mask = attention_mask.to(target_device)

    # 4. Fill embeddings based on mask. ex: ["hey" "<image>", "how", "are"] need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features
    final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
    final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]
    if labels is not None:
      final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_image_indices]

    # 5. Fill the embeddings corresponding to the images. Anything that is still zeros needs filling
    image_to_overwrite = torch.all(final_embedding == 0, dim=-1)
    image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None].to(target_device)

    if image_to_overwrite.sum() != image_features.shape[:-1].numel():
      raise ValueError(
        f"The input provided to the model are wrong. The number of image tokens is {torch.sum(special_image_token_mask)} while"
        f" the number of image given to the model is {num_images}. This prevents correct indexing and breaks batch generation."
      )

    final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim).to(target_device)
    final_attention_mask |= image_to_overwrite
    position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

    if labels is None:
      final_labels = None

    return final_embedding, final_attention_mask, final_labels, position_ids

  def forward(
    self,
    input_ids: torch.LongTensor = None,
    pixel_values: torch.FloatTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    vision_feature_layer: Optional[int] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
  ) -> Union[Tuple, LlavaCausalLMOutputWithPast]:

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
      output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )

    inputs_embeds = self.get_input_embeddings()(input_ids)
    image_outputs = self.vision_tower(pixel_values, output_hidden_states=True)
    
    selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
    selected_image_feature = selected_image_feature[:, 1:]
    image_features = self.multi_modal_projector(selected_image_feature)

    inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_image_features(
      image_features, inputs_embeds, input_ids, attention_mask, labels
    )
    
    if labels is None:labels = torch.full_like(attention_mask, self.config.ignore_index).to(torch.long)

    outputs = self.language_model(
      attention_mask=attention_mask,
      position_ids=position_ids,
      past_key_values=past_key_values,
      inputs_embeds=inputs_embeds,
      use_cache=use_cache,
      output_attentions=output_attentions,
      output_hidden_states=output_hidden_states,
      return_dict=return_dict,
    )

    logits = outputs[0]

    loss = None
    if labels is not None:
      # Shift so that tokens < n predict n
      if attention_mask is not None:
        shift_attention_mask = attention_mask[..., 1:]
        shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
        shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
      else:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
      # Flatten the tokens
      loss_fct = nn.CrossEntropyLoss()
      loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
      )

    if not return_dict:
      output = (logits,) + outputs[1:]
      return (loss,) + output if loss is not None else output

    return LlavaCausalLMOutputWithPast(
      loss=loss,
      logits=logits,
      past_key_values=outputs.past_key_values,
      hidden_states=outputs.hidden_states,
      attentions=outputs.attentions,
    )

