import torch
import torch.nn as nn
from transformers import CLIPModel


class DBMGEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.clip = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            torch_dtype=torch.float32
        )
        # 图像特征投影层
        self.image_cls_fc = nn.Linear(512, 96)
        self.image_tokens_fc = nn.Linear(768, 96)
        self.to(args.device)

        print(f"Encoder initialized. Device: {args.device}")
        print("image_cls_fc weight sample:", self.image_cls_fc.weight[0, :5])

    def forward(
            self,
            caption_input_ids=None,
            caption_attention_mask=None,
            pixel_values=None,
            image_description_input_ids=None,
            image_description_attention_mask=None
    ):
        # 编码caption和图像
        clip_output = self.clip(
            input_ids=caption_input_ids,
            attention_mask=caption_attention_mask,
            pixel_values=pixel_values
        )
        text_embeds = clip_output.text_embeds
        image_embeds = self.image_cls_fc(clip_output.image_embeds)
        text_seq_tokens = clip_output.text_model_output[0]
        image_patch_tokens = self.image_tokens_fc(clip_output.vision_model_output[0])

        # 编码图像描述文本
        clip_desc_output = self.clip(
            input_ids=image_description_input_ids,
            attention_mask=image_description_attention_mask,
            pixel_values=pixel_values
        )
        image_description_embeds = clip_desc_output.text_embeds
        image_description_seq_tokens = clip_desc_output.text_model_output[0]  #

        return {
            "text_embeds": text_embeds,
            "image_embeds": image_embeds,
            "text_seq_tokens": text_seq_tokens,
            "image_patch_tokens": image_patch_tokens,
            "image_description_embeds": image_description_embeds,
            "image_description_seq_tokens": image_description_seq_tokens
        }