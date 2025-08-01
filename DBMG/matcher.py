import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DBMG(nn.Module):
    def __init__(self, args):
        super(DBMG, self).__init__()
        self.args = args
        self.fc_query = nn.Linear(512, 96)
        self.fc_key = nn.Linear(512, 96)
        self.fc_value = nn.Linear(512, 96)
        self.fc_cls = nn.Linear(512, 96)
        self.layer_norm = nn.LayerNorm(96)
        self.fc_caption_text = nn.Linear(512, 96)
        self.fc_image_patch = nn.Linear(96, 96)
        self.text_cls_head = nn.Linear(512, 20)
        self.image_cls_head = nn.Linear(512, 20)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self,
                image_description_embeds,
                image_description_seq_tokens,
                cap_text_embeds,
                cap_text_seq_tokens,
                image_embeds,
                image_patch_tokens,
                state=1):
        entity_cls_fc = self.fc_cls(image_description_embeds)
        text_cls_logits = self.text_cls_head(cap_text_embeds)
        image_cls_logits = self.image_cls_head(image_description_embeds)

        entity_cls_fc = entity_cls_fc.unsqueeze(dim=1)
        query = self.fc_query(image_description_seq_tokens).unsqueeze(dim=1)
        key = self.fc_key(cap_text_seq_tokens).unsqueeze(dim=0)
        value = self.fc_value(cap_text_seq_tokens).unsqueeze(dim=0)

        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(96)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        context = torch.matmul(attention_probs, value).mean(dim=-2)
        context = self.layer_norm(context)
        g2l_matching_score = torch.sum(entity_cls_fc * context, dim=-1).transpose(0, 1)

        g2g_matching_score = F.cosine_similarity(
            cap_text_embeds.unsqueeze(1),
            image_description_embeds.unsqueeze(0),
            dim=-1
        )

        matching_score = (g2l_matching_score + g2g_matching_score) / 2

        caption_text_embeds_proj = self.fc_caption_text(cap_text_embeds)
        global_similarity = F.cosine_similarity(
            caption_text_embeds_proj.unsqueeze(1),
            image_embeds.unsqueeze(0),
            dim=-1
        )

        caption_text_seq_tokens_proj = self.fc_caption_text(cap_text_seq_tokens)
        image_patch_tokens_proj = self.fc_image_patch(image_patch_tokens)

        # Restore full token-patch similarity computation
        cap_tokens = F.normalize(caption_text_seq_tokens_proj, dim=-1)
        img_tokens = F.normalize(image_patch_tokens_proj, dim=-1)

        # Compute pairwise similarity: [B, Lt, B, Li]
        l2l_score = torch.einsum('bth,bph->btbp', cap_tokens, img_tokens)
        l2l_t2i = l2l_score.mean(dim=1).mean(dim=2)  # Average over Lt and Li

        fuse_score = matching_score + (l2l_t2i + global_similarity) / 2

        return fuse_score# , text_cls_logits, image_cls_logits
