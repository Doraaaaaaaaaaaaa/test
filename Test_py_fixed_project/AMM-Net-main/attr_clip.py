import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

AADB_PROMPTS_11 = [
    "a photo with interesting content",
    "a photo with clear object emphasis",
    "a photo with good lighting",
    "a photo with good color harmony",
    "a photo with vivid color",
    "a photo with shallow depth of field",
    "a photo with motion blur",
    "a photo following rule of thirds",
    "a photo with balanced elements",
    "a photo with repetition patterns",
    "a photo with symmetry"
]

class RobustClipAttributeEncoder(nn.Module):
    def __init__(self, prompts=AADB_PROMPTS_11, out_dim=2048, clip_name="ViT-B/16", freeze_clip=True, temperature=0.07, device=None):
        super().__init__()
        self.temperature = temperature
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.clip_model, _ = clip.load(clip_name, device=self.device)
        self.clip_model.eval()
        if freeze_clip:
            for p in self.clip_model.parameters():
                p.requires_grad = False

        with torch.no_grad():
            tokens = clip.tokenize(prompts).to(self.device)
            t = self.clip_model.encode_text(tokens)
            t = F.normalize(t, dim=-1)
        self.register_buffer("prompt_emb", t)

        d_clip = t.shape[-1]
        self.proj_T = nn.Linear(d_clip, out_dim)
        self.proj_V = nn.Linear(d_clip, out_dim)
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, img_clip):
        img_clip = img_clip.to(self.prompt_emb.device)
        with torch.no_grad():
            v = self.clip_model.encode_image(img_clip)
            v = F.normalize(v, dim=-1)
            logits = (v @ self.prompt_emb.t()) / self.temperature
            W = torch.softmax(logits, dim=-1)

        T_proj = self.proj_T(self.prompt_emb)
        T_proj = T_proj.unsqueeze(0).expand(v.size(0), -1, -1)
        V_proj = self.proj_V(v)
        V_gated = V_proj.unsqueeze(1) * W.unsqueeze(-1)
        F_a = self.layer_norm(T_proj + V_gated)
        return F_a
