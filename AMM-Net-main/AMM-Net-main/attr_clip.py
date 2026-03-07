import torch
import torch.nn as nn
import torch.nn.functional as F

import clip  # 你项目里本地的 clip 模块（从 CG-IAA 复制过来的 ./clip 目录）

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
    "a photo with symmetry",
]

class RobustClipAttributeEncoder(nn.Module):
    """
    CLIP prompt-bank attribute encoder.
    Output: Fa (B, m, out_dim) where out_dim=2048 for AMM-Net's MIMN.
    Formula: a_i = LN( Proj_T(t_i) + w_i * Proj_V(v) )
    """
    def __init__(
        self,
        prompts=AADB_PROMPTS_11,
        out_dim=2048,
        clip_name="ViT-B/16",
        freeze_clip=True,
        temperature=0.07,
        device="cuda",
        download_root=None,
    ):
        super().__init__()
        self.temperature = temperature
        self.device = torch.device(device)

        # IMPORTANT: load CLIP directly onto the target device (GPU)
        self.clip_model, _ = clip.load(clip_name, device=self.device, download_root=download_root)
        self.clip_model.eval()
        if freeze_clip:
            for p in self.clip_model.parameters():
                p.requires_grad = False

        # Precompute prompt embeddings on the same device
        with torch.no_grad():
            tokens = clip.tokenize(prompts).to(self.device)         # (m,77)
            t = self.clip_model.encode_text(tokens)                 # (m,d_clip)
            t = F.normalize(t, dim=-1)
        self.register_buffer("prompt_emb", t)  # (m, d_clip), follows module.to(...)

        d_clip = t.shape[-1]
        self.proj_T = nn.Linear(d_clip, out_dim)
        self.proj_V = nn.Linear(d_clip, out_dim)
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, img_clip: torch.Tensor) -> torch.Tensor:
        """
        img_clip: (B,3,224,224) CLIP-normalized.
        returns: Fa (B,m,2048)
        """
        img_clip = img_clip.to(self.prompt_emb.device)

        with torch.no_grad():
            v = self.clip_model.encode_image(img_clip)              # (B,d_clip) in most CLIP impls
            v = F.normalize(v, dim=-1)

        # If your CLIP impl returns different dims, uncomment this assert to debug:
        # assert v.shape[-1] == self.prompt_emb.shape[-1], (v.shape, self.prompt_emb.shape)

        logits = (v @ self.prompt_emb.t()) / self.temperature       # (B,m)
        w = torch.softmax(logits, dim=-1)                           # (B,m)

        # Text prior projection
        T_proj = self.proj_T(self.prompt_emb)                       # (m,out_dim)
        T_proj = T_proj.unsqueeze(0).expand(v.size(0), -1, -1)      # (B,m,out_dim)

        # Visual global projection + gating
        V_proj = self.proj_V(v)                                     # (B,out_dim)
        V_gated = V_proj.unsqueeze(1) * w.unsqueeze(-1)             # (B,m,out_dim)

        Fa = self.layer_norm(T_proj + V_gated)                      # (B,m,out_dim)
        return Fa