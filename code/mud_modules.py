import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import numpy as np

class Variational_IB(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, input_dim)
        self.fc_logvar = nn.Linear(hidden_dim, input_dim)
        nn.init.constant_(self.fc_logvar.weight, 0)
        nn.init.constant_(self.fc_logvar.bias, 0)
        
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, x):
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        z = self.reparameterize(mu, logvar)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        kl_loss = kl_loss.mean()
        out = x + z 
        return out, kl_loss
    
class MLP_Block(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.block(x)

class Interaction_Estimator(nn.Module):
    def __init__(self, dim=256, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn_a2v = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, 
                                              dropout=dropout, batch_first=True)
        self.attn_v2a = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, 
                                              dropout=dropout, batch_first=True)
        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, feat_a, feat_v):
        a_out, _ = self.attn_a2v(query=feat_a, key=feat_v, value=feat_v)
        v_out, _ = self.attn_v2a(query=feat_v, key=feat_a, value=feat_a)
        combined = torch.cat([a_out, v_out], dim=-1)
        projected = self.fusion(combined)
        residual = (feat_a + feat_v) / 2
        return self.norm(projected + residual)

class SMED(nn.Module):  # Structured Multimodal Evidence Decomposition
    def __init__(self, dim=256, dropout=0.1, num_heads=4): 
        super().__init__()
        self.audio_spec_mlp = MLP_Block(dim, dim, dropout)
        self.audio_spec_ib = Variational_IB(input_dim=dim, hidden_dim=dim, dropout=dropout)
    
        self.visual_spec_mlp = MLP_Block(dim, dim, dropout)
        self.visual_spec_ib = Variational_IB(input_dim=dim, hidden_dim=dim, dropout=dropout)
        
        self.common_encoder = Interaction_Estimator(dim, num_heads=num_heads, dropout=dropout)
        self.synergy_encoder = Interaction_Estimator(dim, num_heads=num_heads, dropout=dropout)
        
        self.reconstruct = nn.Linear(dim, dim)

    def forward(self, feat_a, feat_v):
        spec_a_pre = self.audio_spec_mlp(feat_a)
        spec_a, kl_a = self.audio_spec_ib(spec_a_pre)
        
        spec_v_pre = self.visual_spec_mlp(feat_v)
        spec_v, kl_v = self.visual_spec_ib(spec_v_pre)
        
        smed_kl_loss = kl_a + kl_v
    
        common = self.common_encoder(feat_a, feat_v)
        synergy = self.synergy_encoder(feat_a, feat_v)

       
        role_loss, role_stats = self.role_separation_loss(
            feat_a, feat_v, common, synergy
        )
        
        refined_a = self.reconstruct(spec_a + common + synergy)
        refined_v = self.reconstruct(spec_v + common + synergy)
        
        out_a = feat_a + refined_a
        out_v = feat_v + refined_v
        
        return out_a, out_v, synergy, role_loss, role_stats, smed_kl_loss

    @staticmethod
    def role_separation_loss(feat_a, feat_v, common, synergy):
        
        def flat_norm(x):
            return F.normalize(x.reshape(-1, x.shape[-1]), dim=-1)

        anchor_a = flat_norm(feat_a.detach())
        anchor_v = flat_norm(feat_v.detach())
        consensus = F.normalize(anchor_a + anchor_v, dim=-1)
        common_norm = flat_norm(common)
        synergy_norm = flat_norm(synergy)

        common_similarity = (common_norm * consensus).sum(dim=-1)
        common_loss = (1.0 - common_similarity).mean()

        common_anchor = common_norm.detach()
        common_synergy_similarity = (
            common_anchor * synergy_norm
        ).sum(dim=-1)
        
        separation_margin = 0.30
        separation_loss = F.relu(
            common_synergy_similarity.abs() - separation_margin
        ).square().mean()

        role_loss = common_loss + 0.5 * separation_loss
        role_stats = {
            'common_alignment': common_similarity.detach().mean(),
            'common_synergy_abs_cos': common_synergy_similarity.detach().abs().mean(),
            'common_synergy_margin_penalty': separation_loss.detach(),
        }
        return role_loss, role_stats

    

class TAUG(nn.Module):  # Text-Anchored Uncertainty Gate
    def __init__(self, dim=256, dropout=0.1):
        super().__init__()
        
        clip_model, _ = clip.load("ViT-B/32", device="cpu") 
        self.clip_model = clip_model.float()
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        self.categories = [
            'Speech', 'Car', 'Cheering', 'Dog', 'Cat', 'Frying food',
            'Basketball bounce', 'Fire alarm', 'Chainsaw', 'Cello', 'Banjo',
            'Singing', 'Chicken rooster', 'Violin fiddle', 'Vacuum cleaner',
            'Baby laughter', 'Accordion', 'Lawn mower', 'Motorcycle', 'Helicopter',
            'Acoustic guitar', 'Telephone bell ringing', 'Baby cry infant cry', 
            'Blender', 'Clapping'
        ]
        
        self.templates = [
            "a video of {}",
            "a sound of {}",
            "the audio of {}",
            "noise of {}",
            "listening to {}",
            "footage of {}"
        ]
        
        with torch.no_grad():
            all_text_feats = []
            for category in self.categories:
                texts = [temp.format(category) for temp in self.templates]
                tokenized = clip.tokenize(texts)
                class_feats = self.clip_model.encode_text(tokenized).float() 
                class_feats = F.normalize(class_feats, dim=-1)
                mean_feat = class_feats.mean(dim=0)
                mean_feat = F.normalize(mean_feat, dim=-1) 
                all_text_feats.append(mean_feat)
            
            text_bank = torch.stack(all_text_feats, dim=0) 
            self.register_buffer('text_features_bank', text_bank)

        self.text_proj = nn.Sequential(
            nn.Linear(512, dim),
            nn.LayerNorm(dim),
            nn.ReLU()
        )
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.uncertainty_head = nn.Sequential(
            nn.Linear(dim * 2, dim // 2), 
            nn.LayerNorm(dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid() 
        )

        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=dim, 
            num_heads=4, 
            dropout=dropout, 
            batch_first=True
        )
        self.temporal_norm = nn.LayerNorm(dim)

    def forward(self, feat_a, feat_v):
        B, T, D = feat_a.shape
        
        text_feat_proj = self.text_proj(self.text_features_bank)
    
        feat_fusion = (feat_a + feat_v) / 2
        feat_flat = feat_fusion.reshape(-1, D)
        
        logit_scale = self.logit_scale.exp()
        sim = logit_scale * torch.mm(F.normalize(feat_flat, dim=-1), 
                                     F.normalize(text_feat_proj, dim=-1).t())
        
        attn_weights = torch.softmax(sim, dim=-1) 
        
        entropy = -(attn_weights * torch.log(attn_weights + 1e-7)).sum(dim=-1) 
        entropy = entropy / np.log(len(self.categories)) 
        entropy_uncertainty = entropy.reshape(B, T, 1)

        consistency_sim = torch.cosine_similarity(feat_a, feat_v, dim=-1)  
        consistency_sim = consistency_sim.unsqueeze(-1)
        norm_a = torch.norm(feat_a, dim=-1, keepdim=True)
        norm_v = torch.norm(feat_v, dim=-1, keepdim=True)
        norm_diff = torch.abs(norm_a - norm_v) / (norm_a + norm_v + 1e-7)  
        consistency_uncertainty = (1 - consistency_sim) * 0.7 + norm_diff * 0.3  
    
        max_sim, _ = attn_weights.max(dim=-1)  
        max_sim = max_sim.reshape(B, T, 1)
        prediction_uncertainty = 1 - max_sim  
    
        final_uncertainty = (
        entropy_uncertainty * 0.4 + 
        consistency_uncertainty * 0.3 + 
        prediction_uncertainty * 0.3
    )
    
        weighted_text_flat = torch.mm(attn_weights, text_feat_proj)
        weighted_text = weighted_text_flat.reshape(B, T, D)
    
        text_refined, _ = self.temporal_attn(
            query=weighted_text,
            key=weighted_text,
            value=weighted_text
        )
        text_refined = self.temporal_norm(weighted_text + text_refined)  
        
        return text_refined, final_uncertainty

class MBSA(nn.Module):  # Memory-Based Synergistic Alignment
    def __init__(self, feature_dim=256, bank_size=4096, num_classes=25, temp=0.2):
        super().__init__()
        self.bank_size = bank_size
        self.temp = temp
        self.num_classes = num_classes
        
        self.register_buffer("feature_bank", torch.randn(bank_size, feature_dim))
        self.register_buffer("label_bank", torch.zeros(bank_size, num_classes))
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))
        
        self.feature_bank = F.normalize(self.feature_bank, dim=1)

    def update_bank(self, features, labels):
        with torch.no_grad():
            batch_len = features.shape[0]
            ptr = int(self.ptr)
            
            if ptr + batch_len >= self.bank_size:
                remain = self.bank_size - ptr
                self.feature_bank[ptr:] = F.normalize(features[:remain], dim=1)
                self.label_bank[ptr:] = labels[:remain]
                self.ptr[0] = 0
            else:
                self.feature_bank[ptr: ptr + batch_len] = F.normalize(features, dim=1)
                self.label_bank[ptr: ptr + batch_len] = labels
                self.ptr[0] = (ptr + batch_len) % self.bank_size

    def forward(self, features, labels, probs=None): 
        
        B, T, D = features.shape
        features_flat = features.reshape(-1, D)
        labels_flat = labels.reshape(-1, self.num_classes)
        if probs is not None:
            probs_flat = probs.view(-1, self.num_classes)
            max_probs, _ = probs_flat.max(dim=1) 
            conf_mask = max_probs > 0.5 
            if conf_mask.sum() == 0:
                return torch.tensor(0.0).to(features.device)
                
            features_flat = features_flat[conf_mask]
            labels_flat = labels_flat[conf_mask]
        
        active_mask = labels_flat.sum(dim=1) > 0
        if active_mask.sum() == 0:
            return torch.tensor(0.0).to(features.device)
            
        feat_active = features_flat[active_mask]
        label_active = labels_flat[active_mask]
        
        feat_norm = F.normalize(feat_active, dim=1)
        
        bank_norm = self.feature_bank.detach().clone()
        
        logits = torch.mm(feat_norm, bank_norm.t()) / self.temp 
        sim_labels = torch.mm(label_active, self.label_bank.t()) 
        positive_mask = (sim_labels > 0).float()
        
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-7)
        
        mask_sum = positive_mask.sum(dim=1)
        valid_rows = mask_sum > 0
        
        if valid_rows.sum() == 0:
            loss = torch.tensor(0.0).to(features.device)
        else:
            log_prob = log_prob[valid_rows]
            positive_mask = positive_mask[valid_rows]
            mask_sum = mask_sum[valid_rows]
            
            mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / mask_sum
            loss = - mean_log_prob_pos.mean()
        
        self.update_bank(feat_active, label_active)
        
        return loss













