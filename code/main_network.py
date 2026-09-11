import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mud_modules import SMED, MBSA, TAUG

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Encoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, hidden_dim):
        super(Encoder, self).__init__()

        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.final_norm_a = nn.LayerNorm(hidden_dim)
        self.final_norm_v = nn.LayerNorm(hidden_dim)

    def forward(self, src_a, src_v):
        output_a = src_a
        output_v = src_v
        for i in range(self.num_layers):
            output_a, map_aa, map_av = self.layers[i](src_a, src_v)
            output_v, map_vv, map_va = self.layers[i](src_v, src_a)
            src_a = output_a
            src_v = output_v

        return output_a, output_v, [map_aa, map_av, map_vv, map_va]


class SECM(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.1):
        super(SECM, self).__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

    def forward(self, x, y=None):
        B, T, K, D = x.shape
        x = x.contiguous().view(B * T, K, D).permute(1, 0, 2)
        if y is not None:
            y = y.contiguous().view(B * T, K, D).permute(1, 0, 2)
            out, map1 = self.attn(x, y, y, attn_mask=None,
                                  key_padding_mask=None)
        else:
            out, map1 = self.attn(x, x, x, attn_mask=None,
                                  key_padding_mask=None)
        out = out.permute(1, 0, 2)

        return out, map1.contiguous().view(B, T, K, K)


class LGSF(nn.Module):

    def __init__(self, d_model, is_cross=False):
        super(LGSF, self).__init__()
        self.relu = nn.ReLU()
        if is_cross:
            self.glob = nn.Linear(d_model, d_model)
        else:
            self.glob1 = nn.Linear(d_model, d_model)
            self.glob2 = nn.Linear(d_model, d_model)

    def forward(self, x, y=None):
        B, T, K, D = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(B * K, T, D)
        if y is not None:
            y = y.permute(0, 2, 1, 3).contiguous().view(B * K, T, D)
            x_y = self.relu(self.glob(x))
            glob = self.relu(self.glob(y)).mean(-2).unsqueeze(-2)

            relevance = torch.cosine_similarity(x_y, glob, dim=-1)
            relevance = torch.sigmoid(relevance.unsqueeze(-1))
            out = relevance * glob
        else:
            x_1 = self.relu(self.glob1(x))
            glob = self.relu(self.glob2(x)).mean(-2).unsqueeze(-2)
            relevance = torch.cosine_similarity(x_1, glob, dim=-1)
            relevance = torch.sigmoid(relevance.unsqueeze(-1))
            out = relevance * glob

        out = out.contiguous().view(B, K, T, D).permute(0, 2, 1, 3)
        out = out.contiguous().view(B * T, K, D)

        return out


class FGSE(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super(FGSE, self).__init__()

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)

        self.SECM_1 = SECM(d_model, nhead, dropout=0.1)
        self.LGSF_1 = LGSF(d_model, is_cross=False)
        self.SECM_2 = SECM(d_model, nhead, dropout=0.1)
        self.LGSF_2 = LGSF(d_model, is_cross=True)
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, d_model)

    def forward(self, src_q, src_v):
        
        B, T, K, D = src_q.shape
        src_q0 = src_q.contiguous().view(B * T, K, D)

        '''intra'''
        src_1_scd, map_aa = self.SECM_1(src_q)
        src_1_scd = src_q0 + src_1_scd  # B * T, K, D
        src_1_scd = src_1_scd.contiguous().view(B, T, K, D)
        src_1_igf = src_q0 + self.LGSF_1(src_1_scd)
        src1 = self.fc1(src_1_igf)

        '''cross'''
        src_2_scd, map_av = self.SECM_2(src_q, src_v)
        src_2_scd = src_q0 + src_2_scd  # B * T, K, D
        src_2_scd = src_2_scd.contiguous().view(B, T, K, D)
        src_2_igf = src_q0 + self.LGSF_2(src_2_scd, src_v)
        src2 = self.fc2(src_2_igf)
        src_q0 = src1 + src2
        src_q0 = self.norm1(src_q0)

        src_q0 = self.linear2(self.dropout(F.relu(self.linear1(src_q0))))

        return src_q0.contiguous().view(B, T, K, D), map_aa, map_av


class Recon(nn.Module):

    def __init__(self, hidden_dim, input_dim):
        super(Recon, self).__init__()
        self.model = nn.Sequential(
            nn.ReLU(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        try:
            x = self.model(x)
        except:
            assert 1 == 2, x.shape
        return x


class Alpha_net(nn.Module):

    def __init__(self, hidden_dim):
        super(Alpha_net, self).__init__()
        self.model = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        try:
            x = self.model(x)
        except:
            assert 1 == 2, x.shape
        return x


class last_Pred(nn.Module):

    def __init__(self):
        super(last_Pred, self).__init__()
        self.fc = nn.ModuleList(
            [nn.Linear(128, 1) for i in range(25)])

    def forward(self, x):
        out = [self.fc[i](x[:, :, :, i, :]) for i in range(25)]
        out = torch.stack(out, dim=-2).squeeze(-1)  # (B, T, 2, k)
        return out


class MUD_Net(nn.Module):  # MUD-Net

    def __init__(self, args, mode='train'):
        super(MUD_Net, self).__init__()

        self.fc_a = nn.Linear(args.input_a_dim, args.hidden_dim)
        
        self.fc_v = nn.Linear(args.input_v_dim, args.hidden_dim)
        self.fc_st = nn.Linear(512, args.hidden_dim)
        self.fc_fusion = nn.Linear(args.hidden_dim * 2, args.hidden_dim)

        self.smed = SMED(dim=args.hidden_dim, dropout=0.1,num_heads=4)
        
        if mode == 'train':
            self.mbsa = MBSA(
                feature_dim=args.hidden_dim, 
                bank_size=4096, 
                num_classes=25
            )
        
        self.taug = TAUG(dim=args.hidden_dim, dropout=0.1)

        self.v_unmix = nn.ModuleList(
            [nn.Sequential(nn.ReLU(), 
                           nn.Linear(args.hidden_dim * 2, 128), 
                           nn.ReLU(),
                           nn.Dropout(0.1), 
                           nn.Linear(128, 128), 
                           nn.ReLU()) for i in range(26)])
        
        self.a_unmix = nn.ModuleList(
            [nn.Sequential(nn.ReLU(), 
                           nn.Linear(args.hidden_dim * 2, 128), 
                           nn.ReLU()) for i in range(26)])
        
        if mode == 'train':
            self.v_recon = Recon(hidden_dim=args.hidden_dim, input_dim=128 * 26)
            self.a_recon = nn.Sequential(
                nn.Linear(128 * 26, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU()
            )
        
        # FGSE 
        self.av_enhance = Encoder(FGSE(d_model=128, nhead=args.nhead, dim_feedforward=512),
                                  num_layers=args.num_layers,
                                  hidden_dim=128)

        
        self.fc_prob = last_Pred()
        self.fc_frame_att = last_Pred()
        self.fc_av_att = last_Pred()

        
        self.reduce_a = nn.Sequential(nn.Linear(128 * 2, 128), nn.ReLU())
        self.reduce_v = nn.Sequential(nn.Linear(128 * 2, 128), nn.ReLU())

        self.alpha_a = Alpha_net(args.hidden_dim)
        self.alpha_v = Alpha_net(args.hidden_dim)

        self.norm_where = args.norm_where
        self.input_v_dim = args.input_v_dim
        self.input_a_dim = args.input_a_dim
        self.hidden_dim = args.hidden_dim

    
    def forward(self, audio, visual, visual_st, audio_pseudo_labels=None, visual_pseudo_labels=None, mode='train'):

        if audio.size(1) == 64: 
            x1 = audio.permute(0, 2, 1).contiguous().view(-1, self.input_a_dim, 2, 32)
            upsampled = F.interpolate(x1, size=(2, 1024), mode='bicubic')
            upsampled = self.fc_a(upsampled.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).mean(dim=2)
            x1 = F.adaptive_avg_pool1d(upsampled, 10).view(-1, self.hidden_dim, 10)
            x1 = x1.permute(0, 2, 1)
        else:
            x1 = self.fc_a(audio) 

        if visual.size(1) == 80:
            vid_s = self.fc_v(visual).permute(0, 2, 1).unsqueeze(-1)
            vid_s = F.avg_pool2d(vid_s, (8, 1)).squeeze(-1).permute(0, 2, 1)
        else:
            vid_s = self.fc_v(visual)

        vid_st = self.fc_st(visual_st)
        x2 = torch.cat((vid_s, vid_st), dim=-1)
        x2 = self.fc_fusion(x2) 

        
        x1_refined, x2_refined, synergy_feat, loss_smed_role, \
            smed_role_stats, smed_kl_loss = self.smed(x1, x2)
        
        text_guidance, uncertainty = self.taug(x1_refined, x2_refined)
        
        x1_text = torch.cat([x1_refined, text_guidance], dim=-1)
        x2_text = torch.cat([x2_refined, text_guidance], dim=-1)
        
        
        features1 = [self.a_unmix[i](x1_text) for i in range(26)] 
        x1_2 = torch.stack(features1, dim=0).permute(1, 2, 0, 3)  
        B, T, K, D = x1_2.shape
        features2 = [self.v_unmix[i](x2_text) for i in range(26)] 
        x2_2 = torch.stack(features2, dim=0).permute(1, 2, 0, 3)  

        apha_x1_raw = torch.sigmoid(self.alpha_a(x1_refined).unsqueeze(-1))
        apha_x2_raw = torch.sigmoid(self.alpha_v(x2_refined).unsqueeze(-1))
        
        unc_mask = uncertainty.clone()
        unc_mask = torch.clamp(unc_mask, 0.0, 0.8)  

        unc_weight = torch.sigmoid(-10 * (unc_mask - 0.5))  
        unc_broadcast = unc_weight.unsqueeze(-1)
      
        apha_x1 = apha_x1_raw * unc_broadcast
        apha_x2 = apha_x2_raw * unc_broadcast

        
        x1_2_bg = x1_2[:, :, -1, :].unsqueeze(-2).repeat(1, 1, 25, 1)
        x2_2_bg = x2_2[:, :, -1, :].unsqueeze(-2).repeat(1, 1, 25, 1)
        x1_2_e = x1_2[:, :, :-1, :]
        x2_2_e = x2_2[:, :, :-1, :]

        x1_new = torch.cat([apha_x1 * x1_2_e, (1 - apha_x1) * x1_2_bg], dim=-1)
        x2_new = torch.cat([apha_x2 * x2_2_e, (1 - apha_x2) * x2_2_bg], dim=-1)

        x1_3 = self.reduce_a(x1_new)
        x2_3 = self.reduce_v(x2_new)
        loss_a_rec, loss_v_rec, loss_a_ort, loss_v_ort = [torch.tensor(0.0).to(x1.device) for _ in range(4)]

        if mode == 'train':
            loss_a_ort = torch.cosine_similarity(x1_2_e, x1_2_bg, dim=-1).mean()
            loss_v_ort = torch.cosine_similarity(x2_2_e.mean(-2), x2_2[:, :, -1, :], dim=-1).mean()
            x1_ = x1_2.contiguous().view(x1_2.size(0), x1_2.size(1), -1)
            x2_ = x2_2.contiguous().view(x2_2.size(0), x2_2.size(1), -1)
            x1_rec = self.a_recon(x1_)
            x2_rec = self.v_recon(x2_)
            
            loss_a_rec = nn.MSELoss()(x1_refined, x1_rec)
            loss_v_rec = nn.MSELoss()(x2_refined, x2_rec)

       
        x1_3, x2_3, [map_aa, map_av, map_vv, map_va] = self.av_enhance(x1_3, x2_3)

        
        x = torch.cat([x1_3.unsqueeze(2), x2_3.unsqueeze(2)], dim=2)
        frame_logits = self.fc_prob(x)
        frame_prob = torch.sigmoid(frame_logits)

        frame_att = torch.softmax(self.fc_frame_att(x), dim=1)
        av_att = torch.softmax(self.fc_av_att(x), dim=2)
        temporal_prob = (frame_att * frame_prob)
        global_prob = (temporal_prob * av_att).sum(dim=2).sum(dim=1)

        a_prob = temporal_prob[:, :, 0, :].sum(dim=1)
        v_prob = temporal_prob[:, :, 1, :].sum(dim=1)
          
        
        loss_mbsa = torch.tensor(0.0).to(x1.device)
        
       
        if mode == 'train' and hasattr(self, 'mbsa') and audio_pseudo_labels is not None:
            av_pseudo_labels = torch.clamp(audio_pseudo_labels + visual_pseudo_labels, 0, 1)
            
            av_probs, _ = frame_prob.max(dim=2) 
            
            loss_mbsa = self.mbsa(synergy_feat, av_pseudo_labels, probs=av_probs)
        
        if mode == 'train':
            return global_prob, a_prob, v_prob, frame_prob, frame_logits, \
                   [loss_a_rec, loss_v_rec, loss_a_ort, loss_v_ort, 
                     loss_mbsa, loss_smed_role, smed_role_stats,
                     uncertainty.mean(), smed_kl_loss], \
                   [map_aa, map_av, map_vv, map_va]
        else:
            return global_prob, a_prob, v_prob, frame_prob, frame_logits












