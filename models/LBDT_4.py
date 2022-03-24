import torch.nn as nn
import torch.nn.functional as F
from .position_encoding import *
from typing import Optional
from torch import Tensor
import copy


def aligned_bilinear(tensor, factor):
    assert tensor.dim() == 4
    assert factor >= 1
    assert int(factor) == factor

    if factor == 1:
        return tensor

    h, w = tensor.size()[2:]
    tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode="replicate")
    oh = factor * h + 1
    ow = factor * w + 1
    tensor = F.interpolate(
        tensor, size=(oh, ow),
        mode='bilinear',
        align_corners=True
    )
    tensor = F.pad(
        tensor, pad=(factor // 2, 0, factor // 2, 0),
        mode="replicate"
    )

    return tensor[:, :, :oh - 1, :ow - 1]


class Decoder(nn.Module):
    def __init__(
            self,
            norm_layer=nn.BatchNorm2d
    ):
        super(Decoder, self).__init__()
        fpn_dims = [256, 512, 1024, 2048]
        dec_dim = 128

        self.refine_spa = nn.ModuleList()
        for dim in fpn_dims:
            self.refine_spa.append(nn.Sequential(
                nn.Conv2d(dim, dec_dim, kernel_size=3, padding=1, bias=False),
                norm_layer(dec_dim),
                nn.ReLU(inplace=True)
            ))

        self.refine_tem = nn.ModuleList()
        for dim in fpn_dims:
            self.refine_tem.append(nn.Sequential(
                nn.Conv2d(dim, dec_dim, kernel_size=3, padding=1, bias=False),
                norm_layer(dec_dim),
                nn.ReLU(inplace=True)
            ))

        num_conv_tower = 4
        spa_tower = []
        for _ in range(num_conv_tower):
            spa_tower.append(nn.Sequential(
                nn.Conv2d(dec_dim, dec_dim, kernel_size=3, padding=1, bias=False),
                norm_layer(dec_dim),
                nn.ReLU(inplace=True)
            ))
        spa_tower.append(nn.Conv2d(dec_dim, 1, kernel_size=1))
        self.add_module('spa_tower', nn.Sequential(*spa_tower))

        tem_tower = []
        for _ in range(num_conv_tower):
            tem_tower.append(nn.Sequential(
                nn.Conv2d(dec_dim, dec_dim, kernel_size=3, padding=1, bias=False),
                norm_layer(dec_dim),
                nn.ReLU(inplace=True)
            ))
        tem_tower.append(nn.Conv2d(dec_dim, 1, kernel_size=1))
        self.add_module('tem_tower', nn.Sequential(*tem_tower))

        self.fusion = Fusion(dec_dim, dec_dim, norm_layer)

        fuse_tower = []
        for _ in range(num_conv_tower):
            fuse_tower.append(nn.Sequential(
                nn.Conv2d(dec_dim, dec_dim, kernel_size=3, padding=1, bias=False),
                norm_layer(dec_dim),
                nn.ReLU(inplace=True)
            ))
        fuse_tower.append(nn.Conv2d(dec_dim, 1, kernel_size=1))
        self.add_module('fuse_tower', nn.Sequential(*fuse_tower))

        self.mask_head = nn.Conv2d(3, 1, kernel_size=3, padding=1)

    def forward(self, spa_feats, tem_feats, sent):
        for i in range(len(spa_feats)):
            if i == 0:
                spa = self.refine_spa[i](spa_feats[i])
                tem = self.refine_tem[i](tem_feats[i])
            else:
                spa_p = self.refine_spa[i](spa_feats[i])
                tem_p = self.refine_tem[i](tem_feats[i])
                target_h, target_w = spa.size()[2:]
                h, w = spa_p.size()[2:]
                assert target_h % h == 0
                assert target_w % w == 0
                factor_h, factor_w = target_h // h, target_w // w
                assert factor_h == factor_w
                spa_p = aligned_bilinear(spa_p, factor_h)
                tem_p = aligned_bilinear(tem_p, factor_h)
                spa = spa + spa_p
                tem = tem + tem_p



        z = self.fusion(spa, tem, sent)
        pred = self.fuse_tower(z)




        pred = aligned_bilinear(pred, 4).squeeze()
        return  pred


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")


class TransformerEncoderLayer(nn.Module):
    def __init__(
            self,
            d_model=512,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            activation="relu",
            normalize_before=True,
    ):
        super().__init__()
        self.self_attn_words = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn_img = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn_flo = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_words_1 = nn.LayerNorm(d_model)
        self.norm_words_2 = nn.LayerNorm(d_model)
        self.norm_flo = nn.LayerNorm(d_model)
        self.norm_flo2 = nn.LayerNorm(d_model)
        self.dropout_words = nn.Dropout(dropout)
        self.dropout_cross_img = nn.Dropout(dropout)
        self.dropout_cross_flo = nn.Dropout(dropout)
        self.dropout_flo2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, img, flo, words, pos_embed_img, pos_embed_txt, key_padding_mask_img, key_padding_mask_txt, vis=False):
        # img: [H*W, B, C]
        # flo: [H*W, B, C]
        # words: [N, B, C]
        # pos_embed_img: [H*W, B, C]
        # pos_embed_txt: [N, B, C]
        # key_padding_mask_img: [B, H*W]
        # key_padding_mask_txt: [B. N]
        query_words = key_words = self.with_pos_embed(words, pos_embed_txt)
        vis_map = {}
        if not vis:
            words_attn, _ = self.self_attn_words(query_words, key_words, value=words, key_padding_mask=key_padding_mask_txt)
        else:
            words_attn, tmp = self.self_attn_words(query_words, key_words, value=words, key_padding_mask=key_padding_mask_txt)
            vis_map['self_words_attn'] = tmp

        words = words + self.dropout_words(words_attn)
        words = self.norm_words_1(words)

        key_img = value_img = self.with_pos_embed(img, pos_embed_img)
        words_cross_attn, _ = self.cross_attn_img(words, key_img, value=value_img,
                                                  key_padding_mask=key_padding_mask_img)
        words = words + self.dropout_cross_img(words_cross_attn)
        words = self.norm_words_2(words)

        query_flo = self.with_pos_embed(flo, pos_embed_img)
        if not vis:
            flo_cross_attn, _ = self.cross_attn_flo(query_flo, words, value=words, key_padding_mask=key_padding_mask_txt)
        else:
            flo_cross_attn, tmp = self.cross_attn_flo(query_flo, words, value=words, key_padding_mask=key_padding_mask_txt)
            vis_map['cross_words_attn'] = tmp
        flo = flo + self.dropout_cross_flo(flo_cross_attn)
        flo = self.norm_flo(flo)

        flo2 = self.linear2(self.dropout(self.activation(self.linear1(flo))))
        flo2 = flo + self.dropout_flo2(flo2)
        flo2 = self.norm_flo2(flo2)

        if not vis:
            return flo2
        else:
            return flo2, vis_map


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(TransformerEncoderLayer())
        self.num_layers = num_layers

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, img, flo, words, pos_embed_img, pos_embed_txt, key_padding_mask_img, key_padding_mask_txt, vis=False):
        # img: [H*W, B, C]
        # flo: [H*W, B, C]
        # words: [N, B, C]
        # pos_embed_img: [H*W, B, C]
        # pos_embed_txt: [N, B, C]
        # key_padding_mask_img: [B, H*W]
        # key_padding_mask_txt: [B. N]
        vis_map = {}
        for layer in self.layers:
            src_flo = flo
            src_img = img

            if not vis:
                flo = layer(src_img, src_flo, words, pos_embed_img, pos_embed_txt, key_padding_mask_img,
                            key_padding_mask_txt)
                img = layer(src_flo, src_img, words, pos_embed_img, pos_embed_txt, key_padding_mask_img,
                            key_padding_mask_txt)
            else:
                flo, vis_map_flo = layer(src_img, src_flo, words, pos_embed_img, pos_embed_txt, key_padding_mask_img,
                            key_padding_mask_txt, vis)
                img, vis_map_img = layer(src_flo, src_img, words, pos_embed_img, pos_embed_txt, key_padding_mask_img,
                            key_padding_mask_txt, vis)

        if not vis:
            return img, flo
        else:
            vis_map['flo'] = vis_map_flo
            vis_map['img'] = vis_map_img
            return img, flo, vis_map


class Fusion(nn.Module):
    def __init__(
            self,
            input_dim,
            feature_dim,
            norm_layer,
    ):
        super(Fusion, self).__init__()
        self.img_flo_fc = nn.Sequential(
            nn.Linear(input_dim * 2, feature_dim),
            nn.ReLU(inplace=True)
        )
        self.img_txt_fc = nn.Linear(feature_dim + 512, input_dim)
        self.flo_txt_fc = nn.Linear(feature_dim + 512, input_dim)
        self.img_enhance_fc = nn.Linear(feature_dim, feature_dim)
        self.flo_enhance_fc = nn.Linear(feature_dim, feature_dim)
        self.fusion_cat_conv = nn.Sequential(
            nn.Conv2d(input_dim * 2, input_dim, kernel_size=3, padding=1, bias=False),
            norm_layer(input_dim),
            nn.ReLU(inplace=True),
        )

    def forward(
            self,
            image,
            flow,
            txt
    ):
        img_avg = image.flatten(2).mean(dim=2)
        flo_avg = flow.flatten(2).mean(dim=2)
        # [B, C]
        img_avg = img_avg.unsqueeze(1)
        flo_avg = flo_avg.unsqueeze(1)
        # [B, 1, C]
        img_flo = torch.cat([img_avg, flo_avg], dim=2)
        # [B, 1, 2C]
        img_flo = F.relu(self.img_flo_fc(img_flo))
        # [B, 1, c]
        img_txt = torch.cat([img_avg, txt], dim=2)
        # [B, 1, c+512]
        img_txt_gate = torch.sigmoid(self.img_txt_fc(img_txt))
        flo_txt = torch.cat([flo_avg, txt], dim=2)
        flo_txt_gate = torch.sigmoid(self.flo_txt_fc(flo_txt))

        img_txt_gate = img_txt_gate.squeeze(1).unsqueeze(2).unsqueeze(3)
        flo_txt_gate = flo_txt_gate.squeeze(1).unsqueeze(2).unsqueeze(3)

        image = image * img_txt_gate
        flow = flow * flo_txt_gate
        #
        img_enhance = torch.sigmoid(self.img_enhance_fc(img_flo))
        flo_enhance = torch.sigmoid(self.flo_enhance_fc(img_flo))

        img_enhance = img_enhance.squeeze(1).unsqueeze(2).unsqueeze(3)
        flo_enhance = flo_enhance.squeeze(1).unsqueeze(2).unsqueeze(3)
        # # [B, c, 1, 1]
        image = image * img_enhance
        flow = flow * flo_enhance
        # image = image * img_txt_gate.squeeze(1).unsqueeze(2).unsqueeze(3)
        # flow = flow * flo_txt_gate.squeeze(1).unsqueeze(2).unsqueeze(3)
        fusion_cat = torch.cat([image, flow], dim=1)
        fusion_cat = self.fusion_cat_conv(fusion_cat)
        return fusion_cat


class JointModel(nn.Module):
    def __init__(
            self,
            image_encoder=None,
            flow_encoder=None,
            num_layers=1,
            norm_layer=nn.BatchNorm2d,
    ):
        super(JointModel, self).__init__()
        resnet_im = image_encoder
        self.conv1_1 = resnet_im.conv1
        self.bn1_1 = resnet_im.bn1
        self.relu_1 = resnet_im.relu
        self.maxpool_1 = resnet_im.maxpool

        self.res2_1 = resnet_im.layer1
        self.res3_1 = resnet_im.layer2
        self.res4_1 = resnet_im.layer3
        self.res5_1 = resnet_im.layer4

        resnet_fl = flow_encoder
        self.conv1_2 = resnet_fl.conv1
        self.bn1_2 = resnet_fl.bn1
        self.relu_2 = resnet_fl.relu
        self.maxpool_2 = resnet_fl.maxpool

        self.res2_2 = resnet_fl.layer1
        self.res3_2 = resnet_fl.layer2
        self.res4_2 = resnet_fl.layer3
        self.res5_2 = resnet_fl.layer4

        self.text_encoder = TextEncoder(num_layers=num_layers)
        self.decoder = Decoder()

        self.conv_r5_1_reduce = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1, bias=False),
            norm_layer(512),
        )
        self.conv_r5_2_reduce = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1, bias=False),
            norm_layer(512),
        )

        self.transformer = nn.ModuleDict()
        self.transformer['stage4'] = TransformerEncoder(4)
        self.transformer['stage5'] = TransformerEncoder(4)

        self.conv_r4_1_reduce = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, bias=False),
            norm_layer(512),
        )
        self.conv_r4_2_reduce = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, bias=False),
            norm_layer(512),
        )
        self.conv_r4_1_up = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=1, bias=False),
            norm_layer(1024),
        )
        self.conv_r4_2_up = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=1, bias=False),
            norm_layer(1024),
        )
        self.conv_r5_1_up = nn.Sequential(
            nn.Conv2d(512, 2048, kernel_size=1, bias=False),
            norm_layer(2048),
        )
        self.conv_r5_2_up = nn.Sequential(
            nn.Conv2d(512, 2048, kernel_size=1, bias=False),
            norm_layer(2048),
        )

    def _forward_transformer(self, img, flo, words, phrase_mask, img_mask, stage, vis=False):
        B, C, H, W = img.shape
        pos_embed_img = positionalencoding2d(B, d_model=C, height=H, width=W)
        pos_embed_img = pos_embed_img.flatten(2).permute(2, 0, 1).contiguous()
        # [H*W, B, 512]
        pos_embed_txt = positionalencoding1d(B, max_len=phrase_mask.shape[-1])
        pos_embed_txt = pos_embed_txt.permute(1, 0, 2).contiguous()
        # [N, B, 512]
        key_padding_mask_img = ~img_mask.bool()
        # [B, H*W]
        key_padding_mask_txt = ~phrase_mask.bool()
        # [B, N]

        f_img = img.flatten(2).permute(2, 0, 1).contiguous()
        f_flo = flo.flatten(2).permute(2, 0, 1).contiguous()
        # [H*W, B, 512]

        if not vis:
            f_img, f_flo = self.transformer[stage](f_img, f_flo, words, pos_embed_img, pos_embed_txt, key_padding_mask_img,
                                               key_padding_mask_txt)
        else:
            f_img, f_flo, vis_map = self.transformer[stage](f_img, f_flo, words, pos_embed_img, pos_embed_txt, key_padding_mask_img,
                                               key_padding_mask_txt, True)

        f_img = f_img.permute(1, 2, 0).contiguous()
        f_flo = f_flo.permute(1, 2, 0).contiguous()
        f_img = f_img.reshape(B, C, H, W)
        f_flo = f_flo.reshape(B, C, H, W)

        if not vis:
            return f_img, f_flo
        else:
            return f_img, f_flo, vis_map

    def forward(self, image, flow, phrase, phrase_mask, img_mask, vis=False):
        # image: [B, 3, H, W]
        # flow: [B, 3, H, W]
        # phrase: [B, N, 300]
        # phrase_mask: [B, N]
        # img_mask: [B, 100]
        if vis:
            vis_dict = {}

        f_text = self.text_encoder(phrase)
        # [B, N, 512]
        sent = f_text.sum(1, keepdim=True)
        # [B, 1, 512]
        words = f_text.permute(1, 0, 2).contiguous()
        # [N, B, 512]

        spa_feats = []
        tem_feats = []

        x1 = self.conv1_1(image)
        x1 = self.bn1_1(x1)
        x2 = self.conv1_2(flow)
        x2 = self.bn1_2(x2)

        r1_1 = self.relu_1(x1)
        r1_2 = self.relu_2(x2)

        r1_1 = self.maxpool_1(r1_1)
        r1_2 = self.maxpool_2(r1_2)
        r2_1 = self.res2_1(r1_1)
        r2_2 = self.res2_2(r1_2)
        spa_feats.append(r2_1)
        tem_feats.append(r2_2)

        r3_1 = self.res3_1(r2_1)
        r3_2 = self.res3_2(r2_2)

        spa_feats.append(r3_1)
        tem_feats.append(r3_2)

        r4_1 = self.res4_1(r3_1)
        r4_2 = self.res4_2(r3_2)

        # ################################## #
        r4_1_reduce = self.conv_r4_1_reduce(r4_1)
        r4_2_reduce = self.conv_r4_2_reduce(r4_2)

        if not vis:
            r4_1_trans, r4_2_trans = self._forward_transformer(r4_1_reduce, r4_2_reduce, words, phrase_mask,
                                                           img_mask.repeat(1, 4), 'stage4')
        else:
            r4_1_trans, r4_2_trans, vis_4 = self._forward_transformer(r4_1_reduce, r4_2_reduce, words, phrase_mask,
                                                               img_mask.repeat(1, 4), 'stage4', True)
            vis_dict['stage4'] = vis_4

        r4_1_up = self.conv_r4_1_up(r4_1_trans)
        r4_2_up = self.conv_r4_2_up(r4_2_trans)

        r4_1 = F.relu(r4_1 + r4_1_up)
        r4_2 = F.relu(r4_2 + r4_2_up)
        # ################################## #

        spa_feats.append(r4_1)
        tem_feats.append(r4_2)
        r5_1 = self.res5_1(r4_1)
        r5_2 = self.res5_2(r4_2)

        # ################################## #
        r5_1_reduce = self.conv_r5_1_reduce(r5_1)
        r5_2_reduce = self.conv_r5_2_reduce(r5_2)

        if not vis:
            r5_1_trans, r5_2_trans = self._forward_transformer(r5_1_reduce, r5_2_reduce, words, phrase_mask, img_mask,
                                                           'stage5')
        else:
            r5_1_trans, r5_2_trans, vis_5 = self._forward_transformer(r5_1_reduce, r5_2_reduce, words, phrase_mask,
                                                                      img_mask, 'stage5', True)
            vis_dict['stage5'] = vis_5

        r5_1_up = self.conv_r5_1_up(r5_1_trans)
        r5_2_up = self.conv_r5_2_up(r5_2_trans)

        r5_1 = F.relu(r5_1 + r5_1_up)
        r5_2 = F.relu(r5_2 + r5_2_up)
        # ################################## #

        spa_feats.append(r5_1)
        tem_feats.append(r5_2)
        pred = self.decoder(spa_feats, tem_feats, sent)

        if not vis:
            return pred
        else:
            return pred, vis_dict


class TextEncoder(nn.Module):
    def __init__(
            self,
            input_size=300,
            hidden_size=512,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=True,
            batch_first=batch_first,
            dropout=dropout,
        )

    def forward(self, input):
        self.lstm.flatten_parameters()
        output, (_, _) = self.lstm(input)
        return output


