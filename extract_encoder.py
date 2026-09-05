import os
import sys
import torch
from models.networks import UnetEncoder

# =========================
# 确保项目根目录在 sys.path
# =========================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
from models.networks import UnetGenerator
#
# # =========================
# # 路径配置
# # =========================
# ckpt_dir = '/home/sda/ls/InfraGAN-ori/checkpoints/generators-finetuned'          # 你存 200_net_G.pth 的地方
# save_dir = '/home/sda/ls/InfraGAN-ori/checkpoints/Encoder_finetuned'       # 拆出来的 encoder
# os.makedirs(save_dir, exist_ok=True)
#
# # =========================
# # 模型参数（必须和训练时一致）
# # =========================
# input_nc = 3       # 可见光
# output_nc = 1      # 红外
# ngf = 64
# use_dropout = False
#
# # =========================
# # 构建生成器
# # =========================
# net_G = UnetGenerator(
#     input_nc=input_nc,
#     output_nc=output_nc,
#     ngf=ngf,
#     use_dropout=use_dropout
# )
# net_G.eval()
#
# # =========================
# # 遍历所有 net_G checkpoint
# # =========================
# for fname in sorted(os.listdir(ckpt_dir)):
#     if not fname.endswith('.pth'):
#         continue
#
#     # 例如：200_net_G.pth → 200
#     epoch = fname.split('_')[0]
#
#     print(f'>>> Extracting encoder from epoch {epoch}')
#
#     ckpt_path = os.path.join(ckpt_dir, fname)
#     state_dict = torch.load(ckpt_path, map_location='cpu')
#
#     encoder_state_dict = {
#         k.replace('encoder.', ''): v
#         for k, v in state_dict.items()
#         if k.startswith('encoder.')
#     }
#
#     net_G.encoder.load_state_dict(encoder_state_dict, strict=True)
#
#     torch.save(
#         net_G.encoder.state_dict(),
#         os.path.join(save_dir, f'Encoder_epoch{epoch}.pth')
#     )
#
#
#
#
#
# print('✅ All encoder weights extracted successfully!')

encoder = UnetEncoder(input_nc=3, ngf=64)
encoder.load_state_dict(torch.load('/home/sda/ls/InfraGAN-ori/checkpoints/Encoder_finetuned/Encoder_epoch100.pth'))
encoder.eval()

x = torch.randn(1, 3, 256, 256)
bottleneck, skips = encoder(x)
print(bottleneck.shape)
