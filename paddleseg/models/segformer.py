# The SegFormer code was heavily based on https://github.com/NVlabs/SegFormer
# Users should be careful about adopting these functions in any commercial matters.
# https://github.com/NVlabs/SegFormer#license

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np

from paddleseg.cvlibs import manager
from paddleseg.models import layers
from paddleseg.utils import utils
from functools import partial
import math
import os
import matplotlib.pyplot as plt
import numpy as np
import paddle.vision.transforms as transforms
from PIL import Image
from osgeo import gdal



#  conv_layer = nn.Conv2D(in_channels=tensor.shape[1], out_channels=3, kernel_size=1, padding=1)
#         output_feature_maps = conv_layer(tensor)
#         tensor=(output_feature_maps)

def save_tensor_as_CONVRGBpng(folder_path, tensor):
    # Ensure we're operating on the CPU and detach the tensor from the graph
    # conv_layer = nn.Conv2D(in_channels=tensor.shape[1], out_channels=3, kernel_size=1, padding=1)
    # output_feature_maps = conv_layer(tensor)
    tensor = F.interpolate(tensor, size=(256, 256), mode='bilinear', align_corners=False)
    # tensor = tensor.cpu().detach()
    tensor = tensor
    # tensor = tensor.float()
    linear_fuse = ConvBNReLU2(
        in_channels=tensor.shape[1],
        out_channels=3,
        kernel_size=1)

    output_feature_maps = linear_fuse(tensor)
    tensor = (output_feature_maps)
    norm = nn.BatchNorm2D(3)
    tensor = tensor.numpy() 
    # tensor = norm(tensor)
    # Normalize tensor to [0, 255]
    tensor = tensor - tensor.min()  # Min normalization to 0
    tensor = tensor / tensor.max()  # Max normalization to 1
    tensor = tensor * 255.0  # Scale to [0, 255]
    # tensor = F.interpolate(
    #         tensor,
    #         size=(512,512),
    #         mode='bilinear',
    #         align_corners=False)

    # tensor = tensor.detach().numpy() .astype('uint8')  # Convert to numpy array and cast to uint8
    tensor = tensor.astype('uint8')
    # tensor = tensor.cpu().detach()

    # Create the folder if it does not exist
    os.makedirs(folder_path, exist_ok=True)
    # Iterate over the batch
    for i in range(tensor.shape[0]):
        # Permute the tensor from (3, H, W) to (H, W, 3)
        img_arr = np.transpose(tensor[i], (1, 2, 0))

        # Convert the array to an Image
        img = Image.fromarray(img_arr)

        # Save the image
        img.save(os.path.join(folder_path, f'image_batch_{i}.png'))


def save_tensor_as_RGBpng(folder_path, tensor):
    # Ensure we're operating on the CPU and detach the tensor from the graph
    # tensor = tensor.cpu().detach()
    tensor = tensor.numpy() 
    # Create the folder if it does not exist
    os.makedirs(folder_path, exist_ok=True)

    # Normalize tensor to [0, 255]
    tensor = tensor - tensor.min()  # Min normalization to 0
    tensor = tensor / tensor.max()  # Max normalization to 1
    tensor = tensor * 255.0  # Scale to [0, 255]
    tensor = tensor.astype('uint8')  # Convert to numpy array and cast to uint8

    # Iterate over the batch
    for i in range(tensor.shape[0]):
        # Permute the tensor from (3, H, W) to (H, W, 3)
        img_arr = np.transpose(tensor[i], (1, 2, 0))

        # Convert the array to an Image
        img = Image.fromarray(img_arr)

        # Save the image
        img.save(os.path.join(folder_path, f'image_batch_{i}.png'))


class ConvBNReLU2(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvBNReLU2, self).__init__()
        self.conv = nn.Conv2D(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias_attr=False)
        self.bn = nn.BatchNorm2D(out_channels)
        self.relu = nn.Sigmoid()
        # (inplace=True

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
def save_tensor_as_Multipng(folder_path, tensor):
    # tensor = tensor.cpu().detach()
    tensor = tensor.numpy() 
    # 检查目录是否存在，如果不存在则创建
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # Normalize tensor to [0, 255]

    tensor = tensor - tensor.min()  # Min normalization to 0
    tensor = tensor / tensor.max()  # Max normalization to 1
    tensor = tensor * 255.0  # Scale to [0, 255]
    # tensor = F.interpolate(
    #         tensor,
    #         size=(512,512),
    #         mode='bilinear',
    #         align_corners=False)
    tensor = tensor.astype('uint8')  # Convert to numpy array and cast to uint8
    # 确保tensor是在CPU上，并将其从计算图中分离出来
    # tensor = tensor.numpy()

    # 计算可以组成多少组RGB图片（每组三个通道）
    num_images = tensor.shape[1] // 3

    # 遍历每张图片的每组通道
    for i in range(tensor.shape[0]):  # 遍历batch中的每个样本
        for j in range(num_images):  # 遍历每个可能的组合
            # 使用连续的三个通道形成RGB图像
            img_arr = tensor[i, j * 3:(j + 1) * 3, :, :].transpose(1, 2, 0)

            # 截断数值到合法图像范围
            # img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)

            # 使用PIL库将Numpy数组保存为图片
            img = Image.fromarray(img_arr, 'RGB')

            # 生成图片的保存路径
            img.save(os.path.join(folder_path, f'batch_{i:02d}-{j + 1:02d}.png'))





    # 检查目录是否存在，如果不存在则创建
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
     # Normalize tensor to [0, 255]

    tensor = tensor - tensor.min()  # Min normalization to 0
    tensor = tensor / tensor.max()  # Max normalization to 1
    tensor = tensor * 255.0  # Scale to [0, 255]
    # tensor = F.interpolate(
    #         tensor,
    #         size=(512,512),
    #         mode='bilinear',
    #         align_corners=False)
    # tensor = tensor.numpy().astype('uint8')  # Convert to numpy array and cast to uint8
    # 确保tensor是在CPU上，并将其从计算图中分离出来
    # tensor = tensor.numpy()
    
    # 计算可以组成多少组RGB图片（每组三个通道）
    num_images = tensor.shape[1] // 3

    # 遍历每张图片的每组通道
    for i in range(tensor.shape[0]):  # 遍历batch中的每个样本
        for j in range(num_images):  # 遍历每个可能的组合
            # 使用连续的三个通道形成RGB图像
            img_arr = tensor[i, j*3:(j+1)*3, :, :].transpose(1, 2, 0)

            # 截断数值到合法图像范围
            # img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)

            # 使用PIL库将Numpy数组保存为图片
            img = Image.fromarray(img_arr, 'RGB')
            
            # 生成图片的保存路径
            img.save(os.path.join(folder_path, f'batch_{i:02d}-{j+1:02d}.png'))

# class CrossAttentionWithResidual(nn.Layer):
#     def __init__(self, in_channels_x, in_channels_y, out_channels):
#         super().__init__()
#         # 继续使用之前的CrossAttention示例中定义的转换层
#         self.query_conv = nn.Conv2D(in_channels_x, out_channels, kernel_size=1)
#         self.key_conv = nn.Conv2D(in_channels_y, out_channels, kernel_size=1)
#         self.value_conv = nn.Conv2D(in_channels_y, out_channels, kernel_size=1)
#         self.softmax = nn.Softmax(axis=-1)

#         # 如果输入和输出通道不同，需要一个转换层，以便能够相加
#         if in_channels_x != out_channels:
#             self.residual_conv = nn.Conv2D(in_channels_x, out_channels, kernel_size=1)
#         else:
#             self.residual_conv = None

#     def forward(self, x, y):
#         original_x = x

#         # 检查并上采样y以匹配x的分辨率
#         if x.shape[2:] != y.shape[2:]:
#             y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)

#         # 生成查询、键和值
#         query = self.query_conv(x).flatten(2).transpose([0, 2, 1])  # [b, hw_x, c]
#         key = self.key_conv(y).flatten(2).transpose([0, 2, 1])  # [b, hw_y, c]
#         value = self.value_conv(y).flatten(2)  # [b, c, hw_y]

#         # 计算注意力分数
#         attention = paddle.bmm(query, key.transpose([0, 2, 1]))  # [b, hw_x, hw_y]
#         attention = self.softmax(attention)

#         # 将注意力权重使用`value`进行加权聚合以获取加权特征，这里应该以原始x的空间维度作为聚合的目标
#         out = paddle.bmm(attention, value.transpose([0, 2, 1]))  # [b, hw_x, c]
#         out = out.transpose([0, 2, 1]).reshape([x.shape[0], -1, x.shape[2], x.shape[3]])  # [b, c, h_x, w_x]

#         # 进行残差连接之前，可能需要调整x的通道数以匹配输出
#         if self.residual_conv is not None:
#             original_x = self.residual_conv(original_x)

#         # 残差连接
#         out += original_x

#         return out




class CrossAttentionWithResidual(nn.Layer):
    def __init__(self, in_channels_x, in_channels_y, out_channels, num_heads=2):
        super().__init__()
        self.num_heads = num_heads
        self.depth = out_channels // num_heads
        
        self.norm = nn.BatchNorm2D(out_channels)
        
        # 定义查询、键和值的卷积层
        self.query_conv = nn.Conv2D(in_channels_x, self.depth * num_heads, kernel_size=1)
        self.key_conv = nn.Conv2D(in_channels_y, self.depth * num_heads, kernel_size=1)
        self.value_conv = nn.Conv2D(in_channels_y, self.depth * num_heads, kernel_size=1)
        
        # Final output transformation convolution
        self.out_projection = nn.Conv2D(self.depth * num_heads, out_channels, kernel_size=1)
        
        self.softmax = nn.Softmax(axis=-1)

        # Residual connection adjustment (if needed)
        if in_channels_x != out_channels:
            self.residual_conv = nn.Conv2D(in_channels_x, out_channels, kernel_size=1)
        else:
            self.residual_conv = None
    
    def transpose_for_scores(self, x, batch_size):
        # Paddle中需要改变视图时，使用reshape函数和transpose函数
        # [b, num_heads * depth, h, w] -> [b, num_heads, h * w, depth]
        new_x_shape = x.reshape([batch_size, self.num_heads, self.depth, -1])
        return new_x_shape.transpose([0, 1, 3, 2])  # [b, num_heads, h * w, depth]

    def forward(self, x, y):
        batch_size = x.shape[0]

        # 分割多头
        query = self.transpose_for_scores(self.query_conv(x), batch_size)
        key = self.transpose_for_scores(self.key_conv(y), batch_size)
        value = self.transpose_for_scores(self.value_conv(y), batch_size)

        # 缩放 query 以防止点积变得过大
        query = query * (self.depth ** -0.5)

        # 计算多头注意力分数
        attention_scores = paddle.matmul(query, key.transpose([0, 1, 3, 2]))
        attention_scores = self.softmax(attention_scores)

        # 合并多头
        attention_output = paddle.matmul(attention_scores, value)

        # 调整回原始大小
        attention_output = attention_output.transpose([0, 1, 3, 2]).reshape([batch_size, -1, x.shape[2], x.shape[3]])

        # 应用最终的输出投影
        attention_output = self.out_projection(attention_output)

        # 进行残差连接
        if self.residual_conv is not None:
            original_x = self.residual_conv(x)
        else:
            original_x = x
        attention_output += original_x
        attention_output = self.norm(attention_output)

        return attention_output






class MLP(nn.Layer):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        # 第二个维度进行展开
        x = x.flatten(2).transpose([0, 2, 1])
        x = self.proj(x)
        return x


@manager.MODELS.add_component
class SegFormer(nn.Layer):
    """
    The SegFormer implementation based on PaddlePaddle.

    The original article refers to
    Xie, Enze, et al. "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers." arXiv pre# print arXiv:2105.15203 (2021).

    Args:
        num_classes (int): The unique number of target classes.
        backbone (Paddle.nn.Layer): A backbone network.
        embedding_dim (int): The MLP decoder channel dimension.
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature.
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769. Default: False.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(self,
                 num_classes,
                 backbone,
                 embedding_dim,
                 align_corners=False,
                 pretrained=None):
        super(SegFormer, self).__init__()

        self.pretrained = pretrained
        self.align_corners = align_corners
        self.backbone = backbone
        self.num_classes = num_classes
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.backbone.feat_channels
        
        
    


        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)
        # self.cov3d = nn.Conv3D()

        self.dropout = nn.Dropout2D(0.1)
        
        # 通带尺度的融合
        self.linear_fuse = layers.ConvBNReLU(
            in_channels=embedding_dim * 4,
            out_channels=embedding_dim,
            kernel_size=1,
            bias_attr=False)

        self.linear_pred = nn.Conv2D(
            embedding_dim, self.num_classes, kernel_size=1)
        
        self.linear_fuse2 = layers.ConvBNReLU(
            in_channels=4,
            out_channels=3,
            kernel_size=1,
            bias_attr=False)
       
        
        # self.c4y = MultiHeadCrossAttentionWithResidual(c4_in_channels,c4_in_channels,c4_in_channels)
        # self.norm4 = nn.BatchNorm2D(c4_in_channels)
        
        # self.c3y = MultiHeadCrossAttentionWithResidual(c3_in_channels,c3_in_channels,c3_in_channels)
        # self.norm3 = nn.BatchNorm2D(c3_in_channels)
        
        # self.c2y = MultiHeadCrossAttentionWithResidual(c2_in_channels,c2_in_channels,c2_in_channels)
        # self.norm2 = nn.BatchNorm2D(c2_in_channels)
        
        # self.c1y = MultiHeadCrossAttentionWithResidual(c1_in_channels,c1_in_channels,c1_in_channels)
        # self.norm1 = nn.BatchNorm2D(c1_in_channels)
        
        
        self.c4y = CrossAttentionWithResidual(c4_in_channels,c4_in_channels,c4_in_channels)
        self.norm4 = nn.BatchNorm2D(c4_in_channels)
        
        self.c3y = CrossAttentionWithResidual(c3_in_channels,c3_in_channels,c3_in_channels)
        self.norm3 = nn.BatchNorm2D(c3_in_channels)
        
        self.c2y = CrossAttentionWithResidual(c2_in_channels,c2_in_channels,c2_in_channels)
        self.norm2 = nn.BatchNorm2D(c2_in_channels)
        
        self.c1y = CrossAttentionWithResidual(c1_in_channels,c1_in_channels,c1_in_channels)
        self.norm1 = nn.BatchNorm2D(c1_in_channels)
        
        
        self.normxy = nn.BatchNorm2D(3)
        self.init_weight()

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)

    def forward(self, x,y):
        
        # # print("x",x.shape)
        # # print("y",y.shape)
        
        # 获取VIT提取的特征
        # 原始x与y进行交叉
        # save_tensor_as_Multipng("./feat/y",y)
        # save_tensor_as_RGBpng("./feat/x",x)
       
        save_tensor_as_RGBpng("./feat/y",y)
        x = self.linear_fuse2(x)

        save_tensor_as_RGBpng("./feat/x",x)
        
        y =self.normxy(y)
      
        # # print(y.shape)   
        
        # yfeats = self.backbone(y)

        y1,y2,y3,y4 = self.backbone(y)

        feats = self.backbone(x)
       
        
        c1, c2, c3, c4 = feats
        # save_tensor_as_Multipng("./feat/rgbc1",c1)
        # save_tensor_as_Multipng("./feat/rgbc2",c2)
        # save_tensor_as_Multipng("./feat/rgbc3",c3)
        # save_tensor_as_Multipng("./feat/rgbc4",c4)
        
        save_tensor_as_CONVRGBpng("./feat/rgbc1",c1)
        save_tensor_as_CONVRGBpng("./feat/rgbc2",c2)
        save_tensor_as_CONVRGBpng("./feat/rgbc3",c3)
        save_tensor_as_CONVRGBpng("./feat/rgbc4",c4)
        
        
        
        
        # save_tensor_as_Multipng("./feat/multi1",y1)
        # save_tensor_as_Multipng("./feat/multi2",y2)
        # save_tensor_as_Multipng("./feat/multi3",y3)
        # save_tensor_as_Multipng("./feat/multi4",y4)
        
        save_tensor_as_CONVRGBpng("./feat/multi1",y1)
        save_tensor_as_CONVRGBpng("./feat/multi2",y2)
        save_tensor_as_CONVRGBpng("./feat/multi3",y3)
        save_tensor_as_CONVRGBpng("./feat/multi4",y4)
        
        
  

        ############## MLP decoder on C1-C4 ###########
        
        # 类似于多尺度
        c1_shape = paddle.shape(c1)
        c2_shape = paddle.shape(c2)
        c3_shape = paddle.shape(c3)
        c4_shape = paddle.shape(c4)
        
        # # print("shape",c1_shape,c2_shape,c3_shape,c4_shape)
        #   # 在这里应用交叉注意力
        
        
        
        # 多头注意力 与  单头注意力 
        # 
    
        c1 = self.c1y(c1,y1)
        c2 = self.c2y(c2,y2)
        c3 = self.c3y(c3,y3)
        c4 =self.c4y(c4,y4)
        # save_tensor_as_Multipng("./feat/combinec1",c1)
        # save_tensor_as_Multipng("./feat/combinec2",c2)
        # save_tensor_as_Multipng("./feat/combinec3",c3)
        # save_tensor_as_Multipng("./feat/combinec4",c4)
        
        save_tensor_as_CONVRGBpng("./feat/combinec1",c1)
        save_tensor_as_CONVRGBpng("./feat/combinec2",c2)
        save_tensor_as_CONVRGBpng("./feat/combinec3",c3)
         
        
        

        _c4 = self.linear_c4(c4).transpose([0, 2, 1]).reshape(
            [0, 0, c4_shape[2], c4_shape[3]])
        _c4 = F.interpolate(
            _c4,
            size=c1_shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        save_tensor_as_CONVRGBpng("./feat/combine_c4",_c4) 
         
        

        _c3 = self.linear_c3(c3).transpose([0, 2, 1]).reshape(
            [0, 0, c3_shape[2], c3_shape[3]])
        _c3 = F.interpolate(
            _c3,
            size=c1_shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        save_tensor_as_CONVRGBpng("./feat/combine_c3",_c3) 

        _c2 = self.linear_c2(c2).transpose([0, 2, 1]).reshape(
            [0, 0, c2_shape[2], c2_shape[3]])
        _c2 = F.interpolate(
            _c2,
            size=c1_shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        save_tensor_as_CONVRGBpng("./feat/combine_c2",_c2) 

        _c1 = self.linear_c1(c1).transpose([0, 2, 1]).reshape(
            [0, 0, c1_shape[2], c1_shape[3]])
        save_tensor_as_CONVRGBpng("./feat/combine_c1",_c1) 
        # # print("_c1",_c1.shape)
        _c = self.linear_fuse(paddle.concat([_c4, _c3, _c2, _c1], axis=1))
        save_tensor_as_CONVRGBpng("./feat/combine_ccccccccccc",_c ) 
        
        

        logit = self.dropout(_c)
        logit = self.linear_pred(logit)
        save_tensor_as_CONVRGBpng("./feat/combine_logit",logit ) 
        return [
            F.interpolate(
                logit,
                size=paddle.shape(x)[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        ]

# if __name__ == "__main__":
#     SegFormerNet = SegFormer()