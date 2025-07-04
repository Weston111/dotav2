# 测试DOTA数据集得到mAP指标
## 数据集情况
模型：DetCLIPv3
使用数据集：DOTAv2

处理好的数据集压缩包名称为：DOTAv2_val_data.tar.gz

解压之后：
1. DOTAv2_val.json  （标注文件）
2. val_images       （处理好的测试图像）

## Code based mmdet
基于mmdetection库，仅需要修改configs文件即可

以下为使用groundingdino进行zero-shot推理
### 数据集config
dota_detection.py已经放置在_base_/datasets里面
修改一下内容

这里引用的dota_detection修改里面的路径即可
例如：data_root
```python
#================DOTAv2==================
data_root = '../data/LAE-FOD/DOTAv2/'
metainfo = dict(
    classes = ('plane', 'ship', 'storage tank', 'baseball diamond', 'tennis court', 'basketball court', 'ground track field', 'harbor', 'bridge', 'large vehicle', 'small vehicle', 'helicopter', 'roundabout', 'soccer ball field', 'swimming pool', 'container crane', 'airport', 'helipad')
)
```

### 模型定义config
加一个定义模型的新文件(configs/mm_grounding_dino)

grounding_dino_swin-t_pretrain_obj365.py定义模型结构以及其他的东西

修改开头的内容：
```python
_base_ = [
    # '../_base_/datasets/coco_detection.py',
    '../_base_/datasets/dota_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
```

## 测试
测试命令
```bash
./tools/dist_test.sh configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365.py ../weights/groundingdino_swint_ogc_mmdet-822d7e9d.pth 2
```
结果belike

groundingdino的mAP结果为0.013
![](./image.png)