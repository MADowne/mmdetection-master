from mmdet.apis import init_detector, inference_detector
import mmcv

# 指定模型的配置文件和 checkpoint 文件路径
config_file = '../configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = '../work_dirs/faster_rcnn_r50_fpn_1x_voc2012/epoch_6.pth'

# 根据配置文件和 checkpoint 文件构建模型
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# 测试单张图片并展示结果
img3 = '3.jpg'  # 或者 img = mmcv.imread(img)，这样图片仅会被读一次
img4 = '4.jpg'  # 或者 img = mmcv.imread(img)，这样图片仅会被读一次
img5 = '5.jpg'  # 或者 img = mmcv.imread(img)，这样图片仅会被读一次
result1 = inference_detector(model, img3)
result2 = inference_detector(model, img4)
result3 = inference_detector(model, img5)
# 或者将可视化结果保存为图片
model.show_result(img3, result1, out_file='13.jpg')
model.show_result(img4, result2, out_file='14.jpg')
model.show_result(img5, result3, out_file='15.jpg')