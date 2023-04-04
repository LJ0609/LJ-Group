import sys
sys.path.append(".")
from train_net import *

cfg=Config('volleyball')

cfg.use_multi_gpu = False
cfg.device_list="0"
cfg.training_stage=1

cfg.train_backbone=True
cfg.test_before_train = False #True

# VGG16
cfg.backbone = 'vgg16'
cfg.image_size = 720, 1280
cfg.out_size = 22, 40
cfg.emb_features = 512

cfg.num_before = 5  #fid=random.randint(src_fid-self.num_before, src_fid+self.num_after),用来选照片
cfg.num_after = 4  #同上
cfg.stage1_model_path='result/Vstage1_epoch140_89.68%.pth'  #path of the base model, need to be set in stage2
cfg.pose_head_path='result/Tstage1_epoch1.pth'
cfg.batch_size=2 #4  #8
cfg.test_batch_size=1
cfg.num_frames=1
# cfg.train_learning_rate=1e-5
# cfg.lr_plan={}
# cfg.max_epoch=200
cfg.train_learning_rate=1e-4
cfg.lr_plan={50:1e-5, 100:1e-6}
# cfg.lr_plan={30:5e-5, 60:1e-5, 90:1e-6}
# cfg.train_learning_rate=1e-5
# cfg.lr_plan={10:5e-6,20:1e-6}
cfg.max_epoch=1500 #120
cfg.set_bn_eval = False
cfg.actions_weights=([1., 1., 2., 3., 1., 2., 2., 0.2, 1.])
cfg.crop_size = 7, 7 # 5, 5
cfg.exp_note='Volleyball_stage1'
cfg.test_interval_epoch = 150
train_net(cfg)