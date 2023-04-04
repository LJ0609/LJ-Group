import sys
sys.path.append(".")
from train_net_dynamic import *

cfg=Config('volleyball')
cfg.inference_module_name = 'dynamic_volleyball'

cfg.device_list = "0"
cfg.use_gpu = True
cfg.use_multi_gpu = False
cfg.training_stage = 2
cfg.train_backbone = True
cfg.test_before_train = False
cfg.test_interval_epoch = 0

# vgg16 setup
cfg.backbone = 'vgg16'
cfg.stage1_model_path = 'result/Vstage1_epoch46.pth'
cfg.out_size = 22, 40
cfg.emb_features = 512

# res18 setup
# cfg.backbone = 'res18'
# cfg.stage1_model_path = 'result/basemodel_VD_res18.pth'
# cfg.out_size = 23, 40
# cfg.emb_features = 512

# Dynamic Inference setup
cfg.group = 1
cfg.stride = 1
cfg.ST_kernel_size = [(3, 3)]      #[(3, 3),(3, 3),(3, 3),(3, 3)]==============
cfg.dynamic_sampling = True
cfg.sampling_ratio = [1]   #[1,2,4]              #[1] =========
cfg.lite_dim = None # None # 128
cfg.scale_factor = True
cfg.beta_factor = False
cfg.hierarchical_inference = False
cfg.parallel_inference = False
cfg.num_DIM = 1
cfg.train_dropout_prob = 0.3

# cfg.num_before=2
# cfg.num_after=2

cfg.batch_size = 8 
cfg.test_batch_size = 1
cfg.num_frames =  10 #5 缺帧的情况，减少一半
cfg.load_backbone_stage2 = True #True
cfg.load_stage2model=False       #  False 加载模型2
cfg.stage2model = 'result/stage2_epoch1_88.11%.pth'
cfg.train_learning_rate = 1e-4
cfg.lr_plan = {11: 1e-5, 21: 3e-6, 31: 1e-6} #test
cfg.max_epoch = 100 #30
cfg.actions_weights = ([1., 1., 2., 3., 1., 2., 2., 0.2, 1.])

cfg.exp_note = 'Dynamic Volleyball_stage2_res18_litedim128_reproduce_1'
train_net(cfg)
