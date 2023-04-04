from volleyball import *
from collective import *

import pickle


def return_dataset(cfg):
    if cfg.dataset_name=='volleyball':
        train_anns = volley_read_dataset(cfg.data_path, cfg.train_seqs)#data[sid/序列号]：{集合->annotations[fid/照片号] = {'file_name': file_name,'group_activity': activity,'actions': actions,'bboxes': bboxes, }}
        #print("anns", train_anns)
        train_frames = volley_all_frames(train_anns)#序列号加子文件夹号的组合
        # print("frames", len(train_frames))
        test_anns = volley_read_dataset(cfg.data_path, cfg.test_seqs)
        test_frames = volley_all_frames(test_anns)

        all_anns = {**train_anns, **test_anns}
        all_tracks = pickle.load(open(cfg.data_path + '/tracks_normalized.pkl', 'rb'))
        #print("all_tracks",all_tracks)
        #print("all_tracks", len(all_tracks))
        training_set=VolleyballDataset(all_anns,all_tracks,train_frames, #39+16,3493+1337=4830,
                                      cfg.data_path,cfg.image_size,cfg.out_size,cfg.inference_module_name,num_before=cfg.num_before,
                                       num_after=cfg.num_after,is_training=True,is_finetune=(cfg.training_stage==1))

        validation_set=VolleyballDataset(all_anns,all_tracks,test_frames,
                                      cfg.data_path,cfg.image_size,cfg.out_size,cfg.inference_module_name,num_before=cfg.num_before,
                                         num_after=cfg.num_after,is_training=False,is_finetune=(cfg.training_stage==1))
    
    elif cfg.dataset_name=='collective':
        train_anns=collective_read_dataset(cfg.data_path, cfg.train_seqs)
        train_frames=collective_all_frames(train_anns)

        test_anns=collective_read_dataset(cfg.data_path, cfg.test_seqs)
        test_frames=collective_all_frames(test_anns)

        training_set=CollectiveDataset(train_anns,train_frames,
                                      cfg.data_path,cfg.image_size,cfg.out_size,
                                      num_frames = cfg.num_frames, is_training=True,is_finetune=(cfg.training_stage==1))

        validation_set=CollectiveDataset(test_anns,test_frames,
                                      cfg.data_path,cfg.image_size,cfg.out_size,
                                      num_frames = cfg.num_frames, is_training=False,is_finetune=(cfg.training_stage==1))
                              
    else:
        assert False
                                         
    
    print('Reading dataset finished...')
    print('%d train samples'%len(train_frames))
    print('%d test samples'%len(test_frames))
    
    return training_set, validation_set
    