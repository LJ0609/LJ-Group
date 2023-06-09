import torch
import torch.nn as nn

from trans.models.transformer import TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer2, TransformerDecoder,TransformerDecoderLayer
from trans.models.transformer_cluster import TransformerEncoderLayer_cluster,TransformerEncoder_cluster
from torch.nn.init import normal_

#input: config,input_feat,embed_feat
#output: memeory--->size(N,B*T,C_e)
#       tgt-------->size(1,B*T,C_e)
class ST_plus_TR_block_actor(nn.Module):
    def __init__(self,config,embed_feat):
        super(ST_plus_TR_block_actor, self).__init__()
        self.embed_features = embed_feat

        #actor tr encoder
        encoder_layer_actor = TransformerEncoderLayer_cluster(self.embed_features, config.Nhead,total_size=config.total_size,window_size=config.window_size, dropout=config.dropout_porb, normalize_before=True)
        encoder_norm_actor = nn.LayerNorm(self.embed_features)
        self.encoder_actor = TransformerEncoder_cluster(encoder_layer_actor, num_layers=config.num_encoder_layers, norm=encoder_norm_actor)

        #temporal tr encoder
        encoder_layer_temp = TransformerEncoderLayer(self.embed_features, config.Nhead, dropout=config.dropout_porb,
                                                      normalize_before=True)
        encoder_norm_temp = nn.LayerNorm(self.embed_features)
        self.encoder_temp = TransformerEncoder(encoder_layer_temp, num_layers=config.num_encoder_layers,
                                                norm=encoder_norm_temp)
        #actor decoder--first
        decoder_actor=TransformerDecoderLayer2(self.embed_features, config.Nhead, dropout=config.dropout_porb, normalize_before=True)
        decoder_norm_actor= nn.LayerNorm(self.embed_features)
        self.decoder_actor = TransformerDecoder(decoder_actor, num_layers=config.num_decoder_layers, norm=decoder_norm_actor)

        #temporal decoder
        decoder_temp=TransformerDecoderLayer2(self.embed_features,config.Nhead,dropout=config.dropout_porb, normalize_before=True)
        decoder_norm_temp=nn.LayerNorm(self.embed_features)
        self.decoder_temp=TransformerDecoder(decoder_temp,num_layers=config.num_decoder_layers, norm=decoder_norm_temp)

    #x--->N,B*T,C
    #query=1,B*T,C
    def forward(self, x,query):
        N,B,T,C=x.shape
        tgt_len,bsz,dim=query.shape
        actor_o = x.reshape(N,B*T,-1) #(N,B*T,-1)
        #N,B*T,-1
        memory_actor,loss = self.encoder_actor(actor_o)

        #T,B*N,-1
        temp_o=x.permute(2,1,0,3).contiguous().reshape(T,B*N,-1)
        #T,B*N,-1
        memory_temp=self.encoder_temp(temp_o)
        #N,B*T,-1
        memory_temp=memory_temp.reshape(T,B,N,-1).permute(2,1,0,3).contiguous().reshape(N,B*T,-1)
        #N,B*T,-1
        memory=self.decoder_actor(memory_actor,memory_temp)
        memory=memory.reshape(N,B,T,-1)
        #1,B*T,C
        tgt=self.decoder_temp(query,memory)
        tgt=tgt.reshape(tgt_len,bsz,dim)
        return memory,tgt,loss

class ST_plus_TR_block_temp(nn.Module):
    def __init__(self,config,embed_feat):
        super(ST_plus_TR_block_temp, self).__init__()
        self.embed_features = embed_feat


        #actor tr encoder
        encoder_layer_actor = TransformerEncoderLayer_cluster(self.embed_features, config.Nhead,total_size=config.total_size,window_size=config.window_size, dropout=config.dropout_porb, normalize_before=True)
        encoder_norm_actor = nn.LayerNorm(self.embed_features)
        self.encoder_actor = TransformerEncoder_cluster(encoder_layer_actor, num_layers=config.num_encoder_layers, norm=encoder_norm_actor)

        #temporal tr encoder
        encoder_layer_temp = TransformerEncoderLayer(self.embed_features, config.Nhead, dropout=config.dropout_porb,
                                                      normalize_before=True)
        encoder_norm_temp = nn.LayerNorm(self.embed_features)
        self.encoder_temp = TransformerEncoder(encoder_layer_temp, num_layers=config.num_encoder_layers,
                                                norm=encoder_norm_temp)
        #actor decoder--first
        decoder_actor=TransformerDecoderLayer2(self.embed_features, config.Nhead, dropout=config.dropout_porb, normalize_before=True)
        decoder_norm_actor= nn.LayerNorm(self.embed_features)
        self.decoder_actor = TransformerDecoder(decoder_actor, num_layers=config.num_decoder_layers, norm=decoder_norm_actor)

        #temporal decoder
        decoder_temp=TransformerDecoderLayer2(self.embed_features,config.Nhead,dropout=config.dropout_porb, normalize_before=True)
        decoder_norm_temp=nn.LayerNorm(self.embed_features)
        self.decoder_temp=TransformerDecoder(decoder_temp,num_layers=config.num_decoder_layers, norm=decoder_norm_temp)

    #x--->N,B*T,C
    #query=1,B*T,C
    def forward(self, x,query):
        N,B,T,C=x.shape
        tgt_len,bsz,dim=query.shape
        actor_o = x.reshape(N,B*T,-1) #(N,B*T,-1)
        #N,B*T,-1
        memory_actor,loss = self.encoder_actor(actor_o)

        #T,B*N,-1
        temp_o=x.permute(2,1,0,3).contiguous().reshape(T,B*N,-1)
        #T,B*N,-1
        memory_temp=self.encoder_temp(temp_o)
        #T,B*N,-1
        memory_actor=memory_actor.reshape(N,B,T,-1).permute(2,1,0,3).contiguous().reshape(T,B*N,-1)
        #T,B*N,-1
        memory=self.decoder_actor(memory_actor,memory_temp)
        #N,B*T,-1
        memory=memory.reshape(T,B,N,-1).permute(2,1,0,3).contiguous().reshape(N,B,T,-1)
        #1,B*T,C
        tgt=self.decoder_temp(query,memory)
        tgt=tgt.reshape(tgt_len,bsz,dim)
        return memory,tgt,loss

class ST_plus_TR_block_cross(nn.Module):
    def __init__(self,config,embed_feat):
        super(ST_plus_TR_block_cross, self).__init__()

        self.embed_features = embed_feat

        #actor tr encoder
        encoder_layer_actor = TransformerEncoderLayer_cluster(self.embed_features, 8,total_size=12,window_size=3, dropout=0.1, normalize_before=True)
        encoder_norm_actor = nn.LayerNorm(self.embed_features)
        self.encoder_actor = TransformerEncoder_cluster(encoder_layer_actor, num_layers=1, norm=encoder_norm_actor)

        #temporal tr encoder
        encoder_layer_temp = TransformerEncoderLayer(self.embed_features, 8, dropout=0.1,
                                                      normalize_before=True)
        encoder_norm_temp = nn.LayerNorm(self.embed_features)
        self.encoder_temp = TransformerEncoder(encoder_layer_temp, num_layers=1,
                                                norm=encoder_norm_temp)
        #actor decoder  --first
        #actor cross decoder_1
        decoder_actor1=TransformerDecoderLayer2(self.embed_features, 8, dropout=0.1, normalize_before=True)
        decoder_norm_actor1= nn.LayerNorm(self.embed_features)
        self.decoder_actor1 = TransformerDecoder(decoder_actor1, num_layers=1, norm=decoder_norm_actor1)
        #actor cross decoder_2
        decoder_actor2=TransformerDecoderLayer2(self.embed_features, 8, dropout=0.1, normalize_before=True)
        decoder_norm_actor2= nn.LayerNorm(self.embed_features)
        self.decoder_actor2 = TransformerDecoder(decoder_actor2, num_layers=1, norm=decoder_norm_actor2)

        #temporal decoder
        decoder_temp=TransformerDecoderLayer2(self.embed_features,8,dropout=0.1, normalize_before=True)
        decoder_norm_temp=nn.LayerNorm(self.embed_features)
        self.decoder_temp=TransformerDecoder(decoder_temp,num_layers=1, norm=decoder_norm_temp)

    #x--->N,B*T,C
    #query=1,B*T,C
    def forward(self, x,query):
        N,B,T,C=x.shape
        tgt_len,bsz,dim=query.shape
        actor_o = x.reshape(N,B*T,-1) #(N,B*T,-1)
        #N,B*T,-1
        memory_actor,loss = self.encoder_actor(actor_o)

        #T,B*N,-1
        temp_o=x.permute(2,1,0,3).contiguous().reshape(T,B*N,-1)
        #T,B*N,-1
        memory_temp=self.encoder_temp(temp_o)
        #N,B*T,-1
        memory_temp=memory_temp.reshape(T,B,N,-1).permute(2,1,0,3).contiguous().reshape(N,B*T,-1)
        #N,B*T,-1
        memory1=self.decoder_actor1(memory_actor,memory_temp)
        #T,B*N,-1
        memory_actor=memory_actor.reshape(N,B,T,-1).permute(2,1,0,3).contiguous().reshape(T,B*N,-1)
        memory_temp=memory_temp.reshape(N,B,T,-1).permute(2,1,0,3).contiguous().reshape(T,B*N,-1)
        memory2=self.decoder_actor2(memory_actor,memory_temp)         

        # N,B,T,-1
        memory1=memory1.reshape(N,B,T,-1)
        memory2 = memory2.reshape(T,B,N,-1).permute(2,1,0,3).contiguous().reshape(N,B,T,-1)
        memory = memory1 + memory2
        #1,B*T,C
        tgt=self.decoder_temp(query,memory)
        tgt=tgt.reshape(tgt_len,bsz,dim) 
        return memory,tgt,loss

class ST_plus_TR_cross_cluster(nn.Module):
    def __init__(self,config):
        super(ST_plus_TR_cross_cluster, self).__init__()
        self.num_STTR_layers=3#config.num_STTR_layers

        embed_feat=256#config.embed_features
        self.actions_num_classes = 9#config.actions_num_classes
        self.activities_num_classes =8# config.activities_num_classes
        self.embed_features=embed_feat
        self.STTR_module=nn.ModuleList()

        self.group_query = nn.Parameter(torch.randn(embed_feat), requires_grad=True) 
        group_tr_layer = TransformerDecoderLayer2(embed_feat, 8 , dropout=0.1, normalize_before=True)
        group_tr_norm = nn.LayerNorm(embed_feat)
        self.group_tr = TransformerDecoder(group_tr_layer, num_layers=2, norm=group_tr_norm)

        for i in range(self.num_STTR_layers):
            self.STTR_module.append(ST_plus_TR_block_cross(config,embed_feat))

        self.dropout = nn.Dropout(p=0.5)
        self.activities_fc = nn.Linear(self.embed_features, self.activities_num_classes)
        self.actions_fc = nn.Linear(self.embed_features, self.actions_num_classes)
    def forward(self,x,global_token):
        B, T, N, F = x.shape
        #N,B,T,C_e
        x = x.permute(2,0,1,3)
        #1,B*T,C_e
        #token=self.token.repeat(B*T,1).reshape(1,B*T,-1)
        token=global_token.mean(dim=0,keepdim=True)
        token=token.reshape(1,B*T,-1)
        #B=N,B*T,C_e
        memory=x
        #1,B*T,C_e
        group_query = self.group_query.reshape(1,1,-1).repeat(1,B*T,1)
        group = self.group_tr(group_query,x).reshape(1,B*T,-1)
        tgt=token + group

        cluster_losses = []
        for i in range(self.num_STTR_layers):
            memory, tgt, t_loss = self.STTR_module[i](memory, tgt)
            cluster_losses.append(t_loss)
        cluster_loss = sum(cluster_losses)

        #N,B,T,-1
        memory=memory.reshape(N,B,T,-1).permute(1,2,0,3).reshape(-1,self.embed_features)
        #B*T,-1
        tgt=tgt.reshape(-1,self.embed_features)

        memory = self.dropout(memory)
        tgt = self.dropout(tgt)        

        actions_scores=self.actions_fc(memory)
        actions_scores = actions_scores.reshape(B, T, N, -1)
        actions_scores = actions_scores.mean(dim=1).reshape(B * N, -1)

        activities_scores=self.activities_fc(tgt)
        activities_scores=activities_scores.reshape(B,T,-1).mean(dim=1)

        return actions_scores, activities_scores, cluster_loss
