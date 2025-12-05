import torch
from tqdm import tqdm
import os
import logging
from torch.utils.data import DataLoader
from stftdatasets import DASN_VAD_Dataset
from params import HParams
from DASN_model import TCN, add_loss, TCN,Decoder_sigmoid, add_loss_noregular_sigmoid,node_fusion_v2,TCNImpNet
import argparse
from sklearn.metrics import  confusion_matrix
import random
import numpy as np
import torch.nn.functional as F
from torch.distributions import normal

eps = 1e-12
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)        
    random.seed(seed)                                 
    np.random.seed(seed)                            
    torch.manual_seed(seed)                        
    torch.cuda.manual_seed(seed)                     
    torch.cuda.manual_seed_all(seed)                   
    torch.backends.cudnn.deterministic = True        
    torch.backends.cudnn.benchmark = False          

def hard_vq(input_data, codebooks):
    distances = (torch.sum(input_data ** 2, dim=1, keepdim=True)
                 - 2 * (torch.matmul(input_data, codebooks.t()))
                 + torch.sum(codebooks.t() ** 2, dim=0, keepdim=True))
    min_indices = torch.argmin(distances, dim=1)
    quantized_input = codebooks[min_indices]
    remainder = input_data - quantized_input
    return quantized_input, remainder, min_indices



def noise_substitution_vq(input_data, hard_quantized_input):
    normal_dist = normal.Normal(0, 1)
    random_vector = normal_dist.sample(input_data.shape).to(input_data.device)
    norm_hard_quantized_input = (input_data - hard_quantized_input).square().sum(dim=1, keepdim=True).sqrt()
    norm_random_vector = random_vector.square().sum(dim=1, keepdim=True).sqrt()
    vq_error = ((norm_hard_quantized_input / norm_random_vector + eps) * random_vector)
    quantized_input = input_data + vq_error
    return quantized_input



def vq_compress(input_data, codebooks):
    remainder = input_data
    indices_list = []
    codelen = int(codebooks.shape[0])

    for i in range(codelen):
        codebook = codebooks[i]  
        distances = (torch.sum(remainder ** 2, dim=1, keepdim=True)
                     - 2 * torch.matmul(remainder, codebook.t())
                     + torch.sum(codebook.t() ** 2, dim=0, keepdim=True))
        indices = torch.argmin(distances, dim=1) 
        quantized = codebook[indices]  
        remainder = remainder - quantized
        indices_list.append(indices)


    return indices_list


def vq_decompress(indices_list, codebooks):
    quantized_list = []
    for i, indices in enumerate(indices_list):
        codebook = codebooks[i]
        quantized = codebook[indices]  
        quantized_list.append(quantized)
    final_quantized = sum(quantized_list)  
    return final_quantized.detach(),quantized_list  


def per_frame_sqrt_stage(scores, low_stage=2, high_stage=28, score_min=0.0, score_max=1/10):
    norm_scores = torch.clamp((scores - score_min) / (score_max - score_min), 0.0, 1.0)
    weights = torch.sqrt(norm_scores)
    float_stages = low_stage + weights * (high_stage - low_stage)
    stages = torch.floor(float_stages).to(torch.int)
    return stages


def vq_compress_with_stage_map(input_data, codebooks, stage_nums):

    T, D = input_data.shape
    S = codebooks.shape[0]
    remainder = input_data.clone()
    indices_list = []
    quantized_list = []

    for i in range(S):
      
        valid_mask = stage_nums > i  
        if not valid_mask.any():
            break

        codebook = codebooks[i] 
        remainder_valid = remainder[valid_mask]  

        distances = (torch.sum(remainder_valid ** 2, dim=1, keepdim=True)
                     - 2 * torch.matmul(remainder_valid, codebook.t())
                     + torch.sum(codebook.t() ** 2, dim=0, keepdim=True))  

        indices_valid = torch.argmin(distances, dim=1) 
        quantized_valid = codebook[indices_valid]       
        indices = torch.full((T,), -1, dtype=torch.long, device=input_data.device)
        indices[valid_mask] = indices_valid
        quantized = torch.zeros_like(input_data)
        quantized[valid_mask] = quantized_valid
        remainder = remainder - quantized
        indices_list.append(indices)
        quantized_list.append(quantized)

    return indices_list, quantized_list

def test_fc_epoch(encoder_mel_phase, tcn,MelPhaseChoice,melphasecodebook, decoder,  test_loader, loss_fn=add_loss):


    total_loss = 0
    total_f1 = 0
    total_precision = 0
    total_recall = 0
    total_miss_rate = 0
    total_false_alarm_rate = 0
    total_SER = 0
    meltotal_vq_loss = 0
    ph_vq_loss = 0
    counter = 0
    total_tp = total_tn = total_fp = total_fn = 0

    for STFTs, source_feat, source_label, c0clean_feat , c0_label,  features, moc_locs,sources_locs in tqdm(test_loader):
        source_feat = source_feat.to(DEVICE)
        source_label = source_label.to(DEVICE)
        c0clean_feat = c0clean_feat.to(DEVICE)
        c0_label = c0_label.to(DEVICE)
        features = features.to(DEVICE)
        STFTs = STFTs.to(DEVICE)
        phase = torch.angle(STFTs)
        phase = phase[:, :, 1:, :]
        cos_phase = torch.cos(phase)  
        sin_phase = torch.sin(phase)
        MelPha = torch.cat((features,sin_phase,cos_phase),dim=-2)
        MPh ,ph = encoder_mel_phase(MelPha.squeeze(0))
    

        stages_map = per_frame_sqrt_stage(ph,high_stage=args.codebook,score_max=1/args.channel)
        melPhaseF_q_list_difstage = []
        for i in range(args.channel):
            stage_nums = stages_map[i] 
 
            indices_list, quantized_list = vq_compress_with_stage_map(MPh[i].T, melphasecodebook, stage_nums)
            MelPhaseF_quantized = sum(quantized_list) 
            melPhaseF_q_list_difstage.append(MelPhaseF_quantized)
        MelPhaseF_quantized_dif_all = torch.stack(melPhaseF_q_list_difstage,dim=0)
        dq_x_dif, channels_weights_dif = MelPhaseChoice(MelPhaseF_quantized_dif_all.permute(1,0,2).unsqueeze(0))
        wMelF = (dq_x_dif  * channels_weights_dif).sum(dim=-2).permute(0, 2, 1)  

        tcn_embfeddings = tcn(wMelF) 
        postnet_output = decoder(tcn_embfeddings)

        postnet_output = postnet_output.squeeze(1)

        clsloss, _ = loss_fn(source_label, postnet_output)
        total_loss += clsloss.detach().item() 


        y_pred = (postnet_output > 0.5).int().cpu().numpy().flatten()
        y_true = source_label.int().cpu().numpy().flatten()    
        counter += 1   
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        total_tp += tp
        total_tn += tn
        total_fp += fp
        total_fn += fn

    total_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    total_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0

    total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall) if (total_precision + total_recall) > 0 else 0
    total_miss_rate = total_fn / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    total_false_alarm_rate = total_fp / (total_fp + total_tn) if (total_fp + total_tn) > 0 else 0

    total_SER = total_miss_rate + total_false_alarm_rate
    meltotal_vq_loss = meltotal_vq_loss/counter
    ph_vq_loss = ph_vq_loss/counter
    print('Loss: {:.4f}, F1-score: {:.4f}, Precision: {:.4f}, Recall: {:.4f},  SER: {:.4f}, Miss Rate: {:.4f}, False Alarm Rate: {:.4f}'.format(
        total_loss / counter, total_f1, total_precision, total_recall , total_SER , total_miss_rate , total_false_alarm_rate )) 
    return total_SER, total_f1

    
def test_fc_epoch_phase(encoder, vq_mel,vq_phase,melcodebook,vq_phasecodebook,
                                  tcn, nodeChoice, ipdChoice, decoder,  test_loader, loss_fn=add_loss):

    total_loss = 0
    total_f1 = 0
    total_precision = 0
    total_recall = 0
    total_miss_rate = 0
    total_false_alarm_rate = 0
    total_SER = 0
    
    meltotal_vq_loss = 0
    ph_vq_loss = 0
    
    counter = 0
    total_tp = total_tn = total_fp = total_fn = 0
    
    for STFTs, source_feat, source_label, c0clean_feat , c0_label,  features, moc_locs,sources_locs in tqdm(test_loader):
        source_feat = source_feat.to(DEVICE)
        source_label = source_label.to(DEVICE)
        c0clean_feat = c0clean_feat.to(DEVICE)
        c0_label = c0_label.to(DEVICE)
        features = features.to(DEVICE)

        STFTs = STFTs.to(DEVICE)
        MelF = encoder(features.squeeze(0))
        real = STFTs.real
        imag = STFTs.imag
        phase = torch.atan2(imag, real) 
      
        melF_q_list = []
        phase_q_list = []

        for i in range(args.channel):
       
            mel_indices = vq_compress(MelF[i].T,melcodebook)
            MelF_quantized = vq_decompress(mel_indices,melcodebook)
            phase_indices = vq_compress(phase.squeeze(0)[i].T,vq_phasecodebook)
            phase_quantized = vq_decompress(phase_indices,vq_phasecodebook)

            melF_q_list.append(MelF_quantized)
            phase_q_list.append(phase_quantized)
        MelF_quantized_all = torch.stack(melF_q_list,dim=0)
        phase_quantized_all  = torch.stack(phase_q_list,dim=0)
        melvq_loss = F.mse_loss(MelF, MelF_quantized_all.permute(0,2,1))
        phasevq_loss = F.mse_loss(phase.squeeze(0), phase_quantized_all.permute(0,2,1))
        dq_x, channels_weights = nodeChoice(MelF_quantized_all.permute(1,0,2).unsqueeze(0))
        wMelF = (dq_x  * channels_weights).sum(dim=-2).permute(0, 2, 1)  
        phase_quantized_all = phase_quantized_all.permute(0, 2, 1).unsqueeze(0)    
        B, C, f, T = phase_quantized_all.shape              
        phase_perm = phase_quantized_all.permute(0, 3, 1, 2)  

        max_idx = torch.argmax(channels_weights.squeeze(-1), dim=2)  

        index_max_exp = max_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, f) 
        max_phase = torch.gather(phase_perm, dim=2, index=index_max_exp)        

        max_phase_exp = max_phase.expand(-1, -1, C, -1)  
        phase_diff_all = max_phase_exp - phase_perm      

        mask = torch.arange(C, device=phase.device).view(1, 1, C) != max_idx.unsqueeze(-1)  
        mask = mask.unsqueeze(-1).expand(-1, -1, -1, f)  

        phase_diff = phase_diff_all[mask].view(B, T, C - 1, f)  
        ipd, ipdweight = ipdChoice(phase_diff)
        pfeatures = (ipd  * ipdweight).sum(dim=-2).permute(0, 2, 1)  

        finnalF = torch.cat((wMelF,pfeatures),dim=1)
        tcn_embfeddings = tcn(finnalF)
        postnet_output = decoder(tcn_embfeddings)

        postnet_output = postnet_output.squeeze(1)
        clsloss, _ = loss_fn(encoder, source_label, postnet_output)
        total_loss = clsloss.detach().item() +  + melvq_loss.detach().item()+ phasevq_loss.detach().item()
        meltotal_vq_loss += melvq_loss.detach().item()
        ph_vq_loss += phasevq_loss.detach().item()

        y_pred = (postnet_output > 0.95).int().cpu().numpy().flatten()
        y_true = source_label.int().cpu().numpy().flatten()    
        counter += 1   
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        total_tp += tp
        total_tn += tn
        total_fp += fp
        total_fn += fn



    total_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    total_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0

    total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall) if (total_precision + total_recall) > 0 else 0
    total_miss_rate = total_fn / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    total_false_alarm_rate = total_fp / (total_fp + total_tn) if (total_fp + total_tn) > 0 else 0

    total_SER = total_miss_rate + total_false_alarm_rate
    meltotal_vq_loss = meltotal_vq_loss/counter
    ph_vq_loss = ph_vq_loss/counter

    print('Loss: {:.4f}, F1-score: {:.4f}, Precision: {:.4f}, Recall: {:.4f},  SER: {:.4f}, Miss Rate: {:.4f}, False Alarm Rate: {:.4f}'.format(
        total_loss / counter, total_f1, total_precision, total_recall , total_SER , total_miss_rate , total_false_alarm_rate )) 
    return total_SER, total_f1

def parse_args():
    parser = argparse.ArgumentParser(description='Tr-VAD Training Stage')
    parser.add_argument('--test_data', type=str, 
                    default="data/LibriSpeechConcat_2000_node5/test-clean/Features",
                    help='Path to the testing metadata')
    parser.add_argument('--loc_test_data', type=str, 
                    default="data/LibriSpeechConcat_2000_node5/test-clean/Waveforms",
                    help='Path to the testing metadata')
    parser.add_argument('--resume_train', 
                        action='store_true', help='whether continue training using checkpoint')
    parser.add_argument(
        '--stgae1_save_dir',
        type=str,
        default='checkpoint/DASN/node5/stage1/alpha1',
        help='Directory to save model checkpoints'
    )
    parser.add_argument(
        '--stgae2_save_dir',
        type=str,
        default='checkpoint/DASN/node5/Bit_allocation',
        help='Directory to save model checkpoints'
    )
    parser.add_argument(
        '--channel',
        type=int,
        default=5,               
        choices=[5, 10, 20],
        help='The numbers of the nodes'
    )
    parser.add_argument(
        '--codebook',
        type=int,
        default=28,              
        help='The numbers of B'
    )
    return parser.parse_args()



if __name__ == '__main__':
 

    args = parse_args()
    hparams = HParams()
    DEVICE = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    set_seed(42)
    test_da = args.test_data
    loc_test_da = args.loc_test_data

    embedding_dim = 128

    w, u = hparams.w, hparams.u
    winlen = hparams.winlen
    winstep = hparams.winstep
    fft = hparams.n_fft
    batch_size = hparams.batch_size
   
    valid_dataset = DASN_VAD_Dataset(test_da,loc_test_da)

    test_loader = DataLoader(valid_dataset, 1, False,
                             num_workers=0, pin_memory=True)

    torch.cuda.empty_cache()
 

    encoder_mel_phase = TCNImpNet(40+256+256, 1, 1, 3, 1, 128, 128).to(DEVICE) 
    tcn = TCN(128, 1, 1, 3, 3, 128, 128).to(DEVICE)
    MelPhaseChoice  = node_fusion_v2(input_dim=128,att_dim=128).to(DEVICE)
    decoder = Decoder_sigmoid(dim = 128).to(DEVICE)
    best_epoch = 7
    best_performance = 78.28

    
    melphaseencoder_checkpoint_path = f'{args.stgae1_save_dir}/melPhase_encoderweights_26_SER_0.79.pth'
    tcn_checkpoint_path = f'{args.stgae2_save_dir}/tcnweights_{best_epoch}_SER_{best_performance:.2f}.pth'
    melphasechoice_checkpoint_path = f'{args.stgae2_save_dir}/MelPhasechoiceweights_{best_epoch}_SER_{best_performance:.2f}.pth'
    decoder_checkpoint_path = f'{args.stgae2_save_dir}/decoderweights_{best_epoch}_SER_{best_performance:.2f}.pth'
    melphasecodebook_pt = f'{args.stgae2_save_dir}/melphasecodebooks_{best_epoch}_SER_0.24.pt'
   

    encoder_mel_phase.load_state_dict(torch.load(melphaseencoder_checkpoint_path))
    tcn.load_state_dict(torch.load(tcn_checkpoint_path))
    MelPhaseChoice.load_state_dict(torch.load(melphasechoice_checkpoint_path))
    decoder.load_state_dict(torch.load(decoder_checkpoint_path))
    melphasecodebook = torch.load(melphasecodebook_pt, map_location=DEVICE, weights_only=True)  # 或 "cuda"
    if isinstance(melphasecodebook, torch.nn.Parameter):
        melphasecodebook = melphasecodebook.detach().to(DEVICE) 

   
    encoder_mel_phase.eval()  
    tcn.eval()  
    MelPhaseChoice.eval() 
    decoder.eval() 
   

    with torch.no_grad():
        fc_SER, fc_F1 = test_fc_epoch(encoder_mel_phase, tcn, MelPhaseChoice, melphasecodebook, decoder, test_loader, loss_fn=add_loss_noregular_sigmoid)
    print(f'fc_SER:{fc_SER}, fc_F1:{fc_F1}')






 





