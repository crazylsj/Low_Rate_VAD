import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from stftdatasets import DASN_VAD_Dataset
from params import HParams
from DASN_model import TCN, add_loss,  TCN, Decoder_sigmoid,add_loss_noregular_sigmoid,node_fusion_v2,TCNImpNet
import argparse
from sklearn.metrics import  confusion_matrix
import random
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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



def test_fc_epoch(encoder_mel_phase, tcn,MelPhaseChoice, decoder,  test_loader, loss_fn=add_loss,is_spatial=False):

    total_loss = 0
    total_f1 = 0
    total_precision = 0
    total_recall = 0
    total_miss_rate = 0
    total_false_alarm_rate = 0
    total_SER = 0
    counter = 0
    total_tp = total_tn = total_fp = total_fn = 0
    Loss_Im = nn.L1Loss()   
    loss_ims = 0

    for STFTs, source_feat, source_label, c0clean_feat , c0_label,  features, moc_locs,sources_locs in tqdm(test_loader):
        source_feat = source_feat.to(DEVICE)
        source_label = source_label.to(DEVICE)
        c0clean_feat = c0clean_feat.to(DEVICE)
        c0_label = c0_label.to(DEVICE)
        features = features.to(DEVICE)
        STFTs = STFTs.to(DEVICE)

        
        if is_spatial:
            phase = torch.angle(STFTs)
            phase = phase[:, :, 1:, :]
            cos_phase = torch.cos(phase)  
            sin_phase = torch.sin(phase)
            MelPha = torch.cat((features,sin_phase,cos_phase),dim=-2)
            MPh, m_imp  = encoder_mel_phase(MelPha.squeeze(0))
            Melphdq_x, Melphchannels_weights = MelPhaseChoice(MPh.permute(2, 0, 1).unsqueeze(0))
            cw = Melphchannels_weights.squeeze(0).squeeze(-1).permute(1, 0) 
            wMelphF = (Melphdq_x  * Melphchannels_weights).sum(dim=-2).permute(0, 2, 1) 
        else:  
            MPh, m_imp  = encoder_mel_phase(features.squeeze(0))
            Melphdq_x, Melphchannels_weights = MelPhaseChoice(MPh.permute(2, 0, 1).unsqueeze(0))
            cw = Melphchannels_weights.squeeze(0).squeeze(-1).permute(1, 0) 
            wMelphF = (Melphdq_x  * Melphchannels_weights).sum(dim=-2).permute(0, 2, 1)  

        tcn_embfeddings = tcn(wMelphF) 
        postnet_output = decoder(tcn_embfeddings)
        postnet_output = postnet_output.squeeze(1)
        loss_imp = Loss_Im(m_imp, cw.detach()) 
        loss_ims +=loss_imp
        clsloss, _ = loss_fn(source_label, postnet_output)
        total_loss += clsloss.detach().item() 
        y_pred = (postnet_output > 0.50).int().cpu().numpy().flatten()
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

    print('Loss: {:.4f}, F1-score: {:.4f}, Precision: {:.4f}, Recall: {:.4f},  SER: {:.4f}, Miss Rate: {:.4f}, False Alarm Rate: {:.4f}'.format(
        total_loss / counter, total_f1, total_precision, total_recall , total_SER , total_miss_rate , total_false_alarm_rate )) 
    return total_SER, total_f1

  

def parse_args():
    parser = argparse.ArgumentParser(description='Tr-VAD Training Stage')
    parser.add_argument('--test_data', type=str, 
                    default="data/LibriSpeechConcat_2000_node10/test-clean/Features",
                    help='Path to the testing metadata')
    parser.add_argument('--loc_test_data', type=str, 
                    default="data/LibriSpeechConcat_2000_node10/test-clean/Waveforms",
                    help='Path to the testing metadata')
    parser.add_argument(
        '--save_dir',
        type=str,
        default='checkpoint/',
        help='Directory to save model checkpoints'
    )
    parser.add_argument('--resume_train', 
                        action='store_true', help='whether continue training using checkpoint')
    return parser.parse_args()




if __name__ == '__main__':

    args = parse_args()
    hparams = HParams()
    DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    set_seed(42)
    test_da = args.test_data
    loc_test_da = args.loc_test_data
    save_dir = args.save_dir
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
    is_spatial = True
    if is_spatial:
        encoder_mel_phase = TCNImpNet(40+256+256, 1, 1, 3, 1, 128, 128).to(DEVICE) 
    else:
        encoder_mel_phase = TCNImpNet(40, 1, 1, 3, 1, 128, 128).to(DEVICE) 

    tcn = TCN(128, 1, 1, 3, 3, 128, 128).to(DEVICE)
    MelPhaseChoice  = node_fusion_v2(input_dim=128,att_dim=128).to(DEVICE)
    decoder = Decoder_sigmoid(dim = 128).to(DEVICE)
 
    best_epoch = 14
    best_performance = 82.74
    
    melphaseencoder_checkpoint_path = r'checkpoint/DASN/node10/stage1/alpha1/melPhase_encoderweights_28_SER_0.83.pth'
    tcn_checkpoint_path = f'{save_dir}/tcnweights_{best_epoch}_SER_{best_performance}.pth'
    melphasechoice_checkpoint_path = f'{save_dir}/MelPhasechoiceweights_{best_epoch}_SER_{best_performance}.pth'
    decoder_checkpoint_path = f'{save_dir}/decoderweights_{best_epoch}_SER_{best_performance}.pth'
    

    encoder_mel_phase.load_state_dict(torch.load(melphaseencoder_checkpoint_path))
    tcn.load_state_dict(torch.load(tcn_checkpoint_path))
    MelPhaseChoice.load_state_dict(torch.load(melphasechoice_checkpoint_path))
    decoder.load_state_dict(torch.load(decoder_checkpoint_path))
   
    encoder_mel_phase.eval()  
    tcn.eval()  
    MelPhaseChoice.eval()  
    decoder.eval()  
   

    with torch.no_grad():
        fc_SER, fc_F1 = test_fc_epoch(encoder_mel_phase, tcn, MelPhaseChoice, decoder, test_loader, loss_fn=add_loss_noregular_sigmoid,is_spatial=is_spatial)
    print(f'fc_SER:{fc_SER}, fc_F1:{fc_F1}')






 





