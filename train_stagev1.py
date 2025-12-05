import torch
from tqdm import tqdm
import os
import logging
from torch.utils.data import DataLoader
from stftdatasets import DASN_VAD_Dataset
from torch import nn
from params import HParams
from DASN_model import Decoder_sigmoid, TCN,add_loss_noregular_sigmoid,node_fusion_v2,TCNImpNet
from timm.scheduler.cosine_lr import CosineLRScheduler
import numpy as np
import argparse
from sklearn.metrics import  confusion_matrix
import random

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)       # 固定 Python hash 种子
    random.seed(seed)                               # 固定 random 库
    np.random.seed(seed)                            # 固定 numpy
    torch.manual_seed(seed)                         # 固定 CPU 上的 torch
    torch.cuda.manual_seed(seed)                    # 固定当前 GPU
    torch.cuda.manual_seed_all(seed)                # 固定所有 GPU（如果使用多卡）
    
    torch.backends.cudnn.deterministic = True       # 确保每次卷积等操作都一样
    torch.backends.cudnn.benchmark = False          # 禁止 cuDNN 自动优化算法（会引入随机性）

EPS = 1e-6
def mvn(x, dim=-1) -> torch.Tensor:
    """
    Performs mean-variance normalization on a given tensor
    """
    x_norm = (x - torch.mean(x, dim=dim, keepdim=True)) / (
        torch.std(
            x,
            dim=dim,
            keepdim=True,
        )
        + EPS
    )
    return x_norm

def FC(x):
    magnitude = torch.abs(x)            
    phase = torch.angle(x)   

    return magnitude,phase


def train_epoch(encoder_mel_phase, tcn,MelPhaseChoice,  decoder,  train_loader, loss_fn, optimizer, scheduler, batch_size, epoch, start_step,is_spatial,alpha,save_dir):

    step = start_step
    total_loss = 0
    total_f1 = 0
    total_precision = 0
    total_recall = 0
    total_miss_rate = 0
    total_false_alarm_rate = 0
    total_SER = 0
    Loss_Im = nn.L1Loss()    
    num_batches = 0
    loss_imps = 0
    total_tp = total_tn = total_fp = total_fn = 0
    

    for batch_index, batch in enumerate(train_loader):
        STFTs,source_feat, source_label, c0clean_feat , c0_label,  features, moc_locs, sources_locs = batch  
        STFTs = STFTs.to(DEVICE)
        source_feat = source_feat.to(DEVICE)
        source_label = source_label.to(DEVICE)
        c0clean_feat = c0clean_feat.to(DEVICE)
        c0_label = c0_label.to(DEVICE)
        features = features.to(DEVICE) 
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

        loss_imp = Loss_Im(m_imp, cw.detach())  
 

        tcn_embfeddings = tcn(wMelphF) 
        postnet_output = decoder(tcn_embfeddings)
        postnet_output = postnet_output.squeeze(1) 
        cls_loss, _ = loss_fn(source_label, postnet_output)
        loss = cls_loss + alpha*loss_imp 
        loss_imps += alpha*loss_imp
        total_loss +=  loss.detach().item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step_update(step)

        y_pred = (postnet_output > 0.5).int().cpu().numpy().flatten()
        
        y_true = source_label.int().cpu().numpy().flatten()
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        total_tp += tp
        total_tn += tn
        total_fp += fp
        total_fn += fn
        num_batches += 1
        total_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        total_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0    
        total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall) if (total_precision + total_recall) > 0 else 0
        total_miss_rate = total_fn / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        total_false_alarm_rate = total_fp / (total_fp + total_tn) if (total_fp + total_tn) > 0 else 0
        total_SER = total_miss_rate + total_false_alarm_rate

        step += 1
        if step % 10 == 0:
            logging.info('E:{}, S: {}, overall loss: {:.4f}, loss_imp:{:.4f},F1-score: {:.4f}, SER: {:.4f}'.format(
                epoch, step, total_loss / num_batches, loss_imps / num_batches,total_f1, total_SER))
            
    total_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    total_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0    
    total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall) if (total_precision + total_recall) > 0 else 0
    total_miss_rate = total_fn / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    total_false_alarm_rate = total_fp / (total_fp + total_tn) if (total_fp + total_tn) > 0 else 0
    total_SER = total_miss_rate + total_false_alarm_rate
    return total_loss / num_batches, total_f1, total_miss_rate, total_false_alarm_rate, step

def test_epoch(encoder_mel_phase , tcn,MelPhaseChoice, decoder, test_loader, loss_fn,is_spatial,alpha,save_dir):
    logging.info("Validation Begins...")
    counter = 0
    total_loss = 0
    total_f1 = 0
    total_precision = 0
    total_recall = 0
    total_miss_rate = 0
    total_false_alarm_rate = 0
    encoder_mel_phase = encoder_mel_phase.to(DEVICE)
    tcn = tcn.to(DEVICE)
    MelPhaseChoice = MelPhaseChoice.to(DEVICE)

    decoder = decoder.to(DEVICE)
    total_tp = total_tn = total_fp = total_fn = 0
    IPD_M = True
    Loss_Im =  nn.L1Loss()
    loss_imps = 0



    for  STFTs,source_feat, source_label, c0clean_feat , c0_label,  features, moc_locs,sources_locs  in tqdm(test_loader):
        STFTs = STFTs.to(DEVICE)
        source_feat = source_feat.to(DEVICE)
        source_label = source_label.to(DEVICE)
        c0clean_feat = c0clean_feat.to(DEVICE)
        c0_label = c0_label.to(DEVICE)
        features = features.to(DEVICE)

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

        loss_imp = Loss_Im(m_imp, cw.detach()) 

        tcn_embfeddings = tcn(wMelphF) 
        postnet_output = decoder(tcn_embfeddings)
        postnet_output = postnet_output.squeeze(1) 
        loss_imps+=loss_imp.detach().item() 
        loss, _ = loss_fn(source_label, postnet_output)
        total_loss += loss.detach().item() 
        y_pred = (postnet_output > 0.5).int().cpu().numpy().flatten()
        y_true = source_label.int().cpu().numpy().flatten()
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        total_tp += tp
        total_tn += tn
        total_fp += fp
        total_fn += fn
        counter += 1

    total_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    total_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0    
    total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall) if (total_precision + total_recall) > 0 else 0
    total_miss_rate = total_fn / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    total_false_alarm_rate = total_fp / (total_fp + total_tn) if (total_fp + total_tn) > 0 else 0
    total_SER = total_miss_rate + total_false_alarm_rate
    print("Validation Finished...")
    print('Loss: {:.4f}, imLoss: {:.4f}, F1-score: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, Miss Rate: {:.4f}, False Alarm Rate: {:.4f}'.format(
        total_loss / counter, loss_imps / counter, total_f1, total_precision, total_recall,
        total_miss_rate, total_false_alarm_rate))
    
    return total_loss / counter, total_f1 , total_miss_rate, total_false_alarm_rate

def train(encoder_mel_phase , tcn,MelPhaseChoice, decoder,  train_loader, test_loader, loss_fn, optimizer, scheduler, batch_size, start_epoch=0, epochs=20, start_step=0,is_spatial=True,alpha=1,save_dir=""):

    os.makedirs("checkpoint", exist_ok=True)
    SER = 0.0
    best_f1_ser = 0

    for e in range(start_epoch, epochs):
        train_losses = []
        test_losses = []
        train_f1s = []
        test_f1s = []
        train_miss_rates = []
        test_miss_rates = []
        train_false_alarm_rates = []
        test_false_alarm_rates = []
   
        train_loss, train_f1, train_miss_rate, train_false_alarm_rate , start_step = train_epoch(encoder_mel_phase , tcn,MelPhaseChoice,  decoder,  train_loader, loss_fn, optimizer, scheduler, batch_size, e, start_step,is_spatial,alpha,save_dir)
        train_losses.append(train_loss)
        train_f1s.append(train_f1)
        train_miss_rates.append(train_miss_rate)
        train_false_alarm_rates.append(train_false_alarm_rate)

        with torch.inference_mode():
            test_loss, test_f1, test_miss_rate, test_false_alarm_rate = test_epoch(encoder_mel_phase , tcn,MelPhaseChoice,  decoder,  test_loader, loss_fn,is_spatial,alpha,save_dir)

        test_losses.append(test_loss)
        test_f1s.append(test_f1)
        test_miss_rates.append(test_miss_rate)
        test_false_alarm_rates.append(test_false_alarm_rate)

        SER = test_miss_rate + test_false_alarm_rate

        print("Epoch: {}/{}... Train Loss: {:.5f}, Train F1: {:.4f}, Train Miss Rate: {:.4f}, Train False Alarm Rate: {:.4f}, Train SER: {:.4f}, "
        "Test Loss: {:.5f}, Test F1: {:.4f}, Test Miss Rate: {:.4f}, Test False Alarm Rate: {:.4f}, Test SER: {:.4f}".format(
            e + 1, epochs, train_loss, train_f1, train_miss_rate, train_false_alarm_rate, train_miss_rate + train_false_alarm_rate, 
            test_loss, test_f1, test_miss_rate, test_false_alarm_rate, SER))
      

        if test_f1 - SER >=  best_f1_ser:
            best_f1_ser = test_f1 - SER
         
            torch.save(encoder_mel_phase.state_dict(), f'{save_dir}/melPhase_encoderweights_{e + 1}_SER_{best_f1_ser:.2f}.pth')
            torch.save(tcn.state_dict(), f'{save_dir}/tcnweights_{e + 1}_SER_{best_f1_ser:.2f}.pth')
            torch.save(MelPhaseChoice.state_dict(), f'{save_dir}/MelPhasechoiceweights_{e + 1}_SER_{best_f1_ser:.2f}.pth')
            torch.save(decoder.state_dict(), f'{save_dir}/decoderweights_{e + 1}_SER_{best_f1_ser:.2f}.pth')

    return train_losses, test_losses, train_f1s, test_f1s, train_miss_rates, test_miss_rates, train_false_alarm_rates, test_false_alarm_rates

def build_scheduler(optimizer, n_iter_step=1000, total_epoch=5., warmup=0.5):
    num_steps = int(total_epoch * n_iter_step)
    warmup_steps = int(warmup * n_iter_step)

    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=num_steps,
        lr_min=hparams.lr_min,
        warmup_lr_init=hparams.warmup_lr_init,
        warmup_t=warmup_steps,
        cycle_limit=1,
        t_in_epochs=False,
    )
    return lr_scheduler


def parse_args():
    parser = argparse.ArgumentParser(description='Tr-VAD Training Stage')
    parser.add_argument('--train_data', type=str, 
                        default="data/LibriSpeechConcat_2000_node5/train-clean-100/Features",
                        help='Path to the training metadata')
    parser.add_argument('--loc_train_data', type=str, 
                    default="data/LibriSpeechConcat_2000_node5/train-clean-100/Waveforms",
                    help='Path to the testing metadata')
    parser.add_argument('--test_data', type=str, 
                    default="data/LibriSpeechConcat_2000_node5/dev-clean/Features",
                    help='Path to the testing metadata')
    parser.add_argument('--loc_test_data', type=str, 
                    default="data/LibriSpeechConcat_2000_node5/dev-clean/Waveforms",
                    help='Path to the testing metadata')
    parser.add_argument(
        '--alpha',
        type=float,
        default=10,               
        choices=[0.01, 0.1, 1, 10,100],
        help='Weighting factor alpha'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default='checkpoint/DASN/node5/stage1/alpha10',
        help='Directory to save model checkpoints'
    )
    parser.add_argument('--resume_train', 
                        action='store_true', help='whether continue training using checkpoint')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoint_DSAN_alignment/cosine_similarity/single_node_train_0.5alph_vq_clean/weights_10_acc_97.09.pth',
                        help='Path to the checkpoint file, if `resume_train` is set, then resume training from the checkpoint')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    hparams = HParams()
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    set_seed(42)
    RESUME = args.resume_train
    train_da = args.train_data
    loc_train_da = args.loc_train_data
    test_da = args.test_data
    loc_test_da = args.loc_test_data

    alpha = args.alpha
    save_dir = args.save_dir
    w, u = hparams.w, hparams.u
    winlen = hparams.winlen
    winstep = hparams.winstep
    fft = hparams.n_fft
    batch_size = hparams.batch_size

    train_dataset = DASN_VAD_Dataset(train_da,loc_train_da)
    valid_dataset = DASN_VAD_Dataset(test_da,loc_test_da)

    train_loader = DataLoader(train_dataset, 1, True,
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(valid_dataset, 1, False,
                             num_workers=4, pin_memory=True)

    torch.cuda.empty_cache()
    is_spatial = True
    if is_spatial:
        encoder_mel_phase = TCNImpNet(40+256+256, 1, 1, 3, 1, 128, 128).to(DEVICE) 
    else:
        encoder_mel_phase = TCNImpNet(40, 1, 1, 3, 1, 128, 128).to(DEVICE) 
    tcn = TCN(128, 1, 1, 3, 3, 128, 128).to(DEVICE)
    MelPhaseChoice  = node_fusion_v2(input_dim=128,att_dim=128).to(DEVICE)
    decoder = Decoder_sigmoid(dim = 128).to(DEVICE)


    optimizer = torch.optim.AdamW(
        list(encoder_mel_phase.parameters()) + 
        list(tcn.parameters()) + 
        list(MelPhaseChoice.parameters()) +
        list(decoder.parameters()),
        lr=hparams.lr,
        weight_decay=hparams.weight_decay
    )

    scheduler = build_scheduler(
        optimizer, n_iter_step=hparams.n_iter_step, total_epoch=hparams.epochs // 2, warmup=hparams.warmup_factor)

    train_losses, test_losses, train_f1s, test_f1s, train_miss_rates, test_miss_rates, train_false_alarm_rates, test_false_alarm_rates = train(encoder_mel_phase
         , tcn,MelPhaseChoice, decoder, train_loader, test_loader, add_loss_noregular_sigmoid, optimizer, scheduler, batch_size, epochs=hparams.epochs,is_spatial=is_spatial,alpha=alpha,save_dir=save_dir)
        
  




