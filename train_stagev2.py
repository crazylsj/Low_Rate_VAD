import torch
from tqdm import tqdm
import os
import logging
from torch.utils.data import DataLoader
from stftdatasets import DASN_VAD_Dataset
from params import HParams
from DASN_model import  TCN,  Decoder_sigmoid ,add_loss_noregular_sigmoid,node_fusion_v2,TCNImpNet
from timm.scheduler.cosine_lr import CosineLRScheduler
import numpy as np
import torch.nn.functional as F
import argparse
from sklearn.metrics import  confusion_matrix
                             
import random
from residual_vq import StepRVQ


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

def val_vq(data_batch,vector_quantizer,codebooks,num_stages):

    vector_quantizer.eval() 
    codebooks = codebooks.to(DEVICE)
    
    quantized_input_list = []
    remainder_list = []
    min_indices_list = []

    remainder_list.append(data_batch)

    for i in range(num_stages):

        codebook = codebooks[i]
        quantized_input, remainder, min_indices = hard_vq(remainder_list[i], codebook)
        quantized_input_list.append(quantized_input)
        remainder_list.append(remainder)
        min_indices_list.append(min_indices)


    min_indices_np = np.stack([indices.cpu().numpy() for indices in min_indices_list], axis=0).T
    min_indices = torch.tensor(min_indices_np, dtype=torch.long)  
    B, S = min_indices.shape  
    _, N, D = codebooks.shape  

    stage_ids = torch.arange(S).unsqueeze(0).expand(B, -1) 
    selected_codebook_vectors = codebooks[stage_ids, min_indices] 
    summed_vectors = selected_codebook_vectors.sum(dim=1)
    return summed_vectors


def per_frame_sqrt_stage(scores, low_stage=2, high_stage=36, score_min=0.0, score_max=1/10):
    norm_scores = torch.clamp((scores - score_min) / (score_max - score_min), 0.0, 1.0)
    weights = torch.sqrt(norm_scores)
    float_stages = low_stage + weights * (high_stage - low_stage)
    stages = torch.floor(float_stages).to(torch.int)
    return stages

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


def train_epoch(encoder_mel_phase,  vq_melPhase , tcn, MelPhaseChoice, decoder, train_loader, loss_fn, optimizer, scheduler, batch_size, epoch, start_step):
    step = start_step
    total_loss = 0
    total_f1 = 0
    total_precision = 0
    total_recall = 0
    total_miss_rate = 0
    total_false_alarm_rate = 0
    total_SER = 0
    num_batches = 0
    total_tp = total_tn = total_fp = total_fn = 0
    encoder_mel_phase.eval()  
    for param in encoder_mel_phase.parameters():
        param.requires_grad = False
    vq_melPhase.train()
    melvq_losses = 0

    for batch_index, batch in enumerate(train_loader):

        
        STFTs,source_feat, source_label, c0clean_feat , c0_label,  features, moc_locs, sources_locs = batch 
        STFTs = STFTs.to(DEVICE)
        source_feat = source_feat.to(DEVICE)
        source_label = source_label.to(DEVICE)
        c0clean_feat = c0clean_feat.to(DEVICE)
        c0_label = c0_label.to(DEVICE)
        features = features.to(DEVICE)
        phase = torch.angle(STFTs)
        phase = phase[:, :, 1:, :]
        cos_phase = torch.cos(phase) 
        sin_phase = torch.sin(phase)
        MelPha = torch.cat((features,sin_phase,cos_phase),dim=-2)
        with torch.no_grad():
            MPh,scores  = encoder_mel_phase(MelPha.squeeze(0).detach())

        melF_q_list = []
    
        stages = per_frame_sqrt_stage(scores,high_stage=args.codebook,score_max=1/args.channel)
        for i in range(args.channel):
            MelF_quantized, MelF_codebooks  = vq_melPhase(MPh[i].T, used_stage=stages[i], train_mode=True)
            melF_q_list.append(MelF_quantized)

        MelF_quantized_all = torch.stack(melF_q_list,dim=0)
        Melphdq_x, Melphchannels_weights = MelPhaseChoice(MelF_quantized_all.permute(1, 0, 2))   
        wMelphF = (Melphdq_x  * Melphchannels_weights).sum(dim=-2).permute(1,0).unsqueeze(0) 
        tcn_embfeddings = tcn(wMelphF) 
        postnet_output = decoder(tcn_embfeddings)
        postnet_output = postnet_output.squeeze(1) 
        cls_loss, _ = loss_fn(source_label, postnet_output)
        melvq_loss = F.mse_loss(MPh, MelF_quantized_all.permute(0,2,1))
        melvq_losses += melvq_loss
        loss =  melvq_loss + args.beta*cls_loss
        total_loss += loss.detach().item() 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step_update(step)
        num_batches += 1
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
            logging.info('E:{}, S: {}, overall loss: {:.4f}, meltotal loss: {:.4f}, F1-score: {:.4f}, SER: {:.4f}'.format(
                epoch, step, total_loss / num_batches, melvq_losses / num_batches, total_f1, total_SER))
    total_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    total_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0    
    total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall) if (total_precision + total_recall) > 0 else 0
    total_miss_rate = total_fn / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    total_false_alarm_rate = total_fp / (total_fp + total_tn) if (total_fp + total_tn) > 0 else 0
    total_SER = total_miss_rate + total_false_alarm_rate         
    
    return total_loss / num_batches, total_f1, total_miss_rate, total_false_alarm_rate, step

def test_epoch(encoder_mel_phase,  vq_melPhase , tcn, MelPhaseChoice, decoder, test_loader, loss_fn):
    logging.info("Validation Begins...")
    counter = 0
    total_loss = 0
    total_f1 = 0
    total_precision = 0
    total_recall = 0
    total_miss_rate = 0
    total_false_alarm_rate = 0
    total_tp = total_tn = total_fp = total_fn = 0

    encoder_mel_phase = encoder_mel_phase.to(DEVICE)
    MelPhaseChoice = MelPhaseChoice.to(DEVICE)
    tcn = tcn.to(DEVICE)
    vq_melPhase = vq_melPhase.to(DEVICE)
    decoder = decoder.to(DEVICE)
    reconlosses = 0
    for  STFTs,source_feat, source_label, c0clean_feat , c0_label,  features, moc_locs,sources_locs  in tqdm(test_loader):
        STFTs = STFTs.to(DEVICE)
        source_feat = source_feat.to(DEVICE)
        source_label = source_label.to(DEVICE)
        c0clean_feat = c0clean_feat.to(DEVICE)
        c0_label = c0_label.to(DEVICE)
        features = features.to(DEVICE)
        phase = torch.angle(STFTs)
        phase = phase[:, :, 1:, :]
        cos_phase = torch.cos(phase)  
        sin_phase = torch.sin(phase)
        MelPha = torch.cat((features,sin_phase,cos_phase),dim=-2)
        with torch.no_grad():
            MPh,scores  = encoder_mel_phase(MelPha.squeeze(0))
        stages = per_frame_sqrt_stage(scores,high_stage=args.codebook,score_max=1/args.channel)
        melF_q_list = []

        for i in range(args.channel):
            with torch.no_grad():
                MelF_quantized, codebooks ,stage_outputs  = vq_melPhase(MPh[i].T, used_stage=stages[i], train_mode=False,return_stage_outputs=True)
                melF_q_list.append(MelF_quantized)
            
        MelF_quantized_all = torch.stack(melF_q_list,dim=0)
        Melphdq_x, Melphchannels_weights = MelPhaseChoice(MelF_quantized_all.permute(1, 0, 2)) 

        wMelphF = (Melphdq_x  * Melphchannels_weights).sum(dim=-2).permute(1,0).unsqueeze(0)
        tcn_embfeddings = tcn(wMelphF) 
        postnet_output = decoder(tcn_embfeddings)
        postnet_output = postnet_output.squeeze(1) 
        clsloss, _ = loss_fn(source_label, postnet_output)
        reconloss = F.mse_loss(MPh, MelF_quantized_all.permute(0,2,1))
        loss = reconloss + clsloss
        reconlosses += reconloss

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
        total_loss / counter, reconlosses / counter, total_f1, total_precision, total_recall,
        total_miss_rate, total_false_alarm_rate))

    
    return total_loss / counter,reconlosses / counter, total_f1 , total_SER, total_miss_rate, total_false_alarm_rate, codebooks

def train(encoder_mel_phase,  vq_melPhase , tcn, MelPhaseChoice, decoder, train_loader, test_loader, loss_fn, optimizer, scheduler, batch_size, start_epoch=0, epochs=20, start_step=0):
    train_losses = []
    test_losses = []
 
    best_mel_vq_loss = 10
    best_f1_ser = 0

    for e in range(start_epoch, epochs):
        train_loss, train_f1, train_miss_rate, train_false_alarm_rate , start_step = train_epoch(encoder_mel_phase,  vq_melPhase , tcn, MelPhaseChoice, decoder,train_loader, loss_fn, optimizer, scheduler, batch_size, e, start_step)
        train_losses.append(train_loss)
    

        with torch.inference_mode():
            test_loss,test_reconloss, test_f1 , SER, test_miss_rate, test_false_alarm_rate, melphase_codebooks = test_epoch(encoder_mel_phase,  vq_melPhase , tcn, MelPhaseChoice, decoder,  test_loader, loss_fn)

        test_losses.append(test_loss)
        print(f'e:{e+1},test_f1:{test_f1},SER:{SER},test_f1-SER:{test_f1-SER}')
        print("Epoch: {}/{}... Test Loss: {:.5f}".format(
            e + 1, epochs ,
            test_loss))
        if test_reconloss < best_mel_vq_loss:
            best_mel_vq_loss = test_loss
            torch.save(melphase_codebooks, f'{args.stgae2_save_dir}/melphasecodebooks_{e+1}_SER_{best_mel_vq_loss:.2f}.pt')
            torch.save(vq_melPhase.state_dict(), f'{args.stgae2_save_dir}/melphasecodebooks_{e+1}_SER_{best_mel_vq_loss:.2f}.pth')
        if test_f1 - SER >=  best_f1_ser:
            best_f1_ser = test_f1 - SER
            torch.save(tcn.state_dict(),  f'{args.stgae2_save_dir}/tcnweights_{e+1}_SER_{best_f1_ser * 100:.2f}.pth')
            torch.save(MelPhaseChoice.state_dict(),  f'{args.stgae2_save_dir}/MelPhasechoiceweights_{e+1}_SER_{best_f1_ser * 100:.2f}.pth')
            torch.save(decoder.state_dict(),  f'{args.stgae2_save_dir}/decoderweights_{e+1}_SER_{best_f1_ser * 100:.2f}.pth')

    return train_losses, test_losses

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
                        default="data/LibriSpeechConcat_2000_node10/train-clean-100/Features",
                        help='Path to the training metadata')
    parser.add_argument('--loc_train_data', type=str, 
                    default="data/LibriSpeechConcat_2000_node10/train-clean-100/Waveforms",
                    help='Path to the testing metadata')
    parser.add_argument('--test_data', type=str, 
                    default="data/LibriSpeechConcat_2000_node10/dev-clean/Features",
                    help='Path to the testing metadata')
    parser.add_argument('--loc_test_data', type=str, 
                    default="data/LibriSpeechConcat_2000_node10/dev-clean/Waveforms",
                    help='Path to the testing metadata')
    parser.add_argument(
        '--beta',
        type=float,
        default=0.01,               
        choices=[0.01, 0.1, 1, 10],
        help='Weighting factor alpha'
    )
    parser.add_argument(
        '--channel',
        type=int,
        default=10,               
        choices=[5, 10, 20],
        help='The numbers of the nodes'
    )
    parser.add_argument(
        '--codebook',
        type=int,
        default=28,              
        help='The numbers of B'
    )
    parser.add_argument(
        '--codeword',
        type=int,
        default=3,   
        choices=[1,2,3,4,5],           
        help='The numbers of G'
    )
    parser.add_argument(
        '--stgae1_save_dir',
        type=str,
        default='checkpoint/DASN/node10/stage1/alpha1',
        help='Directory to save model checkpoints'
    )
    parser.add_argument(
        '--stgae2_save_dir',
        type=str,
        default='checkpoint/DASN/node10/stage2/G_3/B_36',
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
    DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
   # 使用示例
    set_seed(42)


    RESUME = args.resume_train
    train_da = args.train_data
    loc_train_da = args.loc_train_data
    test_da = args.test_data
    loc_test_da = args.loc_test_data
    stage_save_dir = args.stgae1_save_dir


    w, u = hparams.w, hparams.u
    winlen = hparams.winlen
    winstep = hparams.winstep
    fft = hparams.n_fft
    batch_size = hparams.batch_size
    print(f'hparams.batch_size')

    train_dataset = DASN_VAD_Dataset(train_da,loc_train_da)
    valid_dataset = DASN_VAD_Dataset(test_da,loc_test_da)

    train_loader = DataLoader(train_dataset, 1, True,
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(valid_dataset, 1, False,
                             num_workers=4, pin_memory=True)

    torch.cuda.empty_cache()
    melphase_emdim = 128

    encoder_mel_phase = TCNImpNet(40+256+256, 1, 1, 3, 1, 128, 128).to(DEVICE) 
    tcn = TCN(128, 1, 1, 3, 3, 128, 128).to(DEVICE)
    MelPhaseChoice  = node_fusion_v2(input_dim=128,att_dim=128).to(DEVICE)
    decoder = Decoder_sigmoid(dim = 128).to(DEVICE)
    vq_melPhase = StepRVQ(
            num_stages=args.codebook,
            vq_bitrate_per_stage=args.codeword,
            data_dim = melphase_emdim,
            device=DEVICE
        ).to(DEVICE)
 

    optimizer = torch.optim.AdamW(
        vq_melPhase.parameters(),
        lr=hparams.lr,
        weight_decay=hparams.weight_decay
    )

    scheduler = build_scheduler(
        optimizer, n_iter_step=hparams.n_iter_step, total_epoch=hparams.epochs // 2, warmup=hparams.warmup_factor)
    best_epoch = 28
    best_performance = 0.83
    melphaseencoder_checkpoint_path = f'{stage_save_dir}/melPhase_encoderweights_{best_epoch}_SER_{best_performance}.pth'
    tcn_checkpoint_path = f'{stage_save_dir}/tcnweights_{best_epoch}_SER_{best_performance}.pth'
    melphasechoice_checkpoint_path = f'{stage_save_dir}/MelPhasechoiceweights_{best_epoch}_SER_{best_performance}.pth'
    decoder_checkpoint_path = f'{stage_save_dir}/decoderweights_{best_epoch}_SER_{best_performance}.pth'
   
    
 
    encoder_mel_phase.load_state_dict(torch.load(melphaseencoder_checkpoint_path))
    tcn.load_state_dict(torch.load(tcn_checkpoint_path))
    MelPhaseChoice.load_state_dict(torch.load(melphasechoice_checkpoint_path))
    decoder.load_state_dict(torch.load(decoder_checkpoint_path))

    train_losses, test_losses = train(encoder_mel_phase, 
            vq_melPhase , tcn, MelPhaseChoice, decoder, train_loader, test_loader, add_loss_noregular_sigmoid, optimizer, scheduler, batch_size, epochs=20)
        
 




