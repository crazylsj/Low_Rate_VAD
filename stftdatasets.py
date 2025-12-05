import torch
import os
from torch.utils.data import Dataset
from pathlib import Path
import re
import numpy as np
import torchaudio
import torch.nn.functional as F


def check_labels_consistency(labels):
    first_label = labels[0]  
    for label in labels[1:]: 
        if not torch.equal(label, first_label):
            return False  
    return True  

class MultichannelSTFT(torch.nn.Module):


    def __init__(
        self,
        in_channels,
        n_fft=512,
        win_length=400,
        hop_length=160,
        center=True,
        pad=0,
    ):
        super(MultichannelSTFT, self).__init__()

        self.stft_kw = dict(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=center,
            return_complex=True,
        )
        self.in_channels = in_channels
        self.pad = pad

    def forward(self, x):
        x_padded = F.pad(x, (self.pad, self.pad))
        X = [
            torch.stft(x_padded[i, :], **self.stft_kw)
            for i in range(self.in_channels)
        ]
        return torch.stack(X) 

class DASN_VAD_Dataset(Dataset):
    def __init__(self, root_dir,roc_dir):
        self.root_dir = root_dir
        self.roc_dir = roc_dir
        self.sample_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.roc_dirs = [os.path.join(self.roc_dir, d) for d in os.listdir(self.roc_dir) if os.path.isdir(os.path.join(self.roc_dir, d))]
        self.stft = MultichannelSTFT(in_channels=5)
    
    def __len__(self):
        return len(self.sample_dirs)
    
    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]
        roc_dir = self.roc_dirs[idx]
        feature_files = sorted([os.path.join(sample_dir, f) for f in os.listdir(sample_dir) if f.endswith('.pt')])
        roc_files = sorted([os.path.join(roc_dir, f) for f in os.listdir(roc_dir) if f.endswith('.npy')])
        stft_files = sorted([os.path.join(roc_dir, f) for f in os.listdir(roc_dir) if f.endswith('.flac')])
    
        order_stfts = [f for f in stft_files if f.split('/')[-1].startswith('cn')]
        order_stfts = sorted(order_stfts,key=self.extract_cn_index)
        waveforms = []
        for path in order_stfts:
            waveform, sr = torchaudio.load(path) 
            waveforms.append(waveform)
        lengths = [w.shape[1] for w in waveforms]
        assert len(set(lengths)) == 1, "The audio lengths are inconsistent"
        mulchannel_wave = torch.cat(waveforms, dim=0)
        MSTFT = self.stft(mulchannel_wave) 
        moc_locs = np.load(roc_files[0])
        sources_locs = np.load(roc_files[1])

        features = []
        labels = []

        source_file = [f for f in feature_files if Path(f).name == "target-feature-label.pt"][0]
        c0clean_file = [f for f in feature_files if Path(f).name == "c0-cleanfeature-label.pt"][0]
        filtered_files = [f for f in feature_files if Path(f).name not in ["target-feature-label.pt", "c0-cleanfeature-label.pt"]]
        filtered_files.sort(key=self.get_channel_index)
        for f in filtered_files:
            feat, label = torch.load(f, weights_only=True)
            features.append(feat)
            labels.append(label)

        source_feat, source_label, s_loc = torch.load(source_file, weights_only=True)
        c0clean_feat, c0_label = torch.load(c0clean_file, weights_only=True)
        features = torch.stack(features, dim=0).squeeze(1)  
        labels = torch.stack(labels, dim=0)      

        len_noise_label = labels[0].shape[-1]
        len_noise_feature = features[0].shape[-1]
        len_source = source_label.shape[-1]
        len_source_feature = source_feat.shape[-1]
        len_c0 = c0_label.shape[-1]
        len_c0_feature = c0clean_feat.shape[-1]

        if len_noise_label > len_source:
            labels = labels[:,0:len_source]
            features = features[:,:,0:len_source]
        else:
            source_label = source_label[0:len_noise_label]
            source_feat = source_feat[:,:,0:len_noise_label]
        _,__,alslen = features.shape

        if alslen > len_c0:
            source_label = source_label[0:len_c0]
            source_feat = source_feat[:,:,0:len_c0]
            labels = labels[:,0:len_source]
            features = features[:,:,0:len_source]
        else:
            c0_label = c0_label[0:alslen]
            c0clean_feat = c0clean_feat[:,:,0:alslen]
        
        if MSTFT.shape[-1] >= features.shape[-1]:
            MSTFT = MSTFT[:,:,:features.shape[-1]]
        else:
            source_label = source_label[0:MSTFT.shape[-1]]
            c0_label = c0_label[0:MSTFT.shape[-1]]
            features = features[:,:,:MSTFT.shape[-1]]

        return MSTFT, source_feat, source_label, c0clean_feat , c0_label,  features, moc_locs,sources_locs
    
    def extract_cn_index(self,path):
        match = re.search(r'cn(\d+)\.flac', path)
        return int(match.group(1)) if match else -1  
    
    def get_channel_index(self, f):
        match = re.search(r'c(\d+)-feature-label\.pt', Path(f).name)
        return int(match.group(1)) if match else float('inf')
    
    

