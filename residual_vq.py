import torch
import torch.nn as nn
from torch.distributions import Normal, Uniform




class ProgressiveVQ(nn.Module):
    def __init__(self, num_stages, vq_bitrate_per_stage, data_dim, discard_threshold=0.01, device=torch.device('cpu')):
        super(ProgressiveVQ, self).__init__()
        self.num_stages = num_stages
        self.num_codebooks = int(2 ** vq_bitrate_per_stage)
        self.data_dim = data_dim
        self.device = device
        self.discard_threshold = discard_threshold
        self.eps = 1e-12
        self.normal_dist = Normal(0, 1)

        # 可选：训练后期用于推理时的衰减系数
        self.alpha = (1 - 0.1) / (num_stages - 1)

        # 初始化码本
        initial_codebooks = torch.zeros(num_stages, self.num_codebooks, data_dim, device=device)
        for k in range(num_stages):
            initial_codebooks[k] = Uniform(-1 / self.num_codebooks, 1 / self.num_codebooks).sample(
                [self.num_codebooks, data_dim])
        self.codebooks = nn.Parameter(initial_codebooks, requires_grad=True)
        self.codebooks_used = torch.zeros((num_stages, self.num_codebooks), dtype=torch.int32, device=device)

    def forward(self, input_data, stage_map=None, train_mode=False, min_stage=6):

        T, D = input_data.shape
        residual = input_data.clone()
        quantized_input_list = []
        intermediate_reconstructions = []
        min_indices_list = []

        for i in range(self.num_stages):
            # 当前阶段硬编码量化
            q_input, new_residual, min_indices = self.hard_vq(residual, self.codebooks[i])  # [T, D]

            # 掩码：是否使用第 i+1 阶段
            if stage_map is not None:
                mask = (stage_map >= (i + 1)).float().unsqueeze(-1)  # [T, 1]
            else:
                mask = torch.ones_like(q_input)
            

            # 应用掩码
            q_input = q_input * mask
            residual = new_residual * mask

            quantized_input_list.append(q_input)
            intermediate_reconstructions.append(sum(quantized_input_list))  # [T, D]
            min_indices_list.append(min_indices)

            if train_mode:
                with torch.no_grad():
                    self.codebooks_used[i, min_indices] += 1

        final_output = intermediate_reconstructions[-1]

        # 计算每阶段MSE，方便分析
        with torch.no_grad():
            mse_list = []
            for recon in intermediate_reconstructions:
                mse = torch.mean((recon - input_data) ** 2).item()
                mse_list.append(mse)

        if train_mode:
            return final_output, self.codebooks_used.cpu().numpy(), self.codebooks, mse_list
        else:
            return final_output.detach(), self.codebooks_used.cpu().numpy(), self.codebooks, mse_list


   

    def hard_vq(self, input_data, codebooks):
        distances = (torch.sum(input_data ** 2, dim=1, keepdim=True)
                     - 2 * (input_data @ codebooks.t())
                     + torch.sum(codebooks.t() ** 2, dim=0, keepdim=True))
        min_indices = torch.argmin(distances, dim=1)
        quantized_input = codebooks[min_indices]
        remainder = input_data - quantized_input
        return quantized_input, remainder, torch.unique(min_indices)

    def noise_substitution_vq(self, input_data, hard_quantized_input):
        random_vector = self.normal_dist.sample(input_data.shape).to(input_data.device)
        norm_hard = (input_data - hard_quantized_input).square().sum(dim=1, keepdim=True).sqrt()
        norm_rand = random_vector.square().sum(dim=1, keepdim=True).sqrt()
        vq_error = ((norm_hard / (norm_rand + self.eps)) * random_vector)
        return input_data + vq_error


    def replace_unused_codebooks(self, num_batches):

        with torch.no_grad():
            for k in range(self.num_stages):

                unused_indices = torch.where((self.codebooks_used[k].cpu() / num_batches) < self.discard_threshold)[0]
                used_indices = torch.where((self.codebooks_used[k].cpu() / num_batches) >= self.discard_threshold)[0]

                unused_count = unused_indices.shape[0]
                used_count = used_indices.shape[0]

                if used_count == 0:
                    print(f'####### used_indices equals zero / shuffling whole codebooks ######')
                    self.codebooks[k] += self.eps * torch.randn(self.codebooks[k].size(), device=self.device).clone()
                else:
                    used = self.codebooks[k, used_indices].clone()
                    if used_count < unused_count:
                        used_codebooks = used.repeat(int((unused_count / (used_count + self.eps)) + 1), 1)
                        used_codebooks = used_codebooks[torch.randperm(used_codebooks.shape[0])]
                    else:
                        used_codebooks = used[torch.randperm(used.shape[0])]

                    self.codebooks[k, unused_indices] *= 0
                    self.codebooks[k, unused_indices] += used_codebooks[range(unused_count)] + self.eps * torch.randn(
                        (unused_count, self.data_dim), device=self.device).clone()

                # prints out the number of unused codebook vectors for each individual codebook
                print(f'************* Replaced ' + str(unused_count) + f' for codebook {k+1} *************')
                self.codebooks_used[k, :] = 0.0

from torch.distributions import Normal, Uniform

class RVQ(torch.nn.Module):
    def __init__(self, num_stages, vq_bitrate_per_stage, data_dim, discard_threshold=0.01, device=torch.device('cpu')):
        super(RVQ, self).__init__()

        self.num_stages = num_stages # number of stages used for vector quantization
        self.num_codebooks = int(2 ** vq_bitrate_per_stage) # number of codebook entries for each stage
        self.data_dim = data_dim # data samples or codebook entries dimension
        self.eps = 1e-12
        self.device = device
        self.dtype = torch.float32
        self.normal_dist = Normal(0, 1)
        self.discard_threshold = discard_threshold

        initial_codebooks = torch.zeros(self.num_stages, self.num_codebooks, self.data_dim, device=self.device)
        

        for k in range(num_stages):
            initial_codebooks[k] = Uniform(-1 / self.num_codebooks, 1 / self.num_codebooks).sample(
                [self.num_codebooks, self.data_dim])
    
        self.codebooks = torch.nn.Parameter(initial_codebooks, requires_grad=True)
     
        self.codebooks_used = torch.zeros((num_stages, self.num_codebooks), dtype=torch.int32, device=self.device)

    def forward(self, input_data, train_mode=True, used_stage=None):
        if used_stage is None:
            used_stage = self.num_stages

        quantized_input_list = []
        remainder_list = [input_data]
        min_indices_list = []

        for i in range(used_stage):
            quantized_input, remainder, min_indices = self.hard_vq(remainder_list[i], self.codebooks[i])
            quantized_input_list.append(quantized_input)
            remainder_list.append(remainder)
            min_indices_list.append(min_indices)

        final_input_quantized = sum(quantized_input_list)

        if train_mode:
            with torch.no_grad():
                for i in range(used_stage):
                    self.codebooks_used[i, min_indices_list[i]] += 1

            return final_input_quantized, self.codebooks_used.cpu().numpy(), self.codebooks
        else:
            return final_input_quantized.detach(), self.codebooks_used.cpu().numpy(), self.codebooks


    


    def noise_substitution_vq(self, input_data, hard_quantized_input):
        random_vector = self.normal_dist.sample(input_data.shape).to(input_data.device)
        norm_hard_quantized_input = (input_data - hard_quantized_input).square().sum(dim=1, keepdim=True).sqrt()
        norm_random_vector = random_vector.square().sum(dim=1, keepdim=True).sqrt()
        vq_error = ((norm_hard_quantized_input / norm_random_vector + self.eps) * random_vector)
        quantized_input = input_data + vq_error
        return quantized_input



    def hard_vq(self, input_data, codebooks):
        distances = (torch.sum(input_data ** 2, dim=1, keepdim=True)
                     - 2 * (torch.matmul(input_data, codebooks.t()))
                     + torch.sum(codebooks.t() ** 2, dim=0, keepdim=True))
        min_indices = torch.argmin(distances, dim=1)
        quantized_input = codebooks[min_indices]
        remainder = input_data - quantized_input
        return quantized_input, remainder, torch.unique(min_indices)



    # codebook replacement function: used to replace inactive codebook entries with the active ones
    def replace_unused_codebooks(self, num_batches):
        """
        This function is used to replace the inactive codebook entries with the active ones, to make all codebooks
        entries to be used for training. The function has to be called periodically with the periods of "num_batches".
        In more details, the function waits for "num_batches" training batches and then discards the codebook entries
        which are used less than a specified percentage (self.discard_threshold) during this period, and replace them
        with the codebook entries which were used (active).

        Recommendation: Call this function after a specific number of training batches. In the beginning the number of
        replaced codebooks might increase (the number of replaced codebooks will be printed out during training).
        However, the main trend must be decreasing after some training time. If it is not the case for you, increase the
        "num_batches" or decrease the "discard_threshold" to make the trend for number of replacements decreasing.
        Stop calling the function at the latest stages of training in order not to introduce new codebook entries which
        would not have the right time to be tuned and optimized until the end of training.

        Play with "self.discard_threshold" value and the period ("num_batches") you call the function. A common trend
        could be to select the self.discard_threshold from the range [0.01-0.1] and the num_batches from the set
        {100,500,1000,...}. For instance, as a commonly used case, if we set the self.discard_threshold=0.01 and
        num_batches=100, it means that you want to discard the codebook entries which are used less than 1 percent
        during 100 training batches. Remember you have to set the values for "self.discard_threshold" and "num_batches"
        in a logical way, such that the number of discarded codebook entries have to be in a decreasing trend during
        the training phase.

        :param num_batches: period of training batches that you want to replace inactive codebooks with the active ones

        """
        with torch.no_grad():
            for k in range(self.num_stages):

                unused_indices = torch.where((self.codebooks_used[k].cpu() / num_batches) < self.discard_threshold)[0]
                used_indices = torch.where((self.codebooks_used[k].cpu() / num_batches) >= self.discard_threshold)[0]

                unused_count = unused_indices.shape[0]
                used_count = used_indices.shape[0]

                if used_count == 0:
                    print(f'####### used_indices equals zero / shuffling whole codebooks ######')
                    self.codebooks[k] += self.eps * torch.randn(self.codebooks[k].size(), device=self.device).clone()
                else:
                    used = self.codebooks[k, used_indices].clone()
                    if used_count < unused_count:
                        used_codebooks = used.repeat(int((unused_count / (used_count + self.eps)) + 1), 1)
                        used_codebooks = used_codebooks[torch.randperm(used_codebooks.shape[0])]
                    else:
                        used_codebooks = used[torch.randperm(used.shape[0])]

                    self.codebooks[k, unused_indices] *= 0
                    self.codebooks[k, unused_indices] += used_codebooks[range(unused_count)] + self.eps * torch.randn(
                        (unused_count, self.data_dim), device=self.device).clone()

                # prints out the number of unused codebook vectors for each individual codebook
                print(f'************* Replaced ' + str(unused_count) + f' for codebook {k+1} *************')
                self.codebooks_used[k, :] = 0.0


class StepRVQ(nn.Module):
    def __init__(self, num_stages, vq_bitrate_per_stage, data_dim, discard_threshold=0.01, device=torch.device('cpu')):
        super(StepRVQ, self).__init__()

        self.num_stages = num_stages
        self.num_codebooks = int(2 ** vq_bitrate_per_stage)
        self.data_dim = data_dim
        self.eps = 1e-12
        self.device = device
        self.dtype = torch.float32
        self.normal_dist = Normal(0, 1)
        self.discard_threshold = discard_threshold

        initial_codebooks = torch.zeros(self.num_stages, self.num_codebooks, self.data_dim, device=self.device)
        for k in range(num_stages):
            initial_codebooks[k] = Uniform(-1 / self.num_codebooks, 1 / self.num_codebooks).sample(
                [self.num_codebooks, self.data_dim])
        self.codebooks = nn.Parameter(initial_codebooks, requires_grad=True)
        self.codebooks_used = torch.zeros((num_stages, self.num_codebooks), dtype=torch.int32, device=self.device)

    def forward(self, input_data, train_mode=True, used_stage=None,
                return_stage_outputs=False, return_indices=False):
        if used_stage is None:
            used_stage = torch.full((input_data.shape[0],), self.num_stages, dtype=torch.int, device=input_data.device)

        T, D = input_data.shape
        remainder = input_data
        final_quantized = torch.zeros_like(input_data)
        min_indices_list = []
        stage_outputs = []

        for i in range(self.num_stages):
           
            mask = (used_stage > i).float().unsqueeze(-1)  
            if mask.sum() == 0:
                break

            quantized, residual, min_indices = self.hard_vq(remainder, self.codebooks[i])
            quantized = quantized * mask
            remainder = (remainder - quantized) * mask + remainder * (1 - mask)
            final_quantized += quantized

            if train_mode:
                with torch.no_grad():
                    self.codebooks_used[i, min_indices] += 1

            min_indices_list.append(min_indices)
            stage_outputs.append((quantized, remainder))

        if return_stage_outputs and return_indices:
            return final_quantized, self.codebooks, stage_outputs, min_indices_list
        elif return_stage_outputs:
            return final_quantized, self.codebooks, stage_outputs
        else:
            return final_quantized, self.codebooks

    def hard_vq(self, input_data, codebooks):
        distances = (torch.sum(input_data ** 2, dim=1, keepdim=True)
                     - 2 * torch.matmul(input_data, codebooks.t())
                     + torch.sum(codebooks.t() ** 2, dim=0, keepdim=True))
        min_indices = torch.argmin(distances, dim=1)
        quantized_input = codebooks[min_indices]
        remainder = input_data - quantized_input
        return quantized_input, remainder, min_indices

    def compute_entropy_loss(self, indices_list):
        total_entropy = 0.0
        for indices in indices_list:
            usage = torch.bincount(indices.flatten(), minlength=self.num_codebooks).float()
            prob = usage / (usage.sum() + self.eps)
            total_entropy += -torch.sum(prob * torch.log(prob + self.eps))
        return total_entropy

    def replace_unused_codebooks(self, num_batches):
        with torch.no_grad():
            for k in range(self.num_stages):
                unused_indices = torch.where((self.codebooks_used[k].cpu() / num_batches) < self.discard_threshold)[0]
                used_indices = torch.where((self.codebooks_used[k].cpu() / num_batches) >= self.discard_threshold)[0]

                unused_count = unused_indices.shape[0]
                used_count = used_indices.shape[0]

                if used_count == 0:
                    print(f'####### used_indices equals zero / shuffling whole codebooks ######')
                    self.codebooks[k] += self.eps * torch.randn(self.codebooks[k].size(), device=self.device).clone()
                else:
                    used = self.codebooks[k, used_indices].clone()
                    if used_count < unused_count:
                        used_codebooks = used.repeat(int((unused_count / (used_count + self.eps)) + 1), 1)
                        used_codebooks = used_codebooks[torch.randperm(used_codebooks.shape[0])]
                    else:
                        used_codebooks = used[torch.randperm(used.shape[0])]

                    self.codebooks[k, unused_indices] *= 0
                    self.codebooks[k, unused_indices] += used_codebooks[range(unused_count)] + \
                        self.eps * torch.randn((unused_count, self.data_dim), device=self.device).clone()

                print(f'************* Replaced {unused_count} for codebook {k + 1} *************')
                self.codebooks_used[k, :] = 0.0