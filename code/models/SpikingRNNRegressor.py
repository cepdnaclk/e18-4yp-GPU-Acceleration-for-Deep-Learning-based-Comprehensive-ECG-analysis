import snntorch as snn
from snntorch import surrogate
import torch.nn as nn

class SpikingRNNRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_steps):
        super(SpikingRNNRegressor, self).__init__()
        self.num_steps = num_steps
        self.hidden_size = hidden_size
        self.spike_grad = surrogate.fast_sigmoid()

        # Layer 1
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=0.9, threshold=1.0, spike_grad=self.spike_grad, learn_beta=True)

        # Layer 2
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.lif2 = snn.Leaky(beta=0.9, threshold=1.0, spike_grad=self.spike_grad, learn_beta=True)

        # Output layer
        self.fc_out = nn.Linear(hidden_size, 1)
        self.li_out = snn.Leaky(beta=0.9, threshold=1.0, spike_grad=self.spike_grad, learn_beta=True, reset_mechanism="none")

    def forward(self, x):
        batch_size = x.size(0)
        mem1 = self.lif1.init_leaky(batch_size)
        mem2 = self.lif2.init_leaky(batch_size)
        mem_out = self.li_out.init_leaky(batch_size)

        out_mem = []

        for step in range(self.num_steps):
            cur1 = self.fc1(x[:, step, :])
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            out = self.fc_out(spk2)
            _, mem_out = self.li_out(out, mem_out)

            out_mem.append(mem_out)

        out_mem = torch.stack(out_mem, dim=1)

        return out_mem