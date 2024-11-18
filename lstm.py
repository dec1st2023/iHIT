import torch
import torch.nn as nn

__all__ = ['lstm']
input_size = 4
hidden_size = 32
num_layers = 2

class LSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=32, num_layers=2, batch_first=True, bidirectional=True):
        super(LSTM, self).__init__()
        self.position_len = 15
        self.fc_size = hidden_size * 2 if bidirectional else hidden_size
        self.vor_lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers,
                            bidirectional=bidirectional,
                            batch_first=batch_first)
        self.vor_fc = nn.Linear(self.fc_size*2, 2)

        self.lvl_lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers,
                            bidirectional=bidirectional,
                            batch_first=batch_first)
        self.lvl_fc = nn.Linear(self.fc_size, 2) 
        self.dropout = nn.Dropout(0.5)  
    
    def argmax(self, x):
        return torch.max(x, 1)[1]

    def forward(self, dir_input, position_input):
        # x shape: (batch, seq_len, input_size)
        dirs = self.argmax(dir_input)
        mask_vor = (dirs == 1) + (dirs == 4)
        mask_r = (dirs == 0) + (dirs == 5) # LA RP
        mask_l = (dirs == 2) + (dirs == 3) # RA LP

        self.vor_lstm.flatten_parameters()
        self.lvl_lstm.flatten_parameters()

        x_r, (h_n, h_c) = self.lvl_lstm(position_input[:, self.position_len:, :])
        x_r = self.dropout(x_r[:, -1, :])
        x_r = self.lvl_fc(x_r) * mask_r.unsqueeze(-1).float()

        x_l, (h_n, h_c) = self.lvl_lstm(position_input[:, :self.position_len, :])
        x_l = self.dropout(x_l[:, -1, :])
        x_l = self.lvl_fc(x_l) * mask_l.unsqueeze(-1).float()


        x_vor_l, (h_n, h_c) = self.vor_lstm(position_input[:, :self.position_len, :])
        x_vor_r, (h_n, h_c) = self.vor_lstm(position_input[:, self.position_len:, :])
        x_vor = torch.cat([x_vor_l[:, -1, :], x_vor_r[:, -1, :]], dim=1)
        x_vor = self.dropout(x_vor)
        x_vor = self.vor_fc(x_vor) * mask_vor.unsqueeze(-1).float()

        x = x_r + x_l + x_vor

        return x


def lstm(**kwargs):
    return LSTM(**kwargs)


if __name__ == "__main__":
    batch_size = 16
    seq_len = 30 
    model1 = LSTM(input_size, hidden_size, num_layers)

    input = torch.randn(batch_size, seq_len, input_size) 
    dirs = torch.randn(batch_size, 6) 

    output = model1(dir_input=dirs, position_input=input)
    print(output.shape)