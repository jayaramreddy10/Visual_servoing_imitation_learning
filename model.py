import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F


class VisualServoingLSTM(nn.Module):
  def __init__(self, rnn_type, vel_dims=6, lstm_units=6, layers=5, batch_size=1, seq_len=5):
    super(VisualServoingLSTM, self).__init__()
    self.vel_dims = vel_dims
    self.lstm_units = lstm_units
    self.layers = layers
    self.batch_size = batch_size
    self.seq_len = seq_len
    self.f_interm= []
    self.v_interm= []
    if rnn_type == 'LSTM':
      #nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
      self.lstm = nn.LSTM(vel_dims, lstm_units, layers, batch_first=True)
    elif rnn_type == 'GRU':
      self.lstm = nn.GRU(vel_dims, lstm_units, layers, batch_first=True)
    self.hidden = self.init_hidden(rnn_type)
    
  def init_hidden(self, rnn_type):
    cell = torch.randn(self.layers, self.batch_size, self.lstm_units)# (n_layers, batch_size, hidden_size)
    cell = Variable(cell.cuda())
    if rnn_type == 'LSTM':
      hidden = torch.randn(self.layers, self.batch_size, self.lstm_units)# (n_layers, batch_size, hidden_size)
      hidden = Variable(hidden.cuda())
      return hidden, cell
    else:
      return cell
  
  def reset_hidden(self):
    #(h0, c0) --- (hidden, cell state at time step=0)
    self.hidden = self.init_hidden('LSTM')

  def forward(self, vel, Lsx, Lsy):
    #vel:[1,1,6], Lsx: [384, 512, 6], Lsy: [384, 512, 6]
    vels = None
    for i in range(self.seq_len):
        if i == 0:
            #out, _ = self.lstm(x, (h0, c0))
            #x:[batch_size, seq_len, input_size] but here we estimate out sequentially, so seq_len is 1, not 5
            out, hidden = self.lstm(vel.view(1, 1, self.vel_dims), self.hidden)
            #out -- all the hidden states , so dim is [batch_size, seq_len, hidden_size]
            #out - [1,1,6]
            vels = out.unsqueeze(0)  #increases dimnesion at 0 index
            # [1,1,1,6]
        else:
            out, hidden = self.lstm(out, hidden)
            vels = torch.cat([vels, out.unsqueeze(0)], dim=0)
        
        self.v_interm.append(out.data.cpu().numpy())

    #finally v_interm is list of size 5 where each element is tensor of size [1,1,6]  -- velocities till next 5 time steps
    L = torch.cat((Lsx, Lsy), -1)      #concatenated along depth -- (384, 512, 12)
    vels = vels.repeat(1, 1, 1, 2)    #(5,1,1,12) -- so predicting velocities upto next 5 time steps
    
    #now we need to multiply interaction matrix with predicted velocities to get intermediate flows for next 5 time steps, sum if up to get final predicted flow b/w I(t) and I*.
    f_hat = L*vels   #(5, 384, 512, 12)
    f_hat = torch.sum(f_hat, 0)  #(384, 512, 12)
    f1, f2 = torch.split(f_hat, [6,6], -1) #(384, 512, 6), (384, 512, 6)
    f1 = torch.sum(f1, -1).unsqueeze(-1)    #(384, 512, 1)
    f2 = torch.sum(f2, -1).unsqueeze(-1)  #(384, 512, 1)
    f_hat = torch.cat((f1, f2), -1)    #(384, 512, 2) -- concatenate along the last dimesnsion
    '''
    f_temp_5 = torch.sum(Lsx*out,-1).unsqueeze(-1)

    f_hat = torch.cat((torch.sum(Lsx*out,-1).unsqueeze(-1) , \
                  torch.sum(Lsy*out,-1).unsqueeze(-1)),-1)
    self.f_interm.append(f_hat.data.cpu().numpy())
    print(f_hat.shape)
    '''
    return f_hat #final predicted flow b/w I(t) and I* where first one is u(dot), second is v(dot)