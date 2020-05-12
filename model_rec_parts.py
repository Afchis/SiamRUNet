import torch
import torch.nn as nn

from args import *


'''
Recurrent cells
'''
class ConvRnnCell(nn.Module):    
    def __init__(self, in_channels, out_channels):
        super(ConvRnnCell, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels+out_channels, out_channels, kernel_size=1))
             
    def forward(self, x, hidden):
        out = torch.cat([x, hidden],dim=1)
        out = self.conv1(out)
        hidden = out
        return out, hidden


class ConvGruCell(nn.Module):    
    def __init__(self, in_channels, out_channels):
        super(ConvGruCell, self).__init__()
        self.conv_for_input = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        
        self.conv_for_hidden = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=1))
        
        self.conv_2x_update = nn.Sequential(nn.Conv2d(in_channels+out_channels, out_channels, kernel_size=1))
        self.conv_2x_reset = nn.Sequential(nn.Conv2d(in_channels+out_channels, out_channels, kernel_size=1))
        
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

     
    def forward(self, x, hidden):
        input = torch.cat([x, hidden],dim=1)

        update_gate = self.conv_2x_update(input)
        update_gate = self.sig((update_gate)) ### output after update gate
        reset_gate = self.conv_2x_reset(input)
        reset_gate = self.sig((reset_gate)) ### output after reset gate
        
        
        memory_for_input = self.conv_for_input(x)
        memory_for_hidden = self.conv_for_hidden(hidden)

        memory_content = memory_for_input + (reset_gate * memory_for_hidden) ### output for reset gate(affects how the reset gate do work)
        memory_content = self.relu(memory_content)

        hidden = (update_gate * hidden) + ((1 - update_gate) * memory_content) # torch.ones(input_size, hidden_size)

        return hidden, hidden

'''
Recurrent part
'''
class ConvRnn(nn.Module):
	def __init__(self, in_channels, out_channels, input_size, cell_model=CELL_MODEL, mode=MODE):
		super(ConvRnn, self).__init__()
		self.cell_model = cell_model
		self.mode = mode
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.input_size = input_size
		self.cell_dict = {
		'Rnn' : ConvRnnCell(in_channels, out_channels),
		'Gru' : ConvGruCell(in_channels, out_channels),
		}
		self.conv_rnn = self.cell_dict[self.cell_model]

		self.hidden_size = (BATCH_SIZE, self.out_channels, self.input_size, self.input_size)
		self.init_hidden = torch.zeros(self.hidden_size).to(device)

	def forward(self, x):
		x_cells = torch.tensor([]).to(device)

		x = x.reshape(TIMESTEPS, BATCH_SIZE, self.in_channels, self.input_size, self.input_size)
		if self.mode == 'Standart':
			for i in range(TIMESTEPS):
				x_cell, _ = self.conv_rnn(x[i], self.init_hidden)
				x_cell = x_cell.unsqueeze(0)
				x_cells = torch.cat([x_cells, x_cell], dim=0)
		else:
			hidden = self.init_hidden
			for i in range(TIMESTEPS):
				x_cell, hidden = self.conv_rnn(x[i], hidden)
				x_cell = x_cell.unsqueeze(0)
				x_cells = torch.cat([x_cells, x_cell], dim=0)
		x_cells = x_cells.reshape(TIMESTEPS*BATCH_SIZE, self.out_channels, self.input_size, self.input_size)
		return x_cells


if __name__ =='__main__':
	tensor = torch.rand([TIMESTEPS, BATCH_SIZE, 256, 17, 17])
	model = ConvRnn(in_channels=256, out_channels=1024, input_size=17, cell_model='Gru')
	model = model
	out = model(tensor)
	print(out.shape)
