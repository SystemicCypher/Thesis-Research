import torch as t

class controllerLSTM(t.nn.Module):

    def __init__(self, input_vector_size, memory_size, num_layers):
        super(controllerLSTM, self).__init__()

        self.input_vector_size = input_vector_size
        self.memory_size = memory_size

        #The number of layers to create
        self.num_layers = num_layers
        
        # Creating L lstm layers for a single time step
        single_cell = t.nn.LSTMCell(self.input_vector_size, self.memory_size)
        self.controller_layers = [single_cell for _ in range(self.num_layers)]


    def forward(self, input_vec, hidden, state, step):
        '''
        This function will return the output vector of all the
        hidden states per time step
        
        PARAMETERS

        input_vec : This is the vector [x_t, r{1}_t-1, ..., r{R}_t-1]
        or specifically, the input data for this time step concatenated with
        the read vectors from memory of the previous time step

        state : This is the previous state s{l}_t-1 from the previous
        time step

        step : This is the current time step that the LSTM is on 
        '''
        new_hidden = []
        new_state = []
        
        #TODO Need to pass lower layer's state and hidden to higher layers 
        
        for l in self.controller_layers:
            h, c = l(input_vec, (hidden, state))
            new_hidden.append(h)
            new_state.append(c)
        
        return t.Tensor(new_hidden), t.Tensor(new_state)
        
    
 