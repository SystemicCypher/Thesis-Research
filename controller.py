import tensorflow as tf
import tensorboard 

class LSTM:

    def __init__(self, input_vector_size, memory_size, n_layers, out_vec_size=None, initializer=tf.random.normal, initial_stddev=None):
        self.input_vector_size = input_vector_size
        self.memory_size = memory_size
        self.n_layers = n_layers
        self.out_layer_exists = False
        self.out_vec_size = self.memory_size
        
        if out_vec_size is not None:
            self.out_vec_size = out_vec_size
            self.out_layer_exists = True

            self.weights = tf.Variable(initializer([self.memory_size, self.out_vec_size ], stddev=initial_stddev), name="out_weights")
            self.biases = tf.Variable(initializer([self.out_vec_size], stddev=initial_stddev), name="out_bias")
        
        #
        single_cell = tf.nn.rnn_cell.LSTMCell
        # Constructs N LSTM cells, stacked
        self.lstm_cell = tf.nn.rnn_cell.MultiRNNCell([single_cell(self.memory_size) for _ in range(self.n_layers)])


    def time_step(self, input_vec, state, step):
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
        with tf.variable_scope("LSTM_Step"):
            hidden, new_state = self.lstm_cell(x, state)
        
        return hidden, new_state
        
    
 