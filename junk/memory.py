import torch as t 

class Memory:

    def __init__(self, memory_size):
        super(Memory, self).__init__()

        self.memory_size = memory_size

    def memory_action(self, interface_vec, writeable_data=None):
        '''
        This function receives the interface vector and then 
        either erases/writes or reads data

        PARAMETERS

        interface_vec --  The interface vector that determines
        what needs to be done in the memory

        writeable_data -- The data to be written, by default there's 
        none.
        '''
        


    def erase_and_write(self):
        pass

    def read(self):
        pass