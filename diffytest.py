import torch as T
from dnc import DNC
import pickle

print('Loading data...')
file = open('./IMDb-data/imdbwords.pickle', 'rb')
words = pickle.load(file)
file.close()

file = open('trainset', 'rb')
trainset = pickle.load(file)
file.close()

file = open('trainlabels', 'rb')
trainlables = pickle.load(file)
file.close()

train_data_loader = T.utils.data.DataLoader(dataset=trainset, batch_size=1, shuffle=False)
trainset = iter(train_data_loader)



print('Defining model...')
diffy  = DNC(25, 128, num_layers=2, independent_linears=True)

loss_fn = T.nn.MSELoss()

optimizer = T.optim.Adam(diffy.parameters(), lr=0.0001, betas=[0.9, 0.98])

maxVal = 0

maxItem = []

print('Finding max...')
for item in trainset:
    if maxVal < len(item):
        maxVal = len(item)
        maxItem = item

print('Padding values...')
for i in range(len(trainset)):
    if len(trainset[i]) <maxVal:
        while len(trainset[i]) < maxVal:
            trainset[i].append(0)

inputs = T.tensor(trainset)

inputs = inputs.reshape((1, 25000, 73, 25))

inputs = inputs.to(T.float)

inloader = T.utils.data.DataLoader(dataset=inputs, batch_size=1, shuffle=False)
inputset = iter(inloader)

(controller_hidden, memory, read_vectors) = (None, None, None)



ranges = 2 * len(trainset)




print('Beginnning training loop...')
for it in range(ranges):
    
    optimizer.zero_grad()
    seq = next(inputset)
    
    #Forward pass
    
    output, (controller_hidden, memory, read_vectors) = diffy(seq, (None, memory, None), reset_experience=False)
    

    final_out = T.sum(output,(1,2), keepdim=True) #outer(mid_out)
    
    loss = loss_fn(final_out, trainlabels[it].to(T.float).reshape((1,1,1)))
    
    loss.backward()
    optimizer.step()
    
    memory = {k : (v.detach() if isinstance(v, T.autograd.Variable) else v) for k, v in memory.items()}
    
    if it % 10 == 9:    
        print('Step: {}, Loss: {}'.format(it+1, loss))
        
         
