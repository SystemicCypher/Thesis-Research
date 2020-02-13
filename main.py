import torch as T
from dnc import DNC
import torchvision as tv

train = tv.datasets.MNIST('.', train=True, transform=tv.transforms.ToTensor())
test = tv.datasets.MNIST('.', transform=tv.transforms.ToTensor())

batch_size = 1

train_data_loader = T.utils.data.DataLoader(dataset=train, batch_size = batch_size, shuffle=True)
trainset = iter(train_data_loader)
trainsize = len(train_data_loader)

diffy = DNC(28, 128, num_layers=1, independent_linears=True)

loss_fn = T.nn.MSELoss()

optimizer = T.optim.Adam(diffy.parameters(), lr=0.0001, eps=1e-9, betas=[0.9, 0.98])

(controller_hidden, memory, read_vectors) = (None, None, None)

ranges = 2 * trainsize

for it in range(ranges):
	optimizer.zero_grad()
	img, true_out = next(trainset)
	img = T.squeeze(img, 1)

	output, (controller_hidden, memory, read_vectors) = diffy(img, (None, memory, None), reset_experience=True)

	newt = T.sum(output, (1, 2), keepdim=True)	

	loss = loss_fn(newt, true_out.to(T.float).reshape((1,1,1)))
	
	loss.backward()
	T.nn.utils.clip_grad_norm_(diffy.parameters(), 2)
	optimizer.step()

	memory = { k : (v.detach() if isinstance(v, T.autograd.Variable) else v) for k, v in memory.items()}

	if it % 10 == 9:
		print('Step: {}, Loss: {}'.format(it+1, loss.item()))

	if it % 60000 == 59999:
		trainset = iter(train_data_loader)

correct = 0

for i in range(60000):
	out, (c, m, r) = diffy(test[i][0], (None, memory, None), reset_experience=False)
	y_pred = T.sum(out, (1, 2)).round()
	if (train[i][1] == y_pred) == T.tensor(True):
		correct = correct + 1
acc = correct / 60000
print('Accuracy: {}'.format(acc))
 
