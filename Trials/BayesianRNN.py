import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class BayesianRNN(nn.Module):
	def __init__(self):
		super(BayesianRNN, self).__init__()
		
		self.lstm1 = nn.LSTM(input_size = 28, hidden_size = 128, batch_first = True)
		self.lstm2 = nn.LSTM(input_size = 128, hidden_size = 128, batch_first = True)
		self.linear = nn.Linear(in_features = 128, out_features = 1)
		self.dropout = nn.Dropout(0.2)
	
	def forward(self, x):
		x, hidden = self.lstm1(x)
		x = self.dropout(x)
		x, hidden = self.lstm2(x)
		x = x[:,-1,:]
		x = self.dropout(x)
		x = self.linear(x)
		
		return x


import torch
import numpy as np


## Base Sampler class
## -----------------------------------------------------------------------------

class _BaseSampler:

	def __init__(self, loss_module):
		self.loss_module = loss_module
		self.sampled_weights = []

	def sample(self, nsamples=1, **args):
		raise NotImplementedError()

	def get_weights(self):
		params = self.loss_module.parameters()
		return tuple(p.data.clone().detach().cpu().numpy() for p in params)

	def set_weights(self, weights):
		for p, sample in zip(self.loss_module.parameters(), weights):
			p.copy_(torch.from_numpy(sample))

	def predict(self, x):
		with torch.no_grad():
			f_samples = np.ndarray((x.shape[0], len(self.sampled_weights)))
			for i, weights in enumerate(self.sampled_weights):
				self.set_weights(weights)
				f = self.loss_module.predict(x)
				if f.ndim==1 or f.shape[1]==1:
					f_samples[:,i] = f.flatten()
				else:
					f_samples[:,:,i] = f
			return f_samples

	def get_sampled_weight_matrix(self):
		if len(self.sampled_weights)==0:
			return None
		n_params = 0
		for w in self.sampled_weights[0]:
			n_params += w.size
		w_matrix = np.zeros([len(self.sampled_weights), n_params])
		for idx, weights in enumerate(self.sampled_weights):
			j0 = 0
			for w in weights:
				w_vec = w.flatten()
				length = w_vec.size
				w_matrix[idx, j0:(j0+length)] = w_vec
				j0 += length
		return w_matrix


## (Adaptive) Stochastic Gradient HMC Sampler classes
## -----------------------------------------------------------------------------

class AdaptiveSGHMC(torch.optim.Optimizer):
	""" Stochastic Gradient Hamiltonian Monte-Carlo Sampler that uses a burn-in
		procedure to adapt its own hyperparameters during the initial stages
		of sampling.

		References:
		[1] http://aad.informatik.uni-freiburg.de/papers/16-NIPS-BOHamiANN.pdf
		[2] https://arxiv.org/pdf/1402.4102.pdf
	"""

	def __init__(self, params, lr=1e-2, num_burn_in_steps=3000,
				 epsilon=1e-10, mdecay=0.05, scale_grad=1.):
		""" Set up a Adaptive SGHMC Optimizer.

		Args:
			params: iterable, parameters serving as optimization variable.
			lr: float, base learning rate for this optimizer.
				Must be tuned to the specific function being minimized.
			num_burn_in_steps: int, bumber of burn-in steps to perform.
				In each burn-in step, this sampler will adapt its own internal
				parameters to decrease its error. Set to `0` to turn scale
				adaption off.
			epsilon: float, per-parameter epsilon level.
			mdecay:float, momentum decay per time-step.
			scale_grad: float, optional
				Value that is used to scale the magnitude of the epsilon used
				during sampling. In a typical batches-of-data setting this
				usually corresponds to the number of examples in the
				entire dataset.
		"""
		if lr < 0.0:
			raise ValueError("Invalid learning rate: {}".format(lr))
		if num_burn_in_steps < 0:
			raise ValueError("Invalid num_burn_in_steps: {}".format(
				num_burn_in_steps))

		defaults = dict(
			lr=lr, scale_grad=float(scale_grad),
			num_burn_in_steps=num_burn_in_steps,
			mdecay=mdecay,
			epsilon=epsilon
		)
		super().__init__(params, defaults)

	def step(self, closure=None):
		loss = None

		if closure is not None:
			loss = closure()

		for group in self.param_groups:
			for parameter in group["params"]:

				if parameter.grad is None:
					continue

				state = self.state[parameter]

				if len(state) == 0:
					state["iteration"] = 0
					state["tau"] = torch.ones_like(parameter)
					state["g"] = torch.ones_like(parameter)
					state["v_hat"] = torch.ones_like(parameter)
					state["momentum"] = torch.zeros_like(parameter)
				state["iteration"] += 1

				mdecay = group["mdecay"]
				epsilon = group["epsilon"]
				lr = group["lr"]
				scale_grad = torch.tensor(group["scale_grad"],
										  dtype=parameter.dtype)
				tau, g, v_hat = state["tau"], state["g"], state["v_hat"]

				momentum = state["momentum"]
				gradient = parameter.grad.data * scale_grad

				tau_inv = 1. / (tau + 1.)

				# Update parameters during burn-in
				if state["iteration"] <= group["num_burn_in_steps"]:
					# Specifies the moving average window, see Eq 9 in [1] left
					tau.add_(- tau * (
							g * g / (v_hat + epsilon)) + 1)

					# Average gradient see Eq 9 in [1] right
					g.add_(-g * tau_inv + tau_inv * gradient)

					# Gradient variance see Eq 8 in [1]
					v_hat.add_(-v_hat * tau_inv + tau_inv * (gradient ** 2))

				# Preconditioner
				minv_t = 1. / (torch.sqrt(v_hat) + epsilon)  

				epsilon_var = (2. * (lr ** 2) * mdecay * minv_t - (lr ** 4))

				# Sample random epsilon
				sigma = torch.sqrt(torch.clamp(epsilon_var, min=1e-16))
				sample_t = torch.normal(mean=torch.zeros_like(gradient),
										std=torch.ones_like(gradient) * sigma)

				# Update momentum (Eq 10 right in [1])
				momentum.add_(
					- (lr ** 2) * minv_t * gradient - mdecay * momentum + sample_t
				)

				# Update parameters (Eq 10 left in [1])
				parameter.data.add_(momentum)

		return loss



class SGHMCSampler(_BaseSampler):

	def __init__(self, loss_module, num_burn_in_steps=3000, lr=0.001, 
		keep_every=1, mdecay=0.01):
		super(SGHMCSampler, self).__init__(loss_module)
		self.N = loss_module.X.shape[0]
		self.walker = AdaptiveSGHMC(loss_module.parameters(), lr=lr, 
			num_burn_in_steps=num_burn_in_steps, mdecay=mdecay, scale_grad=self.N)
		self.num_burn_in_steps = num_burn_in_steps
		self.keep_every = keep_every

	def sample(self,nsamples=1, nchains=1, **args):
		keep_every = self.keep_every
		num_burn_in_steps = self.num_burn_in_steps
		n_samples_per_chain = nsamples // nchains

		for step in range(nsamples * keep_every + num_burn_in_steps):
			#if (step+1)%(nsamples*keep_every/10)==0:
			print("Sampling {}/{}".format(step+1, nsamples * keep_every + num_burn_in_steps))
			self.walker.zero_grad()
			loss = self.loss_module() / self.N 
			loss.backward()
			## NOTE: do not use this; it rescales the gradient and it is ugly!
			## Instead, use larger mdecay or smaller lr
			##torch.nn.utils.clip_grad_norm_(self.loss_module.parameters(), 100.)
			self.walker.step()

			if (step > num_burn_in_steps) and \
					((step - num_burn_in_steps) % keep_every == keep_every-1):
				self.sampled_weights.append(self.get_weights())
				if len(self.sampled_weights) % n_samples_per_chain == 0:
					self.loss_module.initialise()
					

class LossModule(nn.Module):
	def __init__(self, model,train, loss):
		super(LossModule, self).__init__()
		self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
		self.model = model.to(self.device)
		self.X = train[0].to(self.device)
		self.labels = train[1].to(self.device)
		self.loss_fn = loss
		
	def negative_log_prior(self,params):
		regularization_term = 0
		for name, W in params:
			regularization_term += W.norm(2)
		return 0.5*regularization_term

	def parameters(self):
		return self.model.parameters()
	
	def forward(self):
		batch = random.randint(0, self.X.size()[0]-256)
		outputs = self.model(self.X[batch:batch+256])
		target = torch.unsqueeze(self.labels[batch:batch+256],1)
		loss = self.loss_fn(outputs, target) + self.negative_log_prior(self.model.named_parameters())
		return loss
	
	def initialise(self):
		pass
		