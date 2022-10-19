class Hyper_params:
	def __init__(self, **kwargs):
		self.hyper_params = {}

		for key, value in kwargs.items():
			self.hyper_params[key] = value

	def __getattr__(self, key):
		if key not in self.hyper_params:
			raise AttributeError("'HParams' object has no attribute %s" % key)
		return self.hyper_params[key]

	def set_hyper_param(self, key, value):
		self.hyper_params[key] = value


# Default hyperparameters
h_params = Hyper_params(
	batch_size = 16,
	l_rate = 1e-4,
    n_epochs = 20,
    img_size_x = 384,
	img_size_y = 512,
    n_features = 2048,
	max_frames = 50
)

def hparams_debug_string():
	values = h_params.hyper_params.values()
	hp = ["  %s: %s" % (name, values[name]) for name in sorted(values) if name != "sentences"]
	return "Hyperparameters:\n" + "\n".join(hp)