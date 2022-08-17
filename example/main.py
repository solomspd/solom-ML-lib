from pathlib import Path
from PIL import Image
import cupy as xp
import numpy as np
import cupy as cp
from tqdm import tqdm
import pdb
import matplotlib.pyplot as plt
from CNN import CNN


def load_pic(x, y, pic):
	img = Image.open(pic)
	img = img.resize(resolution)
	img = np.array(img, dtype="float64")
	if per_pic_mean:
		temp = np.mean(img, axis=2)
		temp = np.repeat(temp[:, :, np.newaxis], 3, axis=2)
		img -= temp
	x.append(img)


resolution = (32, 32)

#### uncomment for more memory (unified memory)
# pool = cp.cuda.MemoryPool(cp.cuda.memory.malloc_managed) # get unified pool
# cp.cuda.set_allocator(pool.malloc) # set unified pool as default allocator

seed = 42
xp.random.seed(seed)

per_pic_mean = False

with Path("/mnt/ramdisk/data") as data:
	x, y = [], []
	t_x, t_y = [], []
	flrs = []
	classes = 0

	for i, I in tqdm(list(enumerate(data.iterdir())), desc="loading images"):
		classes += 1
		flrs.append(I.name)
		for j in I.glob("*.jpg"):
			load_pic(x, y, j)
			y.append(i)
		for j in (I / "test").iterdir():
			load_pic(t_x, t_y, j)
			t_y.append(i)

	x, y = xp.array(x, dtype="float64"), xp.array(y)
	t_x, t_y = xp.array(t_x, dtype="float64"), xp.array(t_y)

	# normalize data
	if not per_pic_mean:
		x -= xp.mean(x, axis=0)
	x /= xp.std(x, axis=0)
	if not per_pic_mean:
		t_x -= xp.mean(t_x, axis=0)
	t_x /= xp.std(t_x, axis=0)

	tune = False

	if not tune:
		nn = CNN(x.shape,
		         classes, [(2,3,1),(2,2)], [500] * 3,
		         step_sz=3e-5,
				 reg=1e-2,
		         act_f="leaky_relu",
		         rand_seed=seed,drop_rate=0.3)
		res = nn.train(x, y, t_x, t_y)
		nn.plot()
		nn.log_results(resolution, seed)

		loss, acc, evaluator = nn.eval(t_x, t_y)
		eval_x = ["Average"]
		eval_y = [acc * 100]
		print("Loss of testing data:", loss)
		print("Accuracy:", acc)
		for i, I in enumerate(flrs):
			eval_x.append(I)
			eval_y.append(float(xp.mean(evaluator[t_y == i] == t_y[t_y == i]) * 100))
			plt.bar(eval_x, eval_y)
		plt.show()
	else:
		test = []
		itt = []
		idx = xp.random.permutation(x.shape[0])
		x = x[idx]
		y = y[idx]
		s_x = xp.array_split(x, 5)
		s_y = xp.array_split(y, 5)
		t_x = s_x.pop(0)
		t_y = s_y.pop(0)
		x = xp.concatenate((s_x), axis=0)
		y = xp.concatenate((s_y), axis=0)
		for i in range(50):
			lr = 3e-3#10**np.random.uniform(-5, -4)
			reg = 1e-3#10**np.random.uniform(-4, -2)
			leaky = 0.01#10**np.random.uniform(-4, -1)
			momentum = 0.9#10**np.random.uniform(-0.2, 0)
			accumulator = 0.99#np.random.uniform(0.8, 1)
			drop = 10**np.random.uniform(-2, -0.5)
			FC = [500] * 2
			conv = [(4, 3, 1)]
			nn = CNN(x.shape,
			         classes,
			         conv,
			         FC,
			         step_sz=lr,
			         reg=reg,
			         leaky_relu_alpha=leaky,
			         momentum=momentum,
			         accumulator=accumulator,
			         drop_rate=drop,
			         act_f="leaky_relu",
			         rand_seed=seed)
			print("\nreg:", reg, "lr:", lr, "leaky:", leaky, "momentum:", momentum, "accumulator:", accumulator,"drop:", drop)
			test.append(nn.train(x, y, t_x, t_y, 10))
			# nn.plot()
			itt.append([reg, lr, leaky, momentum, accumulator, drop])
		test = np.array(test)
		itt = np.array(itt)
		idx = test[:, 0].argsort()[:5]
		print("Best validation (loss):", test[idx], itt[idx])
		print("Average of best 5 (loss):", np.mean(itt[idx], axis=0))
		idx = test[:, 1].argsort()[-5:]
		print("Best validation (accuracy):", test[idx], itt[idx])
		print("Average of best 5 (accuracy):", np.mean(itt[idx], axis=0))
		plt.xscale("log")
		plt.xlabel("parameter")
		plt.ylabel("loss")
		plt.scatter(itt[:,5], test[:,0])
		plt.show()