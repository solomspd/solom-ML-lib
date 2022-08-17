import cupy as xp
import cupy as cp
import numpy as np
import functools
import pdb
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import time
import datetime
import json
import scipy
from scipy import ndimage


class CNN:

    # (data_dimentions) number of data pints by number of features
    # (convolution dimentions) = how the convolutionoal part of the network is laid out. where it is an array of tupes . if a tupe is 3 longth then it is a filer with (output_dim,kernel_sz,stride) and if it is 2 long (kernel_sz,stride) then it is to be considered a maxpooling layer. Assuming all kernels and pooling layers are square.
    # (graph_dimentions) g_dim = dimentions of hidden layers. ex: 2 hidden layers (excluding input and output layers) with 10 and 5 neurons respectivley g_dim =[10,5]
    # (batch size) batch_sz can be up to input data size
    def __init__(
        self,
        d_dim,
        classes,
        c_dim,
        g_dim,
        step_sz=3e-3,
        batch_sz=None,
        loss_f="NLL",
        act_f="relu",
        in_reg_t="L2",
        reg=1e-3,
        hinge_delta=1,
        leaky_relu_alpha=0.01,
        momentum=0.9,
        accumulator=0.99,
        drop_rate=0.1,
        rand_seed=0,
    ):

        xp.random.seed(rand_seed)
        self.batch_num = 1 if batch_sz is None else math.ceil(
            d_dim[0] / batch_sz)

        self.beta_m = momentum
        self.beta_a = accumulator
        self.drop_rate = drop_rate

        # initialize wights and bias for all levels
        self.step_sz = step_sz
        self.c_dim = c_dim
        self.g_dim = g_dim
        self.W = []  # array of weights for all levels
        self.b = []  # array of biases for all levels
        self.a = []  # array of accumulators
        self.m = []  # array of momentums

        self.bm = []
        self.ba = []

        prev_res = d_dim[1]
        # previous dimention is depth of imput image
        prev_dim = d_dim[3]
        for i in range(len(c_dim)):
            if len(c_dim[i]) == 2:
                # since maxpool layers do not have weight we add a place holder value in the weights array to represent them so we can still normally index the weights later when forward and backward passing
                p_holder = xp.array(0)
                self.W.append(p_holder)
                self.a.append(p_holder)
                self.m.append(p_holder)
                self.b.append(p_holder)
                self.ba.append(p_holder)
                self.bm.append(p_holder)
                prev_res = (prev_res - c_dim[i][0])/c_dim[i][1] + 1
                continue
            f_in = prev_dim * c_dim[i][1]**2
            if act_f == "relu":
                self.W.append(xp.random.randn(
                    c_dim[i][0], c_dim[i][1], c_dim[i][1], prev_dim) / xp.sqrt(f_in / 2))
            else:
                self.W.append(xp.random.randn(
                    c_dim[i][0], c_dim[i][1], c_dim[i][1], prev_dim) / xp.sqrt(f_in))
            self.a.append(
                xp.zeros((c_dim[i][0], c_dim[i][1], c_dim[i][1], prev_dim)))
            self.m.append(
                xp.zeros((c_dim[i][0], c_dim[i][1], c_dim[i][1], prev_dim)))
            self.b.append(xp.zeros(c_dim[i][0]))
            self.ba.append(xp.zeros(c_dim[i][0]))
            self.bm.append(xp.zeros(c_dim[i][0]))
            prev_dim = c_dim[i][0]

        # temp dim to build FNN with
        temp_d_dim = [int(prev_res)**2*prev_dim]
        temp_d_dim.extend(g_dim)
        temp_d_dim.append(classes)
        self.reg = reg
        for i in range(len(temp_d_dim) - 1):
            if act_f == "relu":
                self.W.append(xp.random.randn(
                    temp_d_dim[i], temp_d_dim[i + 1]) / xp.sqrt(temp_d_dim[i] / 2))
            else:
                self.W.append(xp.random.randn(
                    temp_d_dim[i], temp_d_dim[i + 1]) / xp.sqrt(temp_d_dim[i]))
            self.a.append(xp.zeros((temp_d_dim[i], temp_d_dim[i + 1])))
            self.m.append(xp.zeros((temp_d_dim[i], temp_d_dim[i + 1])))
            self.b.append(xp.zeros((1, temp_d_dim[i + 1])))
            self.ba.append(xp.zeros((1, temp_d_dim[i + 1])))
            self.bm.append(xp.zeros((1, temp_d_dim[i + 1])))

        if in_reg_t == "L1":
            self.reg_t = lambda: functools.reduce(lambda a, b: a + self.reg * xp.sum(xp.absolute(b)), self.W,
                                                  0.0)
            self.reg_t_back = lambda x: self.reg
        elif in_reg_t == "L2":
            self.reg_t = lambda: functools.reduce(
                lambda a, b: a + self.reg * xp.sum(b * b), self.W, 0.0)
            self.reg_t_back = lambda x: self.reg * x

        if loss_f == "hinge":
            self.loss_f = self._hinge
            self.loss_f_back = self._hinge_back
            self.hinge_delta = hinge_delta
        elif loss_f == "NLL":
            self.loss_f = self._NLL
            self.loss_f_back = self._NLL_back

        if act_f == "sigmoid":
            self.act_f = self._sig
            self.act_f_back = self._sig_back
        elif act_f == "relu":
            self.act_f = lambda x: xp.maximum(0, x)
            self.act_f_back = self._relu_back
        elif act_f == "leaky_relu":
            self.act_f = lambda x, lk=leaky_relu_alpha: xp.maximum(lk * x, x)
            self.leaky_relu_alpha = leaky_relu_alpha
            self.act_f_back = self._leaky_relu_back
        elif act_f == "tanh":
            self.act_f = lambda x: xp.tanh(x)
            self.act_f_back = self._tanh_back

        # result tracking
        self.dat = []
        self.t_dat = []

        self.log = [in_reg_t, loss_f, act_f]
        self.best_t = [100, 0]
        self.best_v = [100, 0]

    # Beginning of boiler plate functions
    # TODO find a nicer way to go about boiler plate functions. theyre ugly

    # warpper function to let me test different convolution apporaches # TODO experiment with fast fourier transform
    def _convolv(self, img, kernel_f, padder, stride=1, dim=1):
        # pdb.set_trace()
        # TODO consider doing width and height independent

        padding = int(padder / 2)
        padded = xp.zeros(
            (img.shape[0], img.shape[1] + padding * 2, img.shape[2] + padding * 2, img.shape[3]))
        padded[:, padding:img.shape[1] + padding,
               padding:img.shape[2] + padding] = img

        k_sz = kernel_f.shape[1]

        n = int((img.shape[1] - k_sz + 2*padding)/stride + 1)
        res = xp.zeros((img.shape[0], n, n, dim))
        for f in range(dim):
            I = 0
            for i in range(0, padded.shape[1] - k_sz, stride):
                J = 0
                for j in range(0, padded.shape[2] - k_sz, stride):
                    # TODO find out if broadcasting will work with multiple filters (will have wonky sizes) so maybe the outer loop can be removed
                    res[:, I, J, f] = xp.sum(
                        padded[:, i:i + k_sz, j:j + k_sz] * kernel_f[f], axis=(1, 2, 3)).reshape(img.shape[0])
                    J += 1
                I += 1
        return res

    def _back_convolv(self, img, kernel_f, padder, stride, in_dim, out_dim=None):
        out_dim = img.shape[0] if out_dim is None else out_dim
        padding = int(padder / 2)
        padded = xp.zeros(
            (img.shape[0], img.shape[1] + padding * 2, img.shape[2] + padding * 2, img.shape[3]))
        padded[:, padding:img.shape[1] + padding,
               padding:img.shape[2] + padding] = img

        k_sz = kernel_f.shape[1]

        n = int((img.shape[1] - k_sz + 2*padding)/stride + 1)
        res = xp.zeros((img.shape[0], n, n, in_dim))
        for f in range(out_dim):
            I = 0
            for i in range(0, padded.shape[1] - k_sz, stride):
                J = 0
                for j in range(0, padded.shape[2] - k_sz, stride):
                    # TODO find out if broadcasting will work with multiple filters (will have wonky sizes) so maybe the outer loop can be removed
                    deriv = padded[:, i:i + k_sz, j:j + k_sz, f]
                    k = kernel_f[f]
                    res[:, I, J] += xp.sum(deriv.reshape(*deriv.shape, 1)
                                           * k.reshape(1, *k.shape), axis=(1, 2))
                    J += 1
                I += 1
        return res

    def _back_convolv2(self, img, kernel_f, padder, stride, in_dim, out_dim=None):
        out_dim = img.shape[0] if out_dim is None else out_dim
        padding = int(padder / 2)
        padded = xp.zeros(
            (img.shape[0], img.shape[1] + padding * 2, img.shape[2] + padding * 2, img.shape[3]))
        padded[:, padding:img.shape[1] + padding,
               padding:img.shape[2] + padding] = img

        k_sz = kernel_f.shape[1]

        n = int((img.shape[1] - k_sz + 2*padding)/stride + 1)
        res = xp.zeros((img.shape[0], out_dim, n, n, in_dim))
        for f in range(out_dim):
            I = 0
            for i in range(0, padded.shape[1] - k_sz, stride):
                J = 0
                for j in range(0, padded.shape[2] - k_sz, stride):
                    # TODO find out if broadcasting will work with multiple filters (will have wonky sizes) so maybe the outer loop can be removed
                    pad = padded[:, i:i + k_sz, j:j + k_sz]
                    deriv = kernel_f[:, :, :, f]
                    res[:, f, I, J] = xp.sum(
                        pad * deriv.reshape(*deriv.shape, 1), axis=(1, 2))
                    J += 1
                I += 1
        return res

    def _max_pool(self, img, k_sz, stride):
        res = xp.zeros((img.shape[0], int(
            img.shape[1] / k_sz), int(img.shape[2] / k_sz), img.shape[3]))
        I = 0
        for i in range(0, img.shape[1], stride):
            J = 0
            for j in range(0, img.shape[2], stride):
                res[:, I, J] = xp.max(
                    img[:, i:i + k_sz, j:j + k_sz], axis=(1, 2))
                J += 1
            I += 1
        return res

    def _softmax(self, x):
        exp = xp.exp(x)
        return exp / xp.sum(exp, axis=1, keepdims=True)

    def _softmax_back(self, x):
        # TODO figure out softmax backpropagation
        pass

    def _hinge(self, sc, y):
        ret = xp.maximum(0, sc - sc[y] + self.hinge_delta)
        ret[y] = 0
        return xp.sum(ret, axis=0, keepdims=True)

    def _hinge_back(self, sc):
        # TODO figure out hinge loss backpropagation function
        pass

    def _NLL(self, sc, y):
        ret = self._softmax(sc)
        return -xp.log(ret[list(range(ret.shape[0])), y])

    def _NLL_back(self, sc, y):
        ret = self._softmax(sc)
        ret[list(range(ret.shape[0])), y] -= 1
        return ret

    def _sig(self, x):
        return 1 / (1 + xp.exp(-x))

    def _sig_back(self, x, sc):
        sig = self._sig(sc)
        x *= (1 - sig) * sig

    def _relu_back(self, x, sc):
        x[sc <= 0] = 0

    def _leaky_relu_back(self, x, sc):
        x[sc <= 0] *= self.leaky_relu_alpha

    def _tanh_back(self, x, sc):
        x *= 1 - xp.tanh(sc)**2

    # End of boiler plate functions

    def forward(self, X, Y, drop):
        # we store all scores by all levels so we dont need to compute them again when back propagating
        self.sc = [X]
        # convolutional forward pass
        for i in range(len(self.c_dim)):
            if (len(self.c_dim[i]) == 2):
                self.sc.append(self._max_pool(
                    self.sc[-1], self.c_dim[i][0], self.c_dim[i][1]))
            else:
                self.sc.append(self.act_f(self._convolv(
                    self.sc[-1], self.W[i], self.c_dim[i][1], self.c_dim[i][2], self.c_dim[i][0]) + self.b[i]))

        # fully connected forward pass

        # convert from CNN form to FNN form
        # self.sc.append(self.sc[-1].reshape(self.sc[-1].shape[0], -1))
        # self.sc[-1].flatten()
        self.boundary_shape = self.sc[-1].shape
        self.sc[-1] = self.sc[-1].reshape(self.sc[-1].shape[0], -1)
        # compute all scores for hidden layers
        for i in range(len(self.c_dim), len(self.c_dim) + len(self.g_dim)):
            self.sc.append((self.act_f(
                xp.dot(self.sc[-1], self.W[i]) + self.b[i])) * drop[i - len(self.c_dim)])

            # last layer is exeption since we do not apply acitication function on it so it is not in the for loop
        self.sc.append(xp.dot(self.sc[-1], self.W[-1]) + self.b[-1])
        # compute score for output layer
        guess = xp.mean(xp.argmax(self.sc[-1], axis=1) == Y)
        return float(xp.mean(self.loss_f(self.sc[-1], Y)) + self.reg_t()), float(guess)

    def backward(self, Y, drop):
        # first, get drivative of loss function
        deriv = self.loss_f_back(self.sc[-1], Y)
        deriv /= deriv.shape[0]
        # for convenience remove final scores (we wont use them again and makes indexing easier later)
        self.sc.pop()
        dW = []
        db = []
        # calculate dW from loss function derivative
        dW.insert(0, xp.dot(self.sc[-1].T, deriv) +
                  self.reg_t_back(self.W[-1]))
        # calculate db from loss function derivative
        db.append(xp.sum(deriv, axis=0, keepdims=True))
        for i in reversed(range(len(self.c_dim), len(self.g_dim) + len(self.c_dim))):
            # get derivative of activations of next layer
            deriv = xp.dot(deriv, self.W[i + 1].T)
            # use activation derivatives same way we used scores
            self.act_f_back(deriv, self.sc[i + 1])
            deriv *= drop[i - len(self.c_dim)]
            # store dW for later application so we do not affect the currect value that would change the result of the next itteration
            dW.insert(0, xp.dot(self.sc[i].T, deriv) +
                      self.reg_t_back(self.W[i]))
            # like dW, store db for later application
            db.insert(0, xp.sum(deriv, axis=0, keepdims=True))

        deriv = xp.dot(deriv, self.W[i].T)
        deriv = deriv.reshape(self.boundary_shape)
        self.sc[i] = self.sc[i].reshape(self.boundary_shape)
        for i in reversed(range(len(self.c_dim))):
            if len(self.c_dim[i]) == 2:
                tmp = xp.zeros(self.sc[i].shape)
                if i != len(self.c_dim)-1:
                    fliped = xp.flip(self.W[i+1], axis=(1, 2))
                    deriv = self._back_convolv(
                        deriv, fliped, self.c_dim[i+1][1], self.c_dim[i+1][2], tmp.shape[3], self.c_dim[i+1][0])
                k_sz = self.c_dim[i][0]
                stride = self.c_dim[i][1]
                for m in range(tmp.shape[3]):
                    for j in range(0, tmp.shape[1], stride):
                        for k in range(0, tmp.shape[2], stride):
                            tmp[:, i:i+k_sz, j:j+k_sz, m] = xp.tile(
                                xp.max(self.sc[i][:, i:i+k_sz, j:j+k_sz, m], axis=(1, 2)), (2, 2, 1)).T
                tmp = tmp == self.sc[i]
                tmp = tmp.astype(float)
                for m in range(tmp.shape[3]):
                    I = 0
                    for j in range(0, tmp.shape[1], stride):
                        J = 0
                        for k in range(0, tmp.shape[2], stride):
                            tmp[:, i:i+k_sz, j:j+k_sz,
                                m] *= xp.tile(deriv[:, I, J, m].T, (2, 2, 1)).T
                            J += 1
                        I += 1
                deriv = tmp
                dW.insert(0, xp.array(0))
                db.insert(0, xp.array(0))
                continue
            if i < len(self.c_dim)-1 and (self.W[i+1] != 0).all():
                fliped = xp.flip(self.W[i+1], axis=(1, 2))
                deriv = self._back_convolv(
                    deriv, fliped, self.c_dim[i+1][1], self.c_dim[i+1][2], tmp.shape[3], self.c_dim[i+1][0])
            self.act_f_back(deriv, self.sc[i + 1])
            dW.insert(0, xp.sum(self._back_convolv2(self.sc[i], deriv, self.c_dim[i][1], self.c_dim[i]
                      [2], self.sc[i].shape[3], self.c_dim[i][0]), axis=0) + self.reg_t_back(self.W[i]))
            db.insert(0, xp.sum(deriv))

        # apply dW and db on each of their respective weights and baises
        for i in range(len(self.c_dim) + len(self.g_dim) + 1):
            if i < len(self.c_dim) and len(self.c_dim[i]) == 2:
                continue
            self.m[i] = self.beta_m * self.m[i] + (1 - self.beta_m) * dW[i]
            self.a[i] = self.beta_a * self.a[i] + \
                (1 - self.beta_a) * (dW[i] * dW[i])
            self.W[i] -= self.step_sz * self.m[i] / (xp.sqrt(self.a[i]) + 1e-7)

            self.bm[i] = self.beta_m * self.bm[i] + (1 - self.beta_m) * db[i]
            self.ba[i] = self.beta_a * self.ba[i] + \
                (1 - self.beta_a) * (db[i] * db[i])
            self.b[i] -= self.step_sz * self.bm[i] / \
                (xp.sqrt(self.ba[i]) + 1e-7)

    # A nice wrapper to do both forward and back propagations
    def epoch(self, X, Y, t_X, t_Y):
        # pdb.set_trace()
        shuff = xp.random.permutation(len(X))
        shuff_X, shuff_Y = X[shuff], Y[shuff]

        # get indecies of random nodes to drop
        drop = []
        for i in self.g_dim:
            temp_drop = xp.ones(i)
            temp_drop[:int(self.drop_rate * i)] = 0
            xp.random.shuffle(temp_drop)
            drop.append(temp_drop)

        shuff_X, shuff_Y = xp.array_split(
            shuff_X, self.batch_num), xp.array_split(shuff_Y, self.batch_num)
        prog = tqdm(range(len(shuff_X)), leave=False)
        for i in prog:
            res = self.forward(shuff_X[i], shuff_Y[i], drop)
            self.backward(shuff_Y[i], drop)
            if t_X is not None:
                valid = self.eval(t_X, t_Y)
            else:
                valid = (float("nan"), float("nan"))
            self.t_dat.append(valid[:2])
            self.dat.append(res)
            prog.set_description(
                "loss: {:10.5f}, Acc: {:5.2f}%, test loss: {:10.5f}, test Acc: {:4.2f}%".format(
                    float(res[0]),
                    float(res[1]) * 100,
                    float(valid[0]),
                    float(valid[1]) * 100,
                ))

        return res, valid

    def train(self, X, Y, t_X, t_Y, itter=-1, time_limit=math.inf):
        # pdb.set_trace()
        prog = tqdm(True) if itter == -1 else tqdm(range(itter))
        time_limit += time.time()
        try:
            i = 0
            while i != itter and time.time() < time_limit:
                res, valid = self.epoch(X, Y, t_X, t_Y)
                prog.set_description(
                    "loss: {:10.5f}, Acc: {:5.2f}%, test loss: {:10.5f}, test Acc: {:4.2f}%".format(
                        float(res[0]),
                        float(res[1]) * 100,
                        float(valid[0]),
                        float(valid[1]) * 100,
                    ))
                prog.update(1)
                i += 1
        except KeyboardInterrupt:
            print("\nend of trianing")
        self.i = i
        t = np.array(self.dat)
        v = np.array(self.t_dat)
        return v.min(0)[0], v.max(0)[1]

    def eval(self, X, Y=None):
        eval_sc = [X]
        # convolutional forward pass
        for i in range(len(self.c_dim)):
            if (len(self.c_dim[i]) == 2):
                eval_sc.append(self._max_pool(
                    eval_sc[-1], self.c_dim[i][0], self.c_dim[i][1]))
            else:
                eval_sc.append(self.act_f(self._convolv(
                    eval_sc[-1], self.W[i], self.c_dim[i][1], self.c_dim[i][2], self.c_dim[i][0]) + self.b[i]))

                # fully connected forward pass

        # convert from CNN form to FNN form
        eval_boundary_shape = eval_sc[-1].shape
        eval_sc[-1] = eval_sc[-1].reshape(eval_sc[-1].shape[0], -1)
        # compute all scores for hidden layers
        for i in range(len(self.c_dim), len(self.c_dim) + len(self.g_dim)):
            eval_sc.append(
                (self.act_f(xp.dot(eval_sc[-1], self.W[i]) + self.b[i])))

        # last layer is exeption since we do not apply acitication function on it so it is not in the for loop
        eval_sc.append(xp.dot(eval_sc[-1], self.W[-1]) + self.b[-1])
        # compute score for output layer
        guess = xp.argmax(eval_sc[-1], axis=1)

        if Y is None:
            return guess
        else:
            return (float(xp.mean(self.loss_f(eval_sc[-1], Y)) + self.reg_t()), float(xp.mean(guess == Y)), guess)

    def plot(self):
        self.fig, (self.loss, self.acc) = plt.subplots(2)
        t = np.array(self.dat)
        v = np.array(self.t_dat)
        self.loss.set_title("Loss")
        self.loss.plot(range(t.shape[0]), t[:, 0], label="training")
        self.loss.plot(range(v.shape[0]), v[:, 0], label="testing")
        self.loss.set_xlabel("Epoch")
        self.loss.set_ylabel("Loss")
        self.loss.legend()
        self.acc.set_title("Accuracy")
        self.acc.plot(range(t.shape[0]), t[:, 1] * 100, label="training")
        self.acc.plot(range(v.shape[0]), v[:, 1] * 100, label="testing")
        self.acc.set_xlabel("Epoch")
        self.acc.set_ylabel("% Accuracy")
        self.acc.legend()
        self.fig.tight_layout()
        plt.show()
        plt.pause(0.01)

    def log_results(self, resolution, seed):
        t = np.array(self.dat)
        v = np.array(self.t_dat)
        now = datetime.datetime.now()
        self.fig.savefig("./results/" + now.strftime("%dth-%H_%M_%S.png"))
        log_line = f"resolution: {resolution[0]}x{resolution[1]}\nconvolutin shape: {self.c_dim}\nhidden_layer shapes: {self.g_dim}\nActivation function: {self.log[2]}, Loss function: {self.log[1]}\nbest traning loss: {t.min(0)[0]}, best training accuracy: {t.max(0)[1]}, best validation loss: {v.min(0)[0]} ({v.argmin(0)[0]}), best validation accuracy: {v.max(0)[1]} ({v.argmax(0)[0]})\nItteration: {self.i}\nlearning rate: {self.step_sz}, regularisation: {self.reg}, regularisation type: {self.log[0]}, momentum: {self.beta_m}, accumulator: {self.beta_a}, drop out: {self.drop_rate}, leaky relu: {self.leaky_relu_alpha if 'self.leaky_relu_alpha' in locals() else 'n/a'}, hinge delata: {self.hinge_delta if 'self.hinge_delta' in locals() else 'n/a'}, seed: {seed}\n\n\n\n\n\n\n"
        print(log_line)
        with open("results.txt", "a") as res_file:
            res_file.write(log_line)
