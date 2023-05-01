import numpy as np
from scipy.special import gamma, loggamma, binom
from sklearn.metrics import pairwise_distances
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import matplotlib
import glob

rng = np.random.default_rng()

font = {'family': 'normal',
        'size': 22}

matplotlib.rc('font', **font)


def log_wasserstein(n, m, R, d):
    return 0.5 * np.log(m / 6) + (np.log(2) + (m + 1) / 2 * np.log(np.pi) + np.log(d) - loggamma((m + 1) / 2)) / m \
        + np.log(R) - np.log(n) / m


def count_eigenvalues(m, M):
    K = (np.sqrt((m - 1) * (m - 1) + 4 * M * M) - (m - 1)) / 2
    return K, binom(m + K, m) + binom(m + K - 1, m)


def difficulty(error, n, m, R, prod_d, delta, D, B):
    '''
    Compute the difficulty of a task in bits.
    params:
        error: desired error rate, used as epsilon / L_L
        n: size of training set
        m: intrinsic dimensionality of the data
        R: maximum norm of data points
        prod_d: prod_i M(d_i), the total number of combinations of discrete features
        delta: spatial resolution of f, the minumum distance between classes
        D: dimensionality of output
        B: upper bound on L-infinity norm of f's output
    '''
    logW = log_wasserstein(n, m, R, prod_d)
    K, num_eigenvalues = count_eigenvalues(m, 2 * np.pi * R / delta)
    Lf = K * np.sqrt(D) / R
    dim_Theta = 2 * D * prod_d * num_eigenvalues
    dim_Theta_q = dim_Theta - n * D
    log_ratio = np.log2(B) + 0.5 * np.log2(D) + 0.5 * np.log2(prod_d) - np.log2(error) + np.log2(Lf) + logW / np.log(2)
    return dim_Theta_q * log_ratio


def evt_delta(data, labels, num_samples=4000000, k=2000):
    '''
    Estimate delta using extreme value theory.
    '''
    n = data.shape[0]
    num_classes = len(set(labels))
    counter = 0
    samples = []
    while counter < num_samples:
        i = rng.integers(n)
        j = rng.integers(n)
        if labels[i] != labels[j]:
            dist = np.linalg.norm(data[i] - data[j])
            samples.append(1 / dist)
            counter += 1
    samples.sort()
    order_k = samples[-k - 1]
    extreme_samples = np.array(samples[-k:])
    print(f'Sample minimum = {1 / extreme_samples[-1]}')
    diff_logs = np.log(extreme_samples) - np.log(order_k)
    M1 = np.sum(diff_logs) / k
    M2 = np.sum(diff_logs * diff_logs) / k
    evi = M1 + 1 - 0.5 / (1 - M1 * M1 / M2)
    print(f'gamma = {evi}')
    pairs = n * n * (1 - 1 / num_classes)
    a = k / num_samples * pairs
    delta = 1 / ((a ** evi - 1) / evi * order_k * M1 + order_k)
    return min(delta, 1 / extreme_samples[-1])


def sample_delta(data, labels, num_samples=30000):
    '''
    Estimate delta by computing it for a random subset of the data.
    '''
    n = data.shape[0]
    indices = rng.choice(n, num_samples, replace=False)
    sample_data = data.reshape((n, -1))[indices]
    sample_labels = labels[indices]
    distances = pairwise_distances(sample_data)
    X, Y = np.meshgrid(sample_labels, sample_labels, indexing='ij')
    masked = np.ma.array(distances, mask=np.logical_or(X == Y, distances < 1))
    return np.min(masked)


def intrinsic_dim(data, num_samples=10000, k=5):
    '''
    Compute the MLE estimate of intrinsic dimensionality.
    '''
    sample_data = rng.choice(data, num_samples, replace=False).reshape((num_samples, -1))
    distances = pairwise_distances(sample_data, data.reshape((data.shape[0], -1)))
    distances.sort()
    log_ratios = np.log(distances[:, k:k + 1]) - np.log(distances[:, 1:k])
    return np.mean(log_ratios) ** -1


def classification_task_difficulty(data, labels, error, m=None, delta='evt', verbose=True):
    '''
    Compute the difficulty of a classification task in bits.
    params:
        data: 3D numpy array of data points along first dimension
        labels: 1D numpy array of the corresponding labels
        error: desired error rate
        m: intrinsic dimensionality of the data, will be estimated from the data if not provided
        delta: method used to estimate delta, can be 'evt', 'sample', or a number
        verbose: prints values used to compute difficulty if True
    '''
    n = data.shape[0]
    R = np.max(np.linalg.norm(data, axis=(1, 2)))
    num_classes = len(set(labels))
    prod_d = num_classes
    D = num_classes - 1
    if verbose:
        print(f'{n} data points')
        print(f'R = {R}')
        print(f'{num_classes} classes')

    if m is None:
        m = np.round(intrinsic_dim(data))
    if verbose:
        print(f'Intrinsic dim. = {m}')

    # estimating delta
    if delta == 'sample':
        delta = sample_delta(data, labels)
    elif delta == 'evt':
        delta = evt_delta(data, labels)
    if verbose:
        print(f'delta = {delta}')
        print(f'Max frequency = {2 * np.pi * R / delta}')

    return difficulty(error, n, m, R, prod_d, delta, D, 1)


def omniglot_difficulty(error, n, m0, m1, R, delta):
    '''
    n: number of training points (sets of 20 images)
    m0: dimensionality of images within an alphabet
    m1: dimensionality of images across alphabets
    '''
    dim_g = 2 * 20 * 19 * count_eigenvalues(m0, 2 * np.pi * R / delta)[1]
    return difficulty(error, n, m1 + 19 * m0, R * np.sqrt(20), 1, delta, dim_g, np.sqrt(20 * 19))


def combined_task_diff(n1, n2, R1, R2, m1, m2, D1, D2, delta, error, verbose=True):
    """
    Compute the difficulty of combinations of classification tasks in bits.
    params:
        data: 3D numpy array of data points along first dimension
        labels: 1D numpy array of the corresponding labels
        error: desired error rate
        m: intrinsic dimensionality of the data, will be estimated from the data if not provided
        delta: method used to estimate delta, can be 'evt', 'sample', or a number
        verbose: prints values used to compute difficulty if True
    """
    if verbose:
        print(f'{n1} data points')
        print(f'R = {R1}')
        print(f'{D1} classes')

    if verbose:
        print(f'Intrinsic dim. = {m1}')

    if verbose:
        print(f'delta = {delta}')
        print(f'Max frequency = {2 * np.pi * np.sqrt(R1 ** 2 + R2 ** 2) / delta}')

    return difficulty(error, n1 * n2, m1 + m2, np.sqrt(R1 ** 2 + R2 ** 2), D1 * D2, delta, D1, 1)


if __name__ == '__main__':
    # Task difficulty computation for Omniglot

    omniglot_folder = './Omniglot/'
    alphabets = [folder[len(omniglot_folder):] for folder in glob.iglob(omniglot_folder + '*')]
    chars_per_alph = [0 for _ in range(len(alphabets))]
    for file in glob.iglob(omniglot_folder + '**/*.png', recursive=True):
        _, alph, char, _ = file.split('\\')
        alph = alphabets.index(alph)
        char = int(char[9:])
        chars_per_alph[alph] = max(char, chars_per_alph[alph])
    n = sum([binom(max(x, 20), 20) * 20 ** 20 for x in chars_per_alph])
    m0 = 22
    m1 = 29
    R = 49.4064773081425
    delta = 5.477225575051661
    omni_diff = omniglot_difficulty(0.01, n, m0, m1, R, delta)
    print(omni_diff)

    # Task difficulty computation for image classification benchmarks

    mnist = datasets.MNIST(
        root='./datasets',
        transform=ToTensor(),
        download=True
    )
    mnist_data = mnist.data.numpy() / 255
    mnist_labels = mnist.targets.numpy()
    print('Computing MNIST difficulty')
    mnist_difficulty = classification_task_difficulty(mnist_data, mnist_labels, 0.001, m=14, delta=2.3987084824335305)
    print(f'MNIST difficulty = {"{:.2e}".format(mnist_difficulty)} bits')

    svhn = datasets.SVHN(
        root='./datasets',
        transform=ToTensor(),
        download=True
    )
    data_shape = svhn.data.shape  # (73257, 3, 32, 32)
    svhn_data = svhn.data.reshape((data_shape[0], data_shape[1], -1)) / 255
    svhn_labels = svhn.labels
    print('Computing SVHN difficulty')
    svhn_difficulty = classification_task_difficulty(svhn_data, svhn_labels, 0.01, m=19, delta=1.5766998174969105)
    print(f'SVHN difficulty = {"{:.2e}".format(svhn_difficulty)} bits')

    cifar10 = datasets.CIFAR10(
        root='./datasets',
        transform=ToTensor(),
        download=True
    )
    data_shape = cifar10.data.shape  # (50000, 32, 32, 3)
    cifar10_data = cifar10.data.reshape((data_shape[0], -1, data_shape[-1])) / 255
    cifar10_labels = np.array(cifar10.targets)
    print('Computing CIFAR10 difficulty')
    cifar10_difficulty = classification_task_difficulty(cifar10_data, cifar10_labels, 0.01, m=27,
                                                        delta=2.7506414616587023)
    print(f'CIFAR10 difficulty = {"{:.3e}".format(cifar10_difficulty)} bits')

    print('Computing ImageNet difficulty')
    n, num_classes, m, R, delta = 1280901, 1000, 48, 943.5923, 65
    imagenet_difficulty = difficulty(0.1, n, m, R, num_classes, delta, num_classes - 1, 1)
    print(f'ImageNet difficulty = {"{:.3e}".format(imagenet_difficulty)} bits')

    diffs = []
    for logn in range(2, 20):
        n = 10 ** logn
        print(f'{n} data points')
        print(f'R = {R}')
        print(f'{num_classes} classes')
        print(f'Intrinsic dim. = {m}')
        print(f'delta = {delta}')
        print(f'Max frequency = {2 * np.pi * R / delta}')
        imagenet_difficulty = difficulty(0.1, n, m, R, num_classes, delta, num_classes - 1, 1)
        print(f'ImageNet difficulty = {"{:.2e}".format(imagenet_difficulty)} bits')
        diffs.append(imagenet_difficulty)
    print(diffs)

    plt.plot([10 ** logn for logn in range(2, 20)], diffs, linewidth=3)
    plt.xlabel('Number of Samples')
    plt.ylabel('Task Difficulty (bits)')
    plt.xscale('log')
    plt.tight_layout()
    plt.show()

    n = 1280901
    diffs = []
    for m in range(48, 100):
        print(m)
        imagenet_difficulty = difficulty(0.1, n, m, R, num_classes, delta, num_classes - 1, 1)
        print(imagenet_difficulty)
        diffs.append(imagenet_difficulty)

    plt.plot([m for m in range(48, 100)], diffs, linewidth=3)
    plt.xlabel('Intrinsic Dimensions')
    plt.ylabel('Task Difficulty (bits)')
    plt.yscale('log')
    plt.tight_layout()
    plt.show()

    # Inductive bias information content for models achieving different error rates

    errors = [0.97, 0.375, 0.2502, 0.2285, 0.2258, 0.219, 0.2142, 0.1246, 0.1145, 0.0906]
    for error in errors:
        print(difficulty(error, n, m, R, num_classes, delta, num_classes - 1, 1))

    # Example code to estimate delta for CIFAR-10

    from scipy.stats import t


    def confidence_interval(data):
        n = data.shape[1]
        means = data.mean(axis=1)
        ses = data.std(axis=1, ddof=1) / n ** 0.5
        tp = t.ppf(0.975, n - 1)
        return means, means - tp * ses, means + tp * ses


    nums = [100, 200, 500, 1000, 2000, 5000, 10000]
    ests = []
    for num_samples in nums:
        trials = []
        for i in range(30):
            if i % 10 == 0 and num_samples >= 2000:
                print(num_samples, i)
            trials.append(sample_delta(cifar10_data, cifar10_labels, num_samples=num_samples))
        ests.append(trials)
    ests = np.array(ests)
    means, lowers, uppers = confidence_interval(np.log(ests))
    plt.plot(nums, np.exp(means), color='darkturquoise')
    plt.fill_between(nums, [max(x, 2.7506414616587023) for x in np.exp(lowers)], np.exp(uppers), color='paleturquoise')
    plt.plot([100, 10000], [2.7506414616587023, 2.7506414616587023], 'k--')
    plt.xlabel('Number of Samples')
    plt.ylabel('Sample delta')
    plt.xscale('log')
    plt.show()
    ks = [3, 5, 10, 20]
    nums = [100, 200, 500, 1000, 2000]
    ests = []
    for num_samples in nums:
        to_add = []
        for k in ks:
            print(num_samples, k)
            trials = []
            for _ in range(10):
                trials.append(intrinsic_dim(cifar10_data, num_samples=num_samples, k=k))
            to_add.append(trials)
        ests.append(to_add)
    ests = np.array(ests)
    vals = ests.mean(axis=2)
    colors = ['springgreen', 'cyan', 'dodgerblue', 'navy']
    for i in range(len(ks)):
        k = ks[i]
        plt.plot(nums, vals[:, i], color=colors[i], label=f'k={k}')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4)
    plt.xlabel('Number of Samples')
    plt.ylabel('Dimensionality Estimate')
    plt.xscale('log')
    plt.show()
    ks = [3, 4, 5, 10, 15, 20, 25]
    ests = []
    for k in ks:
        trials = []
        for i in range(30):
            if i % 10 == 0:
                print(k, i)
            trials.append(intrinsic_dim(cifar10_data, num_samples=1000, k=k))
        ests.append(trials)
    ests = np.array(ests)
    means, lowers, uppers = confidence_interval(ests)
    plt.plot(ks, means, color='darkturquoise')
    plt.fill_between(ks, lowers, uppers, color='paleturquoise')
    plt.xlabel('k')
    plt.ylabel('Dimensionality Estimate')
    plt.show()


    # Task difficulty computation for simplified Cartpole task
    def rl_difficulty(T):
        # n: 100 episodes with length 100
        # delta: 0.001 radians = 0.057 degrees
        # error: 1 mistake every 1000 timesteps

        n, delta, error = 10000, 0.001, 0.001
        return (2 * (2 * np.pi / delta) ** (2 * T) * np.pi ** T / gamma(T + 1) - n) * \
            (np.log2(4 * np.pi * np.pi / np.sqrt(3)) + 0.5 * np.log2(T)
             - np.log2(delta) - np.log2(n) / (2 * T) - np.log2(error))


    diffs = [rl_difficulty(T) for T in range(1, 6)]
    print(diffs)
    plt.plot(range(1, 6), diffs, marker='o', linewidth=3)
    plt.xticks(range(1, 6), fontsize=22)
    plt.xlabel('Number of Observations', fontsize=22)
    plt.ylabel('Task Difficulty (bits)', fontsize=22)
    plt.yscale('log')
    plt.yticks([10.0 ** x for x in range(8, 44, 4)], fontsize=22)
    plt.tight_layout()
    plt.show()


    # Task difficulty computation for MuJoCo tasks
    def mujoco_difficulty(m, D, n=1000000, delta=0.001, error=0.001):
        # params:
        # m: dimensionality of observation space
        # D: dimensionality of action space
        # n: number of timesteps seen
        # delta: spatial resolution
        # error: desired distance from optimal action

        return D * (2 * (2 * np.pi / delta) ** m * np.pi ** (m / 2) / gamma(m / 2 + 1) - n) * \
            (np.log2(4 * np.pi * np.pi / np.sqrt(6)) + np.log2(D) + 0.5 * np.log2(m) - np.log2(delta) - np.log2(
                n) / m - np.log2(error))


    reacher_diff = mujoco_difficulty(6, 2)
    hopper_diff = mujoco_difficulty(15, 4)
    cheetah_diff = mujoco_difficulty(17, 6)
    tasks = ['Reacher', 'Hopper', 'Half-Cheetah']
    difficulties = [reacher_diff, hopper_diff, cheetah_diff]
    print(difficulties)
    plt.bar(tasks, difficulties, log=True)
    plt.xticks(tasks, fontsize=16)
    plt.ylabel('Task Difficulty (bits)', fontsize=16)
    plt.yticks([10.0 ** x for x in range(25, 70, 5)], fontsize=16)
    plt.tight_layout()
    plt.show()

    reacher_diffs = []
    error_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001]
    for error_rate in error_rates:
        reacher_diff = mujoco_difficulty(6, 2, error=error_rate)

        reacher_diffs.append(reacher_diff)

    plt.plot(error_rates, reacher_diffs, linewidth=3)
    plt.xlabel('Desired Error Rate', fontsize=22)
    plt.xscale('log')
    plt.ylabel('Task Difficulty (bits)', fontsize=22)
    plt.tight_layout()
    plt.show()

    # Task difficulty computation for task unions

    # MNIST, SVHN, CIFAR10, ImageNet
    task_data = [
        (60000, 17.179045827183984, 14, 10, 2.3987084824335305, 0.001),
        (73257, 53.86932485133339, 19, 10, 1.5766998174969105, 0.01),
        (50000, 54.9156786783691, 27, 10, 2.7506414616587023, 0.01),
        (1280901, 943.5923, 48, 1000, 65, 0.1)]
    for i in range(len(task_data)):
        diffs = []
        for j in range(len(task_data)):
            n1, R1, m1, D1, delta1, error1 = task_data[j]
            n2, R2, m2, D2, delta2, error2 = task_data[i]
            diff = combined_task_diff(n1, n2, R1, R2, m1, m2, D1, D2, delta1, error1, verbose=False)
            diffs.append(str(diff))
        print(','.join(diffs))

    # Task difficulty computation with a varying number of classes on ImageNet

    n, num_classes, m, R, delta = 1280901, 1000, 48, 943.5923, 65
    diffs = []
    for num_classes in range(10, 1000):
        print(num_classes)
        imagenet_difficulty = difficulty(0.1, n, m, R, num_classes, delta, num_classes - 1, 1)
        diffs.append(imagenet_difficulty)

    plt.plot([num_classes for num_classes in range(10, 1000)], diffs, linewidth=3)
    plt.xlabel('Number of Classes')
    plt.ylabel('Task Difficulty (bits)')
    plt.yscale('log')
    plt.tight_layout()
    plt.show()

    # Task difficulty computation with a varying spatial resolution on ImageNet

    n, num_classes, m, R, delta = 1280901, 1000, 48, 943.5923, 65
    diffs = []
    for delta in range(30, 120):
        imagenet_difficulty = difficulty(0.1, n, m, R, num_classes, delta, num_classes - 1, 1)
        diffs.append(imagenet_difficulty)

    plt.plot([delta for delta in range(30, 120)], diffs, linewidth=3)
    plt.xlabel(r'$\delta$')
    plt.ylabel('Task Difficulty (bits)')
    plt.yscale('log')
    plt.gca().get_xaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.tight_layout()
    plt.show()
