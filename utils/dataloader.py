import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataLoader(object):
    def __init__(self, name):
        self.name = name
        if name == 'susy':
            self.df = pd.read_csv(f'uci/large/SUSY.csv', header=None)
            self.preprocess_susy()
        if name == 'higgs':
            self.df = pd.read_csv(f'uci/large/HIGGS.csv', header=None)
            self.preprocess_susy()
        if name == 'heart':
            self.df = pd.read_csv(f'uci/heart.csv')
            self.preprocess_heart()
        elif name == 'breast':
            self.df = pd.read_csv(f'uci/breast-cancer.data', header=None, sep=',')
            self.preprocess_breast()
        elif name == 'breast2':
            self.df = pd.read_csv(f'uci/breast.csv')
            self.preprocess_breast2()
        elif name == 'german':
            self.df = pd.read_csv('uci/german.data', header=None, sep=' ')
            self.preprocess_german()
        elif name == 'banana':
            self.df = pd.read_csv('uci/banana.csv')
        elif name == 'image':
            self.df = pd.read_csv('uci/image.csv')
        elif name == 'titanic':
            self.df = pd.read_csv('uci/titanic.csv')
        elif name == 'thyroid':
            self.df = pd.read_csv('uci/thyroid.csv')
        elif name == 'twonorm':
            self.df = pd.read_csv('uci/twonorm.csv')
        elif name == 'waveform':
            self.df = pd.read_csv('uci/waveform.csv')
        elif name == 'flare-solar':
            self.df = pd.read_csv('uci/flare-solar.csv')
            self.categorical()
        elif name == 'waveform':
            self.df = pd.read_csv('uci/waveform.csv')
        elif name == 'splice':
            self.df = pd.read_csv('uci/splice.csv')
            self.categorical()
        elif name == 'diabetes':
            self.df = pd.read_csv('uci/diabetes.csv')
            self.preprocess_diabetes()

    def load(self, path):
        df = open(path).readlines()
        df = list(map(lambda line: list(map(float, line.split())), df))
        self.df = pd.DataFrame(df)
        return self

    def categorical(self):
        self.df = onehot(self.df, [col for col in self.df.columns if col != 'target'])

    def preprocess_susy(self):
        self.df.rename(columns={0: 'target'}, inplace=True)

    def preprocess_heart(self):
        self.df = onehot(self.df, ['cp', 'slope', 'thal', 'restecg'])

    def preprocess_german(self):
        self.df.rename(columns={20: 'target'}, inplace=True)
        self.df.target.replace({2: 0}, inplace=True)
        cate_cols = [i for i in self.df.columns if self.df[i].dtype == 'object']
        self.df = onehot(self.df, cate_cols)

    def preprocess_breast(self):
        self.df.rename(columns={0: 'target'}, inplace=True)
        self.df.target.replace({'no-recurrence-events': 0, 'recurrence-events': 1}, inplace=True)
        self.categorical()

    def preprocess_breast2(self):
        self.df.replace({'M': 1, 'B': 0}, inplace=True)
        self.df.rename(columns={'diagnosis': 'target'}, inplace=True)
        self.df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)

    def preprocess_diabetes(self):
        self.df.rename(columns={'Outcome': 'target'}, inplace=True)

    def equalize_prior(self, target='target'):
        pos = self.df.loc[self.df[target] == 1]
        neg = self.df.loc[self.df[target] == 0]
        n = min(pos.shape[0], neg.shape[0])
        pos = pos.sample(n=n)
        neg = neg.sample(n=n)
        self.df = pd.concat([pos, neg], axis=0)
        return self

    def train_test_split(self, test_size=0.25, normalize=True):
        X = self.df.drop(['target'], axis=1).values
        y = self.df.target.values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size)
        sc = StandardScaler()
        if normalize:
            self.X_train = sc.fit_transform(self.X_train)
            self.X_test = sc.transform(self.X_test)
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_test_val_split(self, e0, e1, test_size=0.2, val_size=0.1, normalize=True):
        X = self.df.drop(['target'], axis=1).values
        y = self.df.target.values
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y.astype(int), test_size=0.2)
        # self.X_train, self.y_train = X, y.astype(int)
        if normalize:
            sc = StandardScaler()
            self.X_train = sc.fit_transform(self.X_train)
            self.X_test = sc.fit_transform(self.X_test)
        self.y_train_noisy = make_noisy_data(self.y_train, e0, e1)
        self.y_test_noisy = make_noisy_data(self.y_test, e0, e1)
        return self.X_train, self.X_test, self.y_train, self.y_test, self.y_train_noisy, self.y_test_noisy


    def prepare_train_test_val(self, kargs):
        if kargs['equalize_prior']:
            print('Prior is equalized')
            self.equalize_prior()
        X_train, X_test, y_train, y_test, y_train_noisy, y_test_noisy = self.train_test_val_split(
            e0=kargs['e0'], e1=kargs['e1'],
            test_size=kargs['test_size'],
            val_size=kargs['val_size'],
            normalize=kargs['normalize'],
        )
        return X_train, X_test, y_train, y_test, y_train_noisy, y_test_noisy


class TextDataLoader(object):
    def __init__(self, dataset: str, root='/data/BERT_embeddings/'):
        if dataset.lower() == 'agnews':
            train_file = 'AG_NEWS_train.npy'
            test_file = 'AG_NEWS_test.npy'
        elif dataset.lower() == 'yelp':
            train_file = 'Yelp_train.npy'
            test_file = 'Yelp_test.npy'
        elif dataset.lower() == 'dbpedia':
            train_file = 'DBpedia_train.npy'
            test_file = 'DBpedia_test.npy'
        elif dataset.lower() == 'amazon':
            train_file = 'Amazon_train.npy'
            test_file = 'Amazon_test.npy'
        elif dataset.lower() == 'imdb':
            train_file = 'IMDB_train.npy'
            test_file = 'IMDB_test.npy'
        elif dataset.lower() in ["jigsaw", "jigsaw_balanced"]:
            train_file = 'JigsawToxic_train.npy'
            test_file = 'JigsawToxic_train.npy'
        elif dataset.lower() in ["jigsaw_glove", "jigsaw_glove_balanced"]:
            train_file = 'Jigsaw_Glove_train.npy'
            test_file = 'Jigsaw_Glove_test.npy'
        else:
            raise NotImplementedError
        self.X_train, self.Y_train = self.load(os.path.join(root, train_file))
        self.X_test, self.Y_test = self.load(os.path.join(root, test_file))

        if 'balanced' in dataset.lower():
            self.X_train, self.Y_train = self.balance_subsample(self.X_train, self.Y_train)

    def load(self, path):
        with open(path, 'rb') as f:
            X, Y = np.load(f), np.load(f)
        assert X.shape[0] == Y.shape[0]
        return X, Y
    
    def prepare_train_test(self, kargs=None):
        return self.X_train, self.X_test, self.Y_train, self.Y_test

    def balance_subsample(self, x, y, subsample_size=1.0):
        """
        balance the sample size of each class
        """

        class_xs = []
        min_elems = None

        # find the class with minimum number of elements
        for yi in np.unique(y):
            elems = x[(y == yi)]
            class_xs.append((yi, elems))
            if min_elems == None or elems.shape[0] < min_elems:
                min_elems = elems.shape[0]

        # decide the sample size
        use_elems = min_elems
        if subsample_size < 1:
            use_elems = int(min_elems*subsample_size)

        xs = []
        ys = []

        # resample
        for ci,this_xs in class_xs:
            if len(this_xs) > use_elems:
                np.random.shuffle(this_xs)

            x_ = this_xs[:use_elems]
            y_ = np.empty(use_elems)
            y_.fill(ci)

            xs.append(x_)
            ys.append(y_)

        xs = np.concatenate(xs)
        ys = np.concatenate(ys)

        # verify the data size
        assert xs.shape[0] == ys.shape[0], f"The number of X is {xs.shape[0]}, while the number of Y is {ys.shape[0]}."
        
        # shuffle data examples again
        num_examples = ys.shape[0]
        index = np.random.permutation(num_examples)
        return xs[index], ys[index].astype(int)


def onehot(df, cols):
    dummies = [pd.get_dummies(df[col]) for col in cols]
    df.drop(cols, axis=1, inplace=True)
    df = pd.concat([df] + dummies, axis=1)
    return df


def make_noisy_data(y, e0, e1):
    num_neg = np.count_nonzero(y == 0)
    num_pos = np.count_nonzero(y == 1)
    flip0 = np.random.choice(np.where(y == 0)[0], int(num_neg * e0), replace=False)
    flip1 = np.random.choice(np.where(y == 1)[0], int(num_pos * e1), replace=False)
    flipped_idxes = np.concatenate([flip0, flip1])
    y_noisy = y.copy()
    y_noisy[flipped_idxes] = 1 - y_noisy[flipped_idxes]
    return y_noisy
