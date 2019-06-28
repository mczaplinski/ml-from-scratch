import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

pd.set_option('display.max_columns', 800)

MNIST_TRAIN_PATH = '../data/mnist/train.csv'
MNIST_TEST_PATH = '../data/mnist/test.csv' #without labels -> submit to kaggle

class MNIST():
    
    def __init__(self,
                 validation_split=0.3,
                 batch_size=100,
                 train_path=MNIST_TRAIN_PATH,
                 test_path=MNIST_TEST_PATH,
                 batch_iteration=0):
        
        self.__validation_split = validation_split
        self.__batch_size = batch_size
        self.__train_path = train_path
        self.__test_path = test_path
        self.__batch_iteration = batch_iteration
        self.__pca_applied = False
        
        print('Initializing MNIST data pipeline...')
        if os.path.exists(self.__train_path) == False:
            raise Exception('Path not found: {}'.format(self.__train_path))
        if os.path.exists(self.__test_path) == False:
            raise Exception('Path not found: {}'.format(self.__test_path))
        
        train_data, train_labels = self.__load_train_csv()
        self.X_test = self.__load_test_csv()
        
        X_train, y_train, X_validation, y_validation = self.__train_validation_split(train_data, train_labels)
        self.X_train = X_train
        self.y_train = y_train
        self.X_validation = X_validation
        self.y_validation = y_validation
        
    def __load_train_csv(self):
        print('Loading data from: {}'.format(self.__train_path))
        df = pd.read_csv(self.__train_path)
        train_labels = pd.DataFrame(df['label'],index=df.index)
        train_data = df.drop('label',axis=1)
        print('\tData shape: ',train_data.shape,'\n\tLabels shape: ',train_labels.shape)
        return train_data,train_labels
    
    def __load_test_csv(self):
        print('Loading data from: {}'.format(self.__test_path))
        df = pd.read_csv(self.__test_path)
        print('\tData shape: ',df.shape)
        return df
    
    def normalize_input_data(self, norm_method='min_max'):
        if norm_method == 'min_max':
            print('Normalizing to 0..1 by min-max-scaling')
            # fit to train dataset
            if self.__pca_applied: #scale each principal component independently
                min_max_lower = self.X_train.min()
                min_max_upper = self.X_train.max()
            else: #pixel values. scale all pixels equally
                min_max_lower = self.X_train.min().min()
                min_max_upper = self.X_train.max().max()
            # transform all input datasets
            self.X_train = (self.X_train - min_max_lower) / (min_max_upper - min_max_lower)
            self.X_validation = (self.X_validation - min_max_lower) / (min_max_upper - min_max_lower)
            self.X_test = (self.X_test - min_max_lower) / (min_max_upper - min_max_lower)
            return
        if norm_method == 'zero_mean_unit_variance':
            print('Normalizing to zero mean and unit variance')
            # fit to train dataset
            if self.__pca_applied: #scale each principal component independently
                mean = self.X_train.mean()
                std = self.X_train.std()
            else: #pixel values. scale all pixels equally
                mean = self.X_train.mean().mean()
                std = self.X_train.std().max()
            # transform all input datasets
            self.X_train = (self.X_train - mean) / std
            self.X_validation  = (self.X_validation  - mean) / std
            self.X_test = (self.X_test - mean) / std
            return
        
        print('Normalization method \'{}\' is not valid.'.format(norm_method))
        print('\tAvailable methods: min_max, zero_mean_unit_variance')
        return
        
    def __train_validation_split(self, data, labels):
        if data.shape[0] == labels.shape[0]: #make sure the dimensions are aligned
            print('Splitting into train and validation set...')
            num_train = int(data.shape[0]*(1-self.__validation_split))
            X_train = data[:num_train]
            y_train = labels[:num_train]
            X_validation = data[num_train:]
            y_validation = labels[num_train:]
            print('\tValidation split factor {}'.format(self.__validation_split))
            print('\t-> Train data:',X_train.shape)
            print('\t-> Validation data:',X_validation.shape)
            return X_train, y_train, X_validation, y_validation
        else:
            raise Exception('Data and Labels are not lined up! Check dimensions!')
            
    def get_mnist_full(self):
        return self.X_train, self.y_train, self.X_validation, self.y_validation, self.X_test
    
    def get_next_train_batch(self, batch_iteration=None):
        if batch_iteration != None:
            self.__batch_iteration = batch_iteration #reset iteration if needed
            
        train_size = self.X_train.shape[0]
        if ((self.__batch_iteration+1)*self.__batch_size+1) > train_size:
            #last batch over the train set -> go back to head
            self.__batch_iteration = 0
        
        cur_idx_from = self.__batch_iteration*self.__batch_size
        cur_idx_to = cur_idx_from + self.__batch_size
        
        self.__batch_iteration = self.__batch_iteration + 1
        
        return self.X_train[cur_idx_from:cur_idx_to], self.y_train[cur_idx_from:cur_idx_to]
    
    def get_random_image(self, idx=None, flatten=True):
        print('Retrieving a random image from train set...')
        if idx == None:
            idx = np.random.randint(self.X_train.shape[0])
        if flatten == True:
            return np.array(self.X_train.loc[idx,:])
        else:
            return np.array(self.X_train.loc[idx,:]).reshape(28,28)
        
    def plot_image(self, img):
        print('Plotting image...')
        img = np.array(img).reshape(28,28) #reshape if not already
        plt.imshow(img,cmap='gray')
        plt.show()
        
    def plot_cumulative_variances_of_pca(self):
        print('Plotting cumulative variances of components...')
        pca = PCA(svd_solver='full')
        X_train_temp = pca.fit_transform(self.X_train)        
        cumulative_variances = []
        cumulated_val = 0.0
        for var in pca.explained_variance_:
            cumulated_val = cumulated_val + var
            cumulative_variances.append(cumulated_val)
        plt.plot(cumulative_variances)
        plt.show()
    
    def transform_using_pca(self, num_components=300):
        print('Applying PCA...')
        pca = PCA(n_components=num_components,svd_solver='full')
        new_col = ['pc{}'.format(i) for i in range(num_components)] #new column names, we don't have pixels anymore. so better rename this
        self.X_train = pd.DataFrame(
            pca.fit_transform(self.X_train),
            index=self.X_train.index,
            columns=new_col)
        self.X_validation = pd.DataFrame(
            pca.fit_transform(self.X_validation),
            index=self.X_validation.index,
            columns=new_col)
        self.X_test = pd.DataFrame(
            pca.fit_transform(self.X_test),
            index=self.X_test.index,
            columns=new_col)
        self.__pca_applied = True
        print('New Shapes:\n\tTrain:\t\t{}\n\tValidation:\t{}\n\tTest:\t\t{}'.format(self.X_train.shape,self.X_validation.shape,self.X_test.shape))