'''数据集转换之预处理数据：
      将输入的数据转化成机器学习算法可以使用的数据。包含特征提取和标准化。
      原因：数据集的标准化（服从均值为0方差为1的标准正态分布（高斯分布））是大多数机器学习算法的常见要求。
      如果原始数据不服从高斯分布，在预测时表现可能不好。在实践中，我们经常进行标准化（z-score 特征减去均值/标准差）。

1.1 标准正态分布(均值为0，方差为1) Scale函数的使用 对列进行z-score 

[python] view plain copy
from sklearn import preprocessing  
import numpy as np  
#1、数据标准化 '''   
  
X = np.array([[ 1., -1.,  2.],  
             [ 2.,  0.,  0.],  
             [ 0.,  1., -1.]])  
X_scaled = preprocessing.scale(X)  
X_scaled  
""" 
输出标准化的结果： 
array([[ 0.        , -1.22474487,  1.33630621], 
       [ 1.22474487,  0.        , -0.26726124], 
       [-1.22474487,  1.22474487, -1.06904497]]) 
"""  
X_scaled.mean(axis=0) #用于计算均值和标准偏差的轴。如果为0，独立规范每个特征，否则（如果为1）标准化每个样品。  
""" 
输出归一化后的均值： 
array([ 0.,  0.,  0.]) 
"""  
X_scaled.std(axis=0)  
""" 
输出标准化后的标准差： 
array([ 1.,  1.,  1.]) 
"""  

'''1.2 预处理模块StandardScaler

其实现Transformer API以计算训练集上的平均值和标准偏差，以便以后能够在测试集上重新应用相同的变换。

[python] view plain copy'''
#StandardScaler()的参数  
""" 
StandardScaler() 的参数with_mean 默认为True 表示使用密集矩阵，使用稀疏矩阵则会报错 ，with_mean= False 适用于稀疏矩阵 
with_std 默认为True 如果为True，则将数据缩放为单位方差（单位标准偏差） 
copy 默认为True 如果为False，避免产生一个副本，并执行inplace缩放。 如果数据不是NumPy数组或scipy.sparse CSR矩阵，则仍可能返回副本 
"""  
scaler = preprocessing.StandardScaler().fit(X)   
scaler  
""" 
输出： 
StandardScaler(copy=True, with_mean=True, with_std=True) 
"""  
#StandardScaler()的属性  
scaler.mean_   
""" 
输出X（原数据）每列的均值： 
array([ 1.        ,  0.        ,  0.33333333]) 
"""  
scaler.scale_  
""" 
输出X（原数据）每列的标准差（标准偏差）： 
array([ 0.81649658,  0.81649658,  1.24721913]) 
"""  
scaler.var_  
""" 
输出X（原数据）每列的方差： 
array([ 0.66666667,  0.66666667,  1.55555556]) 
"""  
#StandardScaler()的方法  
scaler.transform(X)   
""" 
输出X（原数据）标准化（z-score）： 
rray([[ 0.        , -1.22474487,  1.33630621], 
       [ 1.22474487,  0.        , -0.26726124], 
       [-1.22474487,  1.22474487, -1.06904497]]) 
"""  
#  StandardScaler().fit(X) 输入数据用于计算以后缩放的平均值和标准差  
#  StandardScaler().fit_transform(X)输入数据，然后转换它  
scaler.get_params() #获取此估计量的参数  
""" 
输出: 
{'copy': True, 'with_mean': True, 'with_std': True} 
"""  
scaler.inverse_transform(scaler.transform(X))#将标准化后的数据转换成原来的数据  
""" 
输出: 
array([[ 1., -1.,  2.], 
       [ 2.,  0.,  0.], 
       [ 0.,  1., -1.]]) 
"""  
#scaler.partial_fit(X) 在X缩放以后 在线计算平均值和std  
#scaler.set_params(with_mean=False)设置此估计量的参数  

'''2、归一化 将特征缩放到一个范围内（0，1）
缩放特征到给定的最小值到最大值之间，通常在0到1之间。或则使得每个特征的最大绝对值被缩放到单位大小。这可以分别使用MinMaxScaler或MaxAbsScaler函数实现。
[python] view plain copy'''
""" 
#训练集数据 例如缩放到[0-1] 
"""  
MinMaxScaler 参数feature_range=(0, 1)数据集的分布范围, copy=True 副本  
计算公式如下：  
X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))  
X_scaled = X_std * (max - min) + min  
""" 
 
X_train = np.array([[ 1., -1.,  2.], 
                   [ 2.,  0.,  0.], 
                   [ 0.,  1., -1.]]) 
min_max_scaler = preprocessing.MinMaxScaler() 
X_train_minmax = min_max_scaler.fit_transform(X_train) 
X_train_minmax 
"""  
输出训练集：  
array([[ 0.5       ,  0.        ,  1.        ],  
       [ 1.        ,  0.5       ,  0.33333333],  
       [ 0.        ,  1.        ,  0.        ]])  
""" 
#测试集数据 
X_test = np.array([[ -3., -1.,  4.]]) 
X_test_minmax = min_max_scaler.transform(X_test) 
X_test_minmax 
"""  
输出测试集：  
array([[-1.5       ,  0.        ,  1.66666667]])  
""" 
"""  
MaxAbsScaler 通过其最大绝对值来缩放每个特征,范围在[-1,1]。它用于已经以零或稀疏数据为中心的数据，应用于稀疏CSR或CSC矩阵。  
X_std = X/每列的最大绝对值  
""" 
X_train = np.array([[ 1., -1.,  2.], 
                   [ 2.,  0.,  0.], 
                   [ 0.,  1., -1.]]) 
max_abs_scaler = preprocessing.MaxAbsScaler() 
X_train_maxabs = max_abs_scaler.fit_transform(X_train) 
X_train_maxabs 
"""  
输出训练集：  
array([[ 0.5, -1. ,  1. ],  
       [ 1. ,  0. ,  0. ],  
       [ 0. ,  1. , -0.5]])  
""" 
max_abs_scaler.scale_   
"""  
输出训练集的缩放数据：  
array([ 2.,  1.,  2.])  
""" 
 
X_test = np.array([[ -3., -1.,  4.]]) 
X_test_maxabs = max_abs_scaler.transform(X_test) 
X_test_maxabs  
"""  
输出测试集：  
array([[-1.5, -1. ,  2. ]])  
"""  

'''3、关于稀疏矩阵
MaxAbsScaler和maxabs_scale是专门为缩放稀疏数据而设计的。scale和StandardScaler可以接受scipy.sparse矩阵作为输入，只要将with_mean = False显式传递给构造函数即可。否则，将抛出ValueError，因为静默中心将打破稀疏性，并且通常会由于无意分配过多的内存而导致执行崩溃。RobustScaler不能适用于稀疏输入，但是您可以对稀疏输入使用变换方法。
请注意，缩放器接受压缩的稀疏行和压缩的稀疏列的格式（请参阅scipy.sparse.csr_matrix和scipy.sparse.csc_matrix）。任何其他稀疏输入将被转换为压缩稀疏行表示。为了避免不必要的内存复制，建议选择CSR或CSC表示。最后，如果中心数据预期足够小，使用稀疏矩阵的toarray方法将输入显式转换为数组是另一个好的选择。

4 缩放具有异常值的数据
如果您的数据包含许多异常值，使用数据的均值和方差的缩放可能无法很好地工作。在这些情况下，您可以使用robust_scale和RobustScaler作为替代替换。 它们对数据的中心和范围使用更稳健的估计。可以使用sklearn.decomposition.PCA或sklearn.decomposition.RandomizedPCA与whiten = True进一步删除特征之间的线性相关。

5 、归一化
归一化是缩放单个样本以具有单位范数的过程。 如果您计划使用二次形式（如点积或任何其他内核）来量化任何样本对的相似性，则此过程可能很有用。这个假设基于经常被用于文本分类和聚类上下文的空间向量模型上。函数normalize提供了一个快速和简单的方法来在单个数组类数据集上执行此操作，使用l1或l2范数。
[python] view plain copy'''
X = [[ 1., -1.,  2.],  
    [ 2.,  0.,  0.],  
     [ 0.,  1., -1.]]  
X_normalized = preprocessing.normalize(X, norm='l2')  
X_normalized  
""" 
输出l2归一化： 
array([[ 0.40824829, -0.40824829,  0.81649658], 
       [ 1.        ,  0.        ,  0.        ], 
       [ 0.        ,  0.70710678, -0.70710678]]) 
"""  
  预处理模块还提供了一个实用类Normalizer，它使用Transformer API实现相同的操作（其中fit方法无用，因为该操作独立处理样本）。transform(X[, y, copy])将X的每个非零行缩放为单位范数。单独归一化样本为单位标准，具有至少一个非零分量的每个样本（即数据矩阵的每一行）独立于其他样本被重新缩放，使得其范数（l1或l2）等于1。
能够使用密集numpy数组和scipy.sparse矩阵（如果避免复制/转换，使用CSR格式）。 例如文本分类或聚类的常见操作。 例如，两个l2归一化的TF-IDF向量的点积是向量的余弦相似性，并且是信息检索团体通常使用的向量空间模型的基本相似性度量。
[python] view plain copy
normalizer = preprocessing.Normalizer(norm='l1').fit(X)  # fit 无用  
normalizer.transform(X)   
""" 
输出： 
array([[ 0.25, -0.25,  0.5 ], 
       [ 1.  ,  0.  ,  0.  ], 
       [ 0.  ,  0.5 , -0.5 ]]) 
"""  

'''6 、二值化
6.1 特征二值化
特征二值化是将数值特征阈值化以获得布尔值的过程。 这对于假设输入数据根据多变量伯努利分布而分布的下游概率估计器可能是有用的。例如，这是sklearn.neural_network.BernoulliRBM 的情况。在文本处理中经常使用二值特征（可能简化概率推理），即使归一化计数（也称为词项频率）或TF-IDF值特征在实践中经常表现得更好。二元化和二元化接受来自scipy.sparse的密集阵列样和稀疏矩阵作为输入。对于稀疏输入，数据将转换为压缩稀疏行表示形式（请参阅scipy.sparse.csr_matrix）。为了避免不必要的内存复制，建议选择CSR。
[python] view plain copy'''
X = [[ 1., -1.,  2.],  
      [ 2.,  0.,  0.],  
     [ 0.,  1., -1.]]  
  
binarizer = preprocessing.Binarizer().fit(X)  # fit does nothing  
binarizer  
""" 
输出： 
Binarizer(copy=True, threshold=0.0) 
"""  
binarizer.transform(X)  
""" 
输出： 
array([[ 1.,  0.,  1.], 
       [ 1.,  0.,  0.], 
       [ 0.,  1.,  0.]]) 
"""  
#可以调整二值化器的阈值  
binarizer = preprocessing.Binarizer(threshold=1.1)  
binarizer.transform(X)  
""" 
输出： 
array([[ 0.,  0.,  1.], 
       [ 1.,  0.,  0.], 
       [ 0.,  0.,  0.]]) 
"""  

'''7、分类特征编码

     通常来说，特征不都是连续的值而是由分类给出的。例如，一个人可以具有如下特征：
       ["male", "female"]
       ["from Europe", "from US", "from Asia"]
       ["uses Firefox", "uses Chrome", "uses Safari", "uses Internet Explorer"]
      这样的特征可以被有效地整合在一起然后进行编码，比如：
       ["male", "from US", "uses Internet Explorer"] 可以用[0, 1, 3]表示
       ["female", "from Asia", "uses Chrome"] 可以用[1, 2, 1]表示
  但是，这样的表示不能用于Sklearn进行估计，因为离散（分类）特征，将特征值转化成数字时往往是不连续的。OneHotEncoder函数通过one-of-K （k之一）和 one-hot(独热)编码来解决这个问题。
'''
[python] view plain copy
enc = preprocessing.OneHotEncoder()  
enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])   
""" 
输出： 
OneHotEncoder(categorical_features='all', dtype=<class 'float'>, 
       handle_unknown='error', n_values='auto', sparse=True) 
"""  
enc.transform([[0, 1, 3]]).toarray()  
""" 
输出： 
array([[ 1.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  1.]]) 
"""  
  
""" 
    默认情况下，每个要素可以自动从数据集中推断出多少值。可以使用参数n_values显式地指定它。 
    在我们的数据集中有两个性别，三个可能的大陆和四个网络浏览器。然后我们拟合估计器，并变换数据点。 
结果:前两个数字编码性别，三个数字的大陆和四个数字的为网络浏览器。 
"""  
#注意，如果存在训练数据可能缺少分类特征的可能性，则必须显式地设置n_value。例如，  
enc = preprocessing.OneHotEncoder(n_values=[2, 3, 4])  
#请注意，第2个和第3个特征缺少分类值 第一个特征不缺少（有0，1）  
enc.fit([[1, 2, 3], [0, 2, 0]])  
""" 
输出： 
OneHotEncoder(categorical_features='all', dtype=<class 'float'>, 
       handle_unknown='error', n_values=[2, 3, 4], sparse=True) 
"""  
enc.transform([[1, 0, 0]]).toarray()  
""" 
输出： 
array([[ 0.,  1.,  1.,  0.,  0.,  1.,  0.,  0.,  0.]]) 
"""  

'''8 缺失值的插补
由于各种原因，许多现实世界数据集包含缺失值，通常编码为空白，NaN或其他占位符。然而，这样的数据集与scikit-learn估计器不兼容，scikit-learn估计器假定数组中的所有值都是数字的，并且都具有和保持意义。使用不完整数据集的基本策略是丢弃包含缺少值的整个行和/或列。然而，这是以丢失可能有价值的数据（即使不完全）为代价。一个更好的策略是插补缺失值，即从数据的已知部分推断它们。 Imputer类提供了输入缺失值的基本策略，使用缺失值所在的行或列的平均值，中值或最常见的值。这个类还允许不同的缺失值编码。
[python] view plain copy'''
#以下代码段演示了如何使用包含缺少值的列（轴0）的平均值替换编码为np.nan的缺失值：  
import numpy as np  
from sklearn.preprocessing import Imputer  
imp = Imputer(missing_values='NaN', strategy='mean', axis=0) # missing_values：integer/“NaN”, strategy：mean/median/most_frequent  
imp.fit([[1, 2], [np.nan, 3], [7, 6]])  
""" 
输出： 
Imputer(axis=0, copy=True, missing_values='NaN', strategy='mean', verbose=0) 
"""  
X = [[np.nan, 2], [6, np.nan], [7, 6]]  
imp.transform(X)  
""" 
输出： 
array([[ 4.        ,  2.        ], 
       [ 6.        ,  3.66666667], 
       [ 7.        ,  6.        ]]) 
"""  
#Imputer类还支持稀疏矩阵：  
import scipy.sparse as sp  
X = sp.csc_matrix([[1, 2], [0, 3], [7, 6]])  
imp = Imputer(missing_values=0, strategy='mean', axis=0)  
imp.fit(X)  
""" 
Imputer(axis=0, copy=True, missing_values=0, strategy='mean', verbose=0) 
"""  
X_test = sp.csc_matrix([[0, 2], [6, 0], [7, 6]])  
imp.transform(X_test)  
""" 
输出： 
array([[ 4.        ,  2.        ], 
       [ 6.        ,  3.66666667], 
       [ 7.        ,  6.        ]]) 
"""  

'''9 生成多项式特征
 通常，通过考虑输入数据的非线性特征来增加模型的复杂性是有用的。使用的一种简单和常见的方法是多项式特征，其可以获得特征的高阶和交互项。它在PolynomialFeatures中实现。注意，当使用多项式核函数时，多项式特征在内核方法（例如，sklearn.svm.SVC，sklearn.decomposition.KernelPCA）中被隐含地使用。
[python] view plain copy'''
from sklearn.preprocessing import PolynomialFeatures  
X = np.arange(6).reshape(3, 2)  
X  
""" 
输出： 
array([[0, 1], 
       [2, 3], 
       [4, 5]]) 
"""  
poly = PolynomialFeatures(2)  
poly.fit_transform(X)          
""" 
输出： 
array([[  1.,   0.,   1.,   0.,   0.,   1.], 
       [  1.,   2.,   3.,   4.,   6.,   9.], 
       [  1.,   4.,   5.,  16.,  20.,  25.]]) 
        
从X(X_1, X_2) 到X(1, X_1, X_2, X_1^2, X_1X_2, X_2^2). 
"""  
  
#在某些情况下，只需要特征之间的交互项，并且可以通过设置获得  
X = np.arange(9).reshape(3, 3)  
X     
""" 
输出： 
array([[0, 1, 2], 
       [3, 4, 5], 
       [6, 7, 8]]) 
"""  
poly = PolynomialFeatures(degree=3, interaction_only=True)  
poly.fit_transform(X)   
""" 
输出： 
array([[   1.,    0.,    1.,    2.,    0.,    0.,    2.,    0.], 
       [   1.,    3.,    4.,    5.,   12.,   15.,   20.,   60.], 
       [   1.,    6.,    7.,    8.,   42.,   48.,   56.,  336.]]) 
        
从(X_1, X_2, X_3) 到 (1, X_1, X_2, X_3, X_1X_2, X_1X_3, X_2X_3, X_1X_2X_3). 
"""  

'''10 自定义转化器
[python] view plain copy'''
from sklearn.preprocessing import FunctionTransformer  
transformer = FunctionTransformer(np.log1p)  
X = np.array([[0, 1], [2, 3]])  
transformer.transform(X)  
""" 
输出： 
array([[ 0.        ,  0.69314718], 
       [ 1.09861229,  1.38629436]]) 
"""  
