# Tensorflow - 从入门到脱发

---
## 写在前面

### 本文目的
 1. 使用tensorflow(Python)的 **高层API** 快速灵活构建可复用且具有较高定制性的深度学习模型，通过走一遍构建深度学习网络的流程，**整合并呈现网**上的资料（主要来源为较为零散的官方文档），本文主要起到**索引**的作用。
 2. 用尽可能短的篇幅展现流程，主要做到**会用即可**，在此基础上尽可能的详细，考虑到不同的模型定制需求
 3. 减少在寻找文档的过程中产生的迷茫感，减少因茫然无助产生的挠头、久坐、贪食、失眠等易导致脱发、肥胖及体虚的行为。

### 本文所使用的例子
自动编码机+DNN分类器，识别mnist手写字体，源代码([demo.py][1])，**建议对照代码食用本文**

简单来说，我们需要通过tensorflow构建两个模型，并定制自动编码机的预测输出结果，作为输入训练DNN分类器

### 一些其他的碎碎念
由于本文更多的起到索引的作用，请务必多关注学习本文的链接，以更详细更好的掌握tensorflow的基本用法。
目前tensorflow的版本是`1.5`版，tensorflow是一个快速迭代的库，各个版本的API调用方式可能会与本文有点出入，出现问题时请参考[官方文档][2]。
此外，由于tensorflow需要大量的底层软硬件支持，各个版本所基于的软硬件（尤其是软件）也会有所变化，建议像关注小姐姐/小鲜肉一样时不时去关注一下Tensorflow的[国内镜像官网][3]和[Github上的Release Note][4]
希望这篇东西能帮助大家对Tensorflow有更多的认识，减少大家在了解使用Tensorflow的过程中产生的踩坑、脱发等情况。欢迎大家去[issues][5]中交流问题或见解，或者参与到编辑改进中来。

## **May the ~~force~~ HAIR be with you**

## 目录
按照构建完整的深度学习模型过程组织，包括：


* [0 一些基础的东西](#0)
    * [0.0 Tensorflow数学运算](#0.0)
    * [0.1 高层API流程简介](#0.1)
* [1 数据读取](#1)
    * [1.0 基本读取规则](#1.0)
    * [1.1 读取mnist示例](#1.1)
    * [1.2 从numpy中读取](#1.2)
    * [1.3 从文本文件中读取](#1.3)
    * [1.4 从TFRecord data](#1.4)
* [2 自定义Estimator构建](#2)
* [3 可视化](#3)
    * [3.1 高层API下的tensorboard构建](#3.1)
    * [3.2 tensorboard示例](#3.2)
* [4 通过Tensorflow Debugger调试](#4)

<h2 id="0">0 一些基础</h2> 
<h3 id="0.0">0.0 Tensorflow数学运算</h3>
这个部分可以先跳过不看，从总的来说tensorflow的数学运算更接近numpy的语法，也和别的计算库大同小异，要用的时候看一下即可。

[博客参考][6]
[官方参考][7]

<h3 id="0.1">0.1 高层API流程简介</h3>
下面的这个视频（中英字幕都有）挺好的介绍了利用tensorflow中的高层API构建模型的主题思路，解释了API是如何相互作用的。建议先看一波

[视频 - TensorFlow High-Level APIs: Models in a Box (TensorFlow Dev Summit 2017)][8]

在正式的介绍下面的 数据读取 和 模型构建 部分之前，首先需要从高层API的流程上解释为什么这两个步在用tensorlfow构建模型的过程中非常重要。我们来看下面这段代码。
```python
# Build the auto encoder
auto_encoder = tf.estimator.Estimator(
    model_fn=sparase_autoencoder,
    model_dir='../log/SAE',
    params={
        'encoder_units': [],
        'encoder_result_units': 200,
        'decoder_units': [],
    }
)

# Train the model
auto_encoder.train(
    input_fn=lambda : input_fn(
        'train', 
        mnist.train.images, 
        mnist.train.images, 
        batch_size=128),
    steps=3000
)

# Evaluate the model
eval_result = auto_encoder.evaluate(
    input_fn=lambda : input_fn(
        function='eval',
        features=mnist.test.images[:20],
        labels=mnist.test.images[:20],
        batch_size=20
    )
)
```
这是一个自动编码机训练、验证所需要的核心流程，我们只需要自己完成两个部分：

 - 第3行出现的 **`model_fn`**
 - 第14行和24行出现的 **`input_fn`**
 
其中input\_fn指的就是数据读取的部分，model\_fn就是模型构建的部分。

从上面的代码中我们可以看出，要用tensorflow自定义构建一个完整的神经网络模型，我们需要用model\_fn来定义自己的模型，用input\_fn来向模型输入其所需要的数据。

<h2 id="1">1. 数据读取</h2>
该节介绍如何自定义`input_fn`来得到想要的输入数据集

参考材料

 - [Importing Data][9]
 - [Datasets Quick Start][10]
 - [TensorFlow 数据集和估算器介绍][11]（数据集部分）

*参考资料提供了几乎所有的数据读取所需要的基本内容，建议阅读*
 
tensorflow提供了大量数据读取相关的API，与一些其它的库结合(numpy,pandas等),tensorflow可以读取基本上所有的常用数据格式，就像在最前面提到的那样，本文不会介绍所有的数据格式读取，请合理谷歌([**eg.** read .mat tensorflow][12])
<p id="feature_columns"></p>
此外，tensorflow还提供了强大的数据特征列(`feature_columns`)来更好的组织数据输入，有关特征列的信息，目前我还处在为这个掉发中的阶段（用得了讲不清），有问题可以在[issues][13]中交流。官方文档对这部分的内容有非常详细的说明，请参考[官方中文博客-特征列][14]与[特征列官方文档][15]

<h3 id="1.0">1.0 基本读取</h3>

#### Dataset
tensorflow中的数据集载体主要为`tf.data.Dataset`实例，Dataset类似于一个python里的字典（或者别的语言里的map之类），为了更好的了解Dataset，建议在使用的时候查看[Dataset的API][16]

从简单易用的角度来说，以下几种方法会经常被用到（这里方法的行为更像是配置）：

 - `batch(batch_size)` --- 批量梯度下降必备方法
 - `repeat(count=None)` --- 指定数据集重复的次数，如果指定的一次话数据集仅迭代一次，迭代到末尾就会触发out of index之类的错误，默认为迭代无限次
 - `shuffle(buffer_size, seed=None, reshuffle_each_iteration=None)` --- 按照buffer_size的大小打乱数据集

#### 自定义input_fn
由之前[0.1](#0.1)部分的代码可以看出，`input_fn`在模型需要数据输入的时候被调用（即训练、交叉验证、测试、预测时），所以自定义`input_fn`的return值需要满足两个条件：
 
 1. return的值（Dataset）的结构要与自定义的model_fn里需要的输入相符
 2. 对于不同的用途，需要不同的input_fn来return不同的Dataset（例如有监督分类时，训练需要标签，预测不需要）

对于第一点而言，tensorflow的官方手法是用[`feature_column`](#feature_columns)的形式来return一个带有features字典的`Dataset`，在model\_fn中解析`Dataset`来获得正确的输入features和lables。这是目前tensorflow中的示例代码[custom\_estimator.py][17]和[iris\_data.py][18]所使用的规范，并且也是tensorflow预制的estimator中所接受的数据格式。从对不同数据集重复使用模型，和模型的规范性角度出发，推荐官方手法。并且许多时候我们需要调用预制的estimator来做基准测试，所以还是很有必要按照谷歌官方给的方法来定义return的数据集的。

但是从偷懒的角度来说，包装原本的数据集略显繁琐。所以展示一个不按照官方文档包装数据集的版本，我们也可以通过与官方示例代码的比较，来看出两者之间细节上的不同，以此来更好的了解在Estimator的实例化和调用过程中，到底发生了什么。

***后续我也会把代码稍微重构一下，改成官方文档建议的样子***

对于第二点而言则比较宽松，我们可以定义多个input\_fn函数来针对不同情况的调用，也可以将不同情况都定义在一个input\_fn中，反正只要确保能按照情况调用到想要的input_fn就可以了

<h3 id="1.1">1.1 读取mnist示例</h3>
先上代码

```python
def my_input_fn(function, features, labels=None, batch_size=None):
    """Custom input function to input data for training, evaluation or prediction step
    Args:
        function: selected function, must be train, predict or eval
        features: input_x
        batch_size: batch size to return
        labels: input_y

    Returns:
        a Dataset
    """
    if function == 'train':
        data_set = tf.data.Dataset.from_tensor_slices((features, labels))
        data_set = data_set.shuffle(512).repeat().batch(batch_size)
    elif function == 'eval':
        data_set = tf.data.Dataset.from_tensor_slices((features, labels))
        data_set = data_set.batch(batch_size)
    else:
        data_set = tf.data.Dataset.from_tensor_slices(features).batch(batch_size)

    return data_set
    
mnist = input_data.read_data_sets('../MNIST_data/', one_hot=True)

auto_encoder.train(
        input_fn=lambda : my_input_fn(
            'train',
            mnist.train.images,
            mnist.train.images,
            batch_size=128),
        steps=3000
    )
```

因为我的目的是不包装数据集，通过`my_input_fn`直接读取mnist数据集，并返回Dataset类型的结果（(feature, label)形式的数据集组合），所以直接通过`tf.data.Dataset.from_tensor_slices((features, labels))`得到想要的数据集并设置打乱抽样之后返回即可，这里的features其实就是输入的x值。如果要得到官方文档推荐的数据集样式，则需要先通过`dict(features)`将features字典化，然后再通过`from_tensor_slices`返回一个按照特征组织的数据集，当然这里的features其实并不是x值，而是x值按照特征列组合的字典。

上面的话可能很是绕口，其实总结一下的话就一点：**一定要保证input\_fn的返回值，和我们的model\_fn里需要的输入格式相符合！**。此外，我们可以结合[Tensorboard](#3)和[Tensorflow Debugger(tfdbg)](#4)来对tenforflow程序进行调试，保证程序内的值和我们预想的一样

<h3 id="1.2">1.2 从.npy中读取</h3>
因为tensorflow的计算、数据类型格式，差不多是基于numpy的，所以从numpy数据集合里读取数据，算是最简单的方式之一。我们可以通过`Dataset.from_tensor_slices()`来很简单的生成一个dataset：

```python
# Load the training data into two NumPy arrays, for example using `np.load()`.
with np.load("/var/data/training_data.npy") as data:
  features = data["features"]
  labels = data["labels"]

# Assume that each row of `features` corresponds to the same row as `labels`.
assert features.shape[0] == labels.shape[0]

dataset = tf.data.Dataset.from_tensor_slices((features, labels))
```

不过这会引起比较严重的内存泄漏问题，因为features和labels会被`tf.constant()`操作组合进计算图，其中需要大量的拷贝工作。[一个解决办法][19]用`tf.placeholder()`来先占位，然后通过迭代器将数据集迭代输入。

通过迭代器读取而不是直接通过列表读取也是python优化中的一大要点之一，迭代器虽然在读取效率上略低，但能极大程度的避免内存泄漏的问题（努力搬砖多买内存，觉得内存够用的话那就可以随意了。。）

<h3 id="1.3">1.3 从csv文件中读取</h3>
参考材料
[Reading a CSV File][20]

<h3 id="1.4">1.4 从TFRecord data</h3>
参考材料
[Consuming TFRecord data][21]

<h2 id="2">2. 模型构建</h2>
通过高层API Estimator构建模型的好处：[Advantages of Estimators
][22]

实现一个自定义的学习模型，需要实现对model\_fn的定义，具体定义流程在谷歌的[官方文档][23]和[博客][24]里都已经有很详细的说明了，have fun learning it~

下面我们通过一段示例代码来具体看下如何完整的定义一个model\_fn

```python
def pic_classifier(features, labels, mode, params):
    """some note"""
    net = tf.layers.dense(features, features.shape[1], activation=None, name='INPUT')

    # define classifier network
    for index, units in enumerate(params['units']):
        net = tf.layers.dense(net, units, activation=tf.nn.sigmoid, name='hidden_'+str(index))

    logits = tf.layers.dense(net, params['n_classes'], activation=None, name='logits')

    predicted_class = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_class[:, tf.newaxis],
            'logits': logits
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute losses
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))

    # Compute evaluation metrics
    labels = tf.argmax(labels, 1)
    accuracy = tf.metrics.accuracy(labels, predictions=predicted_class, name='classifier_acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    # create train op
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
```

和官方文档里有不同的是，因为官方文档里要用到feature_columns来组织输入数据，所以输入层定义为`input_layer = tf.feature_column.input_layer(features, feature_columns)`,但是我这边代码偷懒（不规范）了一下，所以输入层其实就直接是feature\_columns，加了个`net = tf.layers.dense(features, features.shape[1], activation=None, name='INPUT')`主要是为了给输入层命个名

之后解析params参数（这个参数也是前后对应的，一般定义成字典）。读里面自己定义的units参数，也就是隐含层的数量大小。接着递归定义了隐含层

```python
# define classifier network
for index, units in enumerate(params['units']):
    net = tf.layers.dense(net, units, activation=tf.nn.sigmoid, name='hidden_'+str(index))
```
这其中`enumerate()`得到一个索引+值的组合，这样就可以用`name='hidden_'+str(index)`来给这个隐含层命名了，方便后续可视化和debug过程中的查找，我们也可以自己随便想个东西来命名节点，反正只要最后能找得到就行。如果不命名的话，tensorflow一般会自己按照模型名+01234来命名，比如所有的dense会命名为dense1，denses2之类的，不方便查找，所以还是建议合理命名。

最后我们定义输出层如下
```python
logits = tf.layers.dense(net, params['n_classes'], activation=None, name='logits')
```

这样定义完了前向传播的过程，然后我们定义预测时也就是mode==PREDICT值时的调用

```python
predicted_class = tf.argmax(logits, 1)
if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
        'class_ids': predicted_class[:, tf.newaxis],
        'logits': logits
    }
    return tf.estimator.EstimatorSpec(mode,predictions=predictions)
```
在这之后因为我们要定义验证和训练时的行为，所以我们要先定义损失函数

```python
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
```
这里损失函数loss被定义为交叉熵的平均值，我们也可以根据自己的模型需求，换成[别的损失函数][25]

在损失函数定义完之后，我们就可以对模型进行评估了

```python
# Compute evaluation metrics
labels = tf.argmax(labels, 1)
accuracy = tf.metrics.accuracy(labels, predictions=predicted_class, name='classifier_acc_op')
metrics = {'accuracy': accuracy}
tf.summary.scalar('accuracy', accuracy[1])
if mode == tf.estimator.ModeKeys.EVAL:
    return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
```

评估（验证）所返回的值包括一个会被tensorflow自动处理的loss（自动绘图自动记录），和一个其它的值的字典metrics，这个字典作为调用eval过程实际的显式返回值，可以根据自己的需要调用使用。比如在

```python
classifier_eval_result = mnist_classifier.evaluate(
    input_fn=lambda : input_fn_gen('eval', 
        classifier_gen_test, 
        mnist.test.labels, 
        mnist.test.labels.shape[0])
)
```
中，mnist\_classifier这个estimator调用了`evaluate()`方法，触发了EVAL值，返回的metrics就会被赋于`classifier_eval_result`

最后为了训练我们定义的模型，我们需要指定一个优化器并让优化器去最小化loss函数，这里tensorflow提供了大量的[常用预制优化器(xxxxOptimizer)][26]来自动快速实现反向传播会梯度下降的过程。当然我们也可以[自定义一个自己的优化器][27]。

训练部分代码如下

```python
optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
```

至此，我们就成功的定义完成了自己的模型，接着只需通过`tf.estimator.Estimator()`构建实例并调用即可，代码如下

```python
mnist_classifier = tf.estimator.Estimator(
    model_fn=pic_classifier,
    model_dir='../log/classifier',
    params={
        'units': [128],
        'n_classes': 10,
    }
)

# Train classifier
mnist_classifier.train(
    input_fn=lambda : input_fn_gen('train', classifier_gen, mnist.train.labels, batch_size=128),
    steps=3000
)

# Evaluate classifier
classifier_eval_result = mnist_classifier.evaluate(
    input_fn=lambda : input_fn_gen('eval', classifier_gen_test, mnist.test.labels, mnist.test.labels.shape[0])
)
```

有关Estimator的API及更多用法的详细介绍[在这][28]

总的来说我们自定义实现一个pic\_classifier（也就是model\_fn），然后作为`Estimator`的参数`model_fn`，然后返回一个estimator实例赋值给`mnist_estimator`，这个自定义的estimator通过调用自身的`train()`、`evaluate()`、`preditct()`方法，完成模型的训练过程

<h2 id="3">3. 可视化</h2>

tensorflow下的可视化工具被称为tensorboard，这个工具是和tensorflow一起安装的所以不需要去下载安装别的东西，建议先看下tensorboard的视频介绍：[Hands-on TensorBoard][29]

同样地，谷歌也给出了tensorboard的详细教程
[TensorBoard: Visualizing Learning][30] - 创建summary op，保存数据文件，打开tensorboard 
[TensorBoard: Graph Visualization][31] - 可视化计算图
[TensorBoard Histogram Dashboard][32] - 可视化数值分部表，跟踪内部数据分布和变化趋势

以上这些就是基本上所有需要掌握的创建一个良好可视化tensorflow程序的技巧。如果要从比较底层的API构建一个清晰内容全面的tensorboard，差不多就是要对所有的tensorflow操作，都进行合理的命名即记录处理，而这个过程是比较繁琐的，我觉得在搭建模型的初始阶段没人会有心思去这么做。正如视频里所说，Tensorboard对模型构建阶段一个最主要的作用是检查计算图有无错误，方便研究者理清思路，专注于模型的调试和优化，而如果用底层API构建，则会用不少的心思考虑操作的命名和数据的保存，所以不怎么可取（至少在模型初始构建的阶段）

<h3 id="3.1">3.1 高层API下的tensorboard构建</h3>
这时候就可以体现出高层API的作用了，高层API不需要（或者说很少需要）显式的构建计算图流程，或者创建繁琐的数据保存流程。比如

```python
# Build the auto encoder
auto_encoder = tf.estimator.Estimator(
    model_fn=sparase_autoencoder,
    model_dir='../log_test/SAE',
    params={
        'encoder_units': [400,400],
        'encoder_result_units': 200,
        'decoder_units': [400,400],
    }
)
```

会自动根据`sparase_autoencoder`中定义的模型，创建计算图并保存所需要的数据

还记得我们在`logits = tf.layers.dense(net, params['n_classes'], activation=None, name='logits')`类似的操作中定义的`name`参数，我们可以通过这个参数，来修改节点的命名。此外，我们可以通过简简单单添加一句`tf.summary.scalar('accuracy', accuracy)`来告诉高层API要记录accuracy变量，并以‘accuracy’命名。相比以前的低层API而言，高层API显然在可视化方面更加容易使用。

<h3 id="3.2">3.2 tensorboard示例</h3>
tensorboard的重要作用是帮助我们理解模型、看出模型构建除否出了问题、数据是否按照正确的方式流动。下面我就以自己构建模型过程中出了一个简单错误的情况，来说明tensorboard在其中的作用

还是以`sparase_autoencoder`模型为例，假设我需要定义一个100\*100\*100 的编码机，和100\*100\*100的解码机，我们的代码如下

```python
# Build the auto encoder
auto_encoder = tf.estimator.Estimator(
    model_fn=sparase_autoencoder,
    model_dir='../log_test/SAE',
    params={
        'encoder_units': [100,100,100],
        'encoder_result_units': 200,
        'decoder_units': [100,100,100],
    }
)
```
我们通过`model_dir`参数，将模型保存到../log_test/SAE下。

```python
net = tf.layers.dense(features, features.shape[1], activation=None, name='INPUT')

# define encoder
encoder_net = net
for index, units in enumerate(params['encoder_units']):
    encoder_net = tf.layers.dense(encoder_net,
                                  units,
                                  activation=tf.nn.sigmoid,
                                  name='encoder_'+str(index))

# Define encoder output
en_result = tf.layers.dense(encoder_net, params['encoder_result_units'], activation=tf.nn.sigmoid, name='ENCODER_OUTPUT')

# define decoder
# decoder_net = en_result
for index, units in enumerate(params['decoder_units']):
    decoder_net = tf.layers.dense(en_result, units, activation=tf.nn.sigmoid, name='decoder_'+str(index))

```
在代码的解码机单元定义阶段（第17行），我把`en_result`错误的作为了每一层`decoder_net`的输入而不是递归的调用`decoder_net`作为输入

在代码所在目录运行命令行
```
tensorboard --logdir=../log_test
```
tensorboard会到所示目录行下查找可以在网页中显式的文件，得到如下图

![wrong][33]
可以看到计算图被错误的创建了，encoder_output被用作输入，创建了两个根本没用的decoder层

我们更改代码如下
```python
decoder_net = en_result
for index, units in enumerate(params['decoder_units']):
    decoder_net = tf.layers.dense(decoder_net, 
                                  units, 
                                  activation=tf.nn.sigmoid, 
                                  name='decoder_'+str(index))
```

让我们来看看正确的tensorboard计算图
![right][34]

可以看到这时候的计算图就是对的了。

tensorboard可以帮助我们快速的查找并定位模型层面的错误（包括权值和偏置的链接等），直观的加深我们对于模型的理解，极大程度上减少debug的时间。此外，它还有一个非常重要的作用，那就是它可以直观的显式程序中各个节点的命名情况，方便Tensorflow Debugger(tfdbg)中对各个量的查找与调用

<h2 id="4">4. Debugger(tfdbg)</h2>
[Debugger官方文档][35]

对于这个文档有一点要说明的是，这个文档对于Estimator中Debugger hooker的调用有点老旧，建议结合[Estimator的API][36]使用

正如Debugger文档里所描述的那样，要对tenforflow程序debug，在[高层API中][37]我们需要首先创建debughook
```python
# Create a LocalCLIDebugHook and use it as a monitor when calling train.
hooks = [tf_debug.LocalCLIDebugHook()]
```

然后将hook作为参数传入train()中，hook的意思也比较鲜明了，就是在程序运行的时候拿个钩子钩一下，钩出来想要的东西，在这里的话因为是个debug钩子，就起到debug的作用（返回变量值，暂停程序）

在`train()`中添加hook

```python
# Train the model
auto_encoder.train(
    input_fn=lambda : input_fn('train',
                               mnist.train.images[:10],
                               mnist.train.images[:10],
                               batch_size=128),
    steps=3,
    hooks=debug_hooks,
)
```

然后我们在**命令行**里运行程序
```
python3 xxx.py
```
得到如下所示的界面

![tfdbug_interface][38]

输入`run`来跑一次训练，得到一批数据

![run][39]

结合tensorboard所绘制的计算图，我们可以查看指定的节点的数值，比如说我们想看之前的第一层解码机的输出，我们通过打开tensorboard选择第一层解码机：

![decoder_board][40]

可以看到解码机的输出包括一个数据流和六个tensorflow的操作，我们关注数据流（也就是第二层解码机的输入），可以看到它的名字是decoder_1/MatMul

为了查看这个节点的数值情况，我们输入（有好用的自动补全，可以结合按TAB输入）（或者结合滚轮和鼠标点，，但是一大堆节点里去找有点辣眼睛）
```
pt decoder_1/MatMul
```
![result][41]

然后就可以看到这个节点的数据类型，形状，和具体的值啦

### 其它
当然，tfdbg的作用远不止这些，结合tfdbg的[其它常用命令][42]可以完成更多的调试作用，基本满足日常的程序调试需求

## 写在最后
快速入门版的tensorflow高级API教程暂时就到此结束了，tensorflow作为一个快速发展的大型深度学习开源库，其目前能做到的东西以及未来的潜力，远远超过了本文所涵盖的内容。推荐大家在享受高级api的同时，也要尝试着用比较低层的api来构建一次网络，加深tensorflow计算图构建的理解。此外，针对不同的领域，不同的算法模型，tensorflow所实现、包含的内通也远比本文来的多得多，这也是本文很少像官方文档那样讲述详细具体的函数功能，而只给出文档的索引、注重整体思路的原因。就像开头说的那样，希望这篇东西能对大家的入门以及日常的使用有所帮助。从我个人的经验来讲，刚刚入门的时候往往是最新奇又最蛋疼的，希望大家能借着这篇东西少踩点坑，少掉一点头发。

也欢迎大家对这篇东西进行鼓励或者批评，有什么问题也可以在[issues][5]里提。或者发邮件，我的邮箱是skywalkeratlas@gmail.com。大家一起互相学习互相分享才能~~掉更多的头发~~成为更好的程序员。

谢谢大家有兴趣看，么么哒

  [1]: https://github.com/SkywalkerAtlas/tf-implement/blob/master/demo.py
  [2]: https://tensorflow.google.cn/api_docs/python/
  [3]: https://tensorflow.google.cn/
  [4]: https://github.com/tensorflow/tensorflow/blob/master/RELEASE.md
  [5]: https://github.com/SkywalkerAtlas/tf-implement/issues "issues"
  [6]: http://blog.csdn.net/zywvvd/article/details/78593618
  [7]: https://tensorflow.google.cn/api_guides/python/math_ops
  [8]: https://youtu.be/t64ortpgS-E
  [9]: https://tensorflow.google.cn/programmers_guide/datasets#preprocessing_data_with_datasetmap
  [10]: https://tensorflow.google.cn/get_started/datasets_quickstart
  [11]: https://developers.googleblog.cn/2017/09/tensorflow.html
  [12]: https://www.google.com.hk/search?safe=strict&ei=u3aKWpDMBIab0gSm2bDgAg&q=read%20.mat%20tensorflow&oq=read%20.mat%20tensorflow
  [13]: https://github.com/SkywalkerAtlas/tf-implement/issues
  [14]: https://developers.googleblog.cn/2017/12/tensorflow.html
  [15]: https://tensorflow.google.cn/get_started/feature_columns
  [16]: https://tensorflow.google.cn/api_docs/python/tf/data/Dataset
  [17]: https://github.com/tensorflow/models/blob/master/samples/core/get_started/custom_estimator.py
  [18]: https://github.com/tensorflow/models/blob/master/samples/core/get_started/iris_data.py
  [19]: https://tensorflow.google.cn/programmers_guide/datasets#consuming_numpy_arrays
  [20]: https://tensorflow.google.cn/get_started/datasets_quickstart#reading_a_csv_file
  [21]: https://tensorflow.google.cn/programmers_guide/datasets#consuming_tfrecord_data
  [22]: https://tensorflow.google.cn/programmers_guide/estimators#advantages_of_estimators
  [23]: https://tensorflow.google.cn/get_started/custom_estimators
  [24]: https://developers.googleblog.cn/2018/01/tensorflow.html
  [25]: https://tensorflow.google.cn/api_docs/python/tf/losses
  [26]: https://tensorflow.google.cn/api_docs/python/tf/train
  [27]: https://www.bigdatarepublic.nl/custom-optimizer-in-tensorflow/
  [28]: https://tensorflow.google.cn/api_docs/python/tf/estimator/Estimator
  [29]: https://youtu.be/eBbEDRsCmv4
  [30]: https://tensorflow.google.cn/programmers_guide/summaries_and_tensorboard
  [31]: https://tensorflow.google.cn/programmers_guide/graph_viz
  [32]: https://tensorflow.google.cn/programmers_guide/tensorboard_histograms
  [33]: http://static.zybuluo.com/skywalkerAtlas/nt2ngcwfc2emgwyurjcav4tp/Screen%20Shot%202018-02-21%20at%2016.01.38.png
  [34]: http://static.zybuluo.com/skywalkerAtlas/o1d0eygabosd1jnhp8x8pa12/Screen%20Shot%202018-02-21%20at%2016.06.04.png
  [35]: https://tensorflow.google.cn/programmers_guide/debugger
  [36]: https://tensorflow.google.cn/api_docs/python/tf/estimator/Estimator#train
  [37]: https://tensorflow.google.cn/programmers_guide/debugger#debugging_tf-learn_estimators_and_experiments
  [38]: http://static.zybuluo.com/skywalkerAtlas/nyrtjlndvtj0s331iqh4vf5r/Screen%20Shot%202018-02-21%20at%2016.44.20.png
  [39]: http://static.zybuluo.com/skywalkerAtlas/g0eatyvhs1trtmncixdzg7i0/Screen%20Shot%202018-02-21%20at%2016.46.03.png
  [40]: http://static.zybuluo.com/skywalkerAtlas/ftzwrqxll0tgyr4e5zawpi75/Screen%20Shot%202018-02-21%20at%2016.42.07.png
  [41]: http://static.zybuluo.com/skywalkerAtlas/03l52bymfvbu28luzvzq8kr3/Screen%20Shot%202018-02-21%20at%2016.51.29.png
  [42]: https://tensorflow.google.cn/programmers_guide/debugger#tfdbg_cli_frequently-used_commands