# TensorRT
(7.1.3)

这一本主要讲了如何在已知网络结构的情况下使用TensorRT构造引擎, 包括一些基本的构造方法和某些特殊结构的构造方法

如果已经有了onnx的模型, 则不需要一步一步的构造, 直接解析即可构造引擎

也提到了已知引擎的时候怎么用它来推理

## What is

### Benefits

一般用来衡量网络性能的五个因素
* 吞吐量(Throughput): inference/s
* 能效(Efficiency): amount of throughput per power
* 延时(Latency)
* Accuracy
* 内存(memory usage)

通过结合layer, 优化kernel, 进行normalization和想办法对精度和矩阵运算进行优化

### 使用的阶段

需要将训练好的网络生成tensorRT的推理引擎, 并且把这个引擎部署到实际程序里面(两步)
[![生成引擎示意](https://s1.ax1x.com/2022/11/28/zadjZd.png)](https://imgse.com/i/zadjZd)
推理引擎生成主要分成两步, 分别是
* 将网络架构进行解析(parse), 丢进对应的`parser`里面, 例如`pytorch`对应`onnx`
* 根据网络的架构和需要优化的方式构造`builder`, 并进一步构造`engine`

实际应用的时候一般是实际应用和TensorRT的库链接, 把对应的引擎的文件反序列化为引擎, 在实际推理的时候用入队函数把输入放进 ***缓冲区*** 里面 ***异步*** 的推理

每一次using TensorRT构造都要包括

* 从model创建definition
* 用`TensorRT builder`创造一个优化过运行时间的网络
* 序列化和反序列化引擎方便recreated
* feeding engine with input data
### 如何优化

通过网络的定义, 优化的目标来做针对pigtail的优化以及生成推理引擎

生成的引擎不能跨越TensorRT的版本和嵌入式平台使用

### 常用的API和库

详见教程

## C++ API

作用和Python的差不多, 更加适用于 ***对性能要求高*** 和 ***对安全性要求高*** 的场合

## Python API

如何使用Python完成整体的流程

### Creating Network Definition

可以自己用TensorRT手动构造, 可以用解析器自动生成

如何手动构造姑且略过, 需要一层一层把网络内容生成

用解析器生成指根据已有的解析好的文件, 例如`onnx`文件, 直接生成引擎, 生成步骤可以见`TensorRT/samples/python/introductory_parser_samples`里面的示例给出的方法, 或者是在`trt.Runtime().deserialize_cuda_engine`里面给出的集成好的方法

在这种方法里面可以直接根据onnx一步生成network, 并且还可以顺带生成引擎

### Building Engine

同上, 但是在使用Builder构建引擎的时候可以选择引擎的Configuration, 详见官方给出的实例, 一般需要着重定义的设定是
* max batch size: smaller
* max workspace

### Serialize model

可以将上一步生成的引擎序列化, 也可以直接使用引擎, 序列化可以将引擎保存在系统里面下次调用, 上面的`trt.Runtime().deserialize_cuda_engine`就是直接使用序列化的引擎的方法

### Infer

使用引擎进行推理, 可以分为两步, 分别是
* 根据引擎的定义, 从device中预留出输入和输出的空间, 创建输入输出buffer
* 创建context来存储中间计算得到的activation value并在context中进行推理
  * 理论上可以创建多个推理的context, 然后同时处理并行的几个cuda流

## 使用自己定义的层来扩展TensorRT

理论上有两种方式来使用自定义的层, 分别是从解析好的网络到TensorRT上在build的过程中加layer, 或者是在解析网络的过程中加入新的层

在build中的没看懂...姑且略过

在解析网络的过程中比较复杂, 可能需要自己重写相关的层, 再看了...略

## 使用混合精度定义TensorRT

同样在TensorRT中可以以
* 存储的权重以及激活变量
* 和执行层的方式
来储存不同数据类型的变量, 例如说`FP32`, `FP16`

若使用Python来实现的话, 可以通过改变 ***网络中层*** 的数据类型或者 ***builder*** 的数据类型来说实现, 需要修改允许改变数据类型的flag以及使用的数据类型

修改的时候可以逐层修改, 也可以使用 ***calibrator*** 作为引擎的`config`来修改推理引擎

最简单的实现方式可以直接使用`TensorRT`中修改 ***builder*** 的设置为`fp16`可以直接得到使用`fp16`的模型。

## Working With Reformat-Free Network I/O Tensors
不知道在干啥 略

## Working with dynamic shapes

此处恐怕使用onnx的解析器生成的网络不可以, 只有自己用TensorRT一层层推出的结构可以往上叠加
此处指将某些维度的定义推迟引擎推理的时候, 可以将步骤概括为4步
* 在确保网络没有隐式的batch dimension定义的时候设定载入网络的格式
* 把之后确定的网络维度用 `-1` 表示
* 生成优化文件
* 创建引擎的context, 把上一步的优化的文件放到输入维度上面, 确定确定的输入维度, 得到输出; 不同的输出维度对应不同的优化文件

具体的每一步的描述为

### Runtime dimension

* 在定义网络的时候, 把目标的定义成`-1`: `network_definition.add_input("foo", trt.float32,(3, -1, -1))`
* 设置优化配置文件
* 在设置计算context的时候: `context.set_binding_shape(0, (3, 150, 250))`

### Optimization Profile
优化配置文件, 把可能覆盖的输入范围全部列在文件里面, 包括输入的最大维度, 最小维度和优化维度, 在设置实际输入维度之前需要调用优化配置文件

考虑到可能有不止一个配置文件, 在和context以及引擎绑定的时候需要指定编号

### 层的扩展

一些层允许某些输入有特定的动态输入信息, 一些允许计算新的形状, 理论上可以新建拥有这种类型的层

### 针对的Tensor的类型

可以分为Execution tensor 和 shape tensor, 取决于它的tensor的维度能不能变, 某些层对输入的tensor的类型有要求, 但是这两个的维度都必须在build的时候确定

## Work with Quantized Network

可以使用`QAT`技术构造`Quantized ONNX models`, 并使TensorRT构造一个具有特定精度的网络来实现应用这样的onnx网络， 理论上可以直接由`Tensorflow`转化为`onnx`，但是`pytorch`目前转不过去，只能通过`pytorch-quantization`来进行转化

同时对于***quantized***模型，在TensorRT中运行的时候需要运行***explicit precision network***，也就是`trtexec`时需要加上`fp16`，否则获得的提升有限

## 后面的

后面大部分是讲如何初始化具有目标的功能的网络结构的, 例如具有循环结构, 具有空的张量输入, 具有量化的网络等等, 但是这些都涉及到直接使用TensorRT构建网络, 暂时用不上

或者直接使用DLA, 这个nano上没有

## DALI

提到的某种NVIDIA提供的可以在GPU上进行数据预处理的pipeline上, 由于预处理部分大部分都是`math-limit`, 使用GPU计算可以节省时间

## 目前看来似乎提到的可能的方法

* With Mixed Precision(C5)
* DALI(没看)
* 量化模型(没看)
* trtexec本身可以挖掘的参数(没看)
* TensorRT plugin是什么东西

# Optimizing Performance With TensorRT

## Measurements

### 测量指标

* 延时(Latency)
* 吞吐量(Troughput): 在固定时间内完成的推理数量

### 测量方法

* 使用`trtexec`测试, 有可以选的参数, 详见`trtexec`的说明文件
* 测试`CPU`时间: 使用C++中`chrono`库的内容, 可以测量高精度的wall-clock时间, 使用示例见文档, 需要注意
  * 使用时需要手动为测试时间进行 ***同步***
  * 缺点是如果是不同进程见有重叠的数据操作的话, 可能会损失原本的并行运行的性能
* 测试`CUDA`事件: 使用cuda的`EVENT API`, 可以通过给不同的事件盖时间戳的形式来测试, 对于在推理过程中会填充流的, 可以通过设置来模拟这个过程, 这样可以把在推理过程中可能并行执行的动作的时间也计算进来 ==> 和trtexec生成的trace是一种(就是没办法可视化)
* 使用`TensorRT Profile`: 使用C++中提供的测试TensorRT的性能的接口 ==> 和trtexec生成的profile是一种
  * 在`IExecutionContext`接口类的`setProfiler`, 可以初始化`IProfiler`类
  * 可以计算每一层的花费的时间
  * 没办法分别计算每一次循环的时间
* 使用`CUDA Profile`: 看不懂, 反正是使用`Nsight Systems`
* Memory: 创建一个GPU allocator, 可以使用`cudaMalloc`, `cudaFree`为cuda分配memory
  * 使用`IBuilder`for 网络, 如果已经建好反序列化的引擎就是`IRuntime`

## Optimize TensorRT

* Mixed Precision: 写过很多遍了, 不写了
* Batch: 理论上不要使用太小的batch, 因为直接做矩阵运算的运算效率往往更高
  * 可以尝试build不同大小的batch size的engine, 最后选择性能最好的
  * 在使用builder新建engine可以调节最大的batch size
* Streaming: 理论上减少层之间的同步 ,在单独的stream中并行的, 重叠的调度可以提升性能
  * 似乎更适用于多个batch的情景, 每个batch都有独立的stream, 异步的进行推理
* Fusion: 实际上是builder自动完成的 => 似乎是在进行TensorRT推理的时候回自动进行
  * layer fusion: 将需要调动多个核的操作想办法一步到位; 相关的操作记录在`ILogger`对象里面, 在`kINFO`, 会创建不同名字的层
  * MLP fusion: 对多层一次性进行融合
  * QDQ: 不懂?

## Optimize layer

* Concatenation Layer: 当拼接的时候, 不能跨batch的broadcast, 只能复制粘贴, 导致速度减慢
* FC: 使用`1 * 1 conv`代替FC 或者使用矩阵乘法
* Reduce Layer: 尽量对最后一维操作, 保证读到的memory的连续性 或者尽量使用有可能可以融合的操作

## 提到的方法

* 如何测试
* 分batch
* 没看懂怎么操作的layer fusion