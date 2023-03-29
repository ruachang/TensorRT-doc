# Record.md

## technque blog 

* 472 GFLOPS of compute performance
* a quad-core 64-bit ARM CPU (ARM64)
* a 128-core integrated NVIDIA GPU
* 4GB LPDDR4 memory => 换成2GB了
* JetPack 4.2 SDK provides a complete desktop Linux environment for Jetson Nano based on Ubuntu 18.04 with accelerated graphics, support for NVIDIA CUDA Toolkit 10.0, and libraries such as cuDNN 7.3 and TensorRT 5 => 这些版本应该都是和JetPack SDK的版本相关, 但是下好之后应该版本就对了

## Developer Kit

### setup

没啥 似乎提到了能耗的两个模式

### JetPack

solution for building AI Applications, component includes

* Operation System Ubuntu
* Lib: 既然自带了这些东西型号和版本应该都定死了吧??
  * TensorRT cuDNN 
  * CUDA for  GPU
  * OpenCV

## TeGRA LINUX DRIVER PACKAGE

似乎是一些出现过的常见的问题, 记录一下我可能会用到的

* power-based throttling: 如果电源适配器"exceeded", 会自动关机
* /de/root: 14GB 不能扩展
* 设置 NTP server来保持时间设置的正确性
* `nvarguscamerasrc` `maxperf=1` using camera capture with scaling/color conversion 可能对性能有影响
* nvgstcapture-1.0 has an image encoding issue using the nvjpegenc plugin for default YUY2 video capture format for a USB camera. - nvgstcapture-1.0在使用nvjpegenc插件为USB相机的默认YUY2视频捕获格式时存在图像编码问题(如果遇到这个问题就直接搜原文)
* `sudo -H pip install jetson-stats`下载监控jetson性能的插件, 使用`jtop`指令检查jetson工作状态, 例如温度等等
大部分关于Jetson Nano的官方文档都是关于硬件参数的, 似乎找不到什么和"适合于哪种运算", 需要在查一查别的

## Vision Programming Interface

官网上查到的, 不知道是什么东西, 似乎是从软件上提升性能的东西


* 英伟达高性能计算机视觉/图像处理算法库
  * 提供了OpenCV接口
  * 不同硬件设备的接口一样
* Optimized VPI algorithms include 
  * background subtraction
  * perspective warp
  * temporal noise reduction
  * histogram equalization
  * lens distortion.

## Hello AI World

* 辅助搭建了一个`jetson inference`的测试环境, 该环境集成了一些现成的网络以及nano上很多资源, 并且集成了TensorRT的使用
* 然而因为下载模型需要翻墙, 可能没有办法实际跑, 不过最后反正也还是得自己写, 看看算了
* 不过这个开源的包可以在Python里面直接引用, 包括和高效使用摄像头相关的, 可以用
* 直接使用视频文件流的方法
  * 在`jetson.utils`里面
  * 输入流: `videoSource()`, 输出流: `videoOutput()`, 参数是摄像头或者是文件夹,或者是完整的图片和视频的位置
  * 读入下一帧`Capture()`, 显示`Render()`
  * [完整的说明API](https://rawgit.com/dusty-nv/jetson-inference/dev/docs/html/python/jetson.utils.html) (似乎打不开)

## Deep Learning Performance file

### Get started

GPU 通过并行计算加速, 可以通过调整参数来优化性能

* Operating In Math-Limited Regime Where Possible: data movement speed: limit => 加速运算只有在`math-limited`才有用, 也即调整参数对于`memory-bound`是没有用的, 这样的操作经常是哪些不包括矩阵乘法的; 对于包括乘法的, 取决于层的大小和计算的密度, 运算密度足够大, 那么才是`math-bound`
* Using Tensor Cores Efficiently With Alignment: Tensor cores对于操作参数是8的倍数, 或者说维度时16比特时性能最高, 关键参数和层相关. 
  * Tips: TF32是一种NVIDIA的datatype, 可以使用混合精度(Nano 用不了)
* Choosing Parameters To Maximize Execution Efficiency: 使得操作可以更好的平均分的参数有利于并行化, 最好都是2的指数级, 但是不要大于256, 这里的参数包括: 
  * batchsize
  * 输入输出尺寸
  * 通道数

### Background: GPU performance

#### 基本结构

并行的, 可以分成三层
* 流处理器(主要用来计算) Stream Multiprocessor(SMs)
  * 类似CPU, 内部有自己的ISA和计算方式
  * 最常作的就是 ***乘加运算***
* cache
* DRAM

GPU执行function的时候采用的是 ***二级线程结构***, 遵循的规则是
* 一个SM执行多个Thread
* 输入的function完成需要多个Thread
二级机构指将多个Thread分成 ***多个Thread Block***, 每个block中又分成 ***多个Thread***, 因此上述规则改为
* 每个block分配给一个SM
* Block的数量大于SM
对于SM来说, 其本身就具有执行多个threads的能力, 在SM内部, threads可以内部切换来减少延迟, 并且同步和共享内存
[![图示说明](https://s1.ax1x.com/2022/11/26/zNZhY6.png)](https://imgse.com/i/zNZhY6)

#### Performance

在处理器上的性能和三个方面相关
* 数学计算带宽
* memory 带宽
* 延迟

在workload比较大或者并行度很大的时候, 处理器可能呢被math或者memory限制, 假设将存取速度表示成$BW_{mem}$, 代表其带宽, 将计算速度表示成$BW_{math}$, 则可以将math和memory的时间表示成$ops / BW_{math}$和$bytes / BW_{mem}$, 哪个大就是哪个时间长

相应的可以把这个比较改成比较$ops / bytes$ 和$BW_{math} / BW_{mem}$, 前半截是`arithmetic intensity`, 后半截是`ops:byte`在GPU中用这两个来衡量计算和访存的相对时间, 也即计算次数和需要存取的数据量

但是这个值是一个 ***相对值***, 因为理论上一个数据不一定访存一次, 而且假定所有的流水线都在用, 不存在任何延迟

#### DNN中常见的操作

##### element-wise

* tensor中的每个元素单独运算
* 一般是非线性运算
  * Reluctant, scale, bias, add
* memory-limited: 计算量很小, 但是需要取很多值

##### Reduction

* 从一摞输入tensor中拿出来一组值
  * pooling, batch normalization, softmax
* memory-limited

##### dot-product

* tensor之间的点乘
  * fc, Convolution: 矩阵-向量 or 矩阵-矩阵的相差
* 在系数阵很大的时候是math-limited, 如果不够大就是memory-limited

#### Summary

估计GPU性能的方法
* 处理器的数量和`ops:byte`的数量
* 并行度: number and size of Thread block
  * 至少应该是处理器数量的 ***4x*** 倍
  * 一个block里面有几百个Thread
* 并行度决定了是 ***延时*** 还是 ***算力和存储*** 作为性能的主要限制因素

### Background: 矩阵乘法

GEMM: general Matrix-Matrix calculate, 一般形式
$
C = \alpha \ AB + \beta \ C 
$
其中$\alpha$, $\beta$是标量输入, ABC是矩阵, 如果不考虑标量且$\beta$为0, 维度分别为`M * N`, `N * K`, 最后计算一次的乘加运算的次数为`M * N * K`, 计算本身的次数为`2 * M * N * K`, 访问的bytes数为ABC三个的总bytes数, 计算`ops:bytes`为
$
arithmetic intensity = \frac{FLOPS}{bytes \ access} = \frac{2 \cdot M \cdot N \cdot K}{2 \cdot (M \cdot N + N \cdot K + M \cdot N)}
$

#### GPU中的操作

矩阵运算在GPU中的分割是将矩阵分成一小块一小块的, 然后把这样的分块放进不同的tread里面, 每次算完一个tiles组合后, 继续读取下一个 ==> 相当于直接使用分块矩阵的运算

[![示意图](https://s1.ax1x.com/2022/11/27/zNW41A.png)](https://imgse.com/i/zNW41A)

虽然Nano用不了所谓的Tensor Core, 但是如果可以对其存取的大小会快一点, 例如使用`FP16`的时候, 矩阵大小是 ***8x***, 所有维度都满足会更好, 或者退而求其次, 是 ***2的幂指数*** 总之是可以使用更高版本的`cuBLAS`会更快(在NVIDIA的GPU中负责高性能计算矩阵运算的是`cuBLAS`库)

在对矩阵进行分块的时候需要注意通过控制 ***tile*** 的大小来进行性能的取舍
* tile大: 取数据次数少; 并行性较差
* 当Mat本身很大的时候, tradeoff不重要, 因为tile再大都可以分成很多块; Mat本身很小的时候, 很容易GPU性能变差 
* 一般来讲矩阵的大小都满足使用大的tiles

#### 维度量化效应

##### tile量化
当矩阵被tile切开一部分的时候, 可能出现不完整的, 导致补零, 这时有一些本来不用算的计算也被算了进去, 因此复杂度变大了

一般`cuBLAS`库会自动选择较小的tile尺寸来防止出现tile量化

##### 波量化
由于tile的总数SM可以计算的最大的tile阵列的大小的差异, 例如所有的SM支持计算108个tile中的计算, 而最终一次计算需要110个, 则只能分两次算, 导致算108个和后面的2个所需要的时间是差不多, 降低了GPU的利用率

这两种都会导致GPU性能下降, 不够出现的情况不同, 一个是当矩阵的大小和tile的大小不兼容时, 一个是得到总的tiles多少和SMs数量不兼容时

[![示意图](https://s1.ax1x.com/2022/11/27/zN4rXn.png)](https://imgse.com/i/zN4rXn)

### Linear/ FC Layers Optimize

#### 将fc中的运算转化为矩阵运算

fc中的运算可以分成三类, 分别是
* forward propagate: 前推运算, 由已知输入和weight求输出
* activition: 激活运算, 由已知输出和weight求输入activation
* gradient: 梯度倒推, 由已知输入和输出求weight矩阵

将上述过程转化成一般的矩阵运算`GEMMs`的矩阵的维度如下图所示

[![维度变化示意图](https://s1.ax1x.com/2022/11/27/zNj6BD.md.png)](https://imgse.com/i/zNj6BD)
[![说明示意图](https://s1.ax1x.com/2022/11/27/zNjqEQ.png)](https://imgse.com/i/zNjqEQ)

因此, `input`, `output`, `batchsize`分别是fc层中可能造成影响的参数, 他们对性能的影响方式分别是

* I/O: 和`GEMMs`中的一般形式的限制条件一样, 越大越可能math-limited, 但是也更容易充分使用GPU性能.
* batchsize: 在计算梯度的时候, 不影响tile的分配以及其是否高效, 但是会影响单个tile中的计算效率; 在前两种计算中, 会直接影响tile的分配, 越大tile越多, 有可能充分利用或者是造成量化效应

#### 优化示例

文档中展示了对于`Transform`模块中如何优化fc层, 在transform结构的网络中, 使用到fc层的部位有两个, 分别是在`attention`模块中使用以及正常的前推中, 在二者中优化的思路类似, 首先是

* 对于输入输出的约束: 理论上要足够大来尽量充分使用GPU, 但是这个也取决于实际的问题, 对齐要满足
* 对于batchsize的约束: 在求前推和激活的时候, batchsize可以决定tile的个数, 因此可以计算batchsize的大小, 保证不要出现量化(尤其是在前推中, 因为这一步在训练和测试中都很重要)

### Convolution Optimize

如何使用卷积计算此处姑且略过, 只讲述如何挑选参数以及如何结合cuDNN来实现更高的效率

#### 使用cuDNN来加速卷积

cuDNN中包含一些可以估计性能以及调整filter的大小参数的函数, 同时cuDNN中有两种计算卷积的方式, 分别是
* implicit-GEMMs: 就是直接计算卷积, 但是中间所有的乘加运算都以 ***矩阵*** 的方式来计算, implicit把计算转化成向量的矩阵运算
* transform: 把矩阵先进行空间的映射, 经过更简单的计算后在映射回来
里面很多函数都是和如何根据不同卷积或者计算的实际性能来筛选计算方法和计算的, 典型的函数包括
* `cudnnConvolutionForward()`: specify general algorithm used
* `cudnnGet`: 前缀, 都是用来估计目前算法的性能的, 但是相对来说不太准确
* `cudnnFind`: 在计算的过程中挑选最优的卷积方法, 但是对时间和资源的消耗都很大

#### 使用Tensor Core来计算? 和一些和性能有关的分析
(似乎也不是都用到了Tensor Core)

主要使用 `implicit GEMMs`, 在计算中间的构造方法如下图所示
[![implicit mat构造示意](https://s1.ax1x.com/2022/11/27/zUFfZq.png)](https://imgse.com/i/zUFfZq)

示意图中展示的是进行前推计算中的构造, 实际上在进行后向激活和bp的时候也会有类似的构造, 其构造的implicit Matrix的结构如下图所示
[![所以情况对应的矩阵](https://s1.ax1x.com/2022/11/27/zUFbQJ.png)](https://imgse.com/i/zUFbQJ)
因为没有实际存在内存中, 所以不需要那么多次的读取, 读取到之后进行复制即可, 所以可以表示出其Arithmetic Intensity表达为
$
Arithmetic \ Intensity = \frac{2 * N (K \cdot H \cdot W) * (C \cdot R \cdot S)}{2 * (N \cdot C \cdot H \cdot W + C \cdot R \cdot S \cdot K + N \cdot K \cdot P \cdot Q)}
$

##### 从memory中的layout角度

一般就是`NCHW`或早或晚`NHWC`, 其中字母的顺序代表不同维度从memory中拿出来的速度, 不同的layout性能不同, 在 Tensor Core是中`NHWC`最快, 因为`NCHW`会包含自动的转置, 这样的设置可以用过框架来调整[pytorch中调整方法](https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html)

* ? 没有Tensor Core有用吗, 对于cuda core是这样的吗

##### 量化效应的优化

此处分配tiles的方式是根据输出的矩阵来进行划分, 因此是否产生量化效应和输出有关, 更容易出现`wave quantization`, 所以输出越小, 每次进行的卷积越慢越容易拖慢速度

#### 卷积参数和性能的关系

##### 输出参数

也即`N * P * Q`, 输出的大小, 在前推和激活里面作为N比较重要
* 可以提升性能, 越大性能越好
* 在求梯度的时候虽然不是很重要, 但是比较大的时候可以减少过多的存取, 也可以节省开销
* 一般提高N
* 在输出`N * P * Q`默认很小的情况, 也支持tile的算法, 但是可能并行度会不够 ==> 怎么办??

##### filter size

也即`C * R * S`, 在前推里面作为K, 在implicit Mat比较小的时候比较重要
* 对于`1 * 1`filter, channels越多越好, 无论是输入还是输出
* 对于梯度前推的情况, 只有`C`影响量化效应
 
##### Channel

主要的操作时需不需要对channel进行填充, 主要目的是启动Tensor Core, 是否需要手动填充和数据类型和cuDNN版本相关
* cuDNN 晚期: 自动填充channel为4的倍数
* 早期
  * 使用NCHW: 自动填充
  * 使用NHWC: 手动填充为8的倍数
* 在网络的第一层, 常常是1/3层, 最好填充到4
* 在三种计算中输入输出通道都可以决定性能(作为N), 所以一般来说越大越好, 但是比tile size大之后提升的性能的速度降低, 也即return减少

##### stride, dilation
相当于修改输入输出尺寸, 都作用不明显

### 对于Memory-limit Layer的优化

一般来说对于比较小的网络来说, ***memory*** 是主要的性能限制因素, 且这些层往往会花费大部分时间, 对这些曾进行参数调整没有意义

在这样的网络中不同的操作的性能分析如下

#### 标准化

Normalization大部分由于计算量比较小都是memory-limited, 此时输入比较小更有利于性能的提升, 从算法上区分是`non-persistent`和`persistent`的差别, 因为此时输入数据可以存储在GPU芯片的内存里面, 节省了搬运数据的时间

#### 激活

激活层的计算量同样也很少, 基本所有的运算速度都只取决于内存带宽, 因此持续时间只和激活次数相关, 也即进行运算的矩阵的大小

#### 池化

同样池化的计算量也很小, 都是受到内存限制的, 大部分都是被重用, 所以目标是尽量提升内存带宽利用率

### 使用混合精度进行训练

取得最大优化效果的前提: 在 Volta 或者 TURing 架构上

混合精度指的是不是所有的数据都用float32, 而是某些采用低精度的形式, 在`Pascal`框架引入(不知道支不支持nano), 使用`tensor core`可以进一步加速这样的速度, 需要对模型中数据进行类型的改变并增加`loss scale`

常常使用的精度包括
* single/double precision: `FP32`或者`FP64`, 一般会采用的精度
* half precision: `FP16`, 只使用16bit来表示数据, 所以传输起来比single precision快, 但是由于表示的bit数较少, 很可能会在由`FP32`向其转化的时候一些比较大或者比较小的数被忽略
  * 常常使用的弥补方法是对原始数据做移位, 把太小的数移动到可表示范围内, 但是太大的可能被移出去

因此, 如果想要使用`FP16`对原本`FP32`的网络做简化, 需要考虑上面说的精度损失的问题, 对于某些网络, 可能在这个过程损失的细节过过多导致网络判断不准确, 因此需要进行调整

#### 训练中的调整

在训练中为了防止`FP32`中训练出的网络的某些权重参数被省略掉, 需要在运算过程中给所有的参数集体scale, 由于最后判断结果会softmax, 所以这一步影响不太大, 但是在权重更新的时候需要把scale因子乘回去

可以将训练过程概括为

* copy一份`FP32`的结果
* 把结果copy为`FP16`的形式
* FP
* 乘以放大因子
* BP
* 乘回去
* 更新权值

选择放大因子可以根据梯度因子分布的直方图估计一个常数, 也可以每次迭代跟着更新, 算法可以概括为下面的
[![算法示意](https://s1.ax1x.com/2022/11/28/zaYGd0.png)](https://imgse.com/i/zaYGd0)
总的来说, 要是要使用混合精度的估计需要注意三点
* 把模型中可以转化数据类型的地方转成`FP16`
* 保存`FP32`的数据方便迭代更新
* 使用loss scale来保护小的梯度值

这个步骤虽然比较复杂, 但是很多框架都可以直接用, 主流的三种都可以直接用

#### 在tensor core中的进一步优化
(因为 jetson nano里面没有, 姑且略过)

要想最大化Tensor Core提供的性能的增益需要做到三点
* 满足shape的约束, 也就是满足整个byte传输的要求
* 提高算术运算的密度
  * 带有时序运算的: 拼接序列中不同时间的激活运算
  * architecture: dense math operation; wider layer
* 减少不是Tensor-Core的运算

#### 在pytorch中的优化方法

使用PyTorch在1.6之后提供的API`Automatic Mixed Precision`可以快速的转化 ==> 然而目前只能用pytorch1.2