# 程序功能介绍
## KNN.java
在hadoop上实现了KNN算法，优化了从N个值中选取K个最小的算法。


## evaluation.cpp
针对KNN.java输出的结果，计算结果的准确率。

## run.sh
是放在hadoop里，自动编译运行KNN.java，再运行评估程序的脚本。


# 问题应用场景
KNN用过的人应该有这样一个共识：
1. K通常是比较小的，从1-500，一般视训练数据规模和分布而定，但是很少有超过100的其实，因为太大的K值会加强噪声从而降低分类准确率（参考：http://www.cnblogs.com/fengfenggirl/archive/2013/05/27/knn.html）。
2. 训练数据量大的时候很慢，这是一个O(M(N + Nlog K))的算法，其中N是训练数据的规模，M是测试数据的规模，K就是KNN的K值。公式的意义是，对于每个测试数据（贡献了M），需要扫描一遍训练数据得到N个距离值（贡献了N），从这N个距离值从选取K个最小的（贡献了Nlog K，而且得用堆才能达到这样的复杂度）。

之前在kaggle上做san francisco crime classification这个比赛的时候，它的输入数据规模是88万左右，这个用单机版的KNN跑起来就慢的要命，所以就想做一个分布式版本的KNN，在**训练数据量很大而测试数据量不多**的情况下能够快速并行完成。

# KNN.java的设计
## Mapper的输入
使用默认的分割方式，按行输入，比如下面的数据
```text
5.1,3.5,1.4,0.2,Iris-setosa
4.6,3.1,1.5,0.2,Iris-setosa
4.7,3.2,1.3,0.2,Iris-setosa
4.9,3.0,1.4,0.2,Iris-setosa
```
对于Mapper会变成：
```text
1: 5.1,3.5,1.4,0.2,Iris-setosa
2: 4.6,3.1,1.5,0.2,Iris-setosa
3: 4.7,3.2,1.3,0.2,Iris-setosa
4: 4.9,3.0,1.4,0.2,Iris-setosa
```
**注意，在这篇README中，始终使用Key: Value来表示键值对。**

即Mapper的输入键值对类型为：(LongWritable, Text)。


## Mapper的中间处理
注意输入进来的训练数据是Text类型，所以应该先将其分割成：（Double特征向量，类标）。
然后对于每个测试数据（已经通过别的方式预先放在内存里了），分别计算一下自己（这个训练数据）到它们的距离。


## Mapper的输出
类型为（IntWritable, Elem），其中IntWritable表示是与第几个test数据的结果，这样做是为了让结果在Combine的时候，对于同个test数据的结果，能够被Combine到一起来计算。因为我们本来就是要从N个train数据对于同个test数据的距离中，选出K个最小的，针对其类别投票决定这个test数据的类别。

Elem是我自己定义的一个数据类型，从上面的描述中，它至少应该包含两个数据：
1. 训练数据到测试数据的距离；
2. 该训练数据的类别。


## Combiner的使用
为什么要设置Combiner？
> 每一个map都可能会产生大量的本地输出 ，Combiner的作用就是对map端的输出先做一次合并，以减少在map和reduce节点之间的数据传输量，以提高网络IO性能，是MapReduce的一种优化手段之一

更多例子和解释请参考：
1. http://www.zhangrenhua.com/2015/12/19/hadoop-MapReduce-Combiner/
2. http://www.cnblogs.com/edisonchou/p/4297786.html
3. http://blog.csdn.net/jokes000/article/details/7072963

Combiner其实就是本地的一个reducer，把本地mapper的输出中，键值相同的数据组合到一起，找出本地最小的K个距离后，作为当前自己所在这个结点的输出给reducer，试想本地有100000个mapper，而K只有10，如果不使用Combiner，那么你就得通过网络传输这100000个结果给reducer；而使用Combiner的话，只需要将合并后的10个结果传给reducer即可！在这种情况下效率的提高可见一斑！


### Combiner的输入输出
输入类型是(IntWritable, Iterable<Elem>)，IntWritable还是表示第几个测试数据，而后面是一个迭代器，表示对于同个测试数据的距离计算结果。

处理过程就是从N个中选K个最小的。

输出类型是(IntWritable, Elem)，选择本地K个最小的结果之后，将其输出给reducer。


## Reducer
类型是(IntWritable, Iterable<Elem>)，其实跟Combiner是一模一样的，中间处理过程也是从N个中选K个最小的。
不过输出就不一样了，我想要的结果是`已知正确的结果 预测结果`，这样最后才能计算分类准确率，所以reducer的输出类型是(Text, Text)。


# 从N个中选取K个最小的算法
目前我想到的可以有三种方案，本次采用的是：维护一个大小为K的数据，使用插入排序来做，复杂度为O(KN)，另外，我在我的博客里还讨论了更好的方法，具体见：http://blog.csdn.net/jacketinsysu/article/details/51764305


## 小细节
特征向量的各个维度要归一化处理，这里采用的是min-max，即`x' = (x-small)/(big-small)`：
```java
public static void Min_Max_Norm(Vector<BigDecimal> X) {
        BigDecimal small = X.get(0), big = X.get(0);
        for (int i = 0; i < X.size(); ++i) {
            BigDecimal now = X.get(i);
            if (small.compareTo(now) > 0)
                small = now;
            if (big.compareTo(now) < 0)
                big = now;
        }

        BigDecimal interval = big.subtract(small);
        for (int i = 0; i < X.size(); ++i) {
            X.set(i, (X.get(i).subtract(small)).divide(interval, MathContext.DECIMAL64));
        }
    }
```