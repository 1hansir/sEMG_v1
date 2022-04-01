# sEMG_v1

# 肌肉电控制新想法

1. 脑电、肌肉电不同channel中引入注意力机制，选择最重要的channel，并且能够根据attention层的权重值反解出不同channel在解码过程中的重要性

2. encode的方法：

   * 数据降维：PLS--有监督回归算法   
   * Auto-encoder（SAE、AE）
   * CNN等其他网络  

3. 自由度：  肩胛 3 （旋转（内外），外展，前转）

   ​                 肘关节  1 屈伸

   ​				前臂     1    旋转

   ​				手腕     1    屈伸

   ​				手    暂定1（受电极数目限制）   抓取、放开    

   输出1*7的向量，每个分量代表一个自由度，对不同电极的注意力不同

4. 卡尔曼滤波方法对预测结果进行修正

5. 不需要考虑位置，速度，加速度的三个自由度：只需要考虑一个，其他两个可以通过微分形式获得；最好是加速度，可以对标到机器人的扭矩

6. ==预处理==：滤波 20Hz-500Hz、（去噪？），（离散小波变换？）

7. 不能对多个channel的数据进行归一化，因为肌肉电的幅度中包含重要信息

8. Loss 函数的设计：  target replication loss(of LSTM)

9. CNN输入形状：STFT,小波变换-->变成图像            或者将channel*frames，变成图像输出

10. 注意力机制模块的位置：如果采取神经网络作分类器，后，因为CNN网络只需要用于得到每个肌肉是否运动的信息；前，能缩小模型参数量，对【动作】的特异性高，但对于【肌肉】的特异性低。

11. ==如果直接用神经网络训练出幅度比较难的话，是否可以用神经网络结合其他算法，仅仅用神经网络作分类器使用？？==

12. 如何解决电极偏移以及受试者改变的情况: **迁移学习？fine-tuning？卡尔曼滤波？**

13. 如何处理复杂运动（多种运动同时出现）？或许可以设置一个hyperparamter——threshold，当某一维度的输出高于threshold，就认为该运动产生？

14. 如何实现online的训练，切窗？

15. 本问题是一个【多标签问题】，output的形式可以设置为两种（1. 一个向量，每一维的值代表某块肌肉在运动的概率，此时对应一个threshold，超过阈值即代表此处肌肉运动  2. 一个2*n的矩阵，2代表二分类（运动/不运动），n代表肌肉的种类数）

16. 通过机器人设置运动角度，人类模仿





## 两个核心问题：

1. 如何解决电极偏移以及受试者改变的情况
2. 如何设置网络结构，output_size， 以及lossfunction，来确保网络能够【处理多种动作同时出现】的情况，以及如何解码【运动的幅度】





## 设计网络结构

首先，在将数据送入网络之前，需要将数据1. 滤波（20-500Hz） 2. 分为两份（一份经过channel间归一化，送入分类器网络，另一份专用于衡量运动的幅度）



---

### 分类器

**网络一：机器学习方法   CNN + 注意力机制（后） + multiscale  +   threshold设定     LOSS_FN**   

---

对于每一个通道各设置一个CNN网络（1d-conv）（Multi-scale），每个CNN网络输出一个

（1）特征向量，最终的向量是根据各个通道的向量使用注意力机制求解而成，某个维度上的值超过threshold则说明此处肌肉参与运动

（2）2*n的张量，表征每一块肌肉动or不动。这种情况下loss_fn可以直接中crossentropy， 很爽

**扩大输入channel时或许可以考虑使用小波变换的近似解，与原信号作cat**



优点：

* Multi-scale设计能够提取来自不同观察窗口的特性
* 注意力机制置后能保证各块肌肉分离分析，然后通过注意力分析每一个运动中不同肌肉的重要性

缺点：

* 未考虑到不同肌肉电之间的空间耦合特性，没有将不同channel的信息一起卷积分析（不知道有没有）



---

**网络二：机器学习方法   CNN + 注意力机制（前） + multiscale  +   （threshold设定 ）    LOSS_FN **

---

基本与网络一相同，只不过把注意力模块放在了前端，

注意力机制实现的原理：1）由于此种情况下输入是channel*1 * time的张量，对于time维度进行1 x 1 的卷积相当于对于channel维度求权重

2）将每一个channel的数据作pooling，cat成新向量，用这个向量进行FC求注意力权重（channel数 *  肌肉数的矩阵）

优点，缺点与模型一相反





---

**网络三：机器学习方法   CNN + LSTM   注意力机制（前） +  +   （threshold设定 ）    LOSS_FN **

---

在CNN的静态特征提取的基础上加入LSTM，从而引入时序特征，在引入LSTM的基础上便可以不需要multi-scale结构。



优点：

* 可以实现对时序特征的分析
* 更方便进行online的训练，不需要以时间窗口作为检验肌肉运动的基本单元，这样检测的效果更具有连续性。



### 幅度器

---

或许可以使用传统的方法？

1. 离散小波变换+幅度分析
2. PCA+幅度分析



==或许最终可以将幅度器的数据与分类器的结果合并，变成一个类似于控制信号（每个自由度+运动幅度），同样这个过程**的参数也需要进行训练**==



==或许可以把前臂与后臂分离进行分析（前臂可使用nano数据集，后臂自己测量）==



==两个思路：第一个是通过特定肌肉上的电极实现精确控制、另一个是根据高位的电极环进行下肢的粗略控制（残疾人）==
