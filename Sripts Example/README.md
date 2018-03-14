本目录 用来记录AI学习研究过程中 遇到的问题及解决方法
=====================================================
框架选择 
--------------
1.  Keras框架使用example， 新手使用keras； 
2.  TensorLayer 框架使用example， 与keras相比：
    a） 在训练stacked CNN时， 训练速度能提升10%-15%， 不太明显 
    b） 在训练resnet结构时， 训练速度能提升60%左右，非常明显。 所以在实际使用中，以tensorlayer为主， 

DNN 架构
----------------------
3.  不同CNN 架构的实现 example：
    a） Resnet like 
    b） ResNeXT like 
    c） SE-Net

训练优化-loss
-------------------
4.  不同loss的对比， 考虑hard example， easy example
    a） OHEM
    b） Focal Loss 
