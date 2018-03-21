本目录 用来记录AI学习研究过程中 遇到的问题及解决方法
=====================================================
框架选择 
--------------
1.  Keras框架使用example， 新手使用keras； <br>
2.  TensorLayer 框架使用example， 与keras相比:<br>
    a） 在训练stacked CNN时， 训练速度能提升10%-15%， 不太明显 <br>
    b） 在训练resnet结构时， 训练速度能提升60%左右，非常明显。 所以在实际使用中，可以tensorlayer为主来训练模型。 <br>
        然而对于自定义层，tensorlayer需要编写一个Class来定义， 有时显得不太方便， 此时可以使用原生态的tensorflow来编写，速度差不多，但灵活性更好<br>

DNN 架构
----------------------
3.  不同CNN 架构的实现 example：<br>
    a） Resnet like <br>
    b） ResNeXT like <br>
    c） SE-Net<br>

训练优化-loss
-------------------
4.  不同loss的对比， 考虑hard example， easy example<br>
    a） OHEM<br>
    b） Focal Loss <br>
