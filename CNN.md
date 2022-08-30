# Reference
- [How Convolutional Neural Networks work？](https://www.bilibili.com/video/av19231561/)

#  A toy ConvNet: X's and O's
Says whether a picture is of an X or an O
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210214144630309.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTEzOTE2Mjk=,size_16,color_FFFFFF,t_70)

For example

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210214144630348.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTEzOTE2Mjk=,size_16,color_FFFFFF,t_70)

**Trickier Cases**
- translation
- scaling
- rotation
- weight

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210214144630551.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTEzOTE2Mjk=,size_16,color_FFFFFF,t_70)

Deciding is hard

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210214144630443.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTEzOTE2Mjk=,size_16,color_FFFFFF,t_70)

What computers see?

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210214144630442.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTEzOTE2Mjk=,size_16,color_FFFFFF,t_70)

The red area is incorrect

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210214144630360.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTEzOTE2Mjk=,size_16,color_FFFFFF,t_70)

ConvNet match pieces of the image

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210214144630570.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTEzOTE2Mjk=,size_16,color_FFFFFF,t_70)

Features match pieces of the image

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210214144630369.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTEzOTE2Mjk=,size_16,color_FFFFFF,t_70)

## Filtering(The math behind the match)
1. Line up the feature and the image patch
2. Multiply each image pixel by the corresponding feature pixel
3. Add them up
4. Divide by the total number of pixels in the feature
<img src="https://pic1.zhimg.com/v2-6428cf505ac1e9e1cf462e1ec8fe9a68_b.webp" alt="show" />

(output is average value)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210214150017787.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTEzOTE2Mjk=,size_16,color_FFFFFF,t_70#pic_center)
## Pooling(Shrinking the image stack)
1. Pick a window size (usually 2 or 3)
2. Pick a stride (usually 2)
3. Walk your window across your filtered images
4. From each window, take the maximum value

After Filter，data calculation waste much time
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210214144630561.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTEzOTE2Mjk=,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210214144630574.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTEzOTE2Mjk=,size_16,color_FFFFFF,t_70)

**Pooling layer**
A stack of images becomes a stack of smaller images.
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210214144630591.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTEzOTE2Mjk=,size_16,color_FFFFFF,t_70)

**Normalization**
Keep the math from breaking by tweaking each of the values just a bit.
Change everything negative to zero
**ReLu layer**
A stack of images becomes a stack of images with no negative values.
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210214144630605.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTEzOTE2Mjk=,size_16,color_FFFFFF,t_70)

**Layers get stacked**
The output of one becomes the input of the next.
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210214144630567.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTEzOTE2Mjk=,size_16,color_FFFFFF,t_70)

**Deep stacking** 
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210214144630559.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTEzOTE2Mjk=,size_16,color_FFFFFF,t_70)

## Fully connected layer

Every value gets a vote

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210214144630444.png)

Vote depends on how strongly a value predicts X or O.

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210214144630565.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTEzOTE2Mjk=,size_16,color_FFFFFF,t_70)

### Backpropagation
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210214144630647.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTEzOTE2Mjk=,size_16,color_FFFFFF,t_70)

### Gradient descent
For each feature pixel and voting weight, adjust it up and down a bit and see how the error changes.

[![在这里插入图片描述](https://img-blog.csdnimg.cn/20210214144630541.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTEzOTE2Mjk=,size_16,color_FFFFFF,t_70)

## Putting it all together
A set of pixels becomes a set of votes

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210214150246913.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTEzOTE2Mjk=,size_16,color_FFFFFF,t_70#pic_center)
# Hyperparameters(knobs)
(human set parameters)
- Convolution
    - Number of features
    - Size of features
- Pooling
    - Window size
    - Window stride
- Fully Connected
    - Number of neurons

# Application
## Image
Any 2D(or 3D) data
Things closer together are more closely related than things far away.

## Sound
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210214144630421.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTEzOTE2Mjk=,size_16,color_FFFFFF,t_70)

## Text
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210214144630418.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTEzOTE2Mjk=,size_16,color_FFFFFF,t_70)

# Limitations
CNN only capture local "spatial" patterns in data.
If the data can't be made to look like an image, CNN are less useful.
If your data is just as useful after swapping any of your columns with each other, then you can't use Convolutional Neural Network.





