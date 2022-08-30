@[toc]
# Architechture
- PC
	- model training
	- convert model to xxx.tflite
- Device
	- xxx.tflite interpreter
	- inference
		- Neon Kernels
		- Hardware Acceleration interface(GPU/APU...)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210514223405329.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTEzOTE2Mjk=,size_16,color_FFFFFF,t_70)


# Model Structure
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210514220755292.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTEzOTE2Mjk=,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210514223219454.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTEzOTE2Mjk=,size_16,color_FFFFFF,t_70)

## Subgraph
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210514223251873.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTEzOTE2Mjk=,size_16,color_FFFFFF,t_70)

### Operator
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210514221051542.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTEzOTE2Mjk=,size_16,color_FFFFFF,t_70)
# Class Structure
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210514221150296.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTEzOTE2Mjk=,size_16,color_FFFFFF,t_70)
## TfLiteNode/TfLiteRegistration
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210514221306446.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTEzOTE2Mjk=,size_16,color_FFFFFF,t_70)
# Interpreter Implement
## mmap
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210514221735443.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTEzOTE2Mjk=,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210514221719842.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTEzOTE2Mjk=,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210514221419782.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTEzOTE2Mjk=,size_16,color_FFFFFF,t_70)
### TfLiteNode
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210514221443526.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTEzOTE2Mjk=,size_16,color_FFFFFF,t_70)

### TfLiteRegistration
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/20210514221500687.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTEzOTE2Mjk=,size_16,color_FFFFFF,t_70)
 ### TfLiteContext
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210514221638951.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTEzOTE2Mjk=,size_16,color_FFFFFF,t_70)
## Workflow
### InterpreterBuild::operator()
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210514221832963.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTEzOTE2Mjk=,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210514221846427.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTEzOTE2Mjk=,size_16,color_FFFFFF,t_70)
### Subgraph::Invoke()
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210514221931325.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTEzOTE2Mjk=,size_16,color_FFFFFF,t_70)
# Example
This is CNN Model
1. Conv2D
2. MaxPool2D
3. Conv2D
4. MaxPool2D
5. Reshape
6. FullyConnected
7. Softmax
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021051422194911.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTEzOTE2Mjk=,size_16,color_FFFFFF,t_70)
## ParseNodes
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210514222237489.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTEzOTE2Mjk=,size_16,color_FFFFFF,t_70)
## ParseTensors
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210514222434451.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTEzOTE2Mjk=,size_16,color_FFFFFF,t_70)
### SetTensorParameterReadOnly()
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021051422262114.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTEzOTE2Mjk=,size_16,color_FFFFFF,t_70)

### SetTensorParameterReadWrite()
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210514222457672.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTEzOTE2Mjk=,size_16,color_FFFFFF,t_70)
## Subgraph::Invoke()
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210514222410443.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTEzOTE2Mjk=,size_16,color_FFFFFF,t_70)


