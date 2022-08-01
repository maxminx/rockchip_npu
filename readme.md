# 天波AI算法库telpo_algsdk介绍

## 1：基本信息

当前库版本：1.0

编辑日期：2022年 07月 26日 

编辑人：孙永聪

库位置：

依赖：opencv

目录结构如下：

```text
.
├── demo_test_image
├── demo_test_image.cpp
├── include
│   └── telpo_algsdk.h
├── lib
│   ├── libtelpoalgsdk.so
│   └── libtelpoalgsdk.so.1.0
├── opencv
│   ├── bin
│   ├── include
│   ├── lib
│   └── share
├── readme.md
├── telpo_algsdk_model
│   ├── bus.jpg
│   ├── car.json
│   ├── face.jpg
│   ├── face.json
│   ├── hat.json
│   ├── head.json
│   ├── helmet.jpg
│   ├── mask3.jpg
│   ├── mask.jpg
│   ├── nohat.json
│   ├── nomask.json
│   ├── person.json
│   ├── readme.txt
│   ├── result.jpg
│   ├── set_env.sh
│   ├── test.jpg
│   ├── yolov5s_face_1.0.rknn
│   ├── yolov5s_head_1.0.rknn
│   ├── yolov5s_mycoco.rknn
│   └── yolov5s_relu_rv1109_rv1126_out_opt.rknn
└── test.jpg
```

## 2：功能

- [x] 人形检测
- [x] 人脸检测
- [ ] 检测未戴口罩的人脸
- [ ] 人头检测
- [ ] 检测未戴安全帽的人头
- [ ] 车辆检测
- [ ] 吸烟检测
- [ ] 打电话检测

后续完成的算法应用都会以如下形式出现在telpo_algsdk.h头文件的telpo_algsdk_t中，供调用者查看。

```cpp
/**
 * @brief: telpo model application
 * @details: 定义模型算法应用
 */
typedef enum {
    TELPO_ALGSDK_PERSON    = 0,    //detect person
    TELPO_ALGSDK_FACE      = 1,    //detect face 
    TELPO_ALGSDK_NOMASK    = 2,    //detect face without mask 
    TELPO_ALGSDK_HEAD      = 3,    //detect head
    TELPO_ALGSDK_NOHAT     = 4,    //detect head without hat
    
    TELPO_ALGSDK_CAR       = 10,   //detect car
} telpo_algsdk_t;
```



## 3：使用方法示例

一：基本数据类型介绍

-  telpo_algsdk_t，用于指定初始化算法的类型，详细见功能部分介绍

- telpo_object_t，用于保持检测到的结果

- telpo_rect_t，矩形框结构体

  ```cpp
  typedef struct telpo_rect_t
  {
      int left;
      int top;
      int right;
      int bottom;
  }telpo_rect_t;
  /**
   * @brief: define object: location and probability
   * @details: 定义目标物体的位置和置信度的概率
   */
  typedef struct telpo_object_t
  {
      /* data */
      telpo_rect_t box;
      float prob;
      int label=1000;
  }telpo_object_t;
  ```

二：引用头文件

```cpp
#include<telpo_algsdk.h>
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
```

三：算法初始化

```cpp
telpo_algsdk_t algsdk_t= TELPO_ALGSDK_PERSON;
Telpo_algsdk algsdk;
algsdk.init(algsdk_t);
```

四：推理过程

```pwd
std::vector<telpo_object_t> retObjects;
cv::Mat img = cv::imread(img_path);//根据现实情况获取img。这里是根据路径获取
algsdk.process(img, retObjects);
```

***注意事项:

一个算法只需要一次初始化就可以重复调用推理过程。初始化需要消耗较长时间，千万别重复调用初始化。



## 4：demo运行演示

一：在Linux系统中设置环境变量TELPO_ALGSDK_MODEL，为算法应用找到模型和配置文件的路径

```bash
cd telpo_algsdk_model/
export  TELPO_ALGSDK_MODEL=`pwd`
```

二：添加telpoalgsdk.so动态库

```bash
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${TELPO_ALGSDK_MODEL}/../lib
```

三：运行人形检测算法，结果保存在result.jpg

```bash
./demo_test_image  0 test.jpg
```

测试其他算法功能，请选择其他数字，如下：

```cpp
switch (std::stoi(argv[1]))
    {
    case 0:
        algsdk_t = TELPO_ALGSDK_PERSON;
        break;
    
    case 1:
        algsdk_t = TELPO_ALGSDK_FACE;
        break;

    case 2:
        algsdk_t = TELPO_ALGSDK_NOMASK;
        break;

    case 3:
        algsdk_t = TELPO_ALGSDK_HEAD;
        break;

    case 4:
        algsdk_t = TELPO_ALGSDK_NOHAT;
        break;

    case 10:
        algsdk_t = TELPO_ALGSDK_CAR;
        break;
    
    default:
        std::cout<<"arguments errors\n";
        break;
    }
```

## 5：配置文件解析

以***.json文件出现，供使用者方便替换模型和设置阈值参数，face.json 如下：

```json
{
    "modelDet":"yolov5s_face_1.0.rknn",

    "totalLabels":1,

    "labelName":[],

    "threshObj":0.5,

    "threshDet":0.3,

    "threshNMS":0.5,

    "anchors":[4.01953125, 5.1953125, 6.66796875, 8.234375, 10.609375, 13.5546875, 15.8984375, 19.8125, 24.265625, 30.390625, 40.03125, 52.1875, 64.6875, 83.625, 114.0, 148.625, 224.5, 275.0]
}

```

参数介绍：

- modelDet，模型权重参数
- totalLabels，标签个数
- threshObj，可以设置的范围0~1，阈值越大检测出的框越少 
- threshDet，可以设置的范围0~1，阈值越大检测出的框越少 
- threshNMS，可以设置的范围0~1，阈值越大检测出的框越多



