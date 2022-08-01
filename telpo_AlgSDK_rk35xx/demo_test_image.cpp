/*
 * @Copyright: Guangdong Telpo Technologies
 * @Author: sun.yongcong(maxsunyc@gmail.com)
 * @Date: 2022-07-01 13:13:24
 * @Description: Algorithm API for image&video analysis
 */
#include<iostream>
#include<telpo_algsdk.h>
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/opencv.hpp>

using namespace telpo_algsdk;
int main(int argc,  char** argv)
{
    if(argc != 3)
    {
        std::cout<<"arguments errors\n";
        std::cout<<"input example: ./demo_test_image   0    test.jpg\n";
        return 0;
    }

    //选择模型
    std::string model_t = argv[1];
    telpo_algsdk_t algsdk_t;
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

    //读图片
    std::string img_path = argv[2];
    cv::Mat img = cv::imread(img_path);

    //algskd初始化模型，选择模型算法(人脸检测、人形检测等等)
    std::vector<telpo_object_t> retObjects;
    Telpo_algsdk algsdk;
    algsdk.init(algsdk_t);

    //算法检测过程。输入图片img，返回检测结果retObjects
    algsdk.process(img, retObjects);

    //打印检测到的物体个数
    std::cout<<"objects num: "<<retObjects.size()<<std::endl;

    //在图片上画框和置信度
    char prob[1024]={0};
    if(retObjects.size()>0)
    {
        for(auto item:retObjects)
        {
            sprintf(prob,"%.3f", item.prob);
            cv::rectangle(img, cv::Point(item.box.left, item.box.top), cv::Point(item.box.right, item.box.bottom), cv::Scalar(0,0,255),2);
            //cv::putText(img, prob, cv::Point(item.box.left, item.box.top+20), 2, 1, cv::Scalar(0,0,255),1,8);
        }
    }

    //保持检测结果的图片
    cv::imwrite("result.jpg", img);

    return 0;
}
