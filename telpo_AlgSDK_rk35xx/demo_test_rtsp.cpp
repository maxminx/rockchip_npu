/*
 * @Copyright: Guangdong Telpo Technologies
 * @Author: sun.yongcong(maxsunyc@gmail.com)
 * @Date: 2022-07-09 11:13:09
 * @Description: Algorithm API for image&video analysis
 */

#include<iostream>
#include<telpo_algsdk.h>
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/opencv.hpp>
#include<opencv2/videoio.hpp>


using namespace telpo_algsdk;
int main(int argc,  char** argv)
{
    std::string rtsp_url = argv[1];
    cv::VideoCapture rtsp_cap;
    rtsp_cap.open(rtsp_url);
    if(!rtsp_cap.isOpened())
    {
        std::cout<<"fail to open rtsp url"<<std::endl;
        return 0;
    }

    //init
    cv::namedWindow("demoRtsp");
    cv::Mat img;
    std::vector<telpo_object_t> retObjects;
    Telpo_algsdk algsdk;
    algsdk.init(TELPO_APP_FACE);

    //process
    while(true)
    {
        rtsp_cap>>img;
        if(img.empty())
        {
            cv::waitKey();
            continue;
        }
        algsdk.process(img, retObjects);
        char prob[1024]={0};
        if(retObjects.size()>0)
        {
            for(auto item:retObjects)
            {
                sprintf(prob,"%.3f", item.prob);
                cv::rectangle(img, cv::Point(item.box.left, item.box.top), cv::Point(item.box.right, item.box.bottom), cv::Scalar(255,0,0),2);
                cv::putText(img, prob, cv::Point(item.box.left, item.box.top+20), 2, 1, cv::Scalar(0,0,255),2,8);
            }
        }
        cv::imshow("demoRtsp", img);
        cv::waitKey(1);
    }

    return 0;
}
