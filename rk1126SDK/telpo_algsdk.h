/*
 * @Copyright: Guangdong Telpo Technologies
 * @Author: sun.yongcong(maxsunyc@gmail.com)
 * @Date: 2022-06-28 20:21:15
 * @Description: Algorithm API for image&video analysis
 */

#ifndef TELPO_ALGSDK_H
#define TELPO_ALGSDK_H
#include<string>
#include<vector>
#include<opencv2/core/core.hpp>
//#include "logging.h"


namespace telpo_algsdk{
/**
 * @brief: return value of telpo sdk functions
 * @details: 
 */
typedef enum{
    TELPO_RET_SUCCESS = 0,
    TELPO_RET_FAIL    = -1,
}telpo_ret_t;

/**
 * @brief: telpo model application
 * @details: 定义模型算法应用
 */
typedef enum {
    TELPO_ALGSDK_PERSON    = 0,     //detect person
    TELPO_ALGSDK_FACE      = 1,     //detect face 
    TELPO_ALGSDK_NOMASK    = 2,     //detect face without mask检测没戴口罩的人脸
    TELPO_ALGSDK_HEAD      = 3,     //detect head
    TELPO_ALGSDK_NOHAT     = 4,     //detect head without hat检测没戴帽子的人头(安全帽、厨师帽)
    TELPO_ALGSDK_SMOKER    = 5,     //detect smoker吸烟人
    
    TELPO_ALGSDK_CAR       = 10,    //detect car
    TELPO_ALGSDK_EBIKE     = 11,    //detect eBike电动车

    TELPO_ALGSDK_FIRE      = 20,    //detect fire
    TELPO_ALGSDK_SMOG     = 21,    //detect smog烟雾
}telpo_algsdk_t;

/**
 * @brief: define object location
 * @details: 
 */
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

/**
 * @brief: define Telpo_algsdk
 * @details: 
 * @return {*}
 */
class Telpo_algsdk{
    public:
        Telpo_algsdk() = default;
        ~Telpo_algsdk() = default;

        telpo_ret_t init(telpo_algsdk_t appType);
        telpo_ret_t process(cv::Mat &img, std::vector<telpo_object_t> &retObjects);
        telpo_ret_t exit(){return TELPO_RET_SUCCESS;};
    public:
        class AlgImpl{
            public:
                virtual telpo_ret_t init() = 0;
                virtual telpo_ret_t process(cv::Mat &img, std::vector<telpo_object_t> &retObjects) = 0;
                virtual telpo_ret_t exit()=0;
            public:
                std::string getModelPath();
                std::string getAlgCfg(std::string fName);
        };
    private:
        AlgImpl *impl_ = nullptr;
};



}//namespace

#endif //TELPO_ALGSDK_H