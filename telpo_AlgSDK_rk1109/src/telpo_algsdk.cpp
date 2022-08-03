/*
 * @Copyright: Guangdong Telpo Technologies
 * @Author: sun.yongcong(maxsunyc@gmail.com)
 * @Date: 2022-06-28 21:38:04
 * @Description: Algorithm API for image&video analysis
 */
#include<iostream>
#include "telpo_algsdk.h"
#include "telpo_applications.h"

namespace telpo_algsdk{
    telpo_ret_t Telpo_algsdk::init(telpo_algsdk_t appType)
    {
        switch(appType)
        {
            case TELPO_ALGSDK_PERSON:
                impl_ = Person_det::getInstance();
                break;

            case TELPO_ALGSDK_FACE:
                impl_ = Face_det::getInstance();
                break;

            case TELPO_ALGSDK_NOMASK:
                impl_ = NoMask_det::getInstance();
                break;

            case TELPO_ALGSDK_HEAD:
                impl_ = Head_det::getInstance();
                break;
            
            case TELPO_ALGSDK_NOHAT:
                impl_ = NoHat_det::getInstance();
                break;

            case TELPO_ALGSDK_CAR:
                impl_ = Car_det::getInstance();
                break;
            
            default:
                std::cout<<"\ninvalid algorithm app"<<std::endl;
                return TELPO_RET_FAIL;
        }

        return impl_->init();
    }

    telpo_ret_t Telpo_algsdk::process(cv::Mat &img, std::vector<telpo_object_t> &retObjects)
    {
        
        return impl_->process(img, retObjects);
    }

    std::string Telpo_algsdk::AlgImpl::getModelPath()
    {
        char* path=nullptr;
        path = getenv("TELPO_ALGSDK_MODEL");
        if(path==nullptr)
        {
            std::cout<<"fail to get model path\n";
            exit();
            
        }
        std::string temp(path,path+strlen(path));
        return temp;
    }

    std::string Telpo_algsdk::AlgImpl::getAlgCfg(std::string fName)
    {
        std::string fullPath = getModelPath() + "/" +fName;
        return fullPath;

    }

}

