/*
 * @Copyright: Guangdong Telpo Technologies
 * @Author: sun.yongcong(maxsunyc@gmail.com)
 * @Date: 2022-06-28 21:40:24
 * @Description: Algorithm API for image&video analysis
 */

// #ifndef TELPO_APPLICATIONS_H
// #define TELPO_APPLICATIONS_H

// namespace algsdk{

// }//namespace

//#endif

#ifndef TELPO_APPLICATIONS_H
#define TELPO_APPLICATIONS_H
#include<mutex>
#include "rknn_api.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "telpo_algsdk.h"

namespace telpo_algsdk{

typedef struct Box{
    float x,y,w,h;
}box;

typedef struct Detection{
    int classes;
    int sort_class;
    float objectness;
    float *prob=nullptr;
    float probb;
    box bbox;
} detection;


//**********************************************************//
class Telpo_AI_base{
    public:
        unsigned char* load_data(FILE* fp, size_t ofst, size_t sz);
        unsigned char* load_model(const char* filename, int* model_size);
};

//**********************************************************//
class Telpo_det:public Telpo_AI_base{
    public:
        Telpo_det();
        ~Telpo_det();
        telpo_ret_t init(const std::string &algCfg);
        telpo_ret_t infer(cv::Mat &img, std::vector<telpo_object_t> &Tdets);

    public:
        static int nms_comparator(const void *pa, const void *pb);

    private:
        telpo_ret_t postProcess(rknn_output outputs[]);
        telpo_ret_t get_yolo_box(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int netw, int neth, int stride);
        telpo_ret_t get_network_boxes(float *predictions, int netw,int neth,int GRID,int* masks, float* anchors, int box_off);
        telpo_ret_t doNMS();
        float boxIOU(box a, box b);
        float overlap(float x1,float w1,float x2,float w2);

        float sigmoid(float x) { return 1.0 / (1.0 + expf(-x)); }
        float unsigmoid(float y) { return -1.0 * logf((1.0 / y) - 1.0); }
        int  clamp(float val, int min, int max) { return val > min ? (val < max ? val : max) : min; }

        int32_t __clip(float val, float min, float max)
        {
            float f = val <= min ? min : (val >= max ? max : val);
            return f;
        }

        int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale)
        {
            float  dst_val = (f32 / scale) + zp;
            int8_t res     = (int8_t)__clip(dst_val, -128, 127);
            return res;
        }

        float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }

        int process(int8_t* input, float* anchor, int grid_h, int grid_w, int height, int width, int stride,
                   std::vector<float>& boxes, std::vector<float>& objProbs, std::vector<int>& classId, float threshold,
                   int32_t zp, float scale);

        int post_process(int8_t* input0, int8_t* input1, int8_t* input2, int model_in_h, int model_in_w, float conf_threshold,
                 float nms_threshold, float scale_w, float scale_h, std::vector<int32_t>& qnt_zps,
                 std::vector<float>& qnt_scales);

        int quick_sort_indice_inverse(std::vector<float>& input, int left, int right, std::vector<int>& indices);
        int nms(int validCount, std::vector<float>& outputLocations, std::vector<int> classIds, std::vector<int>& order,
               int filterId, float threshold);
        float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1,
                              float ymax1);
        
    
    private:
        const  static int net_hw = 640;
        const  static int grid2 = 20;
        const  static int grid1 = 40;
        const  static int grid0 = 80;
        const  static int nanchor = 3;
        const  static int nboxes_0 = 19200;//80*80*3
        const  static int nboxes_1 = 24000;//(80*80+40*40)*3
        float thresh_obj = 0.15;
        float thresh_det = 0.05;//thresh_object*prob[i]
        float thresh_nms = 0.15;
        int nclasses = 80;
        float anchors[18];
        int nboxes_total = 25200;
        int nboxes_total_useful = 0;
        unsigned char* model_data = nullptr;
        box box_tmp;
        detection* dets = nullptr;

        std::vector<telpo_object_t> detRets;

        rknn_context ctx;

        int PROP_BOX_SIZE;
        int OBJ_CLASS_NUM;

        
};



//**********************************************************//
class Telpo_cls:public Telpo_AI_base{
    public:
        //Telpo_cls();
        ~Telpo_cls();
        telpo_ret_t init(std::string &algCfg);
        telpo_ret_t infer(cv::Mat &img, int &label);
    private:
        unsigned char* _model_data = nullptr;
        float _threshCls = 0.6;
        int _netH = 112;
        int _netW = 112;
        rknn_context _ctx;
        rknn_input _inputs[1];
        rknn_output _outputs[1];
        rknn_input_output_num io_num;
        rknn_tensor_attr _input_attrs[1];
        rknn_tensor_attr _output_attrs[1];
};


//**********************************************************//
class Person_det: public Telpo_algsdk::AlgImpl{
    public:
        static Person_det* getInstance();
        telpo_ret_t init() final;
        telpo_ret_t process(cv::Mat &img, std::vector<telpo_object_t> &retObjects) final;
        telpo_ret_t exit() final;

    private:
        Person_det(){};
        Telpo_det telp_det;
        static Person_det* instance;
        static std::mutex mtx_t;
        //release men resource
        class GarbageCollector
        {
            public:
                ~GarbageCollector()
                {
                    if(Person_det::instance != nullptr)
                    {
                        delete Person_det::instance;
                        Person_det::instance = nullptr;
                    }
                }
        };
        static GarbageCollector gc;
};


//**********************************************************//
class Face_det: public Telpo_algsdk::AlgImpl{
    public:
        static Face_det* getInstance();
        telpo_ret_t init() final;
        telpo_ret_t process(cv::Mat &img, std::vector<telpo_object_t> &retObjects) final;
        telpo_ret_t exit() final;

    private:
        Face_det(){};
        Telpo_det telp_det;
        static Face_det* instance;
        static std::mutex mtx_t;
};


//**********************************************************//
class NoMask_det: public Telpo_algsdk::AlgImpl{
    public:
        static NoMask_det* getInstance();
        telpo_ret_t init() final;
        telpo_ret_t process(cv::Mat &img, std::vector<telpo_object_t> &retObjects) final;
        telpo_ret_t exit() final;

    private:
        NoMask_det(){};
        Telpo_det telp_det;
        Telpo_cls telpo_cls;
        static NoMask_det* instance;
        static std::mutex mtx_t;
};


//**********************************************************//
class Head_det: public Telpo_algsdk::AlgImpl{
    public:
        static Head_det* getInstance();
        telpo_ret_t init() final;
        telpo_ret_t process(cv::Mat &img, std::vector<telpo_object_t> &retObjects) final;
        telpo_ret_t exit() final;

    private:
        Head_det(){};
        Telpo_det telp_det;
        static Head_det* instance;
        static std::mutex mtx_t;
};



//**********************************************************//
class NoHat_det: public Telpo_algsdk::AlgImpl{
    public:
        static NoHat_det* getInstance();
        telpo_ret_t init() final;
        telpo_ret_t process(cv::Mat &img, std::vector<telpo_object_t> &retObjects) final;
        telpo_ret_t exit() final;

    private:
        NoHat_det(){};
        Telpo_det telp_det;
        static NoHat_det* instance;
        static std::mutex mtx_t;
};


//**********************************************************//
class Car_det: public Telpo_algsdk::AlgImpl{
    public:
        static Car_det* getInstance();
        telpo_ret_t init() final;
        telpo_ret_t process(cv::Mat &img, std::vector<telpo_object_t> &retObjects) final;
        telpo_ret_t exit() final;

    private:
        Car_det(){};
        Telpo_det telp_det;
        static Car_det* instance;
        static std::mutex mtx_t;

};



}//namespace

#endif