/*
 * @Copyright: Guangdong Telpo Technologies
 * @Author: sun.yongcong(maxsunyc@gmail.com)
 * @Date: 2022-06-28 21:40:41
 * @Description: Algorithm API for image&video analysis
 */
#include<unistd.h>
#include<iostream>
#include<fstream>
#include<algorithm>
#include<string>
#include<stdlib.h>
#include<set>
#include<mutex>
#include "rknn_api.h"
#include "json.hpp"
#include "telpo_algsdk.h"
#include "telpo_applications.h"
//#include "logging.h"
//using  namespace std;
namespace telpo_algsdk{
    //Telpo_AI_base*******************************************************************************
    /**
     * @brief: 
     * @details: 
     * @param {string} &algCfg
     * @return {*}
     */ 

    unsigned char* Telpo_AI_base::load_data(FILE* fp, size_t ofst, size_t sz){
        unsigned char* data;
        int            ret;

        data = NULL;

        if (NULL == fp) {
            return NULL;
        }

        ret = fseek(fp, ofst, SEEK_SET);
        if (ret != 0) {
            printf("blob seek failure.\n");
            return NULL;
        }

        data = (unsigned char*)malloc(sz);
        if (data == NULL) {
            printf("buffer malloc failure.\n");
            return NULL;
        }
        ret = fread(data, 1, sz, fp);
        return data;
    };

    unsigned char* Telpo_AI_base::load_model(const char* filename, int* model_size){
        FILE*          fp;
        unsigned char* data;

        fp = fopen(filename, "rb");
        if (NULL == fp) {
            printf("Open file %s failed.\n", filename);
            return NULL;
        }

        fseek(fp, 0, SEEK_END);
        int size = ftell(fp);

        data = load_data(fp, 0, size);

        fclose(fp);

        *model_size = size;
        return data;
    };
    //Telpo_det*******************************************************************************
    /**
     * @brief: 
     * @details: 
     * @param {string} &algCfg
     * @return {*}
     */ 
    Telpo_det::Telpo_det()
    {

    }
    Telpo_det::~Telpo_det()
    {
        for(int i=0; i<nboxes_total; ++i)
        {
            free(dets[i].prob);
        }
        free(dets);
    }
    telpo_ret_t Telpo_det::init(const std::string &algCfg){

        //std::cout<<"algCfg= "<<algCfg<<std::endl;
        
        std::ifstream i(algCfg);
        nlohmann::ordered_json alg_j;
        i>>alg_j;
        nlohmann::json algCfg_j =nlohmann::json::parse(alg_j.dump());
        
        if(!algCfg_j.contains("modelDet"))
        {
            std::cout<<"parse config key modelDet error"<<std::endl;
            return TELPO_RET_FAIL;
        }
        char *path=nullptr;
        path = getenv("TELPO_ALGSDK_MODEL");
        std::string model_det(path, path+strlen(path));
        model_det = model_det + "/" + algCfg_j["modelDet"].get<std::string>();

        if(!algCfg_j.contains("totalLabels"))
        {
            std::cout<<"parse config key totalClasses error"<<std::endl;
            return TELPO_RET_FAIL;
        }
        nclasses = algCfg_j["totalLabels"].get<int>();

        if(!algCfg_j.contains("anchors"))
        {
            std::cout<<"parse config key anchors error"<<std::endl;
        }
        std::vector<float> anchros_tmp= algCfg_j["anchors"].get<std::vector<float>>();
        
        for(int i=0; i<anchros_tmp.size()&&i<18; ++i)
        {
            anchors[i] = anchros_tmp[i];
            //std::cout<<anchors[i]<<" ";
        }
        //std::cout<<std::endl;


        if(algCfg_j.contains("threshObj"))
        {
            thresh_obj = algCfg_j["threshObj"].get<float>();
        }

        if(algCfg_j.contains("threshDet"))
        {
            thresh_det = algCfg_j["threshDet"].get<float>();
        }

        if(algCfg_j.contains("threshNMS"))
        {
            thresh_nms = algCfg_j["threshNMS"].get<float>();
        }

        // if(algCfg_j.contains("netHW"))
        // {
        //     net_hw = algCfg_j["netHW"].get<int>();
        // }

        PROP_BOX_SIZE = 5+nclasses;
        OBJ_CLASS_NUM = nclasses;

        int model_data_size=0;
        model_data = load_model(model_det.c_str(), &model_data_size);
        if(rknn_init(&ctx, model_data, model_data_size, 0, NULL) < 0)
        {
            std::cout<<"rknn init error"<<std::endl;
            return TELPO_RET_FAIL;
        }

        return TELPO_RET_SUCCESS;
    };
    /**
     * @brief: 
     * @details: 
     * @param {Mat} &img
     * @param {vector<telpo_object_t>} &retObjects
     * @return {*}
     */
    telpo_ret_t Telpo_det::infer(cv::Mat &img, std::vector<telpo_object_t> &Tdets){
        rknn_input inputs[1];
        rknn_output outputs[3];

        rknn_input_output_num io_num;
        if (rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num)) < 0) {
            printf("rknn_init rknn_query error \n");
            return TELPO_RET_FAIL;
        }

        //rknn_inputs
        memset(inputs, 0, sizeof(inputs));
        inputs[0].index = 0;
        inputs[0].size = 1228800;//==640*640*3;
        inputs[0].pass_through = false;         //需要type和fmt
        inputs[0].type = RKNN_TENSOR_UINT8;
        inputs[0].fmt = RKNN_TENSOR_NHWC;

        //rknn outputs
        memset(outputs, 0, sizeof(outputs));
        for (int i = 0; i < io_num.n_output; i++) {
            outputs[i].want_float = 0;
        }

        //rknn outputs_attr
        rknn_tensor_attr output_attrs[io_num.n_output];
        memset(output_attrs, 0, sizeof(output_attrs));
        for (int i = 0; i < io_num.n_output; i++) {
            output_attrs[i].index = i;
            rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        }

        rknn_tensor_attr input_attrs[io_num.n_input];
        memset(input_attrs, 0, sizeof(input_attrs));
        for (int i = 0; i < io_num.n_input; i++) {
            input_attrs[i].index = i;
            if (rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr)) < 0) {
            printf("rknn_init query  error");
            }
        }
        int channel = 3;
        int width   = 0;
        int height  = 0;
        if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
            //printf("model is NCHW input fmt\n");
            channel = input_attrs[0].dims[1];
            width   = input_attrs[0].dims[2];
            height  = input_attrs[0].dims[3];
        } else {
            //printf("model is NHWC input fmt\n");
            width   = input_attrs[0].dims[1];
            height  = input_attrs[0].dims[2];
            channel = input_attrs[0].dims[3];
        }

        cv::Mat imgResized;
        cv::resize(img, imgResized, cv::Size(net_hw,net_hw), (0, 0), (0, 0), cv::INTER_LINEAR);
        cv::cvtColor(imgResized, imgResized, cv::COLOR_BGR2RGB);
        inputs[0].buf = (void *)imgResized.data;

        if(rknn_inputs_set(ctx, 1, inputs) < 0)
        {
            std::cout<<"rknn_inputs_set fail!"<<std::endl;
            return TELPO_RET_FAIL;
        }

        if(rknn_run(ctx, nullptr) < 0)
        {
            std::cout<<"rknn_run fail!"<<std::endl;
            return TELPO_RET_FAIL;
        }

        if(rknn_outputs_get(ctx, 3, outputs, nullptr) < 0)
        {
            std::cout<<"rknn_outputs_get fail!"<<std::endl;
            return TELPO_RET_FAIL;
        }

        if(outputs[0].buf==NULL || outputs[1].buf==NULL || outputs[2].buf==NULL)
        {
            std::cout<<"outputs is NULL\n";
        }

        float scale_w = float(net_hw)/img.cols;
        float scale_h = float(net_hw)/img.rows;
        std::vector<float>    out_scales;
        std::vector<int32_t>  out_zps;
        for (int i = 0; i < io_num.n_output; ++i) {
            out_scales.push_back(output_attrs[i].scale);
            out_zps.push_back(output_attrs[i].zp);
        }

        post_process((int8_t*)outputs[0].buf, (int8_t*)outputs[1].buf, (int8_t*)outputs[2].buf, height, width,
               thresh_obj, thresh_nms, scale_w, scale_h, out_zps, out_scales);
        
        
        Tdets.swap(detRets);
        return TELPO_RET_SUCCESS;
    };

    telpo_ret_t Telpo_det::postProcess(rknn_output outputs[]){
        dets = (detection*)calloc(nboxes_total, sizeof(detection));
        for(int i=0; i<nboxes_total; ++i)
        {
            dets[i].prob = (float*)calloc(nclasses, sizeof(float));
        }

        float* output_0=(float*)outputs[0].buf;//80*80
        float* output_1=(float*)outputs[1].buf;//40*40
        float* output_2=(float*)outputs[2].buf;//20*20

        // std::cout<<"output_0 size="<<outputs[0].size<<std::endl;
        // std::cout<<"output_1 size="<<outputs[1].size<<std::endl;
        // std::cout<<"output_2 size="<<outputs[2].size<<std::endl;
        int masks_2[3] = {6, 7, 8};
        int masks_1[3] = {3, 4, 5};
        int masks_0[3] = {0, 1, 2};


        get_network_boxes(output_0,net_hw,net_hw,grid0,masks_0,anchors,0);
        get_network_boxes(output_1,net_hw,net_hw,grid1,masks_1,anchors,nboxes_0);
        get_network_boxes(output_2,net_hw,net_hw,grid2,masks_2,anchors,nboxes_1);

        doNMS();


        return TELPO_RET_SUCCESS;
    };

    int Telpo_det::process(int8_t* input, float* anchor, int grid_h, int grid_w, int height, int width, int stride,
                   std::vector<float>& boxes, std::vector<float>& objProbs, std::vector<int>& classId, float threshold,
                   int32_t zp, float scale)
    {
        int    validCount = 0;
        int    grid_len   = grid_h * grid_w;
        float  thres      = unsigmoid(threshold);
        int8_t thres_i8   = qnt_f32_to_affine(thres, zp, scale);

        thres = unsigmoid(thresh_det);
        int8_t thres_i8_det =  qnt_f32_to_affine(thres, zp, scale);
        //std::cout<<"\noutput anchors: "<<" zp="<<zp<<" scale="<<scale<<std::endl;
        for (int a = 0; a < 3; a++) {
            //std::cout<<" "<<anchor[a * 2]<<" "<<anchor[a * 2+1];
            for (int i = 0; i < grid_h; i++) {
            for (int j = 0; j < grid_w; j++) {
                int8_t box_confidence = input[(PROP_BOX_SIZE * a + 4) * grid_len + i * grid_w + j];
                if (box_confidence >= thres_i8) {
                    int     offset = (PROP_BOX_SIZE * a) * grid_len + i * grid_w + j;
                    int8_t* in_ptr = input + offset;

                    int8_t maxClassProbs = in_ptr[5 * grid_len];
                    int    maxClassId    = 0;
                    for (int k = 1; k < OBJ_CLASS_NUM; ++k) {
                        int8_t prob = in_ptr[(5 + k) * grid_len];
                        if (prob > maxClassProbs) {
                        maxClassId    = k;
                        maxClassProbs = prob;
                        }
                    }
                    if(maxClassProbs < thres_i8_det) continue;
                    float conf_class= sigmoid(deqnt_affine_to_f32(maxClassProbs, zp, scale));
                    float conf_obj =sigmoid(deqnt_affine_to_f32(box_confidence, zp, scale));
                    objProbs.push_back(conf_class*conf_obj);
                    classId.push_back(maxClassId);
                    



                    float   box_x  = sigmoid(deqnt_affine_to_f32(*in_ptr, zp, scale)) * 2.0 - 0.5;
                    float   box_y  = sigmoid(deqnt_affine_to_f32(in_ptr[grid_len], zp, scale)) * 2.0 - 0.5;
                    float   box_w  = sigmoid(deqnt_affine_to_f32(in_ptr[2 * grid_len], zp, scale)) * 2.0;
                    float   box_h  = sigmoid(deqnt_affine_to_f32(in_ptr[3 * grid_len], zp, scale)) * 2.0;
                    box_x          = (box_x + j) * (float)stride;
                    box_y          = (box_y + i) * (float)stride;
                    box_w          = box_w * box_w * (float)anchor[a * 2] ;//* (float)width;
                    box_h          = box_h * box_h * (float)anchor[a * 2 + 1] ;//* (float)height;
                    
                    // std::cout<<"\nbox_x="<<box_x*width<<" "<<width;
                    // std::cout<<"\nbox_y="<<box_y*height<<" "<<height;
                    // std::cout<<"\nbox_w="<<box_w*width;
                    // std::cout<<"\nbox_h="<<box_h*height;
                    box_x -= (box_w / 2.0);
                    box_y -= (box_h / 2.0);
                    boxes.push_back(box_x);
                    boxes.push_back(box_y);
                    boxes.push_back(box_w);
                    boxes.push_back(box_h);

                    
                    validCount++;
                }
            }
            }
        }
        return validCount;
    }

    int Telpo_det::post_process(int8_t* input0, int8_t* input1, int8_t* input2, int model_in_h, int model_in_w, float conf_threshold,
                 float nms_threshold, float scale_w, float scale_h, std::vector<int32_t>& qnt_zps,
                 std::vector<float>& qnt_scales)
    {

        std::vector<float> filterBoxes;
        std::vector<float> objProbs;
        std::vector<int>   classId;

        // stride 8
        int stride0     = 8;
        int grid_h0     = model_in_h / stride0;
        int grid_w0     = model_in_w / stride0;
        int validCount0 = 0;
        validCount0 = process(input0, (float*)(&anchors[0]), grid_h0, grid_w0, model_in_h, model_in_w, stride0, filterBoxes, objProbs,
                                classId, conf_threshold, qnt_zps[0], qnt_scales[0]);

        // stride 16
        int stride1     = 16;
        int grid_h1     = model_in_h / stride1;
        int grid_w1     = model_in_w / stride1;
        int validCount1 = 0;
        validCount1 = process(input1, (float*)(&anchors[6]), grid_h1, grid_w1, model_in_h, model_in_w, stride1, filterBoxes, objProbs,
                                classId, conf_threshold, qnt_zps[1], qnt_scales[1]);

        // stride 32
        int stride2     = 32;
        int grid_h2     = model_in_h / stride2;
        int grid_w2     = model_in_w / stride2;
        int validCount2 = 0;
        validCount2 = process(input2, (float*)(&anchors[12]), grid_h2, grid_w2, model_in_h, model_in_w, stride2, filterBoxes, objProbs,
                                classId, conf_threshold, qnt_zps[2], qnt_scales[2]);

        int validCount = validCount0 + validCount1 + validCount2;
        // no object detect
        if (validCount <= 0) {
            return 0;
        }

        std::vector<int> indexArray;
        for (int i = 0; i < validCount; ++i) {
            indexArray.push_back(i);
        }

        quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);

        std::set<int> class_set(std::begin(classId), std::end(classId));

        for (auto c : class_set) {
            nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold);
        }

        for(int i=0; i<indexArray.size(); ++i)
        {
            int id = indexArray[i];
            if(id==-1) continue;
            telpo_object_t tmp;
            tmp.box.left    = (int)(clamp(filterBoxes[id*4], 1, net_hw-2) / scale_w);
            tmp.box.top     = (int)(clamp(filterBoxes[id*4+1], 1, net_hw-2) / scale_h);
            tmp.box.right   = (int)(clamp(filterBoxes[id*4] + filterBoxes[id*4+2], 1, net_hw-2) / scale_w);
            tmp.box.bottom  = (int)(clamp(filterBoxes[id*4+1] + filterBoxes[id*4+3], 1, net_hw-2) / scale_h);
            tmp.prob = objProbs[id];
            tmp.label =classId[id];

            //std::cout<<"\nleft="<<tmp.box.left<<" top="<<tmp.box.top<<" right="<<tmp.box.right<<" bottom="<<tmp.box.bottom<<std::endl;

            detRets.push_back(tmp);
        }

         return 0;
    }

    int Telpo_det::quick_sort_indice_inverse(std::vector<float>& input, int left, int right, std::vector<int>& indices)
    {
        float key;
        int   key_index;
        int   low  = left;
        int   high = right;
        if (left < right) {
            key_index = indices[left];
            key       = input[left];
            while (low < high) {
            while (low < high && input[high] <= key) {
                high--;
            }
            input[low]   = input[high];
            indices[low] = indices[high];
            while (low < high && input[low] >= key) {
                low++;
            }
            input[high]   = input[low];
            indices[high] = indices[low];
            }
            input[low]   = key;
            indices[low] = key_index;
            quick_sort_indice_inverse(input, left, low - 1, indices);
            quick_sort_indice_inverse(input, low + 1, right, indices);
        }
        return low;
    }

    int Telpo_det::nms(int validCount, std::vector<float>& outputLocations, std::vector<int> classIds, std::vector<int>& order,
               int filterId, float threshold)
    {
        for (int i = 0; i < validCount; ++i) {
            if (order[i] == -1 || classIds[i] != filterId) {
            continue;
            }
            int n = order[i];
            for (int j = i + 1; j < validCount; ++j) {
            int m = order[j];
            if (m == -1 || classIds[i] != filterId) {
                continue;
            }
            float xmin0 = outputLocations[n * 4 + 0];
            float ymin0 = outputLocations[n * 4 + 1];
            float xmax0 = outputLocations[n * 4 + 0] + outputLocations[n * 4 + 2];
            float ymax0 = outputLocations[n * 4 + 1] + outputLocations[n * 4 + 3];

            float xmin1 = outputLocations[m * 4 + 0];
            float ymin1 = outputLocations[m * 4 + 1];
            float xmax1 = outputLocations[m * 4 + 0] + outputLocations[m * 4 + 2];
            float ymax1 = outputLocations[m * 4 + 1] + outputLocations[m * 4 + 3];

            float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

            if (iou > threshold) {
                order[j] = -1;
            }
            }
        }
    return 0;
    }

    float Telpo_det::CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1,
                              float ymax1)
    {
        float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
        float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
        float i = w * h;
        float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) + (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
        return u <= 0.f ? 0.f : (i / u);
    }

    telpo_ret_t Telpo_det::get_network_boxes(float *predictions, int netw,int neth,int GRID,int* masks, float* anchors, int box_off)
    {
        int lw=GRID;
        int lh=GRID;
        int nboxes=GRID*GRID*nanchor;
        int LISTSIZE=1+4+nclasses;
        //darkent output排列格式: box顺序为先grid再anchor
        //1个anchor: 7*7*x+7*7*y+7*7*w+7*7*w+7*7*obj+7*7*classes1+7*7*classes2..+7*7*classes80,共3个anchor
        //x和y(先做,或者放在后面转换到实际坐标这一步中也行),以及obj和classes做logisic
        //xy做logistic
        for(int n=0;n<nanchor;n++){
            int index=n*lw*lh*LISTSIZE;
            int index_end=index+2*lw*lh;
            for(int i=index;i<index_end;i++)
                predictions[i]=1./(1.+exp(-predictions[i]));			
        }
        //类别和obj做logistic
        for(int n=0;n<nanchor;n++){
            int index=n*lw*lh*LISTSIZE+4*lw*lh;
            int index_end=index+(1+nclasses)*lw*lh;
            for(int i=index;i<index_end;i++){
                predictions[i]=1./(1.+exp(-predictions[i]));		
            }
        }
        //dets将outpus重排列,dets[i]为第i个框,box顺序为先anchor再grid

        int count=box_off;
        for(int i=0;i<lw*lh;i++){
            int row=i/lw;
            int col=i%lw;
            for(int n=0;n<nanchor;n++){
                int box_loc=n*lw*lh+i;  
                int box_index=n*lw*lh*LISTSIZE+i;            //box的x索引,ywh索引只要依次加上lw*lh
                int obj_index=box_index+4*lw*lh;
                float objectness=predictions[obj_index];
                if(objectness<thresh_obj) continue;
                dets[count].objectness=objectness;
                dets[count].classes=nclasses;
                get_yolo_box(predictions,anchors,masks[n],box_index,col,row,lw,lh,netw,neth,lw*lh);
                dets[count].bbox=box_tmp;
                for(int j=0;j<nclasses;j++){
                //for(int j=0;j<1;j++){
                    int class_index=box_index+(5+j)*lw*lh;
                    float prob=objectness*predictions[class_index];
                    dets[count].prob[j]=prob;
                    //cout<<j<<"==>"<<dets[count].prob[j]<<"\n";
                }
                ++count;
            }
        }
        //cout<<"count: "<<count-box_off<<"\n";
        //return dets;
        return TELPO_RET_SUCCESS;
    }

    telpo_ret_t Telpo_det::get_yolo_box(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int netw, int neth, int stride)
    {

        box_tmp.x = (i + x[index + 0*stride]) / lw;
        box_tmp.y = (j + x[index + 1*stride]) / lh;
        box_tmp.w = exp(x[index + 2*stride]) * biases[2*n]   / netw;
        box_tmp.h = exp(x[index + 3*stride]) * biases[2*n+1] / neth;
    }

    telpo_ret_t Telpo_det::doNMS()
    {
        int total = nboxes_total;

        int i, j, k;
        k = total-1;
        for(i = 0; i <= k; ++i){
            if(dets[i].objectness == 0){
                detection swap = dets[i];
                dets[i] = dets[k];
                dets[k] = swap;
                --k;
                --i;
            }
        }
        total = k+1;
        //cout<<"total after OBJ_THRESH: "<<total<<"\n";

        for(k = 0; k < nclasses; ++k){
            for(i = 0; i < total; ++i){
                dets[i].sort_class = k;
            }
            qsort(dets, total, sizeof(detection), nms_comparator);
            for(i = 0; i < total; ++i){
                if(dets[i].prob[k] == 0) continue;
                box a = dets[i].bbox;
                for(j = i+1; j < total; ++j){
                    box b = dets[j].bbox;

                    if (boxIOU(a, b) > thresh_nms){
                        dets[j].prob[k] = 0;
                    }
                }
            }
        }

        nboxes_total_useful = total;
        return TELPO_RET_SUCCESS;
    }

    int Telpo_det::nms_comparator(const void *pa, const void *pb)
    {
        detection a = *(detection *)pa;
        detection b = *(detection *)pb;
        float diff = 0;
        if(b.sort_class >= 0){
            diff = a.prob[b.sort_class] - b.prob[b.sort_class];
        } else {
            diff = a.objectness - b.objectness;
        }
        if(diff < 0) return 1;
        else if(diff > 0) return -1;
        return 0;
    }

    float Telpo_det::boxIOU(box a, box b)
    {
        float w = overlap(a.x, a.w, b.x, b.w);
        float h = overlap(a.y, a.h, b.y, b.h);
        if(w < 0 || h < 0) return 0;
        float i = w*h;
        float u = a.w*a.h + b.w*b.h - i;
        return i/u;
    }

    float Telpo_det::overlap(float x1,float w1,float x2,float w2)
    {
        float l1=x1-w1/2;
        float l2=x2-w2/2;
        float left=l1>l2? l1:l2;
        float r1=x1+w1/2;
        float r2=x2+w2/2;
        float right=r1<r2? r1:r2;
        return right-left;
    }

    

    //Telpo_cls**************************************************************
    /**
     * @brief: implement of  Telpo_cls member function
     * @details: 
     * @return {*}
     */ 
    telpo_ret_t Telpo_cls::init(std::string &algCfg)
    {
        std::ifstream i(algCfg);
        nlohmann::ordered_json alg_j;
        i>>alg_j;
        nlohmann::json algCfg_j =nlohmann::json::parse(alg_j.dump());


        std::cout<<"telpo_cls  init"<<std::endl;
        
        if(!algCfg_j.contains("modelCls"))
        {
            std::cout<<"parse config key modelCls error"<<std::endl;
            return TELPO_RET_FAIL;
        }
        char *path=nullptr;
        path = getenv("TELPO_ALGSDK_MODEL");
        std::string model_cls(path, path+strlen(path));
        model_cls = model_cls + "/" + algCfg_j["modelCls"].get<std::string>();

        std::cout<<"\n cls model="<<model_cls<<std::endl;


        if(algCfg_j.contains("threshCls"))
        {
            _threshCls = algCfg_j["threshObj"].get<float>();
        }

        if(algCfg_j.contains("clsNetH"))
        {
            _netH = algCfg_j["threshObj"].get<int>();
        }

        if(algCfg_j.contains("clsNetW"))
        {
            _netW = algCfg_j["threshObj"].get<int>();
        }

        //init inputs
        memset(_inputs, 0, sizeof(_inputs));
        _inputs[0].index  = 0;
        _inputs[0].type = RKNN_TENSOR_UINT8;
        _inputs[0].size = 112 * 112 * 3;
        _inputs[0].fmt = RKNN_TENSOR_NHWC;
        _inputs[0].pass_through = 0;

        //init output
        memset(_outputs, 0 ,sizeof(_outputs));
        _outputs[0].want_float = 1;

        //
        memset(_input_attrs, 0, sizeof(_input_attrs));
        memset(_output_attrs, 0, sizeof(_output_attrs));

        int model_data_size=0;
        _model_data = load_model(model_cls.c_str(), &model_data_size);
        if(rknn_init(&_ctx, _model_data, model_data_size, 0, NULL) < 0)
        {
            std::cout<<"rknn init error"<<std::endl;
            return TELPO_RET_FAIL;
        }


        if(rknn_query(_ctx, RKNN_QUERY_INPUT_ATTR, &(_input_attrs[0]), sizeof(rknn_tensor_attr)) < 0)
        {
            std::cout<<"_input_attrs  rknn_query error"<<std::endl;
        }

        if(rknn_query(_ctx, RKNN_QUERY_OUTPUT_ATTR, &(_output_attrs[0]), sizeof(rknn_tensor_attr)) < 0)
        {
            std::cout<<"_output_attrs  rknn_query error"<<std::endl;
        }

        return TELPO_RET_SUCCESS;

    }


    telpo_ret_t Telpo_cls::infer(cv::Mat &img, int &label)
    {
        cv::Mat img_tmp;
        //std::cout<<"img clos= "<<img.cols<<"  img rows= "<<img.rows<<std::endl;
        cv::resize(img, img_tmp, cv::Size(112,112), (0, 0), (0, 0), cv::INTER_LINEAR);
        //std::cout<<"img_tmp size="<<img_tmp.size()<<std::endl;
        //cv::resize(img, img_tmp, cv::Size(_netH, _netW), (0, 0), (0, 0), cv::INTER_LINEAR);
        _inputs[0].buf = (void*)img_tmp.data;

        if(rknn_inputs_set(_ctx, 1, _inputs) < 0)
        {
            std::cout<<"Telpo_cls rknn_inputs_set fail, error code: "<<rknn_inputs_set(_ctx, 1, _inputs)<<std::endl;
            return TELPO_RET_FAIL;
        }

        if(rknn_run(_ctx,   nullptr) < 0)
        {
            std::cout<<"Telpo_cls rknn_run fail\n";
            return TELPO_RET_FAIL;
        }

        if(rknn_outputs_get(_ctx, 1, _outputs, nullptr) < 0)
        {
            std::cout<<"Telpo_cls rknn_outputs_get fail\n";
            return TELPO_RET_FAIL;
        }

        //get label with high score
        float* score = (float*)_outputs[0].buf;
        float tmp=0;
        int num = _outputs[0].size/4;

        for(int i=0; i<num; ++i)
        {
            if(score[i] > tmp)
            {
                tmp = score[i];
                label = i;
            }
        }

        if(tmp < _threshCls)
        {
            label = -1;
        }

        return TELPO_RET_SUCCESS;

    }


    Telpo_cls::~Telpo_cls()
    {
        rknn_outputs_release(_ctx, 1, _outputs);
        // if(_ctx > 0)
        // {
        //     rknn_destroy(_ctx);
        // }

        // if(_model_data != nullptr)
        // {
        //     free(_model_data);
        // }
    }




    //Person_det**************************************************************
    /**
     * @brief: implement of  Person_det member function
     * @details: 
     * @return {*}
     */ 
    std::mutex Person_det::mtx_t;
    Person_det* Person_det::instance = nullptr;
    Person_det::GarbageCollector Person_det::gc;
    Person_det* Person_det::getInstance()
    {
        if(instance == nullptr)
        {
            std::unique_lock<std::mutex> lock(mtx_t);
            if(instance == nullptr)
            {
                instance = new Person_det();
            }
        }
        return instance;
    }

    telpo_ret_t Person_det::init()
    {
        std::string algCfg = getAlgCfg(std::string("person.json"));
        if(telp_det.init(algCfg)!=0)
        {
            std::cout<<"Person_det init fail!\n";
        }
    }

    telpo_ret_t Person_det::process(cv::Mat &img, std::vector<telpo_object_t> &retObjects)
    {
        std::vector<telpo_object_t> tmp_obj;
        if(telp_det.infer(img, tmp_obj)!=0)
        {
            std::cout<<"Person_det process fail!\n";
            return TELPO_RET_FAIL;
        }

        for(auto item:tmp_obj)
        {
            if(item.label != 0) continue;
            retObjects.push_back(item);
        }

        return  TELPO_RET_SUCCESS;
    }


    telpo_ret_t Person_det::exit()
    {
        return TELPO_RET_SUCCESS;
    }

    //Face_det**************************************************************
    /**
     * @brief: implement of  Face_det member function
     * @details: 
     * @return {*}
     */ 
    std::mutex Face_det::mtx_t;
    Face_det* Face_det::instance = nullptr;
    Face_det* Face_det::getInstance()
    {
        if(instance == nullptr)
        {
            std::unique_lock<std::mutex> lock(mtx_t);
            if(instance == nullptr)
            {
                instance = new Face_det();
            }
        }
        return instance;
    }

    telpo_ret_t Face_det::init()
    {
        std::string algCfg = getModelPath();
        algCfg = algCfg+"/face.json";
        if(telp_det.init(algCfg)!=0)
        {
            std::cout<<"Face_det init fail!\n";
        }
    }

    telpo_ret_t Face_det::process(cv::Mat &img, std::vector<telpo_object_t> &retObjects)
    {
        if(telp_det.infer(img, retObjects)!=0)
        {
            std::cout<<"Face_det process fail!\n";
            return TELPO_RET_FAIL;
        }

        return  TELPO_RET_SUCCESS;
    }

    telpo_ret_t Face_det::exit()
    {
        return TELPO_RET_SUCCESS;
    }

//NoMask_det**************************************************************
    /**
     * @brief: implement of  Head_det member function
     * @details: 
     * @return {*}
     */ 
    std::mutex NoMask_det::mtx_t;
    NoMask_det* NoMask_det::instance = nullptr;
    NoMask_det* NoMask_det::getInstance()
    {
        if(instance == nullptr)
        {
            std::unique_lock<std::mutex> lock(mtx_t);
            if(instance == nullptr)
            {
                instance = new NoMask_det();
            }
        }
        return instance;
    }

    telpo_ret_t NoMask_det::init()
    {
        std::string algCfg = getAlgCfg(std::string("nomask.json"));
        if(telp_det.init(algCfg)!=0 || telpo_cls.init(algCfg)!=0)
        {
            std::cout<<"NoMask_det init fail!\n";
        }
        std::cout<<"NoMask_det init success\n";
    }

    telpo_ret_t NoMask_det::process(cv::Mat &img, std::vector<telpo_object_t> &retObjects)
    {
        cv::Mat img_rgb;
        cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);
        //detection  process
        std::vector<telpo_object_t> tmp_obj;
        if(telp_det.infer(img, tmp_obj)!=0)
        {
            //std::cout<<"NoMask_det process fail!\n";
            return TELPO_RET_FAIL;
        }

        
        //classify process
        cv::Mat roi;
        cv::Rect rect;
        int label = -1;
        for(auto item:tmp_obj)
        {
            //std::cout<<"*****************************************\n";
            rect = cv::Rect(item.box.left, item.box.top, item.box.right-item.box.left, item.box.bottom-item.box.top);
            //std::cout<<"\n"<<item.box.left<<" "<<item.box.top<<" "<<item.box.right-item.box.left<<" "<<item.box.bottom-item.box.top<<std::endl;
            //std::cout<<"\nrect"<<rect.x<<"  "<<rect.width<<std::endl;
            roi = img_rgb(rect);
            telpo_cls.infer(roi, label);
            if(label != 0) continue;
            retObjects.push_back(item);
        }

        return  TELPO_RET_SUCCESS;
    }

    telpo_ret_t NoMask_det::exit()
    {
        return TELPO_RET_SUCCESS;
    }




    //Head_det**************************************************************
    /**
     * @brief: implement of  Head_det member function
     * @details: 
     * @return {*}
     */ 
    std::mutex Head_det::mtx_t;
    Head_det* Head_det::instance = nullptr;
    Head_det* Head_det::getInstance()
    {
        if(instance == nullptr)
        {
            std::unique_lock<std::mutex> lock(mtx_t);
            if(instance == nullptr)
            {
                instance = new Head_det();
            }
        }
        return instance;
    }

    telpo_ret_t Head_det::init()
    {
        std::string algCfg = getModelPath();
        algCfg = algCfg+"/head.json";
        if(telp_det.init(algCfg)!=0)
        {
            std::cout<<"Head_det init fail!\n";
        }
    }

    telpo_ret_t Head_det::process(cv::Mat &img, std::vector<telpo_object_t> &retObjects)
    {
        if(telp_det.infer(img, retObjects)!=0)
        {
            std::cout<<"Head_det process fail!\n";
            return TELPO_RET_FAIL;
        }

        return  TELPO_RET_SUCCESS;
    }

    telpo_ret_t Head_det::exit()
    {
        return TELPO_RET_SUCCESS;
    }

//NoHat_det**************************************************************
    /**
     * @brief: implement of  Head_det member function
     * @details: 
     * @return {*}
     */ 
    std::mutex NoHat_det::mtx_t;
    NoHat_det* NoHat_det::instance = nullptr;
    NoHat_det* NoHat_det::getInstance()
    {
        if(instance == nullptr)
        {
            std::unique_lock<std::mutex> lock(mtx_t);
            if(instance == nullptr)
            {
                instance = new NoHat_det();
            }
        }
        return instance;
    }

    telpo_ret_t NoHat_det::init()
    {
        std::string algCfg = getAlgCfg(std::string("nohat.json"));
        if(telp_det.init(algCfg)!=0)
        {
            std::cout<<"NoHat_det init fail!\n";
        }
    }

    telpo_ret_t NoHat_det::process(cv::Mat &img, std::vector<telpo_object_t> &retObjects)
    {
        std::vector<telpo_object_t> tmp_obj;
        if(telp_det.infer(img, tmp_obj)!=0)
        {
            std::cout<<"NoHat_det process fail!\n";
            return TELPO_RET_FAIL;
        }

        for(auto item:tmp_obj)
        {
            if(item.label == 1) continue;
            retObjects.push_back(item);
        }

        return  TELPO_RET_SUCCESS;
    }

    telpo_ret_t NoHat_det::exit()
    {
        return TELPO_RET_SUCCESS;
    }


//Car_det**************************************************************
    /**
     * @brief: implement of  Head_det member function
     * @details: 
     * @return {*}
     */ 
    std::mutex Car_det::mtx_t;
    Car_det* Car_det::instance = nullptr;
    Car_det* Car_det::getInstance()
    {
        if(instance == nullptr)
        {
            std::unique_lock<std::mutex> lock(mtx_t);
            if(instance == nullptr)
            {
                instance = new Car_det();
            }
        }
        return instance;
    }

    telpo_ret_t  Car_det::init()
    {
        std::string algCfg = getAlgCfg(std::string("car.json"));
        if(telp_det.init(algCfg)!=0)
        {
            std::cout<<"Car_det init fail!\n";
        }
    }

    telpo_ret_t Car_det::process(cv::Mat &img, std::vector<telpo_object_t> &retObjects)
    {
        std::vector<telpo_object_t> tmp_obj;
        if(telp_det.infer(img, tmp_obj)!=0)
        {
            std::cout<<"Car_det process fail!\n";
            return TELPO_RET_FAIL;
        }

        for(auto item:tmp_obj)
        {
            if(item.label != 2 && item.label !=5 && item.label !=7) continue;
            retObjects.push_back(item);
        }

        return  TELPO_RET_SUCCESS;
    }

    telpo_ret_t Car_det::exit()
    {
        return TELPO_RET_SUCCESS;
    }


    
}

