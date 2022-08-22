/*
 * @Copyright: Guangdong Telpo Technologies
 * @Author: sun.yongcong(maxsunyc@gmail.com)
 * @Date: 2022-07-01 17:42:50
 * @Description: Algorithm API for image&video analysis
 */

#include<iostream>
#include<fstream>
#include<string>
#include "3rdparty/nlohmann/json.hpp"
int main()
{
    std::string cfg="model/face.json";

    std::ifstream i("model/face.json");
    

    nlohmann::ordered_json algcfg;
    i>>algcfg;

    nlohmann::json json_parse = nlohmann::json::parse(algcfg.dump());
    

    return 0;
}