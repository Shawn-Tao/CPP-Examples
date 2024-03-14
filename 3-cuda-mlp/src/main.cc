/*
 * @Author: Shawn-Tao 1054087304@qq.com
 * @Date: 2024-03-12 22:16:12
 * @LastEditors: Shawn-Tao 1054087304@qq.com
 * @LastEditTime: 2024-03-14 22:50:35
 * @FilePath: /3-cuda-mlp/src/main.cc
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */

#include "mlp.h" 
#include <string>
#include <iostream>
#include <fstream>

int main(){
    // print compile time
    printf("compile time: %s %s\n", __DATE__, __TIME__);
    printCudaVersion();
   
    int input_dim = 70;
    int output_dim = 12;
    std::vector<int> hidden_dim = {512,256,128};

    MLP_Network *mlp_network = new MLP_Network(input_dim, output_dim, hidden_dim);
    mlp_network->load("../weights/model.");

    float *input_data,
    *output_data;

    input_data = (float *)malloc(input_dim * sizeof(float));
    output_data = (float *)malloc(output_dim * sizeof(float));

    // input data from dataset/data.txt, data fromat as :-1.4129668474197388, -0.9265999794006348, 0.8672218322753906, -2.7135589122772217, -2.0724117755889893, -4.636621475219727, -2.9140963554382324, -2.277871608734131, -1.7852810621261597, -2.9832754135131836, -2.3302969932556152, -2.713895797729492, -2.20442795753479, -2.247941255569458, -2.8078951835632324, -1.1597107648849487, -3.0197761058807373, 0.18199443817138672, -0.47706329822540283, -1.4346051216125488, -1.1049365997314453, -3.23655366897583, -1.979907751083374, -2.8308358192443848, -1.831479549407959, -1.4634394645690918, -1.0012321472167969, -3.254819393157959, -2.288618326187134, -1.399328589439392, -2.173872947692871, -2.428089141845703, -2.060429573059082, -1.1480692625045776, -0.016515612602233887, -2.012803316116333, -2.467679023742676, -1.7028706073760986, -2.516176462173462, -1.5330582857131958, -1.868077039718628, -1.5477488040924072, -0.7347134351730347, -2.5074355602264404, -0.9581161737442017, -1.5913506746292114, -1.71095871925354, -3.872709274291992, -2.4011998176574707, -1.3601646423339844, -1.7930248975753784, -2.1834499835968018, -2.7953999042510986, -2.4132063388824463, -1.7903368473052979, -0.8136016130447388, -2.761786699295044, -0.6381787061691284, -2.193225622177124, -1.9163774251937866, -3.174342632293701, -0.8853050470352173, -0.993748664855957, -3.236908435821533, -3.0536985397338867, -1.2433278560638428, -2.1383984088897705, -2.9394044876098633, -0.9253208637237549, -1.975097417831421
    std::string input_file = "../dataset/data.txt";
    std::ifstream input_in(input_file);
    if (!input_in.is_open())
    {
        std::cout << "file open failed" << std::endl;
        return -1;
    }
    // 获取一行数据
    std::string line;
    std::getline(input_in, line);
    // std::cout << line << std::endl;
    // 将一行数据转换为float
    std::string::size_type pos = 0;
    std::string::size_type pre_pos = 0;
    for (int i = 0; i < input_dim; i++)
    {
        pos = line.find(",", pre_pos);
        input_data[i] = std::stof(line.substr(pre_pos, pos - pre_pos));
        pre_pos = pos + 2;
    }

    printf("input data: ");
    for (int i = 0; i < input_dim; i++)
    {
        printf("%f ", input_data[i]);
    }
    printf("\n");

    mlp_network->forward(input_data, output_data);

    // printf output data in printf
    for (int i = 0; i < output_dim; i++)
    {
        printf("%f ", output_data[i]);
    }
    printf("\n");
    
}