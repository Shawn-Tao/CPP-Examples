/*
 * @Author: Shawn-Tao 1054087304@qq.com
 * @Date: 2024-03-12 22:16:12
 * @LastEditors: Shawn-Tao 1054087304@qq.com
 * @LastEditTime: 2024-03-14 14:27:43
 * @FilePath: /3-cuda-mlp/src/main.cc
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include <iostream>
#include "mlp.h" 
#include <fstream>
#include <string>

int main(){
    printCudaVersion();
   
    int input_dim = 70;
    int output_dim = 12;
    std::vector<int> hidden_dim = {512,256,128};

    std::vector<std::vector<float>> weight;
    std::vector<std::vector<float>> bias;

    weight.resize(hidden_dim.size() + 1);
    bias.resize(hidden_dim.size() + 1);

    for (int i = 0; i < hidden_dim.size() + 1; i++)
    {
        if (i == 0)
        {
            weight[i].resize(input_dim * hidden_dim[i]);
            bias[i].resize(hidden_dim[i]);
        }
        else if (i == hidden_dim.size())
        {
            weight[i].resize(hidden_dim[i - 1] * output_dim);
            bias[i].resize(output_dim);
        }
        else
        {
            weight[i].resize(hidden_dim[i - 1] * hidden_dim[i]);
            bias[i].resize(hidden_dim[i]);
        }
    }

    // from weights dir import weight, bias, file_name format as model.0.weight, model.0.bias model.1.weight, model.1.bias ... file fromat as txt
    // weight data fromat as: 5.775797739624977112e-02 9.274608641862869263e-02 9.097371995449066162e-02 -3.896432369947433472e-02 8.233007788658142090e-02 4.302547872066497803e-02 -5.699666216969490051e-02 1.030777990818023682e-01 -8.263161033391952515e-02 -3.941248729825019836e-02 7.196903228759765625e-02 -6.248612329363822937e-02 6.537564098834991455e-02 7.431036978960037231e-02 -8.189787715673446655e-02 3.451202437281608582e-02 1.168887317180633545e-01 -2.860546670854091644e-02 -3.589286282658576965e-02 1.100692525506019592e-01 -1.034498214721679688e-01 7.543692737817764282e-02 -6.475850939750671387e-02 4.281159024685621262e-03 -1.108226738870143890e-02 -1.018114909529685974e-01 -7.698357850313186646e-02 -4.539959132671356201e-02 5.966753140091896057e-02 6.405939906835556030e-02 -1.822711713612079620e-02 -1.113728955388069153e-01 -1.061483919620513916e-01 2.056322759017348289e-03 3.521056100726127625e-02 -3.982116468250751495e-03 -8.193900436162948608e-02 -8.821850270032882690e-02 -8.040712215006351471e-03 -3.663736209273338318e-02 9.805580973625183105e-02 4.960985481739044189e-02 -8.484749495983123779e-02 9.904786199331283569e-02 2.522131986916065216e-02 1.154022142291069031e-01 5.314904451370239258e-02 -1.036855950951576233e-01 -4.955576732754707336e-02 3.657537046819925308e-03 -4.346548020839691162e-02 7.816004008054733276e-02 -7.440515607595443726e-02 -5.466493964195251465e-02 -6.397718191146850586e-02 -7.143815606832504272e-02 -6.956502050161361694e-02 -3.572038933634757996e-02 1.183103397488594055e-01 -2.368948236107826233e-02 1.314053405076265335e-02 1.713620312511920929e-02 -6.055627949535846710e-03 9.986586123704910278e-02 -4.747605696320533752e-02 7.000513374805450439e-02 -3.205957263708114624e-02 5.325207486748695374e-02 -1.074269935488700867e-01 3.986042365431785583e-02
    // bias data fromat as: -1.255860775709152222e-01
    for (int i = 0; i < hidden_dim.size() + 1; i++)
    {
        std::string weight_file = "../weights/model." + std::to_string(i * 2) + ".weight";
        std::string bias_file = "../weights/model." + std::to_string(i * 2) + ".bias";
        std::ifstream weight_in(weight_file);
        std::ifstream bias_in(bias_file);
        if (!weight_in.is_open() || !bias_in.is_open())
        {
            std::cout << "file open failed" << std::endl;
            return -1;
        }
        for (int j = 0; j < weight[i].size(); j++)
        {   
            weight_in >> weight[i][j];
        }
        for (int j = 0; j < bias[i].size(); j++)
        {
            bias_in >> bias[i][j];
        }
    }

    // // print weight and bias
    // for (int i = 0; i < hidden_dim.size() + 1; i++)
    // {
    //     for (int j = 0; j < weight[i].size(); j++)
    //     {
    //         std::cout << weight[i][j] << " ";
    //     }
    //     std::cout << std::endl;
    //     for (int j = 0; j < bias[i].size(); j++)
    //     {
    //         std::cout << bias[i][j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    

    MLP_Network *mlp_network = new MLP_Network(input_dim, output_dim, hidden_dim, weight, bias);

    float *input_data, *output_data;

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

    // input_data[0] = 2;
    // input_data[1] = 2;

    mlp_network->forward(input_data, output_data);

    // printf output data in printf
    for (int i = 0; i < output_dim; i++)
    {
        printf("%f ", output_data[i]);
    }
    printf("\n");
    
}