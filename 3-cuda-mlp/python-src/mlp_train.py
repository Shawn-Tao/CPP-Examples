'''
Author: Shawn-Tao 1054087304@qq.com
Date: 2024-03-14 09:12:23
LastEditors: Shawn-Tao 1054087304@qq.com
LastEditTime: 2024-03-14 14:23:00
FilePath: \3-cuda-mlp\python-src\mlp_train.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
from torch import nn
import numpy as np
import torch.nn.init as init

class MLP_Network(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, activation=torch.nn.ELU(), output_activation=None):
        super().__init__()
        layers = []
        in_dim = input_dim
        iter = 0
        for hd in hidden_dims:
            layers.append(torch.nn.Linear(in_dim, hd))
            layers.append(activation)
            in_dim = hd
            
            # init.constant_(layers[iter].weight, 0)
            # init.constant_(layers[iter].bias, 0)
            # iter += 2
        layers.append(torch.nn.Linear(in_dim, output_dim))
        # init.constant_(layers[iter].weight, 0) 
        # init.constant_(layers[iter].bias, 0)  
        if output_activation is not None:
            layers.append(output_activation)
            
        self.model = torch.nn.Sequential(*layers)
        print(f"MLP: {self.model}")
        
        

    def forward(self, x):
        return self.model(x)
    
if __name__ == "__main__":

    # Example usage
    input_dim = 70
    output_dim = 12
    hidden_dims = [512, 256, 128]
    mlp = MLP_Network(input_dim, output_dim, hidden_dims)


    # training example

    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001)

    loss_list = []

    # init a file to save data and label
    f_data = open("../dataset/data.txt", "w")
    f_label = open("../dataset/label.txt", "w")

    for i in range(2000):
        key = np.random.randint(0, 1000)
        # x = torch.randn(10, input_dim)
        # y = torch.zeros(10, output_dim)
        x = torch.zeros(10, input_dim)
        y = torch.zeros(10, output_dim)
        if key % 2 != 0:
            # x = x add 2
            x = x + 2
            y = torch.ones(10, output_dim)
        else:
            # x = x minus 2
            x = x - 2
            y = torch.zeros(10, output_dim)  
        # save x and y into a file
        for i in range(10) :
            # save data without []
            f_data.write(str(x[i].tolist())[1:-1] + "\n")
            f_label.write(str(y[i].tolist())[1:-1] + "\n")
        y_pred = mlp(x)
        l = loss(y_pred, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        loss_list.append(l.item())
        print(f"Loss: {l.item()}")

    # save the weights of the model
    torch.save(mlp.state_dict(), "../weights/model.pth")

    # # save the weights of the model in numpy format
    # model = mlp.state_dict()
    # for key in model:
    #     np.save(key, model[key].numpy())


    # save the weights of the model as txt
    model = mlp.state_dict()
    for key in model:
        path = "../weights/"+ key
        np.savetxt(path, model[key].numpy())

    # painting example

    import matplotlib.pyplot as plt
    plt.plot(loss_list)
    plt.show()

    # Example usage
