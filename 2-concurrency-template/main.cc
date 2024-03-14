/*
 * @Author: Shawn-Tao 1054087304@qq.com
 * @Date: 2024-03-10 21:30:45
 * @LastEditors: Shawn-Tao 1054087304@qq.com
 * @LastEditTime: 2024-03-11 14:19:43
 * @FilePath: /2-concurrency/main.cc
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include "base_task.h"



int main(){

    std::promise<ID> pro;
    std::future<ID> fut = pro.get_future();
    
    Son son(std::move(pro));
    // Daughter daughter(std::move(fut));

    // task.Run(std::bind(&Base_Task::task, &task));
    son.Run();
    // daughter.Run();

    while (true)
    {
        /* code */
        son.log();
        // fut.wait();
        ID id = fut.get();

        printf("Daughter is running, ID is %d\n", id.value);
    }




    return 0;
}