/*
 * @Author: Shawn-Tao 1054087304@qq.com
 * @Date: 2024-03-10 21:29:43
 * @LastEditors: Shawn-Tao 1054087304@qq.com
 * @LastEditTime: 2024-03-11 14:17:08
 * @FilePath: /2-concurrency/base_task.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#ifndef __BASE_TASK__
#define __BASE_TASK__

#include <thread>
#include <mutex>
#include <functional>
#include <future>

struct ID
{
    /* data */
    int value;
    double longitude;
    double latitude;
};

class Base_Task
{
private:
    /* data */
    std::mutex *lock_;
    std::thread *demon_thread;

public:
    Base_Task(/* args */);
    ~Base_Task();
    void Run(std::function<void ()> task);
    void Run();
    void Stop();
    virtual void task();
};

class Son :public Base_Task
{
public:
    Son(std::promise<ID> pro);
    ~Son() = default;
    void task() override;
    void log();
private:
    int status_count_;
    std::promise<ID> pro_;
    
};

class Daughter :public Base_Task
{
public:
    Daughter(std::future<ID> fut);
    ~Daughter() = default;
    void task() override;
    void log();

private:
    std::future<ID> fut_;
};

#endif