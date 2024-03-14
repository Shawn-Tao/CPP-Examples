#include "base_task.h"
#include <iostream>
#include <chrono>

Base_Task::Base_Task(/* args */)
{
}

Base_Task::~Base_Task()
{
}

void Base_Task::Run()
{
    demon_thread = new std::thread(std::bind(&Base_Task::task, this));
    demon_thread->detach();
}

void Base_Task::Run(std::function<void ()> task){
    // std::function<void()> task_ = task;
    // task_ = std::bind(&Base_Task::task, this);
    demon_thread = new std::thread(task);
    demon_thread->detach();
}

void Base_Task::Stop(){
    demon_thread->~thread();
}

void Base_Task::task(){
    while (true)
    {
        std::cout << "Task is running" << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

}

Son::Son(std::promise<ID> pro) : pro_(std::move(pro))
{
    status_count_ = 0;
}


void Son::task(){
    
    while (true)
    {
        ID id;
        id.value = status_count_;
        id.latitude = 0.0;
        id.longitude = 0.0;
        pro_.set_value(id);
        status_count_++;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

void Son::log()
{
    printf("Son status count is %d\n", status_count_);
    std::this_thread::sleep_for(std::chrono::seconds(1));
}

Daughter::Daughter(std::future<ID> fut) 
{

}

void Daughter::task()
{
    while (true)
    {
        fut_.wait();
        ID id = fut_.get();

        printf("Daughter is running, ID is %d\n", id.value);

        // std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}
