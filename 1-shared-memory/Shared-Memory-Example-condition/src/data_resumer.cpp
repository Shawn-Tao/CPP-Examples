/*
 * @Author: Shawn-Tao 1054087304@qq.com
 * @Date: 2024-03-07 23:04:43
 * @LastEditors: Shawn-Tao 1054087304@qq.com
 * @LastEditTime: 2024-03-08 10:39:36
 * @FilePath: /1-shared-memory/Shared-Memory-Example/main.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include <QCoreApplication>
#include <sys/shm.h>
#include <sys/sem.h>
#include <signal.h>

#include <sys/types.h>
#include <sys/ipc.h>

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    key_t shm_key = ftok("/shm-test", 1);
    
    int shm_flag = shmat()



    int shm_flag = shmget(shm_key, 4096, IPC_CREAT | 0666);
    if (shm_flag == -1)
    {
        perror("shmget");
        return -1;
    }
    void *shm_addr = shmat(shm_flag, NULL, 0);
    if (shm_addr == (void *)-1)
    {
        perror("shmat");
        return -1;
    }
    printf("shm_addr: %p\n", shm_addr);
    printf("shm_key: %d\n", shm_key);
    printf("shm_flag: %d\n", shm_flag);

    return a.exec();
}
