/*
 * @Author: Shawn-Tao 1054087304@qq.com
 * @Date: 2024-03-08 10:43:22
 * @LastEditors: Shawn-Tao 1054087304@qq.com
 * @LastEditTime: 2024-03-08 17:24:54
 * @FilePath: /1-shared-memory/Shared-Memory-Example/svshm_string.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/sem.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define errExit(msg)        \
    do                      \
    {                       \
        perror(msg);        \
        exit(EXIT_FAILURE); \
    } while (0)

union semun
{ /* Used in calls to semctl() */
    int val;
    struct semid_ds *buf;
    unsigned short *array;
#if defined(__linux__)
    struct seminfo *__buf;
#endif
};

#define MEM_SIZE 4096

struct Imu_Data
{
    double x;
    double y;
    double z;
    double roll;
    double pitch;
    double yaw;
};

void SemLock(int* semid)
{
    struct sembuf sop;
    sop.sem_num = 0;
    sop.sem_op = -1;
    sop.sem_flg = SEM_UNDO;
    if (semop(*semid, &sop, 1) == -1)
        errExit("semop");
}

void SemUnLock(int *semid)
{
    struct sembuf sop;
    sop.sem_num = 0;
    sop.sem_op = 1;
    sop.sem_flg = SEM_UNDO;
    if (semop(*semid, &sop, 1) == -1)
        errExit("semop");
}
