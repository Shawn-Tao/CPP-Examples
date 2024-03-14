/*
 * @Author: Shawn-Tao 1054087304@qq.com
 * @Date: 2024-03-08 10:45:23
 * @LastEditors: Shawn-Tao 1054087304@qq.com
 * @LastEditTime: 2024-03-08 14:34:13
 * @FilePath: /1-shared-memory/Shared-Memory-Example/svshm_string_write.c
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */

/* svshm_string_write.c

Licensed under GNU General Public License v2 or later.

*/

#include <thread>
#include <chrono>
#include <iostream>
#include <unistd.h>
#include <random>

#include "svshm_string.h"

std::mt19937 gen;
std::uniform_int_distribution<int> dis(1, 100);

int semid, shmid;
struct sembuf sop;

int main(int argc, char *argv[])
{
    char *shm_addr;
    size_t len;

    key_t shm_key = ftok("/shm-test", 1);
    key_t sem_key = ftok("/sem-test", 1);
    shmid = shmget(shm_key, MEM_SIZE, 0);
    if (shmid == -1)
        errExit("shmget");

    semid = semget(sem_key, 1, 0);
    if (semid == -1)
        errExit("semget");

    // shmid = atoi(argv[1]);
    // semid = atoi(argv[2]);

    /* Attach shared memory into our shm_address space and copy string
       (including trailing null byte) into memory. */

    shm_addr = (char*)shmat(shmid, NULL, 0);
    if (shm_addr == (void *)-1)
        errExit("shmat");

    Imu_Data *imu_data = (Imu_Data*)malloc(sizeof(Imu_Data));


    // random the value of imu data


    while (true)
    {
        imu_data->x = dis(gen);
        imu_data->y = dis(gen);
        imu_data->z = dis(gen);
        imu_data->roll = dis(gen);
        imu_data->pitch = dis(gen);
        imu_data->yaw = dis(gen);

        memcpy(shm_addr, imu_data, sizeof(Imu_Data));

        /* Decrement semaphore to 0 */

        sop.sem_num = 0;
        sop.sem_op = -1;
        sop.sem_flg = 0;

        if (semop(semid, &sop, 1) == -1)
            errExit("semop");

        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    exit(EXIT_SUCCESS);
}
