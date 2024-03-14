/*
 * @Author: Shawn-Tao 1054087304@qq.com
 * @Date: 2024-03-08 10:44:24
 * @LastEditors: Shawn-Tao 1054087304@qq.com
 * @LastEditTime: 2024-03-08 17:27:09
 * @FilePath: /1-shared-memory/Shared-Memory-Example/svshm_string_read.c
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */

/* svshm_string_read.c

Licensed under GNU General Public License v2 or later.

*/

#include <thread>
#include <chrono>
#include <iostream>
#include <unistd.h>
#include <signal.h>

#include "svshm_string.h"

// veriables for shared memory and semaphore
int semid, shmid;
union semun arg, dummy;
struct sembuf sop;

// release the shared memory and semaphore set when the program is terminated
void cleanup(int signo)
{
    printf("Cleaning up\n");
    if (shmctl(shmid, IPC_RMID, NULL) == -1)
        errExit("shmctl");
    if (semctl(semid, 0, IPC_RMID, dummy) == -1)
        errExit("semctl");
    exit(EXIT_SUCCESS);
}

int main(int argc, char *argv[])
{

    char *addr;
    signal(SIGINT, cleanup);

    /* Create shared memory and semaphore set containing one semaphore */

    key_t shm_key = ftok("/shm_test", 10);
    key_t sem_key = ftok("/sem_test", 10);

    shmid = shmget(shm_key, MEM_SIZE, IPC_CREAT | 0600);
    if (shmid == -1){
        cleanup(0);
        errExit("shmget");
    }

    semid = semget(sem_key, 1, IPC_CREAT | 0600);
    if (semid == -1){
        cleanup(0);
        errExit("semget");
    }

    /* Attach shared memory into our address space */

    addr = (char*)shmat(shmid, NULL, SHM_RDONLY);
    if (addr == (void *)-1){
        cleanup(0);
        errExit("shmat");
    }
    /* Initialize semaphore 0 in set with value 1 */

    arg.val = 1;
    if (semctl(semid, 0, SETVAL, arg) == -1){
        cleanup(0);
        errExit("semctl");
    }

    printf("shmid = %d; semid = %d\n", shmid, semid);

    Imu_Data *imu_data = (Imu_Data *)malloc(sizeof(Imu_Data));

    while(true){
        SemLock(&semid);

        memcpy(imu_data, addr, sizeof(Imu_Data));
        
        SemUnLock(&semid);

        printf("%lf, %lf, %lf, %lf, %lf, %lf\n", imu_data->x, imu_data->y, imu_data->z, imu_data->roll, imu_data->pitch, imu_data->yaw);
    }

    free(imu_data);

    /* Remove shared memory and semaphore set */
    if (shmctl(shmid, IPC_RMID, NULL) == -1)
        errExit("shmctl");
    if (semctl(semid, 0, IPC_RMID, dummy) == -1)
        errExit("semctl");

    exit(EXIT_SUCCESS);
}