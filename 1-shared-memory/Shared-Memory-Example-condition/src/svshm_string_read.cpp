/*
 * @Author: Shawn-Tao 1054087304@qq.com
 * @Date: 2024-03-08 10:44:24
 * @LastEditors: Shawn-Tao 1054087304@qq.com
 * @LastEditTime: 2024-03-08 17:24:16
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

    key_t shm_key = ftok("/shm", 3);
    key_t sem_key = ftok("/sem", 3);

    shmid = shmget(shm_key, MEM_SIZE, IPC_CREAT | IPC_EXCL );
    if (shmid == -1)
        errExit("shmget");

    semid = semget(sem_key, 1, IPC_CREAT | IPC_EXCL );
    if (semid == -1)
        errExit("semget");

    /* Attach shared memory into our address space */

    addr = (char*)shmat(shmid, NULL, SHM_RDONLY);
    if (addr == (void *)-1)
        errExit("shmat");

    /* Initialize semaphore 0 in set with value 1 */

    arg.val = 1;
    if (semctl(semid, 0, SETVAL, arg) == -1)
        errExit("semctl");

    printf("shmid = %d; semid = %d\n", shmid, semid);

    Imu_Data *imu_data = (Imu_Data *)malloc(sizeof(Imu_Data));

    while(true){
        /* Wait for semaphore value to become 0 */
        sop.sem_num = 0;
        sop.sem_op = 0;
        sop.sem_flg = 0;
        if (semop(semid, &sop, 1) == -1)
            errExit("semop");
        
        memcpy(imu_data, addr, sizeof(Imu_Data));
        printf("%lf, %lf, %lf, %lf, %lf, %lf\n", imu_data->x, imu_data->y, imu_data->z, imu_data->roll, imu_data->pitch, imu_data->yaw);

        /* Increment semaphore to 1 */

        sop.sem_num = 0;
        sop.sem_op = 1;
        sop.sem_flg = 0;
        if (semop(semid, &sop, 1) == -1)
            errExit("semop");
    }

    free(imu_data);

    /* Remove shared memory and semaphore set */
    if (shmctl(shmid, IPC_RMID, NULL) == -1)
        errExit("shmctl");
    if (semctl(semid, 0, IPC_RMID, dummy) == -1)
        errExit("semctl");

    exit(EXIT_SUCCESS);
}