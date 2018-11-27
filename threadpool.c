#include "threadpool.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#define ARR_SIZE 1000000
static pthread_mutex_t count_mutex = PTHREAD_MUTEX_INITIALIZER;
static int count;
void fast_task(void *ptr)
{
        int *pval = (int*)ptr;
        int i;
        for (i = 0; i < 1000; i++) {
                (*pval)++;
        }
        pthread_mutex_lock(&count_mutex);
        count++;
        pthread_mutex_unlock(&count_mutex);
}
void slow_task(void *ptr)
{
        printf("slow task: count value is %d.\n",count);
        pthread_mutex_lock(&count_mutex);
        count++;
        pthread_mutex_unlock(&count_mutex);
}
int main(int argc, char **argv)
{
        struct threadpool *pool;
        int arr[ARR_SIZE], i, ret, failed_count = 0;
        for (i = 0; i < ARR_SIZE; i++) {
                arr[i] = i;
        }
        /* Create a threadpool of 10 thread workers. */
        if ((pool = threadpool_init(10)) == NULL) {
                printf("Error! Failed to create a thread pool struct.\n");
                exit(EXIT_FAILURE);
        }
        for (i = 0; i < ARR_SIZE; i++) {
                if (i % 10000 == 0) {
                        /* blocking. */
                        ret = threadpool_add_task(pool,slow_task,arr + i,1);
                }
                else {
                        /* non blocking. */
                        ret = threadpool_add_task(pool,fast_task,arr + i,0);
                }
                if (ret == -1) {
                        printf("An error had occurred while adding a task.");
                        exit(EXIT_FAILURE);
                }
                if (ret == -2) {
                        failed_count++;
                }
        }
        /* Stop the pool. */
        threadpool_free(pool,1);
        printf("Example ended.\n");
        printf("%d tasks out of %d have been executed.\n",count,ARR_SIZE);
        printf("%d tasks out of %d did not execute since the pool was overloaded.\n",failed_count,ARR_SIZE);
        printf("All other tasks had not executed yet.");
        return 0;
}
