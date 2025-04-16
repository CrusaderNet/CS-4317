#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

int value = 0;  // Shared variable

void *runner(void *param); // Thread function

int main(int argc, char *argv[]) {
    pid_t pid;              // Process ID
    pthread_t tid;          // Thread ID
    pthread_attr_t attr;    // Thread attributes

    pid = fork();  // Create a child process

    if (pid == 0) {  // Child process
        pthread_attr_init(&attr);
        pthread_create(&tid, &attr, runner, NULL);
        pthread_join(tid, NULL);
        printf("CHILD: value = %d\n", value);
    } else if (pid > 0) {  // Parent process
        wait(NULL);  // Wait for child to finish
        printf("PARENT: value = %d\n", value);
    } else {
        fprintf(stderr, "Fork failed\n");
        return 1;
    }

    return 0;
}

// Thread function
void *runner(void *param) {
    value = 5;
    pthread_exit(0);
}