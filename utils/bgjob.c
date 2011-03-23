/* Run a command in the background such that it doesn't die even if the
 * terminal is killed. This is what nohup is supposed to do!
 */
#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include <syslog.h>
#include <string.h>

int main(int argc, char **argv)
{
        
  /* Our process ID and Session ID */
  pid_t pid, sid;
  char fname[80];
  char cmdline[4096];
  int i;
        
  /* Fork off the parent process */
  pid = fork();
  if (pid < 0) {
    exit(EXIT_FAILURE);
  }
  /* If we got a good PID, then
     we can exit the parent process. */
  if (pid > 0) {
    exit(EXIT_SUCCESS);
  }

  /* Change the file mode mask */
  umask(0);
                
  /* Create a new SID for the child process */
  sid = setsid();
  if (sid < 0) {
    exit(EXIT_FAILURE);
  }

  /* close stdin and redirect stdout and stderr to file */
  fclose(stdin);
  sprintf(fname, "bgjob-%d.out", sid);
  freopen(fname, "wb", stdout);
  sprintf(fname, "bgjob-%d.err", sid);
  freopen(fname, "wb", stderr);
        
  /* build the command line */
  cmdline[0]='\0';
  for (i=1; i<argc; i++)
  {
    strcat(cmdline, argv[i]);
    strcat(cmdline, " ");
  }

  /* print the command line before executing it */
  printf(cmdline);
  printf("\n");
  system(cmdline);
  
  return 0;
}
