#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "constants.h"

PetscBool next_line(FILE *fp, char line[MAXLINE]){
  
  char tmpline[MAXLINE];

  while (fgets(tmpline, MAXLINE, fp) != NULL){
    if ( tmpline[0] == '#' ){
      continue; /* Skip comments */
    } else if (strcmp(tmpline, "\r\n") == 0 || strcmp(tmpline, "\n") == 0) {
      continue; /* Skip blank lines */
    } else {
      strcpy(line,tmpline);
      return true;
    }
  }
  
  strcpy(line,"");
  return false;
}

