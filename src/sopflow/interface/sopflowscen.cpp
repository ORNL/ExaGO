#include "common.h"
#include "petscsys.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <petscsystypes.h>
#include <private/opflowimpl.h>
#include <private/scenariolist.h>
#include <private/scopflowimpl.h>
#include <private/sopflowimpl.h>
#include <private/tcopflowimpl.h>

extern void clean2Char(char *);
extern char **blankTokenizer(const char *str, int *numtok, int maxtokens,
                             int maxchar);
extern char *next_line(FILE *, char *);

/*
  SOPFLOWSetScenarioData - Sets the scenario data

  Input Parameter
+  sopflow - The SOPFLOW object
.  scenfileformat - the scenario file format
.  scenunctype    - type of uncertainty
-  scenfile - The name of the scenario list file

*/
PetscErrorCode SOPFLOWSetScenarioData(SOPFLOW sopflow,
                                      ScenarioFileInputFormat scenfileformat,
                                      ScenarioUncertaintyType scenunctype,
                                      const char scenfile[]) {
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = PetscStrcpy(sopflow->scenfile, scenfile);
  CHKERRQ(ierr);

  sopflow->scenfileformat = scenfileformat;
  sopflow->scenfileset = PETSC_TRUE;
  sopflow->scenunctype = scenunctype;
  PetscFunctionReturn(0);
}

/*
  SOPFLOWGetScenarioFileVersion - Checks the version of the scenario data file

  Input Parameter
+  sopflow - The SOPFLOW object
.  scenfileformat - the scenario file format
.  scenunctype    - type of uncertainty
-  scenfile - The name of the scenario list file

*/
PetscErrorCode
SOPFLOWGetScenarioFileVersion(ScenarioFileInputFormat* scenfileformat,
                                             const char scenfile[]) {
  char line[MAXLINE];
  char sep[] = ",";
  FILE *fp;
  char *tok;

  PetscFunctionBegin;

  fp = fopen(scenfile, "r");
  if (fp == NULL) {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN,
            "Cannot open scenario file %s", scenfile);
  }

  fgets(line, MAXLINE, fp);
  tok = strtok(line, sep);
  if ( !strcmp(tok, "version")) {
    *scenfileformat = SOPFLOW_NATIVE_SINGLEPERIOD_V2;
  } else {
    *scenfileformat = SOPFLOW_NATIVE_SINGLEPERIOD;
  }

  fclose(fp);

  PetscFunctionReturn(0);
}



/*
  SOPFLOWReadScenarioData_Wind_SinglePeriod - Reads the wind data and populates
the scenario list Input Parameters
+ sopflow - SOPFLOW object
. windgenprofile - wind generator profile file

  Note: This function reads the wind scenario data for "single period" format
files.
*/
PetscErrorCode
SOPFLOWReadScenarioData_Wind_SinglePeriod(SOPFLOW sopflow,
                                          const char windgenprofile[]) {
  PetscErrorCode ierr;
  FILE *fp;
  char line[MAXLINE];
  char *out;
  PetscInt ngenwind, nw = 0;
  char *tok, *tok2;
  char sep[] = ",", sep2[] = "_";
  PetscReal pg, weight;
  int scen_num = 0;
  int genid;
  int windgenbus[1000];
  char windgenid[1000][3];
  int i;
  ScenarioList *scenlist = &sopflow->scenlist;
  Scenario *scenario;
  Forecast *forecast;

  PetscFunctionBegin;

  ngenwind = 1000; // This should be increased for larger cases (?)
  fp = fopen(windgenprofile, "r");
  if (fp == NULL) {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN,
            "Cannot open wind generation profile file %s", windgenprofile);
  }

  /* First line -- has the bus numbers */
  out = fgets(line, MAXLINE, fp);
  /* Parse wind generator numbers */
  tok = strtok(line, sep);
  tok = strtok(NULL, sep); /* Skip first token */
  while (tok != NULL) {
    if (strcmp(tok, "weight") == 0 || strcmp(tok, "weight\n") == 0 ||
        strcmp(tok, "weight\r\n") == 0) {
      tok = strtok(NULL, sep);
      continue;
    }
    /* Parse generator info */
    tok2 = strsep(&tok, sep2);
    sscanf(tok2, "%d", &windgenbus[nw]);
    tok2 = strsep(&tok, sep2);
    tok2 = strsep(&tok, sep2); /* Skip string "Wind" */
    sscanf(tok2, "%d", &genid);
    if (nw == ngenwind)
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP,
              "Exceeded max. number of wind generators=%d\n", ngenwind);
    snprintf(windgenid[nw], 3, "%-2d", genid);

    nw++;
    tok = strtok(NULL, sep);
  }

  while ((out = fgets(line, MAXLINE, fp)) != NULL) {
    if (strcmp(line, "\r\n") == 0 || strcmp(line, "\n") == 0) {
      continue; /* Skip blank lines */
    }

    tok = strtok(line, sep);
    sscanf(tok, "%d", &scen_num); /* Scenario number */
    scen_num -= 1; /* Scenario numbers start with 1 in the file, convert to
                      zero-based start */

    if (scen_num < scenlist->Nscen || scen_num == sopflow->Ns) {
      fclose(fp);
      PetscFunctionReturn(0);
    }

    scenario = &scenlist->scen[scen_num];
    forecast = &scenario->forecastlist[scenario->nforecast];
    forecast->num = scen_num;
    forecast->type = FORECAST_WIND;
    forecast->nele = nw;
    ierr = PetscCalloc1(forecast->nele, &forecast->buses);
    CHKERRQ(ierr);
    ierr = PetscCalloc1(forecast->nele, &forecast->id);
    CHKERRQ(ierr);
    for (i = 0; i < forecast->nele; i++) {
      ierr = PetscCalloc1(3, &forecast->id[i]);
    }
    ierr = PetscCalloc1(forecast->nele, &forecast->val);
    CHKERRQ(ierr);

    tok = strtok(NULL, sep);
    for (i = 0; i < nw; i++) {
      forecast->buses[i] = windgenbus[i];
      ierr = PetscStrcpy(forecast->id[i], windgenid[i]);
      CHKERRQ(ierr);
      sscanf(tok, "%lf", &pg);
      forecast->val[i] = pg;
      tok = strtok(NULL, sep);
    }

    /* Read scenario weight */
    sscanf(tok, "%lf", &weight);
    scenario->prob = weight;

    scenario->nforecast++;
    scenlist->Nscen++;
  }
  PetscFunctionReturn(0);
}

/*
  SOPFLOWReadScenarioData_Natvie_SinglePeriodV2 - Reads the wind data and populates
the scenario list Input Parameters
+ sopflow - SOPFLOW object
. scenariofile - wind generator profile file

  Note: This function reads the scenario data for generic "single period" format
files.
*/
PetscErrorCode
SOPFLOWReadScenarioData_Native_SinglePeriodV2(SOPFLOW sopflow,
                                             const char scenariofile[]) {
  PetscErrorCode ierr;
  FILE *fp;
  char line[MAXLINE];
  char *tok, *tok2;
  ScenarioListV2 *scenlist = &sopflow->scenlist_v2;
  ScenarioV2 *scenario;
  ModElement *element;
  Modification *mod;

  PetscFunctionBegin;

  fp = fopen(scenariofile, "r");
  if (fp == NULL) {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN,
            "Cannot open scenario file %s", scenariofile);
  }

  /*Skip version info here...this is handled in a different function*/
  fgets(line, MAXLINE, fp);
  
  while(next_line(fp, line)){

    tok = strtok(line, ",");
      
    PetscInt scen_num = atoi(tok);
    scen_num -= 1; /*convert to 0 index */
    if (scen_num < scenlist->Nscen || scen_num == sopflow->Ns) {
      fclose(fp);
      PetscFunctionReturn(0);
    }
    scenario = &scenlist->scen[scen_num];

    tok = strtok(NULL,",");
    PetscReal scen_weight = atof(tok);
    scenario->prob = scen_weight;
    
    while(next_line(fp, line)){
      tok = strtok(line,",");
      while( tok && strcmp(tok,"END\n")){
        element = &scenario->elementlist[scenario->nelements];
        if(!strcmp(tok,"GEN") || !strcmp(tok,"LOAD")){
          if(!strcmp(tok,"GEN")){
            element->type = ELEMENT_GEN;
          } else if (!strcmp(tok,"LOAD")){
            element->type = ELEMENT_LOAD;
          }
          tok = strtok(NULL,",");
          element->fr_bus = atoi(tok);
          element->to_bus = -1;
        } else if(!strcmp(tok,"LINE") || !strcmp(tok,"XFRMR")){
          if(!strcmp(tok,"LINE")){
            element->type = ELEMENT_LINE;
          } else if(!strcmp(tok,"XFRMR")){
            element->type = ELEMENT_XFRMR;
          }
          tok = strtok(NULL,",");
          element->fr_bus = atoi(tok);
          tok = strtok(NULL,",");
          element->to_bus = atoi(tok);
        } else {
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_READ,
              "Unsupported element type in scenario file: %s", tok);
          fclose(fp);
          PetscFunctionReturn(PETSC_ERR_FILE_READ);
        }
        tok = strtok(NULL,",");
        strcpy(element->id, tok);
        element->nmods = 0;
        tok = strtok(NULL,",");
        while(tok){
          tok2 = strtok(NULL,",");
          mod = &element->modlist[element->nmods];
          if (!strcmp(tok,"P")){
            mod->type = PARAM_P;
          } else if (!strcmp(tok,"Q")){
            mod->type = PARAM_Q;
          } else if (!strcmp(tok,"R")){
            mod->type = PARAM_R;
          } else if (!strcmp(tok,"X")){
            mod->type = PARAM_X;
          } else if (!strcmp(tok,"B")){
            mod->type = PARAM_B;
          }
          mod->val = atof(tok2);
          element->nmods++;
          tok = strtok(NULL,",");
        }
      scenario->nelements++;
      }
    }
    scenlist->Nscen++;
  }
  fclose(fp);
  PetscFunctionReturn(PETSC_SUCCESS);
}


/*
  SOPFLOWReadScenarioData_Wind_MultiPeriod - Reads the wind data and populates
the scenario list Input Parameters
+ sopflow - SOPFLOW object
. windgenprofile - wind generator profile file

  Note: This function reads the wind scenario data for "multi-period" format
files.
*/
PetscErrorCode
SOPFLOWReadScenarioData_Wind_MultiPeriod(SOPFLOW sopflow,
                                         const char windgenprofile[]) {
  PetscErrorCode ierr;
  FILE *fp;
  char line[MAXLINE];
  char *out;
  PetscInt ngenwind, nw = 0;
  char *tok, *tok2;
  char sep[] = ",", sep2[] = "_";
  PetscReal pg;
  int scen_num = 0;
  int genid;
  int windgenbus[1000];
  char windgenid[1000][3];
  int i;
  ScenarioList *scenlist = &sopflow->scenlist;
  Scenario *scenario;
  Forecast *forecast;

  PetscFunctionBegin;

  ngenwind = 1000; // This should be increased for larger cases (?)
  fp = fopen(windgenprofile, "r");
  if (fp == NULL) {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN,
            "Cannot open wind generation profile file %s", windgenprofile);
    CHKERRQ(ierr);
  }

  /* First line -- has the bus numbers */
  out = fgets(line, MAXLINE, fp);
  /* Parse wind generator numbers */
  tok = strtok(line, sep);
  tok = strtok(NULL, sep); /* Skip first token */
  tok = strtok(NULL, sep); /* Skip second token */
  while (tok != NULL) {
    /* Parse generator info */
    tok2 = strsep(&tok, sep2);
    sscanf(tok2, "%d", &windgenbus[nw]);
    tok2 = strsep(&tok, sep2);
    tok2 = strsep(&tok, sep2); /* Skip string "Wind" */
    sscanf(tok2, "%d", &genid);
    if (nw == ngenwind)
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP,
              "Exceeded max. number of wind generators=%d\n", ngenwind);
    snprintf(windgenid[nw], 3, "%-2d", genid);

    nw++;
    tok = strtok(NULL, sep);
  }

  while ((out = fgets(line, MAXLINE, fp)) != NULL) {
    if (strcmp(line, "\r\n") == 0 || strcmp(line, "\n") == 0) {
      continue; /* Skip blank lines */
    }

    tok = strtok(line, sep);
    tok = strtok(NULL, sep);      /* Skip first token */
    sscanf(tok, "%d", &scen_num); /* Scenario number */
    scen_num -= 1; /* Scenario numbers start with 1 in the file, convert to
                      zero-based start */

    if (scen_num < scenlist->Nscen || scen_num == sopflow->Ns) {
      fclose(fp);
      PetscFunctionReturn(0);
    }

    scenario = &scenlist->scen[scen_num];
    forecast = &scenario->forecastlist[scenario->nforecast];
    forecast->num = scen_num;
    forecast->type = FORECAST_WIND;
    forecast->nele = nw;
    ierr = PetscCalloc1(forecast->nele, &forecast->buses);
    CHKERRQ(ierr);
    ierr = PetscCalloc1(forecast->nele, &forecast->id);
    CHKERRQ(ierr);
    for (i = 0; i < forecast->nele; i++) {
      ierr = PetscCalloc1(3, &forecast->id[i]);
    }
    ierr = PetscCalloc1(forecast->nele, &forecast->val);
    CHKERRQ(ierr);

    for (i = 0; i < nw; i++) {
      forecast->buses[i] = windgenbus[i];
      ierr = PetscStrcpy(forecast->id[i], windgenid[i]);
      CHKERRQ(ierr);
    }

    tok = strtok(NULL, sep);
    nw = 0;
    while (tok != NULL) {
      sscanf(tok, "%lf", &pg);
      forecast->val[nw] = pg;
      nw++;
      tok = strtok(NULL, sep);
    }
    scenario->nforecast++;
    scenlist->Nscen++;
  }
  PetscFunctionReturn(0);
}

/*
  SOPFLOWReadScenarioData - Reads the scenario data file

  Input Parameters
+ sopflow - the sopflow object
. scenfileformat - the scenario file format (NATIVE or PSSE)
- scenfile - the scenario file name

*/
PetscErrorCode SOPFLOWReadScenarioData(SOPFLOW sopflow,
                                       ScenarioFileInputFormat scenfileformat,
                                       const char scenfile[]) {
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (sopflow->scenunctype == WIND) {
    switch(scenfileformat) {
      case SOPFLOW_NATIVE_SINGLEPERIOD:
        ierr = SOPFLOWReadScenarioData_Wind_SinglePeriod(sopflow, scenfile);
        break;
      case SOPFLOW_NATIVE_MULTIPERIOD:
        ierr = SOPFLOWReadScenarioData_Wind_MultiPeriod(sopflow, scenfile);
        break;
      case SOPFLOW_NATIVE_SINGLEPERIOD_V2:
        ierr = SOPFLOWReadScenarioData_Native_SinglePeriodV2(sopflow, scenfile);
        break;
    
    CHKERRQ(ierr);

    }
  }
  PetscFunctionReturn(0);
}

/* SOPFLOWGetNumScenarios_Native_MultiPeriod - Gets the number of scenarios from
 * the scenario file
 */
PetscErrorCode SOPFLOWGetNumScenarios_Native_MultiPeriod(const char scenfile[],
                                                         PetscInt *Ns) {
  PetscErrorCode ierr;
  FILE *fp;
  char line[MAXLINE];
  char *out;
  char *tok;
  char sep[] = ",";
  int scen_num, ns = 0;

  PetscFunctionBegin;

  fp = fopen(scenfile, "r");
  if (fp == NULL) {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN,
            "Cannot open wind generation profile file %s", scenfile);
    CHKERRQ(ierr);
  }

  /* First line -- has the bus numbers */
  out = fgets(line, MAXLINE, fp);

  while ((out = fgets(line, MAXLINE, fp)) != NULL) {
    if (strcmp(line, "\r\n") == 0 || strcmp(line, "\n") == 0) {
      continue; /* Skip blank lines */
    }

    tok = strtok(line, sep);
    tok = strtok(NULL, sep);      /* Skip first token */
    sscanf(tok, "%d", &scen_num); /* Scenario number */
    if (ns < scen_num)
      ns = scen_num;
    else
      break;
  }

  fclose(fp);
  *Ns = ns;
  PetscFunctionReturn(0);
}

/* SOPFLOWGetNumScenarios_Native_SinglePeriod - Gets the number of scenarios
 * from the scenario file
 */
PetscErrorCode SOPFLOWGetNumScenarios_Native_SinglePeriod(const char scenfile[],
                                                          PetscInt *Ns) {
  PetscErrorCode ierr;
  FILE *fp;
  char line[MAXLINE];
  char *out;
  char *tok;
  char sep[] = ",";
  int Nscen = 0;

  PetscFunctionBegin;

  fp = fopen(scenfile, "r");
  if (fp == NULL) {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN,
            "Cannot open wind generation profile file %s", scenfile);
    CHKERRQ(ierr);
  }

  /* First line -- has the bus numbers */
  out = fgets(line, MAXLINE, fp);

  while ((out = fgets(line, MAXLINE, fp)) != NULL) {
    if (strcmp(line, "\r\n") == 0 || strcmp(line, "\n") == 0) {
      continue; /* Skip blank lines */
    }

    tok = strtok(line, sep);
    sscanf(tok, "%d", &Nscen); /* Scenario number */
  }

  fclose(fp);
  *Ns = Nscen;
  PetscFunctionReturn(0);
}

/* SOPFLOWGetNumScenarios_Native_SinglePeriodV2 - Gets the number of scenarios
 * from the scenario file
 */
PetscErrorCode
SOPFLOWGetNumScenarios_Native_SinglePeriodV2(const char scenfile[],
                                             PetscInt *Ns) {
  PetscErrorCode ierr;
  FILE *fp;
  char line[MAXLINE];
  char *tok;
  char sep[] = ",";
  int Nscen = 0;
  int val, cnt;
  
  PetscFunctionBegin;

  fp = fopen(scenfile, "r");
  if (fp == NULL) {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN,
            "Cannot open wind generation profile file %s", scenfile);
    CHKERRQ(ierr);
  }

  /* Scenarios are only entry that begins with a number */
  while(fgets(line, MAXLINE, fp)){
    tok = strtok(line,sep);
    cnt = 0;
    sscanf(tok,"%d%n",&val, &cnt);
    if ( cnt == 1 ) Nscen++;
  }

  fclose(fp);
  *Ns = Nscen;
  PetscFunctionReturn(0);

}
/*
  SOPFLOWReadScenarioData - Gets the number of scenarios from the scenario data
file

  Input Parameters
+ sopflow - the sopflow object
. scenfileformat - the scenario file format (NATIVE or PSSE)
. scenfile - the scenario file name
- Ns - number of scenarios given in the scenario file

*/
PetscErrorCode SOPFLOWGetNumScenarios(SOPFLOW sopflow,
                                      ScenarioFileInputFormat scenfileformat,
                                      const char scenfile[], PetscInt *Ns) {
  (void)sopflow;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch(scenfileformat){
    case SOPFLOW_NATIVE_SINGLEPERIOD:
      ierr = SOPFLOWGetNumScenarios_Native_SinglePeriod(scenfile, Ns);
      break;
    case SOPFLOW_NATIVE_SINGLEPERIOD_V2:
      ierr = SOPFLOWGetNumScenarios_Native_SinglePeriodV2(scenfile, Ns);
      break;
    case SOPFLOW_NATIVE_MULTIPERIOD:
      ierr = SOPFLOWGetNumScenarios_Native_MultiPeriod(scenfile, Ns);
      break;
    default:
      ierr = PETSC_SUCCESS;
      *Ns = 1;
      break;
  }
  
  CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
