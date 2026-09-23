#include <private/opflowimpl.h>
#include <private/tcopflowimpl.h>
#include <stdio.h>
#include <stdlib.h>

/* The three readers below took their lines with fgets() into a MAXLINE
   (10000) character buffer. A longer line -- a load profile for
   case_ACTIVSg2000 lists 1125 loads, 11.4 kB per row -- was silently split:
   the period got the first ~980 loads of its row, and the next period was fed
   the remainder of the same row as if it were its own, with its first ~140
   loads set to values belonging to other buses.

   Lines are now read whole, and every row must carry exactly as many values
   as the header lists. */
static PetscErrorCode TCOPFLOWProfileGetLine(FILE *fp, char **line, size_t *cap,
                                             PetscBool *got) {
  ssize_t n;

  PetscFunctionBegin;
  *got = PETSC_FALSE;
  while ((n = getline(line, cap, fp)) >= 0) {
    /* strip the line end, and skip a blank line */
    while (n > 0 && ((*line)[n - 1] == '\n' || (*line)[n - 1] == '\r'))
      (*line)[--n] = '\0';
    if (n == 0)
      continue;
    *got = PETSC_TRUE;
    break;
  }
  PetscFunctionReturn(0);
}

/*
  TCOPFLOWReadPloadProfile - Reads the active power load profile

  Input Parameters:
+ tcopflow - the TCOPFLOW object
- ploadprofile - the file containing the real load power profiles

  Note: TCOPFLOWRReadPloadProfile parses the load profile files created by NREL
*/
PetscErrorCode TCOPFLOWReadPloadProfile(TCOPFLOW tcopflow,
                                        char ploadprofile[]) {
  PetscErrorCode ierr;
  FILE *fp;
  char *line = NULL;
  size_t linecap = 0;
  PetscBool got;
  OPFLOW opflow;
  PS ps;
  PetscInt nload = tcopflow->opflows[0]->ps->nload, *lbus, nl = 0, ncol = 0;
  char *tok;
  char sep[] = ",";
  PSLOAD load;
  PetscInt t = 0;
  PetscReal pl;

  PetscFunctionBegin;

  fp = fopen(ploadprofile, "r");
  if (fp == NULL) {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN,
            "Cannot open P load profile file %s", ploadprofile);
  }

  ierr = PetscMalloc1(nload, &lbus);
  CHKERRQ(ierr);
  /* First line -- has the bus numbers */
  ierr = TCOPFLOWProfileGetLine(fp, &line, &linecap, &got);
  CHKERRQ(ierr);
  if (!got)
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_READ,
            "P load profile file %s is empty", ploadprofile);

  /* Parse load bus numbers */
  tok = strtok(line, sep);
  tok = strtok(NULL, sep); /* Skip first token */
  while (tok != NULL) {
    if (nl >= nload)
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_READ,
              "P load profile file %s lists more than the network's %d "
              "loads",
              ploadprofile, (int)nload);
    sscanf(tok, "%d", &lbus[nl]);
    nl++;
    tok = strtok(NULL, sep);
  }
  ncol = nl;

  while (t < tcopflow->Nt) {
    ierr = TCOPFLOWProfileGetLine(fp, &line, &linecap, &got);
    CHKERRQ(ierr);
    if (!got)
      break;

    opflow = tcopflow->opflows[t];
    ps = opflow->ps;
    /* Parse load values */
    tok = strtok(line, sep);
    tok = strtok(NULL, sep); /* Skip first token */
    nl = 0;
    while (tok != NULL) {
      if (nl >= ncol)
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_READ,
                "P load profile file %s: row %d has more values than the "
                "%d loads of its header",
                ploadprofile, (int)t + 1, (int)ncol);
      ierr = PSGetLoad(ps, lbus[nl], "1 ", &load);
      CHKERRQ(ierr);
      sscanf(tok, "%lf", &pl);
      load->pl = pl / ps->MVAbase;
      nl++;
      tok = strtok(NULL, sep);
    }
    if (nl != ncol)
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_READ,
              "P load profile file %s: row %d has %d values, its header "
              "lists %d loads",
              ploadprofile, (int)t + 1, (int)nl, (int)ncol);

    t++;
  }

  ierr = PetscFree(lbus);
  CHKERRQ(ierr);
  free(line);
  fclose(fp);
  PetscFunctionReturn(0);
}

/*
  TCOPFLOWReadQloadProfile - Reads the reactive power load profile

  Input Parameters:
+ tcopflow - the TCOPFLOW object
- qloadprofile - the file containing the real load power profiles

  Note: TCOPFLOWReadQloadProfile parses the load profile files created by NREL
*/
PetscErrorCode TCOPFLOWReadQloadProfile(TCOPFLOW tcopflow,
                                        char qloadprofile[]) {
  PetscErrorCode ierr;
  FILE *fp;
  char *line = NULL;
  size_t linecap = 0;
  PetscBool got;
  OPFLOW opflow;
  PS ps;
  PetscInt nload = tcopflow->opflows[0]->ps->nload, *lbus, nl = 0, ncol = 0;
  char *tok;
  char sep[] = ",";
  PSLOAD load;
  PetscInt t = 0;
  PetscReal ql;

  PetscFunctionBegin;

  fp = fopen(qloadprofile, "r");
  if (fp == NULL) {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN,
            "Cannot open Q load profile file %s", qloadprofile);
  }

  ierr = PetscMalloc1(nload, &lbus);
  CHKERRQ(ierr);
  /* First line -- has the bus numbers */
  ierr = TCOPFLOWProfileGetLine(fp, &line, &linecap, &got);
  CHKERRQ(ierr);
  if (!got)
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_READ,
            "Q load profile file %s is empty", qloadprofile);

  /* Parse load bus numbers */
  tok = strtok(line, sep);
  tok = strtok(NULL, sep); /* Skip first token */
  while (tok != NULL) {
    if (nl >= nload)
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_READ,
              "Q load profile file %s lists more than the network's %d "
              "loads",
              qloadprofile, (int)nload);
    sscanf(tok, "%d", &lbus[nl]);
    nl++;
    tok = strtok(NULL, sep);
  }
  ncol = nl;

  while (t < tcopflow->Nt) {
    ierr = TCOPFLOWProfileGetLine(fp, &line, &linecap, &got);
    CHKERRQ(ierr);
    if (!got)
      break;

    opflow = tcopflow->opflows[t];
    ps = opflow->ps;
    /* Parse load values */
    tok = strtok(line, sep);
    tok = strtok(NULL, sep); /* Skip first token */
    nl = 0;
    while (tok != NULL) {
      if (nl >= ncol)
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_READ,
                "Q load profile file %s: row %d has more values than the "
                "%d loads of its header",
                qloadprofile, (int)t + 1, (int)ncol);
      ierr = PSGetLoad(ps, lbus[nl], "1 ", &load);
      CHKERRQ(ierr);
      sscanf(tok, "%lf", &ql);
      load->ql = ql / ps->MVAbase;
      nl++;
      tok = strtok(NULL, sep);
    }
    if (nl != ncol)
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_READ,
              "Q load profile file %s: row %d has %d values, its header "
              "lists %d loads",
              qloadprofile, (int)t + 1, (int)nl, (int)ncol);

    t++;
  }

  ierr = PetscFree(lbus);
  CHKERRQ(ierr);
  free(line);
  fclose(fp);
  PetscFunctionReturn(0);
}

/*
  TCOPFLOWReadWindGenProfile - Reads the wind generation

  Input Parameters:
+ tcopflow - the TCOPFLOW object
- windgenprofile - the file containing the wind generation profiles

  Note: TCOPFLOWReadWindGenProfile parses the load profile files created by NREL
*/
PetscErrorCode TCOPFLOWReadWindGenProfile(TCOPFLOW tcopflow,
                                          char windgenprofile[]) {
  PetscErrorCode ierr;
  FILE *fp;
  char *line = NULL;
  size_t linecap = 0;
  PetscBool got;
  OPFLOW opflow;
  PS ps;
  PetscInt ngen = tcopflow->opflows[0]->ps->ngen, *windgenbus, nw = 0, ncol = 0;
  char *tok, *tok2;
  char sep[] = ",", sep2[] = "_";
  PSGEN gen;
  PetscInt t = 0;
  PetscReal pg;
  int scen_num;
  int genid;
  char windgenid[100][3];

  PetscFunctionBegin;

  fp = fopen(windgenprofile, "r");
  if (fp == NULL) {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN,
            "Cannot open wind generation profile file %s", windgenprofile);
  }

  ierr = PetscMalloc1(ngen, &windgenbus);
  CHKERRQ(ierr);

  /* First line -- has the bus numbers */
  ierr = TCOPFLOWProfileGetLine(fp, &line, &linecap, &got);
  CHKERRQ(ierr);
  if (!got)
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_READ,
            "wind generation profile file %s is empty", windgenprofile);

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
    if (nw == 100)
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP,
              "Exceeded max. values set for parsing wind generators\n");
    snprintf(windgenid[nw], 3, "%-2d", genid);

    nw++;
    tok = strtok(NULL, sep);
  }
  ncol = nw;

  while (t < tcopflow->Nt) {
    ierr = TCOPFLOWProfileGetLine(fp, &line, &linecap, &got);
    CHKERRQ(ierr);
    if (!got)
      break;

    opflow = tcopflow->opflows[t];
    ps = opflow->ps;
    /* Parse the row */
    tok = strtok(line, sep);
    tok = strtok(NULL, sep);      /* Skip first token */
    sscanf(tok, "%d", &scen_num); /* Scenario number */
    if (scen_num != 1)
      continue;
    tok = strtok(NULL, sep); /* Scenario number */
    nw = 0;
    while (tok != NULL) {
      if (nw >= ncol)
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_READ,
                "wind generation profile file %s: row %d has more values "
                "than the %d generators of its header",
                windgenprofile, (int)t + 1, (int)ncol);
      ierr = PSGetGen(ps, windgenbus[nw], windgenid[nw], &gen);
      CHKERRQ(ierr);
      sscanf(tok, "%lf", &pg);
      gen->pg = gen->pt =
          pg / ps->MVAbase; /* Set real power generation. Note that Pg upper
                               limit is also set to Pg */
      nw++;
      tok = strtok(NULL, sep);
    }
    if (nw != ncol)
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_READ,
              "wind generation profile file %s: row %d has %d values, its "
              "header lists %d generators",
              windgenprofile, (int)t + 1, (int)nw, (int)ncol);

    t++;
  }

  ierr = PetscFree(windgenbus);
  CHKERRQ(ierr);
  free(line);
  fclose(fp);
  PetscFunctionReturn(0);
}
