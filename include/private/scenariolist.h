/**
 * scenariolist.h
 * Private header file for scenario list
 */

#ifndef SCENARIOLIST_H
#define SCENARIOLIST_H

#include "common.h"
#include "constants.h"
#include <ps.h>

#define MAX_FORECASTS_PER_SCENARIO 5
#define MAX_SCENARIOS 10
#define MAX_MODS_PER_ELEMENT 10
#define MAX_ELEMENTS_PER_SCENARIO 100
#define MAXID 16

struct _p_Forecast {
  PetscInt num;      /* Scenario number */
  ForecastType type; /* Forecast type */
  PetscInt nele;     /* Number of devices/elements involved in this forecast */
  PetscInt *buses;   /* Bus numbers */
  char **id;         /* Device ids */
  PetscScalar *val;  /* forecast values */
};

typedef struct _p_Forecast Forecast;

struct _p_Modification {
  ParamType type; /* Parameter type */
  PetscReal val; /* Prameter value */
};

typedef struct _p_Modification Modification;

struct _p_ModElement {
  PetscInt nmods;  
  ElementType type; /*type of element being modified */
  Modification modlist[MAX_MODS_PER_ELEMENT]; /* List of modifications for
                                                        this scenario */
  PetscInt fr_bus, to_bus;
  char id[MAXID];
};

typedef struct _p_ModElement ModElement;

struct _p_Scenario {
  PetscInt nforecast; /* Each scenario can have one or more forecasts */
  Forecast forecastlist[MAX_FORECASTS_PER_SCENARIO]; /* List of forecasts for
                                                        this scenario */
  PetscScalar prob; /* Probability of the scenario */
};

typedef struct _p_Scenario Scenario;

struct _p_ScenarioList {
  PetscInt Nscen; /* Number of scenarios = number of scenarios */
  Scenario *scen; /* Scenarios */
};

typedef struct _p_ScenarioList ScenarioList;

struct _p_ScenarioV2 {
  PetscInt nelements; /* Each scenario can have one or more elements to be modified */
  ModElement elementlist[MAX_MODS_PER_ELEMENT]; /* List of modifications for
                                                        this scenario */
  PetscScalar prob; /* Probability of the scenario */
};

typedef struct _p_ScenarioV2 ScenarioV2;

struct _p_ScenarioListV2 {
  PetscInt Nscen; /* Number of scenarios = number of scenarios */
  ScenarioV2 *scen; /* Scenarios */
};

typedef struct _p_ScenarioListV2 ScenarioListV2;

#endif
