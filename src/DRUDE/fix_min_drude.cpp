// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "fix_min_drude.h"
#include "fix_drude.h"

#include "atom.h"
#include "comm.h"
#include "compute.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "input.h"
#include "modify.h"
#include "random_mars.h"
#include "update.h"
#include "variable.h"
#include "pointers.h"
#include "min.h"
#include "output.h"
#include "integrate.h"

#include <cstring>
#include <cmath>

using namespace LAMMPS_NS;
using namespace FixConst;

enum{NOBIAS,BIAS};
enum{CONSTANT,EQUAL};

/* ---------------------------------------------------------------------- */

FixMinDrude::FixMinDrude(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
    maxiter = 100;
  //if (narg < 9) error->all(FLERR,"Illegal fix min/drude command");
  
  //comm_reverse = 3;

  // core temperature
  //tstr_core = nullptr;
  //if (utils::strmatch(arg[3],"^v_")) {
  //  tstr_core = utils::strdup(arg[3]+2);
  //  tstyle_core = EQUAL;
  //} else {
  //  t_start_core = utils::numeric(FLERR,arg[3],false,lmp);
  //  t_target_core = t_start_core;
  //  tstyle_core = CONSTANT;
  //}
  //t_period_core = utils::numeric(FLERR,arg[4],false,lmp);
  //int seed_core = utils::inumeric(FLERR,arg[5],false,lmp);

  //// drude temperature
  //tstr_drude = nullptr;
  //if (strstr(arg[7],"v_") == arg[6]) {
  //  tstr_drude = utils::strdup(arg[6]+2);
  //  tstyle_drude = EQUAL;
  //} else {
  //  t_start_drude = utils::numeric(FLERR,arg[6],false,lmp);
  //  t_target_drude = t_start_drude;
  //  tstyle_drude = CONSTANT;
  //}
  //t_period_drude = utils::numeric(FLERR,arg[7],false,lmp);
  //int seed_drude = utils::inumeric(FLERR,arg[8],false,lmp);

  // error checks
  //if (t_period_core <= 0.0)
  //  error->all(FLERR,"Fix langevin/drude period must be > 0.0");
  //if (seed_core  <= 0) error->all(FLERR,"Illegal langevin/drude seed");
  //if (t_period_drude <= 0.0)
  //  error->all(FLERR,"Fix langevin/drude period must be > 0.0");
  //if (seed_drude <= 0) error->all(FLERR,"Illegal langevin/drude seed");

  //random_core  = new RanMars(lmp,seed_core);
  //random_drude = new RanMars(lmp,seed_drude);

  //int iarg = 9;
  //zero = 0;
  //while (iarg < narg) {
  //  if (strcmp(arg[iarg],"zero") == 0) {
  //    if (iarg+2 > narg) error->all(FLERR,"Illegal fix langevin/drude command");
  //    zero = utils::logical(FLERR, arg[iarg + 1], false, lmp);
  //    iarg += 2;
  //  } else error->all(FLERR,"Illegal fix langevin/drude command");
  //}

  //tflag = 0; // no external compute/temp is specified yet (for bias)
  energy = 0.;
  fix_drude = nullptr;
  //temperature = nullptr;
  //id_temp = nullptr;
}

/* ---------------------------------------------------------------------- */

FixMinDrude::~FixMinDrude()
{
  delete random_core;
  delete [] tstr_core;
  delete random_drude;
  delete [] tstr_drude;
}

/* ---------------------------------------------------------------------- */

int FixMinDrude::setmask()
{
  int mask = 0;
  mask |= PRE_FORCE;
  mask |= PRE_FORCE_RESPA;
  mask |= MIN_POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixMinDrude::init()
{
  // check variable-style target core temperature
  //if (tstr_core) {
  //  tvar_core = input->variable->find(tstr_core);
  //  if (tvar_core < 0)
  //    error->all(FLERR,"Variable name for fix langevin/drude does not exist");
  //  if (input->variable->equalstyle(tvar_core)) tstyle_core = EQUAL;
  //  else error->all(FLERR,"Variable for fix langevin/drude is invalid style");
  //}

  // check variable-style target drude temperature
  //if (tstr_drude) {
  //  tvar_drude = input->variable->find(tstr_drude);
  //  if (tvar_drude < 0)
  //    error->all(FLERR,"Variable name for fix langevin/drude does not exist");
  //  if (input->variable->equalstyle(tvar_drude)) tstyle_drude = EQUAL;
  //  else error->all(FLERR,"Variable for fix langevin/drude is invalid style");
  //}

  int ifix;
  for (ifix = 0; ifix < modify->nfix; ifix++)
    if (strcmp(modify->fix[ifix]->style,"drude") == 0) break;
  if (ifix == modify->nfix) error->all(FLERR, "fix min/drude requires fix drude");
  fix_drude = dynamic_cast<FixDrude *>(modify->fix[ifix]);
}

/* ---------------------------------------------------------------------- */

void FixMinDrude::setup(int /*vflag*/)
{
  if (!utils::strmatch(update->integrate_style,"^verlet"))
    error->all(FLERR,"RESPA style not compatible with fix langevin/drude");
  //if (!comm->ghost_velocity)
  //  error->all(FLERR,"fix langevin/drude requires ghost velocities. Use comm_modify vel yes");

  //if (zero) {
  //    int *mask = atom->mask;
  //    int nlocal = atom->nlocal;
  //    int *drudetype = fix_drude->drudetype;
  //    int *type = atom->type;
  //    bigint ncore_loc = 0;
  //    for (int i=0; i<nlocal; i++)
  //        if (mask[i] & groupbit && drudetype[type[i]] != DRUDE_TYPE)
  //            ncore_loc++;
  //    MPI_Allreduce(&ncore_loc, &ncore, 1, MPI_LMP_BIGINT, MPI_SUM, world);
  //}
}

/* ---------------------------------------------------------------------- */
void FixMinDrude::min_post_force(int /*vflag*/)
{
  // zero forces on non-drude atoms

  double **f = atom->f;
  int nlocal = atom->nlocal;

  int *drudetype = fix_drude->drudetype;
  int *type = atom->type;
  int dim = domain->dimension;
  if (setforce_flag){
    for (int i = 0; i < nlocal; i++) {
      if (drudetype[type[i]] != DRUDE_TYPE) {
        for (int k=0; k<dim; k++) f[i][k] = 0;
      }
    }
  }
}


void FixMinDrude::pre_force(int /*vflag*/)
{
  bigint ntimestep_hold = update->ntimestep;
  bigint endstep_hold = update->endstep;
  bigint nsteps_hold = update->nsteps;
  bigint output_next_hold = output->next;
  update->whichflag = 2;
  update->nsteps = maxiter;
  update->endstep = update->laststep = update->firststep + maxiter;
  output->next = update->firststep + maxiter + 1;

  if (update->laststep < 0)
    error->all(FLERR,"Too many iterations");
  
  //update->minimize->setup();
[
  //modify->addstep_compute_all(update->ntimestep);
  setforce_flag = 1;
  update->minimize->setup_minimal(1);

  int ncalls = neighbor->ncalls;

  update->minimize->run(maxiter);

  update->minimize->cleanup();
  
  setforce_flag = false;
  update->ntimestep = ntimestep_hold;
  update->endstep = update->laststep = endstep_hold;
  update->nsteps = nsteps_hold;
  output->next = output_next_hold;
  for (int i = 0; i < modify->ncompute; i++)
    if (modify->compute[i]->timeflag) modify->compute[i]->clearstep();


  update->whichflag = 1;
  update->integrate->setup(1);
  // this may be needed if don't do full init
  modify->addstep_compute_all(update->ntimestep);
}
