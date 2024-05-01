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
#include "angle.h"
#include "atom_vec.h"
#include "bond.h"
#include "comm.h"
#include "compute.h"
#include "dihedral.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "improper.h"
#include "input.h"
#include "kspace.h"
#include "modify.h"
#include "random_mars.h"
#include "update.h"
#include "variable.h"
#include "pointers.h"
#include "min.h"
#include "output.h"
#include "integrate.h"
#include "neighbor.h"
#include "pair.h"

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

  if (modify->get_fix_by_id("package_omp")) external_force_clear = 1;
  torqueflag = extraflag = 0;
  if (atom->torque_flag) torqueflag = 1;
  if (atom->avec->forceclearflag) extraflag = 1;

  if (force->pair && force->pair->compute_flag) pair_compute_flag = 1;
  else pair_compute_flag = 0;

    if (force->kspace && force->kspace->compute_flag) kspace_compute_flag = 1;
  else kspace_compute_flag = 0;
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

void FixMinDrude::force_clear()
{
  if (external_force_clear) return;

  // clear global force array
  // if either newton flag is set, also include ghosts

  size_t nbytes = sizeof(double) * atom->nlocal;
  if (force->newton) nbytes += sizeof(double) * atom->nghost;

  if (nbytes) {
    memset(&atom->f[0][0],0,3*nbytes);
    if (torqueflag) memset(&atom->torque[0][0],0,3*nbytes);
    if (extraflag) atom->avec->force_clear(0,nbytes);
  }
}

/* ---------------------------------------------------------------------- */

// void FixMinDrude::pre_force(int /*vflag*/)
void FixMinDrude::pre_force(int /*vflag*/)
{
  // bigint ntimestep_hold = update->ntimestep;
  // bigint endstep_hold = update->endstep;
  // bigint nsteps_hold = update->nsteps;
  // bigint output_next_hold = output->next;
  // update->whichflag = 2;
  // update->nsteps = maxiter;
  // update->endstep = update->laststep = update->firststep + maxiter;
  // output->next = update->firststep + maxiter + 1;

  // if (update->laststep < 0)
  //   error->all(FLERR,"Too many iterations");
  
  // //modify->addstep_compute_all(update->ntimestep);
  // setforce_flag = 1;
  // update->integrate->cleanup();

  // // update->minimize->setup(0);
  // update->minimize->setup_minimal(0);

  // int ncalls = neighbor->ncalls;

  // update->minimize->run(maxiter);

  // update->minimize->cleanup();
  
  // setforce_flag = false;
  // update->ntimestep = ntimestep_hold;
  // update->endstep = update->laststep = endstep_hold;
  // update->nsteps = nsteps_hold;
  // output->next = output_next_hold;
  // for (int i = 0; i < modify->ncompute; i++)
  //   if (modify->compute[i]->timeflag) modify->compute[i]->clearstep();


  // update->whichflag = 1;
  // update->integrate->setup(1);
  // // this may be needed if don't do full init
  // modify->addstep_compute_all(update->ntimestep);
  // setforce_flag = 0;

  force->setup();
  int vflag = 0;
  int eflag = 1;
  int triclinic = domain->triclinic;
  // ev_set(update->ntimestep); // ???
  for (int iter = 0; iter < 100; iter++){
    // compute forces
    force_clear();
    modify->setup_pre_force(0); // should arg be 0?

    if (pair_compute_flag) force->pair->compute(eflag,vflag);
    else if (force->pair) force->pair->compute_dummy(eflag,vflag);

    if (atom->molecular != Atom::ATOMIC) {
      if (force->bond) force->bond->compute(eflag,vflag);
      if (force->angle) force->angle->compute(eflag,vflag);
      if (force->dihedral) force->dihedral->compute(eflag,vflag);
      if (force->improper) force->improper->compute(eflag,vflag);
    }

    if (force->kspace) {
      force->kspace->setup();
      if (kspace_compute_flag) force->kspace->compute(eflag,vflag);
      else force->kspace->compute_dummy(eflag,vflag);
    }

    modify->setup_pre_reverse(eflag,vflag);
    if (force->newton) comm->reverse_comm();
    // move DOs
    for (int i = 0; i < atom->nlocal; i++){
      if (atom->mask[i] & groupbit && fix_drude->drudetype[atom->type[i]] == DRUDE_TYPE){
        for (int j = 0; j < 3; j++){
          atom->x[i][j] += atom->f[i][j] * 0.0001;
          printf("force %f %f %f\n", atom->f[i][0], atom->f[i][1], atom->f[i][2]);
        }
      }
    }
  
    int nflag = neighbor->decide();

    if (nflag == 0) {
      comm->forward_comm();
    } else {
      if (modify->n_min_pre_exchange) {
        modify->min_pre_exchange();
      }
      if (triclinic) domain->x2lamda(atom->nlocal);
      domain->pbc();
      if (domain->box_change) {
        domain->reset_box();
        comm->setup();
        if (neighbor->style) neighbor->setup_bins();
      }
      comm->exchange();
      if (atom->sortfreq > 0 &&
          update->ntimestep >= atom->nextsort) atom->sort();
      comm->borders();
      if (triclinic) domain->lamda2x(atom->nlocal+atom->nghost);
      if (modify->n_min_pre_neighbor) {
        modify->min_pre_neighbor();
      }
      neighbor->build(1);
      if (modify->n_min_post_neighbor) {
        modify->min_post_neighbor();
      }
    }

    // print the energy :)
    printf("Energy: %f\n", force->pair->eng_vdwl + force->pair->eng_coul);
  }
}
