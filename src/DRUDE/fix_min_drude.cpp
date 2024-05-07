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
#include "memory.h"
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

/* ---------------------------------------------------------------------- */

FixMinDrude::FixMinDrude(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  // Default values
  fix_drude = nullptr;

  nmax = atom->nmax;
  memory->create(prev_force, nmax, 3, "min/drude:prev_force");
  memory->create(prev_dir, nmax, 3, "min/drude:prev_dir");
  memory->create(new_dir, nmax, 3, "min/drude:new_dir");
  memory->create(new_force, nmax, 3, "min/drude:new_force");
  memory->create(min_x, nmax, 3, "min/drude:min_x");
  if (narg == 3){
    maxiter = 15;
    conv_tol = 0.000001;
    alpha = 0.0001;
    line_search_iter=100;
  } else {
    maxiter = utils::inumeric(FLERR,arg[3],false,lmp);
    line_search_iter = utils::inumeric(FLERR,arg[4],false,lmp);
    conv_tol = utils::numeric(FLERR,arg[5],false,lmp);
    alpha = utils::numeric(FLERR,arg[6],false,lmp);
  }
}

/* ---------------------------------------------------------------------- */

FixMinDrude::~FixMinDrude(){
  memory->destroy(prev_force);
  memory->destroy(prev_dir);
  memory->destroy(new_dir);
  memory->destroy(new_force);
  memory->destroy(min_x);
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
    error->all(FLERR,"RESPA style not compatible with fix min drude");
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

void FixMinDrude::compute_forces(int eflag, int vflag){
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

}

/* ---------------------------------------------------------------------- */

// void FixMinDrude::pre_force(int /*vflag*/)
void FixMinDrude::pre_force(int /*vflag*/)
{
  // printf("\n");
  // printf("MINIMIZING...\n");
  // printf("\n");
  int natoms = int(atom->nlocal);
  double beta[3];

  // reallocate arrays if neccesary
  if (atom->nmax > nmax) {
    nmax = atom->nmax;
    memory->destroy(prev_force);
    memory->destroy(prev_dir);
    memory->destroy(new_dir);
    memory->destroy(min_x);
    memory->destroy(new_force);
    memory->create(prev_force, nmax, 3, "min/drude:prev_force");
    memory->create(prev_dir, nmax, 3, "min/drude:prev_dir");
    memory->create(new_dir, nmax, 3, "min/drude:new_dir");
    memory->create(new_force, nmax, 3, "min/drude:new_force");
    memory->create(min_x, nmax, 3, "min/drude:min_x");
  }

  int vflag = 0;
  int eflag = 1;

  force->setup();
  int triclinic = domain->triclinic;

  // initial force calcualtion
  compute_forces(eflag, vflag);  
  for (int i = 0; i < atom->nlocal; i++){
    for (int j = 0; j < 3; j++){
      prev_force[i][j] = atom->f[i][j];
      prev_dir[i][j] = -1.0*atom->f[i][j];
    }
  }
  
  double conv_condition = 0;
  for (int iter = 0; iter < maxiter; iter++){

    // move DOs
    compute_forces(eflag, vflag);

    for (int i = 0; i < atom->nlocal; i++){
      for (int j = 0; j < 3; j++){
        new_force[i][j] = atom->f[i][j];
        new_dir[i][j] = -1.0*atom->f[i][j];
        min_x[i][j] = atom->x[i][j];
      }
    }

    double dot_prev[3] = {0.0, 0.0, 0.0};
    double dot_new[3] = {0.0, 0.0, 0.0};
    for (int i = 0; i < atom->nlocal; i++){
      if (atom->mask[i] & groupbit && fix_drude->drudetype[atom->type[i]] == DRUDE_TYPE){
        for (int j = 0; j < 3; j++){
          dot_new[j] += new_force[i][j]*new_force[i][j];
          dot_prev[j] += prev_force[i][j]*prev_force[i][j];
        }
      }
    }
    if (iter == 0){
        beta[0] = 0.0;
        beta[1] = 0.0;
        beta[2] = 0.0;
    } else {
        for (int j = 0; j < 3; j++){
          if (dot_prev[j] == 0.0){
            beta[j] = 0.0;
          }
          else{
            beta[j] = dot_new[j]/dot_prev[j];
          }
        }
    }

    for (int i = 0; i < atom->nlocal; i++){
      if (atom->mask[i] & groupbit && fix_drude->drudetype[atom->type[i]] == DRUDE_TYPE){
        for (int j = 0; j < 3; j++){
          new_dir[i][j] = new_force[i][j] + beta[j]*prev_dir[i][j];
        }
      }
    }
    for (int i = 0; i < atom->nlocal; i++){
      for (int j = 0; j < 3; j++){
        prev_force[i][j] = new_force[i][j];
        prev_dir[i][j] = new_force[i][j];
      }
    }

    double min_y = 1E10;
    for (int k = 0; k < line_search_iter; k++){
      for (int i = 0; i < atom->nlocal; i++){
        if (atom->mask[i] & groupbit && fix_drude->drudetype[atom->type[i]] == DRUDE_TYPE){
          for (int j = 0; j < 3; j++){
            atom->x[i][j] += new_dir[i][j] * alpha;
          }
        }
      }

      compute_forces(eflag, vflag);
      
      double norm = 0.0;
      for (int i = 0; i < atom->nlocal; i++){
        if (atom->mask[i] & groupbit && fix_drude->drudetype[atom->type[i]] == DRUDE_TYPE){
          for (int j = 0; j < 3; j++){
            norm += sqrt(atom->f[i][0]*atom->f[i][0] + atom->f[i][1]*atom->f[i][1] + atom->f[i][2]*atom->f[i][2]);
          }
        }
      }
      
      double global_norm = 0.0;
      MPI_Allreduce(&norm,&global_norm,1,MPI_DOUBLE,MPI_SUM,world);

      if (global_norm < min_y){
        for (int i = 0; i < atom->nlocal; i++){
          for (int j = 0; j < 3; j++){
            min_x[i][j] = atom->x[i][j];
          }
        }
        min_y = global_norm;
        conv_condition = global_norm;
      } else {
        break;
      }
    }

    for (int i = 0; i < atom->nlocal; i++){
      for (int j = 0; j < 3; j++){
        atom->x[i][j] = min_x[i][j];
      }
      if (atom->mask[i] & groupbit && fix_drude->drudetype[atom->type[i]] == DRUDE_TYPE){
      }
    }

    if (conv_condition < conv_tol) {
      break;
    }

    // re-clear forces
    force_clear();
    // reneighbor and exchange
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
  }
}

/* ---------------------------------------------------------------------- */

double FixMinDrude::memory_usage(){
  double bytes = 0;
  bytes += memory->usage(prev_force, nmax, 3);
  bytes += memory->usage(prev_dir, nmax, 3);
  bytes += memory->usage(new_dir, nmax, 3);
  bytes += memory->usage(new_dir, nmax, 3);
  bytes += memory->usage(min_x, nmax, 3);
  return bytes;
}