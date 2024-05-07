/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(min/drude,FixMinDrude);
// clang-format on
#else

#ifndef LMP_FIX_MIN_DRUDE_H
#define LMP_FIX_MIN_DRUDE_H

#include "fix.h"

namespace LAMMPS_NS {

class FixMinDrude : public Fix {
 public:
  FixMinDrude(class LAMMPS *, int, char **);
  ~FixMinDrude() override;
  int setmask() override;
  void init() override;
  void setup(int vflag) override;
  void pre_force(int vflag) override;
  double memory_usage() override;

 protected:
  void force_clear();
  void compute_forces(int, int);
  int maxiter, line_search_iter;
  double alpha, conv_tol;
  int tstyle_core, tstyle_drude;
  int tvar_core, tvar_drude;
  char *tstr_core, *tstr_drude;
  double energy;
  double **prev_force, **prev_dir, **new_dir, **new_force, **min_x;
  int tflag, torqueflag, extraflag, external_force_clear, pair_compute_flag, kspace_compute_flag;

  class RanMars *random_core, *random_drude;
  int zero;
  bigint ncore;
  class FixDrude *fix_drude;
  class Compute *temperature;
  char *id_temp;
};

}    // namespace LAMMPS_NS

#endif
#endif
