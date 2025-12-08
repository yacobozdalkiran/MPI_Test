//
// Created by ozdalkiran-l on 12/1/25.

#include <iostream>
#include <mpi.h>
#include <Eigen/Dense>
#include <complex>

#define NDIMS 4

using SU3 = Eigen::Matrix3cd;
using Complex = std::complex<double>;

struct lat4d {
    int L; //Taille de la lattice : Lt = Lx = Ly = Lz (actifs+frozen)
    size_t V; //Volume (nombre de sites actifs+frozen)
    size_t V_halo; //Volume halo
    std::vector<Complex> links; //Vecteur de liens
    std::vector<Complex> halo_send; //Vecteur de halo à envoyer
    std::vector<Complex> halo_rec; //Halo à recevoir

    lat4d(int L_) {
        L = L_;
        V = static_cast<size_t>(L)*L*L*L;
        V_halo = static_cast<size_t>(L)*L*L;
        links.resize(V*4*9);
        halo_send.resize(V_halo*4*9);
        halo_rec.resize(V_halo*4*9);
    }

    size_t index(int x, int y, int z, int t) const {
        return (((static_cast<size_t>(t)*L)+z)*L+y)*L+x;
    }

    size_t index_halo(int x, int y, int z) const {
        //Local coordinates x,y,z of the halo
        return ((static_cast<size_t>(z)*L)+y)*L+x;
    }

    Eigen::Map<SU3> view_link(size_t site, int mu) {
        return Eigen::Map<SU3>(&links[site*4+mu]);
    }

    Eigen::Map<const SU3> view_link_const(size_t site, int mu) const {
        return Eigen::Map<const SU3>(&links[site*4+mu]);
    }

    Eigen::Map<SU3> view_halo_send(size_t site, int mu) {
        return Eigen::Map<SU3>(&halo_send[site*4+mu]);
    }

    Eigen::Map<const SU3> view_halo_send_const(size_t site, int mu) const {
        return Eigen::Map<const SU3>(&halo_send[site*4+mu]);
    }

    Eigen::Map<SU3> view_halo_rec(size_t site, int mu) {
        return Eigen::Map<SU3>(&halo_rec[site*4+mu]);
    }

    Eigen::Map<const SU3> view_halo_rec_const(size_t site, int mu) const {
        return Eigen::Map<const SU3>(&halo_rec[site*4+mu]);
    }

    void fill_halo_send(int mu, bool i_mu) {
        //Fills the halo with the cellule defined by mu=0 if i_mu=0 or mu=L-1 if i_mu = 1
            for (int c1 = 0; c1 < L; c1++) {
                for (int c2 = 0; c2 < L; c2++) {
                    for (int c3 = 0; c3 < L; c3++) {
                        size_t index_local_halo = index_halo(c1,c2,c3);
                        size_t index_global_lattice;
                        if (mu == 0 && i_mu == 0) {
                            index_global_lattice = index(0, c1, c2, c3);
                        }
                        else if (mu == 0 && i_mu == 1) {
                            index_global_lattice = index(L-1, c1, c2, c3);
                        }
                        else if (mu == 1 && i_mu == 0) {
                            index_global_lattice = index(c1, 0, c2, c3);
                        }
                        else if (mu == 1 && i_mu == 1) {
                            index_global_lattice = index(c1, L-1, c2, c3);
                        }
                        else if (mu == 2 && i_mu == 0) {
                            index_global_lattice = index(c1, c2, 0, c3);
                        }
                        else if (mu == 2 && i_mu == 1) {
                            index_global_lattice = index(c1, c2, L-1, c3);
                        }
                        else if (mu == 3 && i_mu == 0) {
                            index_global_lattice = index(c1, c2, c3, 0);
                        }
                        else if (mu == 3 && i_mu == 1) {
                            index_global_lattice = index(c1, c2, c3, L-1);
                        }
                        for (int mu = 0; mu < NDIMS; mu++) {
                            view_halo_send(index_local_halo, mu) = view_link_const(index_global_lattice, mu);
                        }
                    }
                }
            }
        }

    void shift_pos(int mu) {
        //Shifts the value of all the links of the lattice in direction +mu
        for (int cshift = L-1; cshift>0; cshift--) {
            for (int c1 = 0; c1 < L; c1++) {
                for (int c2 = 0; c2 < L; c2++) {
                    for (int c3 = 0; c3 < L; c3++) {
                        size_t index_site;
                        size_t index_new;
                        if (mu == 0) {
                            index_site = index(cshift, c1, c2, c3);
                            index_new = index(cshift-1, c1, c2, c3);
                        }
                        else if (mu == 1) {
                            index_site = index(c1, cshift, c2, c3);
                            index_new = index(c1, cshift-1, c2, c3);
                        }
                        else if (mu == 2) {
                            index_site = index(c1, c2, cshift, c3);
                            index_new = index(c1, c2, cshift-1, c3);
                        }
                        else if (mu == 3) {
                            index_site = index(c1, c2, c3, cshift);
                            index_new = index(c1, c2, c3, cshift-1);
                        }
                        for (int mu = 0; mu < NDIMS; mu++) {
                            view_link(index_site, mu) = view_link_const(index_new, mu);
                        }
                    }
                }
            }
        }
    }

    void shift_neg(int mu) {
        //Shifts the value of all the links of the lattice in direction -mu
        for (int cshift = 0; cshift < L-1; cshift++) {
            for (int c1 = 0; c1 < L; c1++) {
                for (int c2 = 0; c2 < L; c2++) {
                    for (int c3 = 0; c3 < L; c3++) {
                        size_t index_site;
                        size_t index_new;
                        if (mu == 0) {
                            index_site = index(cshift, c1, c2, c3);
                            index_new = index(cshift+1, c1, c2, c3);
                        }
                        else if (mu == 1) {
                            index_site = index(c1, cshift, c2, c3);
                            index_new = index(c1, cshift+1, c2, c3);
                        }
                        else if (mu == 2) {
                            index_site = index(c1, c2, cshift, c3);
                            index_new = index(c1, c2, cshift+1, c3);
                        }
                        else if (mu == 3) {
                            index_site = index(c1, c2, c3, cshift);
                            index_new = index(c1, c2, c3, cshift+1);
                        }
                        for (int mu = 0; mu < NDIMS; mu++) {
                            view_link(index_site, mu) = view_link_const(index_new, mu);
                        }
                    }
                }
            }
        }
    }

    void fill_halo(int source, int dest, MPI_Comm comm) {
        //Fills the halo_rec of the dest node with the content of the halo_send of the source node
        MPI_Sendrecv(halo_send.data(), 9*4*V_halo, MPI_DOUBLE, dest, 0, halo_rec.data(), 9*4*V_halo, MPI_DOUBLE, source, 0, comm, MPI_STATUS_IGNORE);
    }

    void fill_lattice_with_halo(int mu, bool i_mu) {
        //Replace the values of the corresponding links of the lattice with those of halo_rec
        //Halo rec was filled the same way halo_send was, hence the same function
        for (int c1 = 0; c1 < L; c1++) {
            for (int c2 = 0; c2 < L; c2++) {
                for (int c3 = 0; c3 < L; c3++) {
                    size_t index_local_halo = index_halo(c1,c2,c3);
                    size_t index_global_lattice;
                    if (mu == 0 && i_mu == 0) {
                        index_global_lattice = index(0, c1, c2, c3);
                    }
                    else if (mu == 0 && i_mu == 1) {
                        index_global_lattice = index(L-1, c1, c2, c3);
                    }
                    else if (mu == 1 && i_mu == 0) {
                        index_global_lattice = index(c1, 0, c2, c3);
                    }
                    else if (mu == 1 && i_mu == 1) {
                        index_global_lattice = index(c1, L-1, c2, c3);
                    }
                    else if (mu == 2 && i_mu == 0) {
                        index_global_lattice = index(c1, c2, 0, c3);
                    }
                    else if (mu == 2 && i_mu == 1) {
                        index_global_lattice = index(c1, c2, L-1, c3);
                    }
                    else if (mu == 3 && i_mu == 0) {
                        index_global_lattice = index(c1, c2, c3, 0);
                    }
                    else if (mu == 3 && i_mu == 1) {
                        index_global_lattice = index(c1, c2, c3, L-1);
                    }
                    for (int mu = 0; mu < NDIMS; mu++) {
                        view_link(index_global_lattice, mu) = view_halo_rec_const(index_local_halo, mu);
                    }
                }
            }
        }
    }

};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::cout << "Rank : " << rank << " Size : " << size << std::endl;
    MPI_Finalize();
    return 0;
}