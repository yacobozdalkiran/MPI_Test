//
// Created by ozdalkiran-l on 12/1/25.

#include <iostream>
#include <mpi.h>
#include <Eigen/Dense>
#include <complex>
#include <random>

#define NDIMS 4

using SU3 = Eigen::Matrix3cd;
using Complex = std::complex<double>;

std::random_device rd;
std::mt19937_64 rng(rd());

SU3 random_su3(std::mt19937_64 &rng) {
    //Génère une matrice de SU3 aléatoire uniformément selon la mesure de Haar en utilisant la décomposition QR
    std::normal_distribution<double> gauss(0.0, 1.0);
    Eigen::Matrix3cd z;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            z(i,j) = Complex(gauss(rng), gauss(rng));

    // Décomposition QR
    Eigen::HouseholderQR<SU3> qr(z);
    SU3 Q = qr.householderQ();

    // Corrige la phase globale pour que det(Q) = 1
    Complex detQ = Q.determinant();
    Q /= std::pow(detQ, 1.0/3.0);

    return Q;
}

struct lat4d {
    int L; //Taille de la lattice : Lt = Lx = Ly = Lz (actifs+frozen)
    size_t V; //Volume (nombre de sites actifs+frozen)
    size_t V_halo; //Volume halo
    std::vector<Complex> links; //Vecteur de liens
    std::vector<Complex> halo_send; //Vecteur de halo à envoyer
    std::vector<Complex> halo_rec; //Halo à recevoir

    explicit lat4d(int L_) {
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
        return Eigen::Map<SU3>(&links[(site*4+mu)*9]);
    }

    Eigen::Map<const SU3> view_link_const(size_t site, int mu) const {
        return Eigen::Map<const SU3>(&links[(site*4+mu)*9]);
    }

    Eigen::Map<SU3> view_halo_send(size_t site, int mu) {
        return Eigen::Map<SU3>(&halo_send[(site*4+mu)*9]);
    }

    Eigen::Map<const SU3> view_halo_send_const(size_t site, int mu) const {
        return Eigen::Map<const SU3>(&halo_send[(site*4+mu)*9]);
    }

    Eigen::Map<SU3> view_halo_rec(size_t site, int mu) {
        return Eigen::Map<SU3>(&halo_rec[(site*4+mu)*9]);
    }

    Eigen::Map<const SU3> view_halo_rec_const(size_t site, int mu) const {
        return Eigen::Map<const SU3>(&halo_rec[(site*4+mu)*9]);
    }

    void fill_halo_send(int mu, bool i_mu) {
        if (mu < 0 || mu > 3) std::cerr << "Wrong value of mu\n";
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
                        for (int nu = 0; nu < NDIMS; nu++) {
                            view_halo_send(index_local_halo, nu) = view_link_const(index_global_lattice, nu);
                        }
                    }
                }
            }
        }

    void shift_pos(int mu) {
        if (mu < 0 || mu > 3) std::cerr << "Wrong value of mu\n";
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
                        for (int nu = 0; nu < NDIMS; nu++) {
                            view_link(index_site, nu) = view_link_const(index_new, nu);
                        }
                    }
                }
            }
        }
    }

    void shift_neg(int mu) {
        //Shifts the value of all the links of the lattice in direction -mu
        if (mu < 0 || mu > 3) std::cerr << "Wrong value of mu\n";
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
                        for (int nu = 0; nu < NDIMS; nu++) {
                            view_link(index_site, nu) = view_link_const(index_new, nu);
                        }
                    }
                }
            }
        }
    }

    void exchange_halos(int source, int dest, MPI_Comm comm) {
        //Fills the halo_rec of the dest node with the content of the halo_send of the source node
        MPI_Sendrecv(halo_send.data(), 2*9*4*V_halo, MPI_DOUBLE, dest, 0, halo_rec.data(), 2*9*4*V_halo, MPI_DOUBLE, source, 0, comm, MPI_STATUS_IGNORE);
    }

    void fill_lattice_with_halo(int mu, bool i_mu) {
        //Replace the values of the corresponding links of the lattice with those of halo_rec
        //Halo rec was filled the same way halo_send was, but if halo_send was at coord=L-1, halo_rec contains coord=0
        if (mu < 0 || mu > 3) std::cerr << "Wrong value of mu\n";
        for (int c1 = 0; c1 < L; c1++) {
            for (int c2 = 0; c2 < L; c2++) {
                for (int c3 = 0; c3 < L; c3++) {
                    size_t index_local_halo = index_halo(c1,c2,c3);
                    size_t index_global_lattice;
                    if (mu == 0 && i_mu == 0) {
                        index_global_lattice = index(L-1, c1, c2, c3);
                    }
                    else if (mu == 0 && i_mu == 1) {
                        index_global_lattice = index(0, c1, c2, c3);
                    }
                    else if (mu == 1 && i_mu == 0) {
                        index_global_lattice = index(c1, L-1, c2, c3);
                    }
                    else if (mu == 1 && i_mu == 1) {
                        index_global_lattice = index(c1, 0, c2, c3);
                    }
                    else if (mu == 2 && i_mu == 0) {
                        index_global_lattice = index(c1, c2, L-1, c3);
                    }
                    else if (mu == 2 && i_mu == 1) {
                        index_global_lattice = index(c1, c2, 0, c3);
                    }
                    else if (mu == 3 && i_mu == 0) {
                        index_global_lattice = index(c1, c2, c3, L-1);
                    }
                    else if (mu == 3 && i_mu == 1) {
                        index_global_lattice = index(c1, c2, c3, 0);
                    }
                    for (int nu = 0; nu < NDIMS; nu++) {
                        view_link(index_global_lattice, nu) = view_halo_rec_const(index_local_halo, nu);
                    }
                }
            }
        }
    }

    double local_mean_trace() const {
        //Computes the local mean retr of the node's lattice
        double res = 0.0;
        for (size_t site =0; site<V; site++) {
            for (int mu = 0; mu<NDIMS; mu++) {
                res += view_link_const(site, mu).trace().real();
            }
        }
        return res/(static_cast<double>(V)*4);
    }

    void hot_start() {
        for (size_t site =0; site<V; site++) {
            for (int mu = 0; mu<NDIMS; mu++) {
                view_link(site, mu) = random_su3(rng);
            }
        }
    }
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    //Création de la topologie 4d
    int dims[4] = {2,2,2,2};
    int period[4] = {1,1,1,1};
    int reorder = 1; //Autorise MPI a réordonner les rank pour optimiser
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 4, dims, period, reorder, &cart_comm);

    //On récupère les coordonnées locales
    int cart_rank;
    MPI_Comm_rank(cart_comm, &cart_rank);
    int coords[4];
    MPI_Cart_coords(cart_comm, cart_rank, 4, coords);
    //std::cout << "Rank " << rank << ", (" << coords[0] << ", " << coords[1] << ", " << coords[2] << ", " << coords[3] << ")" << std::endl;

    //On récupère les rangs des voisins gauche et droite dans chaque direction
    int x0, xL;
    MPI_Cart_shift(cart_comm, 0, 1, &x0, &xL);
    int y0, yL;
    MPI_Cart_shift(cart_comm, 1, 1, &y0, &yL);
    int z0, zL;
    MPI_Cart_shift(cart_comm, 2, 1, &z0, &zL);
    int t0, tL;
    MPI_Cart_shift(cart_comm, 3, 1, &t0, &tL);


    //On crée une lattice 4x4 hot start dans chaque noeud
    lat4d lat(4);
    lat.hot_start();
    if (rank == 0) {
        std::cout << "Lattices créées, hot start effectué\n";
    }
    double l_trace = lat.local_mean_trace();
    double g_trace;

    MPI_Reduce(&l_trace, &g_trace, 1, MPI_DOUBLE, MPI_SUM, 0, cart_comm);

    if (rank == 0) {
        g_trace /= size;
        std::cout << "Trace globale moyenne : " << g_trace << std::endl;
    }

    //Shift dans la direction +x
    if (rank==0) {
        std::cout << "Shifting...\n";
    }
    lat.fill_halo_send(0, 1);
    lat.exchange_halos(x0, xL, cart_comm);
    lat.shift_pos(0);
    lat.fill_lattice_with_halo(0,1);
    if (rank == 0) {
        std::cout << "Shifting done !\n";
    }

    //On recalcule la trace moyenne
    double l_trace2 = lat.local_mean_trace();
    double g_trace2;
    MPI_Reduce(&l_trace2, &g_trace2, 1, MPI_DOUBLE, MPI_SUM, 0, cart_comm);

    if (rank == 0) {
        g_trace2 /= size;
        std::cout << "Trace globale moyenne après shift : " << g_trace2 << std::endl;
    }


    MPI_Finalize();
    return 0;
}