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
    size_t V_halo; //Volume halo -> taille d'une cellule 3d d'un hypercube
    std::vector<Complex> links; //Vecteur de liens
    std::vector<Complex> halo_send; //Vecteur de halo à envoyer
    std::vector<Complex> halo_rec; //Halo à recevoir

    //Halos pour calcul plaquette moyenne
    std::vector<Complex> halo_x0;
    std::vector<Complex> halo_xL;
    std::vector<Complex> halo_y0;
    std::vector<Complex> halo_yL;
    std::vector<Complex> halo_z0;
    std::vector<Complex> halo_zL;
    std::vector<Complex> halo_t0;
    std::vector<Complex> halo_tL;

    explicit lat4d(int L_) {
        L = L_;
        V = static_cast<size_t>(L)*L*L*L;
        V_halo = static_cast<size_t>(L)*L*L;
        links.resize(V*4*9);
        halo_send.resize(V_halo*4*9);
        halo_rec.resize(V_halo*4*9);
        halo_x0.resize(V_halo*4*9);
        halo_xL.resize(V_halo*4*9);
        halo_y0.resize(V_halo*4*9);
        halo_yL.resize(V_halo*4*9);
        halo_z0.resize(V_halo*4*9);
        halo_zL.resize(V_halo*4*9);
        halo_t0.resize(V_halo*4*9);
        halo_tL.resize(V_halo*4*9);

    }

    //Index functions
    size_t index(int x, int y, int z, int t) const {
        return (((static_cast<size_t>(t)*L)+z)*L+y)*L+x;
    }

    size_t index_halo(int x, int y, int z) const {
        //Local coordinates x,y,z of the halo
        return ((static_cast<size_t>(z)*L)+y)*L+x;
    }

    //Link access functions
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

    //Shift functions
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

    void fill_lattice_with_halo_rec(int mu, bool i_mu) {
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

    //Start
    void hot_start(std::mt19937_64 &gen) {
        for (size_t site =0; site<V; site++) {
            for (int mu = 0; mu<NDIMS; mu++) {
                view_link(site, mu) = random_su3(gen);
            }
        }
    }

    void cold_start() {
        for (size_t site =0; site<V; site++) {
            for (int mu = 0; mu<NDIMS; mu++) {
                view_link(site, mu) = SU3::Identity();
            }
        }
    }

    //Remplissage halo observables
    void fill_halo_obs(int x0, int xL, int y0, int yL, int z0, int zL, int t0, int tL, MPI_Comm comm) {
        fill_halo_send(0, 0);
        MPI_Sendrecv(halo_send.data(), 2*9*4*V_halo, MPI_DOUBLE, x0, 0, halo_xL.data(), 2*9*4*V_halo, MPI_DOUBLE, xL, 0, comm, MPI_STATUS_IGNORE);
        fill_halo_send(0, 1);
        MPI_Sendrecv(halo_send.data(), 2*9*4*V_halo, MPI_DOUBLE, xL, 1, halo_x0.data(), 2*9*4*V_halo, MPI_DOUBLE, x0, 1, comm, MPI_STATUS_IGNORE);
        fill_halo_send(1, 0);
        MPI_Sendrecv(halo_send.data(), 2*9*4*V_halo, MPI_DOUBLE, y0, 2, halo_yL.data(), 2*9*4*V_halo, MPI_DOUBLE, yL, 2, comm, MPI_STATUS_IGNORE);
        fill_halo_send(1, 1);
        MPI_Sendrecv(halo_send.data(), 2*9*4*V_halo, MPI_DOUBLE, yL, 3, halo_y0.data(), 2*9*4*V_halo, MPI_DOUBLE, y0, 3, comm, MPI_STATUS_IGNORE);
        fill_halo_send(2, 0);
        MPI_Sendrecv(halo_send.data(), 2*9*4*V_halo, MPI_DOUBLE, z0, 4, halo_zL.data(), 2*9*4*V_halo, MPI_DOUBLE, zL, 4, comm, MPI_STATUS_IGNORE);
        fill_halo_send(2, 1);
        MPI_Sendrecv(halo_send.data(), 2*9*4*V_halo, MPI_DOUBLE, zL, 5, halo_z0.data(), 2*9*4*V_halo, MPI_DOUBLE, z0, 5, comm, MPI_STATUS_IGNORE);
        fill_halo_send(3, 0);
        MPI_Sendrecv(halo_send.data(), 2*9*4*V_halo, MPI_DOUBLE, t0, 6, halo_tL.data(), 2*9*4*V_halo, MPI_DOUBLE, tL, 6, comm, MPI_STATUS_IGNORE);
        fill_halo_send(3, 1);
        MPI_Sendrecv(halo_send.data(), 2*9*4*V_halo, MPI_DOUBLE, tL, 7, halo_t0.data(), 2*9*4*V_halo, MPI_DOUBLE, t0, 7, comm, MPI_STATUS_IGNORE);
    }

    //Accès link lattice+halos
    SU3 get_link_at(int x, int y, int z, int t, int mu) const {
    // interne helper pour mapper un halo-vector -> Map<SU3>
    auto map_halo = [&](const std::vector<Complex> &halo_vec, size_t halo_idx, int mu_local) -> SU3 {
        const Complex* ptr = &halo_vec[(halo_idx * 4 + mu_local) * 9];
        return Eigen::Map<const SU3>(ptr);
    };

    // cas normal (site local)
    if (0 <= x && x < L && 0 <= y && y < L && 0 <= z && z < L && 0 <= t && t < L) {
        size_t site = index(x,y,z,t);
        return view_link_const(site, mu);
    }

    // hors domaine : on regarde laquelle coord est hors bornes (il n'y en aura au plus qu'une si on demande x+1 par ex)
    // pour chaque direction on mappe vers le halo correspondant en utilisant l'ordre (c1,c2,c3) attendu par index_halo
    if (x == -1) {
        // face x0: halo_x0 stores sites ordered as (c1=y, c2=z, c3=t)
        size_t hidx = index_halo(y, z, t);
        return map_halo(halo_x0, hidx, mu);
    }
    if (x == L) {
        // face xL
        size_t hidx = index_halo(y, z, t);
        return map_halo(halo_xL, hidx, mu);
    }
    if (y == -1) {
        // face y0: halo_y0 stores (c1=x, c2=z, c3=t)
        size_t hidx = index_halo(x, z, t);
        return map_halo(halo_y0, hidx, mu);
    }
    if (y == L) {
        size_t hidx = index_halo(x, z, t);
        return map_halo(halo_yL, hidx, mu);
    }
    if (z == -1) {
        // face z0: halo_z0 stores (c1=x, c2=y, c3=t)
        size_t hidx = index_halo(x, y, t);
        return map_halo(halo_z0, hidx, mu);
    }
    if (z == L) {
        size_t hidx = index_halo(x, y, t);
        return map_halo(halo_zL, hidx, mu);
    }
    if (t == -1) {
        // face t0: halo_t0 stores (c1=x, c2=y, c3=z)
        size_t hidx = index_halo(x, y, z);
        return map_halo(halo_t0, hidx, mu);
    }
    if (t == L) {
        size_t hidx = index_halo(x, y, z);
        return map_halo(halo_tL, hidx, mu);
    }

    // cas improbable : coordonnées sortent de ±1
    std::cerr << "get_link_at: coordinate out of expected range\n";
    return SU3::Identity();
}

    //Observables
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

    // Calcule la somme locale des plaquettes (sum ReTr(U_p)) et le nombre local de plaquettes comptées
    // Chaque site produit 6 plaquettes (mu<nu). On compte chaque plaquette une seule fois.
    double local_mean_plaquette() const {
        double sum = 0.0;
        size_t count = 0;

        for (int t = 0; t < L; ++t) {
            for (int z = 0; z < L; ++z) {
                for (int y = 0; y < L; ++y) {
                    for (int x = 0; x < L; ++x) {
                        // pour chaque paire mu<nu
                        for (int mu = 0; mu < NDIMS; ++mu) {
                            for (int nu = mu + 1; nu < NDIMS; ++nu) {
                                // coord de x+mu
                                int xp = x, yp = y, zp = z, tp = t;
                                if (mu == 0) xp = x + 1;
                                if (mu == 1) yp = y + 1;
                                if (mu == 2) zp = z + 1;
                                if (mu == 3) tp = t + 1;
                                // coord de x+nu
                                int xq = x, yq = y, zq = z, tq = t;
                                if (nu == 0) xq = x + 1;
                                if (nu == 1) yq = y + 1;
                                if (nu == 2) zq = z + 1;
                                if (nu == 3) tq = t + 1;

                                // pour accéder, on autorise valeurs L (ces cas seront lus dans les halos)
                                auto U_mu      = get_link_at(x, y, z, t, mu);
                                auto U_nu_xmu  = get_link_at(xp, yp, zp, tp, nu);
                                auto U_mu_xnu  = get_link_at(xq, yq, zq, tq, mu);
                                auto U_nu      = get_link_at(x, y, z, t, nu);

                                SU3 pl = U_mu * U_nu_xmu * U_mu_xnu.adjoint() * U_nu.adjoint();
                                sum += pl.trace().real()/3.0;
                                ++count;
                            }
                        }
                    }
                }
            }
        }
        return sum/static_cast<double>(count);
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
    lat.hot_start(rng);
    //lat.cold_start();
    if (rank == 0) {
        std::cout << "Lattices créées, hot start\n";
    }

    //Calcul plaquette moyenne globale et locale avant shift
    lat.fill_halo_obs(x0, xL, y0, yL, z0, zL, t0, tL, cart_comm);
    double l_plaquette = lat.local_mean_plaquette();
    double g_plaquette;
    MPI_Reduce(&l_plaquette, &g_plaquette, 1, MPI_DOUBLE, MPI_SUM, 0, cart_comm);

    if (rank == 0) {
        g_plaquette /= size;
        std::cout << "Plaquette moyenne locale : " << lat.local_mean_plaquette() << std::endl;
        std::cout << "Plaquette moyenne globale : " << g_plaquette << std::endl;
    }

    //Shift dans la direction +x
    if (rank==0) {
        std::cout << "Shifting...\n";
    }
    lat.fill_halo_send(0, 1);
    lat.exchange_halos(x0, xL, cart_comm);
    lat.shift_pos(0);
    lat.fill_lattice_with_halo_rec(0,1);
    if (rank == 0) {
        std::cout << "Shifting done !\n";
    }

    //Calcul plaquette moyenne globale et locale après shift
    lat.fill_halo_obs(x0, xL, y0, yL, z0, zL, t0, tL, cart_comm);
    l_plaquette = lat.local_mean_plaquette();
    g_plaquette=0.0;
    MPI_Reduce(&l_plaquette, &g_plaquette, 1, MPI_DOUBLE, MPI_SUM, 0, cart_comm);

    if (rank == 0) {
        g_plaquette /= size;
        std::cout << "Plaquette moyenne locale : " << lat.local_mean_plaquette() << std::endl;
        std::cout << "Plaquette moyenne globale : " << g_plaquette << std::endl;
    }

    //TODO: ECMC sur node
    MPI_Finalize();
    return 0;
}