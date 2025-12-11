//
// Created by ozdalkiran-l on 12/1/25.

#include <iostream>
#include <mpi.h>
#include <Eigen/Dense>
#include <complex>
#include <random>
#include <optional>

#define NDIMS 4

using SU3 = Eigen::Matrix3cd;
using Complex = std::complex<double>;

std::random_device rd;
std::mt19937_64 gen(rd());

//Observables


SU3 random_su3(std::mt19937_64 &rng) {
    //Génère une matrice de SU3 aléatoire uniformément selon la mesure de Haar en utilisant la décomposition QR
    std::normal_distribution<double> gauss(0.0, 1.0);
    Eigen::Matrix3cd z;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            z(i, j) = Complex(gauss(rng), gauss(rng));

    // Décomposition QR
    Eigen::HouseholderQR<SU3> qr(z);
    SU3 Q = qr.householderQ();

    // Corrige la phase globale pour que det(Q) = 1
    Complex detQ = Q.determinant();
    Q /= std::pow(detQ, 1.0 / 3.0);

    return Q;
}

SU3 su2_quaternion_to_su3(const std::array<double,4> &su2, int i, int j){
    //Permet d'embedder une matrice SU2 représentation quaternionique en matrice SU3 (sous groupe d'indice i,j)
    if (i==j) std::cerr<<"i = j wrong embedding\n";
    SU3 X;
    int k = 3-i-j;
    X.setZero();
    X(k,k) = Complex(1.0,0.0);
    X(i,i) = Complex(su2[0],su2[3]);
    X(j,j) = Complex(su2[0],-su2[3]);
    X(i,j) = Complex(su2[2],su2[1]);
    X(j,i) = Complex(-su2[2],su2[1]);
    return X;
}

SU3 random_SU3_epsilon(double epsilon, std::mt19937_64 &rng) {
    //Pour générer des matrices de SU3 epsilon proches de l'identité (cf Gattringer)
    std::uniform_real_distribution<double> unif(-0.5,0.5);
    std::array<double,4> x = {0.0, 0.0, 0.0, 0.0};
    SU3 M = SU3::Identity();

    //double r0 = unif(rng);
    double r1 = unif(rng);
    double r2 = unif(rng);
    double r3 = unif(rng);
    double norm = sqrt(r1*r1 + r2*r2 + r3*r3);
    x[0] = sqrt(1-epsilon*epsilon);
    x[1] = epsilon * r1 / norm;
    x[2] = epsilon * r2 / norm;
    x[3] = epsilon * r3 / norm;
    M *= su2_quaternion_to_su3(x, 0,1);

    //r0 = unif(rng);
    r1 = unif(rng);
    r2 = unif(rng);
    r3 = unif(rng);
    norm = sqrt(r1*r1 + r2*r2 + r3*r3);
    x[0] = sqrt(1-epsilon*epsilon);
    x[1] = epsilon * r1 / norm;
    x[2] = epsilon * r2 / norm;
    x[3] = epsilon * r3 / norm;
    M *= su2_quaternion_to_su3(x, 0,2);

    //r0 = unif(rng);
    r1 = unif(rng);
    r2 = unif(rng);
    r3 = unif(rng);
    norm = sqrt(r1*r1 + r2*r2 + r3*r3);
    x[0] = sqrt(1-epsilon*epsilon);
    x[1] = epsilon * r1 / norm;
    x[2] = epsilon * r2 / norm;
    x[3] = epsilon * r3 / norm;
    M *= su2_quaternion_to_su3(x, 1,2);

    return M;
}

std::vector<SU3> ecmc_set(double epsilon, std::vector<SU3> &set, std::mt19937_64 &rng) {
    //Crée un set de matrices SU(3) epsilon-proches de l'identité de taille size avec leurs adjoints
    size_t size = set.size()-1;
    set[0] = SU3::Identity();
    for (int i = 1; i < size+1; i+=2) {
        set[i] = random_SU3_epsilon(epsilon, rng);
        set[i+1] = set[i].adjoint();
    }
    return set;
}

inline int dsign(double x) {
    //fonction signe pour double
    if (x > 0) return 1;
    if (x < 0) return -1;
    return 0;
}

inline SU3 el_3(double xi) {
    /*Generates a exp(i xi lambda_3)*/
    SU3 result;
    double cxi = cos(xi);
    double sxi = sin(xi);
    result << Complex(cxi, sxi), Complex(0.0, 0.0), Complex(0.0, 0.0),
            Complex(0.0, 0.0), Complex(cxi, -sxi), Complex(0.0, 0.0),
            Complex(0.0, 0.0), Complex(0.0, 0.0), Complex(1.0, 0.0);
    return result;
}

void compute_reject(double A, double B, double &gamma, double &reject, int epsilon) {
    if (epsilon == -1) B = -B;
    double R = sqrt(A * A + B * B);
    double phi = atan2(-A / R, B / R);

    if (phi < 0) phi += 2 * M_PI;
    //cout << "phi = " << phi << endl;
    double period = 0.0, p1 = 0.0, p2 = 0.0;
    std::array<double, 4> intervals = {0.0, 0.0, 2 * M_PI, 2 * M_PI};
    if (phi < M_PI / 2.0) {
        //cout << "cas 1"<< endl;
        intervals[1] = M_PI / 2.0 + phi;
        intervals[2] = 3 * M_PI / 2.0 + phi;
        p1 = R * (sin(intervals[1] - phi) - sin(intervals[0] - phi));
        p2 = R * (sin(intervals[3] - phi) - sin(intervals[2] - phi));
        if ((p1 < 0) && (p2 < 0)) std::cerr << "Périodes négatives !" << std::endl;
        period = p1 + p2;
        //cout << "contrib periodique = " << period << endl;
        gamma = gamma - std::floor(gamma / period) * period;
        if (gamma > p1) {
            gamma -= p1;
            double alpha = gamma / R + sin(intervals[2] - phi);
            double theta1 = fmod((phi + asin(alpha) + 2 * M_PI), 2 * M_PI);
            double theta2 = fmod((phi + M_PI - asin(alpha) + 2 * M_PI), 2 * M_PI);
            if ((theta1 < intervals[3]) && (theta1 > intervals[2])) {
                reject = theta1;
            } else {
                reject = theta2;
            }
        } else {
            double alpha = gamma / R + sin(intervals[0] - phi);
            double theta1 = fmod((phi + asin(alpha) + 2 * M_PI), 2 * M_PI);
            double theta2 = fmod((phi + M_PI - asin(alpha) + 2 * M_PI), 2 * M_PI);
            if ((theta1 < intervals[1]) && (theta1 > intervals[0])) {
                reject = theta1;
            } else {
                reject = theta2;
            }
        }
    }
    if (phi > 3 * M_PI / 2.0) {
        //cout << "cas 2" << endl;
        intervals[1] = -3 * M_PI / 2.0 + phi;
        intervals[2] = -M_PI / 2.0 + phi;
        //cout << "[" << intervals[0] << ", " << intervals[1] << "]" << endl;
        //cout << "[" << intervals[2] << ", " << intervals[3] << "]" << endl;
        p1 = R * (sin(intervals[1] - phi) - sin(intervals[0] - phi));
        p2 = R * (sin(intervals[3] - phi) - sin(intervals[2] - phi));
        if ((p1 < 0) && (p2 < 0)) std::cerr << "Périodes négatives !" << std::endl;
        period = p1 + p2;
        //cout << "contrib periodique = " << period << endl;
        gamma = gamma - std::floor(gamma / period) * period;
        if (gamma > p1) {
            gamma -= p1;
            double alpha = gamma / R + sin(intervals[2] - phi);
            double theta1 = fmod((phi + asin(alpha) + 2 * M_PI), 2 * M_PI);
            double theta2 = fmod((phi + M_PI - asin(alpha) + 2 * M_PI), 2 * M_PI);
            if ((theta1 < intervals[3]) && (theta1 > intervals[2])) {
                reject = theta1;
            } else {
                reject = theta2;
            }
        } else {
            double alpha = gamma / R + sin(intervals[0] - phi);
            double theta1 = fmod((phi + asin(alpha) + 2 * M_PI), 2 * M_PI);
            double theta2 = fmod((phi + M_PI - asin(alpha) + 2 * M_PI), 2 * M_PI);
            if ((theta1 < intervals[1]) && (theta1 > intervals[0])) {
                reject = theta1;
            } else {
                reject = theta2;
            }
        }
    }
    if ((phi >= M_PI / 2.0) && (phi <= 3 * M_PI / 2.0)) {
        //cout << "cas 3" << endl;
        intervals[0] = -M_PI / 2.0 + phi;
        intervals[1] = M_PI / 2.0 + phi;
        period = R * (sin(intervals[1] - phi) - sin(intervals[0] - phi));
        if (period < 0) std::cerr << "Période négative !" << std::endl;
        //cout << "contrib periodique = " << period << endl;
        gamma = gamma - std::floor(gamma / period) * period;
        double alpha = gamma / R + sin(intervals[0] - phi);
        double theta1 = fmod((phi + asin(alpha) + 2 * M_PI), 2 * M_PI);
        double theta2 = fmod((phi + M_PI - asin(alpha) + 2 * M_PI), 2 * M_PI);
        if ((theta1 < intervals[1]) && (theta1 > intervals[0])) {
            reject = theta1;
        } else {
            reject = theta2;
        }
    }
}

int selectVariable(const std::vector<double> &probas, std::mt19937_64 &rng) {
    //Choisit un index entre 0 et probas.size()-1 selon la méthode tower of probas
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    double r = unif(rng);
    double s = 0.0;
    for (int i = 0; i < static_cast<int>(probas.size()); i++) {
        s += probas[i];
        if (s > r) {
            return i;
        }
    }
    std::cerr << "SelectVariable Error" << std::endl;
    return -1;
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

    //Utiles pour ECMC
    std::vector<std::array<std::array<std::optional<size_t>, 2>,NDIMS> > neighbors; //Neighbors for all the possible sites of the lattice
    std::vector<std::array<bool,NDIMS> > frozen; //True if link is frozen, false if not
    std::vector<std::array<std::array<std::array<std::pair<std::optional<size_t>, int>, 3>, 6>, 4> > staples;

    explicit lat4d(int L_) {
        L = L_;
        V = static_cast<size_t>(L) * L * L * L;
        V_halo = static_cast<size_t>(L) * L * L;
        links.resize(V * 4 * 9);
        halo_send.resize(V_halo * 4 * 9);
        halo_rec.resize(V_halo * 4 * 9);
        halo_x0.resize(V_halo * 4 * 9);
        halo_xL.resize(V_halo * 4 * 9);
        halo_y0.resize(V_halo * 4 * 9);
        halo_yL.resize(V_halo * 4 * 9);
        halo_z0.resize(V_halo * 4 * 9);
        halo_zL.resize(V_halo * 4 * 9);
        halo_t0.resize(V_halo * 4 * 9);
        halo_tL.resize(V_halo * 4 * 9);
        neighbors.resize(V);
        frozen.resize(V * 4);
        staples.resize(V*4);

        for (int x = 0; x < L; x++) {
            for (int y = 0; y < L; y++) {
                for (int z = 0; z < L; z++) {
                    for (int t = 0; t < L; t++) {
                        size_t site_n = index(x, y, z, t);
                        if (x+1<=L-1) neighbors[site_n][0][0] = index(x + 1, y, z, t);
                        if (x-1>=0) neighbors[site_n][0][1] = index(x - 1, y, z, t);
                        if (y+1<=L-1) neighbors[site_n][1][0] = index(x, y + 1, z, t);
                        if (y-1>=0)neighbors[site_n][1][1] = index(x, y - 1, z, t);
                        if (z+1<=L-1)neighbors[site_n][2][0] = index(x, y, z + 1, t);
                        if (z-1>=0)neighbors[site_n][2][1] = index(x, y, z - 1, t);
                        if (t+1<=L-1)neighbors[site_n][3][0] = index(x, y, z, t + 1);
                        if (t-1>=0)neighbors[site_n][3][1] = index(x, y, z, t - 1);
                    }
                }
            }
        }

        for (int x = 0; x < L; x++) {
            for (int y = 0; y < L; y++) {
                for (int z = 0; z < L; z++) {
                    for (int t = 0; t < L; t++) {
                        size_t site = index(x, y, z, t);
                        for (int mu = 0; mu < NDIMS; mu++) {
                            frozen[site][mu] = is_frozen(x, y, z, t);
                        }
                    }
                }
            }
        }

        for (int x = 1; x < L-1 ; x++) {
            for (int y = 1; y < L-1 ; y++) {
                for (int z = 1; z < L-1 ; z++) {
                    for (int t = 1; t < L-1 ; t++) {
                        size_t site = index(x, y, z, t);
                        for (int mu = 0; mu < NDIMS; mu++) {
                            int j = 0;
                            for (int nu = 0; nu < NDIMS; nu++) {
                                if (nu == mu) continue;

                                std::optional<size_t> xmu = neighbors[site][mu][0].value();

                                staples[site][mu][j][0] = {neighbors[site][mu][0], nu};
                                staples[site][mu][j][1] = {neighbors[site][nu][0], mu};
                                staples[site][mu][j][2] = {site, nu};
                                if (xmu.has_value()) staples[site][mu][j + 1][0] = {neighbors[xmu.value()][nu][1], nu};
                                else std::cerr << "Invalid acces during lat.staples construction \n";
                                staples[site][mu][j + 1][1] = {neighbors[site][nu][1], mu};
                                staples[site][mu][j + 1][2] = {neighbors[site][nu][1], nu};

                                j += 2;
                            }
                        }
                    }
                }
            }
        }
    }

    //Index functions
    size_t index(int x, int y, int z, int t) const {
        return ((static_cast<size_t>(t) * L + z) * L + y) * L + x;
    }

    size_t index_halo(int x, int y, int z) const {
        //Local coordinates x,y,z of the halo
        return ((static_cast<size_t>(z) * L) + y) * L + x;
    }

    //Link access functions
    Eigen::Map<SU3> view_link(size_t site, int mu) {
        return Eigen::Map<SU3>(&links[(site * 4 + mu) * 9]);
    }

    Eigen::Map<const SU3> view_link_const(size_t site, int mu) const {
        return Eigen::Map<const SU3>(&links[(site * 4 + mu) * 9]);
    }

    Eigen::Map<SU3> view_halo_send(size_t site, int mu) {
        return Eigen::Map<SU3>(&halo_send[(site * 4 + mu) * 9]);
    }

    Eigen::Map<const SU3> view_halo_send_const(size_t site, int mu) const {
        return Eigen::Map<const SU3>(&halo_send[(site * 4 + mu) * 9]);
    }

    Eigen::Map<SU3> view_halo_rec(size_t site, int mu) {
        return Eigen::Map<SU3>(&halo_rec[(site * 4 + mu) * 9]);
    }

    Eigen::Map<const SU3> view_halo_rec_const(size_t site, int mu) const {
        return Eigen::Map<const SU3>(&halo_rec[(site * 4 + mu) * 9]);
    }

    //Shift functions
    void fill_halo_send(int mu, bool i_mu) {
        if (mu < 0 || mu > 3) std::cerr << "Wrong value of mu\n";
        //Fills the halo with the cellule defined by mu=0 if i_mu=0 or mu=L-1 if i_mu = 1
        for (int c1 = 0; c1 < L; c1++) {
            for (int c2 = 0; c2 < L; c2++) {
                for (int c3 = 0; c3 < L; c3++) {
                    size_t index_local_halo = index_halo(c1, c2, c3);
                    size_t index_global_lattice;
                    if (mu == 0 && i_mu == 0) {
                        index_global_lattice = index(0, c1, c2, c3);
                    } else if (mu == 0 && i_mu == 1) {
                        index_global_lattice = index(L - 1, c1, c2, c3);
                    } else if (mu == 1 && i_mu == 0) {
                        index_global_lattice = index(c1, 0, c2, c3);
                    } else if (mu == 1 && i_mu == 1) {
                        index_global_lattice = index(c1, L - 1, c2, c3);
                    } else if (mu == 2 && i_mu == 0) {
                        index_global_lattice = index(c1, c2, 0, c3);
                    } else if (mu == 2 && i_mu == 1) {
                        index_global_lattice = index(c1, c2, L - 1, c3);
                    } else if (mu == 3 && i_mu == 0) {
                        index_global_lattice = index(c1, c2, c3, 0);
                    } else if (mu == 3 && i_mu == 1) {
                        index_global_lattice = index(c1, c2, c3, L - 1);
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
        for (int cshift = L - 1; cshift > 0; cshift--) {
            for (int c1 = 0; c1 < L; c1++) {
                for (int c2 = 0; c2 < L; c2++) {
                    for (int c3 = 0; c3 < L; c3++) {
                        size_t index_site;
                        size_t index_new;
                        if (mu == 0) {
                            index_site = index(cshift, c1, c2, c3);
                            index_new = index(cshift - 1, c1, c2, c3);
                        } else if (mu == 1) {
                            index_site = index(c1, cshift, c2, c3);
                            index_new = index(c1, cshift - 1, c2, c3);
                        } else if (mu == 2) {
                            index_site = index(c1, c2, cshift, c3);
                            index_new = index(c1, c2, cshift - 1, c3);
                        } else if (mu == 3) {
                            index_site = index(c1, c2, c3, cshift);
                            index_new = index(c1, c2, c3, cshift - 1);
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
        for (int cshift = 0; cshift < L - 1; cshift++) {
            for (int c1 = 0; c1 < L; c1++) {
                for (int c2 = 0; c2 < L; c2++) {
                    for (int c3 = 0; c3 < L; c3++) {
                        size_t index_site;
                        size_t index_new;
                        if (mu == 0) {
                            index_site = index(cshift, c1, c2, c3);
                            index_new = index(cshift + 1, c1, c2, c3);
                        } else if (mu == 1) {
                            index_site = index(c1, cshift, c2, c3);
                            index_new = index(c1, cshift + 1, c2, c3);
                        } else if (mu == 2) {
                            index_site = index(c1, c2, cshift, c3);
                            index_new = index(c1, c2, cshift + 1, c3);
                        } else if (mu == 3) {
                            index_site = index(c1, c2, c3, cshift);
                            index_new = index(c1, c2, c3, cshift + 1);
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
        MPI_Sendrecv(halo_send.data(), 2 * 9 * 4 * V_halo, MPI_DOUBLE, dest, 0, halo_rec.data(), 2 * 9 * 4 * V_halo,
                     MPI_DOUBLE, source, 0, comm, MPI_STATUS_IGNORE);
    }

    void fill_lattice_with_halo_rec(int mu, bool i_mu) {
        //Replace the values of the corresponding links of the lattice with those of halo_rec
        //Halo rec was filled the same way halo_send was, but if halo_send was at coord=L-1, halo_rec contains coord=0
        if (mu < 0 || mu > 3) std::cerr << "Wrong value of mu\n";
        for (int c1 = 0; c1 < L; c1++) {
            for (int c2 = 0; c2 < L; c2++) {
                for (int c3 = 0; c3 < L; c3++) {
                    size_t index_local_halo = index_halo(c1, c2, c3);
                    size_t index_global_lattice;
                    if (mu == 0 && i_mu == 0) {
                        index_global_lattice = index(L - 1, c1, c2, c3);
                    } else if (mu == 0 && i_mu == 1) {
                        index_global_lattice = index(0, c1, c2, c3);
                    } else if (mu == 1 && i_mu == 0) {
                        index_global_lattice = index(c1, L - 1, c2, c3);
                    } else if (mu == 1 && i_mu == 1) {
                        index_global_lattice = index(c1, 0, c2, c3);
                    } else if (mu == 2 && i_mu == 0) {
                        index_global_lattice = index(c1, c2, L - 1, c3);
                    } else if (mu == 2 && i_mu == 1) {
                        index_global_lattice = index(c1, c2, 0, c3);
                    } else if (mu == 3 && i_mu == 0) {
                        index_global_lattice = index(c1, c2, c3, L - 1);
                    } else if (mu == 3 && i_mu == 1) {
                        index_global_lattice = index(c1, c2, c3, 0);
                    }
                    for (int nu = 0; nu < NDIMS; nu++) {
                        view_link(index_global_lattice, nu) = view_halo_rec_const(index_local_halo, nu);
                    }
                }
            }
        }
    }

    void full_shift_n(int n, int mu, int c0, int cL, MPI_Comm comm) {
        //Effectue n shifts dans la direction mu, c0 et cL sont resp. les sources et destinations de exchange_halos
        MPI_Barrier(comm);
        for (int i = 0; i < n; i++) {
            fill_halo_send(mu, 1);
            exchange_halos(c0, cL, comm);
            shift_pos(mu);
            fill_lattice_with_halo_rec(mu, 1);
        }
        MPI_Barrier(comm);
    }

    void full_shift_n_all_dirs(int n, int x0, int xL, int y0, int yL, int z0, int zL, int t0, int tL, MPI_Comm comm) {
        full_shift_n(n, 0, x0, xL, comm);
        full_shift_n(n, 1, y0, yL, comm);
        full_shift_n(n, 2, z0, zL, comm);
        full_shift_n(n, 3, t0, tL, comm);
    }

    //Start
    void hot_start(std::mt19937_64 &gen) {
        for (size_t site = 0; site < V; site++) {
            for (int mu = 0; mu < NDIMS; mu++) {
                view_link(site, mu) = random_su3(gen);
            }
        }
    }

    void cold_start() {
        for (size_t site = 0; site < V; site++) {
            for (int mu = 0; mu < NDIMS; mu++) {
                view_link(site, mu) = SU3::Identity();
            }
        }
    }

    //Remplissage halo observables
    void fill_halo_obs(int x0, int xL, int y0, int yL, int z0, int zL, int t0, int tL, MPI_Comm comm) {
        fill_halo_send(0, 0);
        MPI_Sendrecv(halo_send.data(), 2 * 9 * 4 * V_halo, MPI_DOUBLE, x0, 0, halo_xL.data(), 2 * 9 * 4 * V_halo,
                     MPI_DOUBLE, xL, 0, comm, MPI_STATUS_IGNORE);
        fill_halo_send(0, 1);
        MPI_Sendrecv(halo_send.data(), 2 * 9 * 4 * V_halo, MPI_DOUBLE, xL, 1, halo_x0.data(), 2 * 9 * 4 * V_halo,
                     MPI_DOUBLE, x0, 1, comm, MPI_STATUS_IGNORE);
        fill_halo_send(1, 0);
        MPI_Sendrecv(halo_send.data(), 2 * 9 * 4 * V_halo, MPI_DOUBLE, y0, 2, halo_yL.data(), 2 * 9 * 4 * V_halo,
                     MPI_DOUBLE, yL, 2, comm, MPI_STATUS_IGNORE);
        fill_halo_send(1, 1);
        MPI_Sendrecv(halo_send.data(), 2 * 9 * 4 * V_halo, MPI_DOUBLE, yL, 3, halo_y0.data(), 2 * 9 * 4 * V_halo,
                     MPI_DOUBLE, y0, 3, comm, MPI_STATUS_IGNORE);
        fill_halo_send(2, 0);
        MPI_Sendrecv(halo_send.data(), 2 * 9 * 4 * V_halo, MPI_DOUBLE, z0, 4, halo_zL.data(), 2 * 9 * 4 * V_halo,
                     MPI_DOUBLE, zL, 4, comm, MPI_STATUS_IGNORE);
        fill_halo_send(2, 1);
        MPI_Sendrecv(halo_send.data(), 2 * 9 * 4 * V_halo, MPI_DOUBLE, zL, 5, halo_z0.data(), 2 * 9 * 4 * V_halo,
                     MPI_DOUBLE, z0, 5, comm, MPI_STATUS_IGNORE);
        fill_halo_send(3, 0);
        MPI_Sendrecv(halo_send.data(), 2 * 9 * 4 * V_halo, MPI_DOUBLE, t0, 6, halo_tL.data(), 2 * 9 * 4 * V_halo,
                     MPI_DOUBLE, tL, 6, comm, MPI_STATUS_IGNORE);
        fill_halo_send(3, 1);
        MPI_Sendrecv(halo_send.data(), 2 * 9 * 4 * V_halo, MPI_DOUBLE, tL, 7, halo_t0.data(), 2 * 9 * 4 * V_halo,
                     MPI_DOUBLE, t0, 7, comm, MPI_STATUS_IGNORE);
    }

    //Accès link lattice+halos
    SU3 get_link_at(int x, int y, int z, int t, int mu) const {
        // interne helper pour mapper un halo-vector -> Map<SU3>
        auto map_halo = [&](const std::vector<Complex> &halo_vec, size_t halo_idx, int mu_local) -> SU3 {
            const Complex *ptr = &halo_vec[(halo_idx * 4 + mu_local) * 9];
            return Eigen::Map<const SU3>(ptr);
        };

        // cas normal (site local)
        if (0 <= x && x < L && 0 <= y && y < L && 0 <= z && z < L && 0 <= t && t < L) {
            size_t site = index(x, y, z, t);
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
        for (size_t site = 0; site < V; site++) {
            for (int mu = 0; mu < NDIMS; mu++) {
                res += view_link_const(site, mu).trace().real();
            }
        }
        return res / (static_cast<double>(V) * 4);
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
                                auto U_mu = get_link_at(x, y, z, t, mu);
                                auto U_nu_xmu = get_link_at(xp, yp, zp, tp, nu);
                                auto U_mu_xnu = get_link_at(xq, yq, zq, tq, mu);
                                auto U_nu = get_link_at(x, y, z, t, nu);

                                SU3 pl = U_mu * U_nu_xmu * U_mu_xnu.adjoint() * U_nu.adjoint();
                                sum += pl.trace().real() / 3.0;
                                ++count;
                            }
                        }
                    }
                }
            }
        }
        return sum / static_cast<double>(count);
    }

    //ECMC
    void compute_list_staples(size_t site, int mu, std::array<SU3, 6> &list_staple) const {
        //Calcule la liste des staples autour un lien de jauge
        int index = 0;
        for (int nu = 0; nu < 4; nu++) {
            if (nu == mu) {
                continue;
            }
            size_t x = site; //x
            size_t xmu = neighbors[x][mu][0].value(); //x+mu
            size_t xnu = neighbors[x][nu][0].value(); //x+nu
            size_t xmunu = neighbors[xmu][nu][1].value(); //x+mu-nu
            size_t xmnu = neighbors[x][nu][1].value(); //x-nu
            auto U0 = view_link_const(xmu, nu);
            auto U1 = view_link_const(xnu, mu);
            auto U2 = view_link_const(x, nu);
            list_staple[index] = U0 * U1.adjoint() * U2.adjoint();
            auto V0 = view_link_const(xmunu, nu);
            auto V1 = view_link_const(xmnu, mu);
            auto V2 = view_link_const(xmnu, nu);
            list_staple[index + 1] = V0.adjoint() * V1.adjoint() * V2;
            index += 2;
        }
    }

    void compute_reject_angles(size_t site, int mu, const std::array<SU3, 6> &list_staple, const SU3 &R, int epsilon,
                               const double &beta, std::array<double, 6> &reject_angles, std::mt19937_64 &rng) const {
        //Calcule la liste des 6 angles de rejets pour le lien (site, mu) (1 angle par plaquette associée au lien)
        std::uniform_real_distribution<double> unif(0.0, 1.0);
        for (int i = 0; i < 6; i++) {
            double gamma = -log(unif(rng));

            SU3 P = R.adjoint() * view_link_const(site, mu) * list_staple[i] * R;
            double A = P(0, 0).real() + P(1, 1).real();
            double B = -P(0, 0).imag() + P(1, 1).imag();
            A *= -(beta / 3.0);
            B *= -(beta / 3.0);

            compute_reject(A, B, gamma, reject_angles[i], epsilon);
        }
    }

    bool is_frozen(int x, int y, int z, int t) const {
        //Returns true if the link is frozen, false if not
        if (x == 0 || x == L - 1 || y == 0 || y == L - 1 || z == 0 || z == L - 1 || t == 0 || t == L - 1) {
            return true;
        }
        return false;
    }

    void ecmc_update(size_t site, int mu, double theta, int epsilon, const SU3 &R) {
        SU3 Uold = view_link_const(site, mu);
        view_link(site, mu) = R * el_3(epsilon * theta) * R.adjoint() * Uold;
        //projection_su3(links, site, mu);
    }

    std::pair<std::pair<size_t, int>, int> lift_improved(size_t site, int mu, int j, SU3 &R, const SU3 &lambda_3,
                                                         std::mt19937_64 &rng, const std::vector<SU3> &set) {
        SU3 U0 = view_link_const(site, mu);
        auto links_staple_j = staples[site][mu][j];
        if (!links_staple_j[0].first.has_value() || !links_staple_j[1].first.has_value() || !links_staple_j[2].first.has_value()) {
            std::cerr << "Trying to compute invalid staple\n";
        }
        SU3 U1 = view_link_const(links_staple_j[0].first.value(), links_staple_j[0].second); //Les 3 matrices SU3 associées
        SU3 U2 = view_link_const(links_staple_j[1].first.value(), links_staple_j[1].second);
        SU3 U3 = view_link_const(links_staple_j[2].first.value(), links_staple_j[2].second);

        std::array<std::pair<size_t, int>, 4> links_plaquette_j; //On rajoute le lien actuel
        links_plaquette_j[0] = std::make_pair(site, mu);
        links_plaquette_j[1] = {links_staple_j[0].first.value(), links_staple_j[0].second};
        links_plaquette_j[2] = {links_staple_j[1].first.value(), links_plaquette_j[1].second};
        links_plaquette_j[3] = {links_staple_j[2].first.value(), links_plaquette_j[2].second};
        std::vector<double> probas(4);
        std::vector<double> abs_dS(4);
        double sum = 0.0;
        std::vector<int> sign_dS(4);
        std::vector<SU3> P(4);

        if (j % 2 == 0) {
            //Forward plaquette
            P[0] = U0 * U1 * U2.adjoint() * U3.adjoint();
            P[1] = U1 * U2.adjoint() * U3.adjoint() * U0;
            P[2] = U2 * U1.adjoint() * U0.adjoint() * U3;
            P[3] = U3 * U2 * U1.adjoint() * U0.adjoint();
        } else {
            //Backward plaquette
            P[0] = U0 * U1.adjoint() * U2.adjoint() * U3;
            P[1] = U1 * U0.adjoint() * U3.adjoint() * U2;
            P[2] = U2 * U1 * U0.adjoint() * U3.adjoint();
            P[3] = U3 * U0 * U1.adjoint() * U2.adjoint();
        }
        for (int i = 0; i < 4; i++) {
            if (!frozen[links_plaquette_j[i].first][links_plaquette_j[i].second]) { //Seuls les liens actifs ont une proba non nulle
                probas[i] = -(Complex(0.0, 1.0) * lambda_3 * R.adjoint() * P[i] * R).trace().real();
                sign_dS[i] = dsign(probas[i]);
                probas[i] = abs(probas[i]);
                abs_dS[i] = probas[i];
                sum += probas[i];
            }
            else {
                probas[i] = 0.0;
                sign_dS[i] = 0.0;
                abs_dS[i] = 0.0;
            }
        }
        for (int i = 0; i < 4; i++) {
            probas[i] /= sum;
        }
        int index_lift = selectVariable(probas, rng); //Donne forcément un lien non frozen car les autres ont proba 0.0

        //On change le R
        std::uniform_int_distribution<size_t> distrib(0, set.size() - 1);
        std::uniform_real_distribution<double> uniform_0_1(0, 1);
        SU3 R_new;
        size_t i_set = distrib(rng);
        R_new = set[i_set] * R;
        double dS_j_R = (-Complex(0.0, 1.0) * lambda_3 * R.adjoint() * P[index_lift] * R).trace().real();
        double dS_j_R_new = (-Complex(0.0, 1.0) * lambda_3 * R_new.adjoint() * P[index_lift] * R_new).trace().real();
        double new_epsilon = -sign_dS[index_lift];
        if (abs(dS_j_R) < abs(dS_j_R_new)) {
            R = R_new;
            new_epsilon = -dsign(dS_j_R_new);
        } else {
            double r = uniform_0_1(rng);
            if (r < abs(dS_j_R_new) / abs(dS_j_R)) {
                R = R_new;
                new_epsilon = -dsign(dS_j_R_new);
            }
        }
        return make_pair(links_plaquette_j[index_lift], new_epsilon);
    }

    std::vector<double> ecmc_samples_improved(double beta, int N_samples,
    double param_theta_sample, double param_theta_refresh, std::mt19937_64 &rng, bool poisson, double epsilon_set) {

        if (param_theta_sample<param_theta_refresh) {
            std::cerr << "Wrong args value, must have param_theta_sample>param_theta_refresh \n";
        }
        //Set de matrices pour refresh R
        int N_set = 100;
        size_t lift_counter=0;
        std::vector<SU3> set(N_set+1);
        ecmc_set(epsilon_set, set, rng);

        //Variables aléatoires
        std::uniform_int_distribution<int> random_coord(1, L-2);
        std::uniform_int_distribution<int> random_eps(0,1);
        std::uniform_int_distribution<int> random_dir(0,3);
        std::exponential_distribution<double> random_theta_sample(1.0/param_theta_sample);
        std::exponential_distribution<double> random_theta_refresh(1.0/param_theta_refresh);

        //Matrice lambda_3 de Gell-Mann
        SU3 lambda_3;
        lambda_3 << Complex(1.0,0.0), Complex(0.0,0.0), Complex(0.0,0.0),
                    Complex(0.0,0.0), Complex(-1.0,0.0), Complex(0.0, 0.0),
                    Complex(0.0,0.0), Complex(0.0,0.0), Complex(0.0, 0.0);

        //Initialisation aléatoire de la position de la chaîne
        size_t site_current = index(random_coord(gen), random_coord(gen), random_coord(gen), random_coord(gen));
        int mu_current = random_dir(rng);
        int epsilon_current = 2 * random_eps(rng) -1;

        //Initialisation aléatoire des theta limites pour sample et refresh
        double theta_sample{};
        double theta_refresh{};
        if (poisson) {
            theta_sample = random_theta_sample(rng);
            theta_refresh = random_theta_refresh(rng);
        }
        else {
            theta_sample = param_theta_sample;
            theta_refresh = param_theta_refresh;
        }

        //Initialisation des angles totaux parcourus à 0.0
        double theta_parcouru_sample = 0.0;
        double theta_parcouru_refresh= 0.0;

        //Angle d'update
        double theta_update = 0.0;

        //Arrays utilisés à chaque étape de la chaîne (évite de les initialiser des milliers de fois)
        std::array<double,6> reject_angles = {0.0, 0.0, 0.0, 0.0, 0.0};
        std::array<SU3,6> list_staple;

        SU3 R = random_su3(rng);
        //std::cout << "beta = " << beta << std::endl;

        int samples = 0;
        std::array<double,2> deltas = {0.0,0.0};
        size_t event_counter = 0;
        std::vector<double> meas_plaquette;
        while (samples < N_samples) {
            if (lift_counter%N_set ==0) {
                ecmc_set(epsilon_set, set, rng);
            }
            compute_list_staples(site_current, mu_current, list_staple);
            compute_reject_angles(site_current, mu_current, list_staple, R, epsilon_current,beta,reject_angles,rng);
            auto it = std::ranges::min_element(reject_angles.begin(), reject_angles.end());
            auto j = static_cast<int>(std::ranges::distance(reject_angles.begin(), it)); //theta_reject = reject_angles[j]
            //cout << "Angle reject : " << reject_angles[j] << endl;
            deltas[0] = theta_sample - reject_angles[j] - theta_parcouru_sample;
            deltas[1] = theta_refresh - reject_angles[j] - theta_parcouru_refresh;

            auto it_deltas = std::ranges::min_element(deltas.begin(), deltas.end());
            auto F = static_cast<int>(std::ranges::distance(deltas.begin(), it_deltas));

            if ((deltas[0]<0)&&(deltas[1]<0)) {
                if (F == 0) {
                    //On sample
                    theta_update = theta_sample - theta_parcouru_sample;
                    ecmc_update(site_current, mu_current, theta_update, epsilon_current, R);
                    //std::cout << "Sample " << samples << ", ";
                    double plaq = local_mean_plaquette();
                    //std::cout << "<P> = " << plaq << ", " << event_counter << " events" << std::endl;
                    //cout << "Q = " << topo_charge_clover(links, lat) << endl;
                    event_counter = 0;
                    meas_plaquette.emplace_back(plaq);
                    samples++;
                    theta_parcouru_sample = 0;
                    if (poisson) theta_sample = random_theta_sample(rng); //On retire un nouveau theta_sample
                    theta_parcouru_refresh += theta_update;
                    //On update jusqu'au refresh
                    theta_update = theta_refresh - theta_parcouru_refresh;
                    ecmc_update(site_current, mu_current, theta_update, epsilon_current, R);
                    theta_parcouru_sample += theta_update;
                    theta_parcouru_refresh = 0;
                    if (poisson) theta_refresh = random_theta_refresh(rng); //On retire un nouveau theta refresh
                    //On refresh
                    event_counter++;
                    site_current = index(random_coord(rng), random_coord(rng),random_coord(rng),random_coord(rng));
                    mu_current = random_dir(rng);
                    epsilon_current = 2* random_eps(rng) -1;
                    R = random_su3(rng);
                }
                if (F == 1) {
                    //On update jusqu'au refresh
                    theta_update = theta_refresh - theta_parcouru_refresh;
                    ecmc_update(site_current, mu_current, theta_update, epsilon_current, R);
                    theta_parcouru_sample += theta_update;
                    theta_parcouru_refresh = 0;
                    if (poisson) theta_refresh = random_theta_refresh(rng); //On retire un nouveau theta_refresh
                    //On refresh
                    event_counter++;
                    site_current = index(random_coord(rng), random_coord(rng),random_coord(rng),random_coord(rng));
                    mu_current = random_dir(rng);
                    epsilon_current = 2* random_eps(rng) -1;
                    R = random_su3(rng);
                }
            }
            else if (deltas[F]<0) {
                if (F == 0) {
                    //On update jusqu'a theta_sample
                    theta_update = theta_sample - theta_parcouru_sample;
                    ecmc_update(site_current, mu_current, theta_update, epsilon_current, R);
                    //On sample
                    //std::cout << "Sample " << samples << ", ";
                    double plaq = local_mean_plaquette();
                    //std::cout << "<P> = " << plaq << ", " << event_counter << " events" << std::endl;
                    //cout << "Q = " << topo_charge_clover(links, lat) << endl;
                    event_counter = 0;
                    meas_plaquette.emplace_back(plaq);
                    samples++;
                    theta_parcouru_sample = 0;
                    if (poisson) theta_sample = random_theta_sample(rng); //On retire un nouveau theta_sample
                    theta_parcouru_refresh += theta_update;
                    //On finit l'update et on lift
                    theta_update = -deltas[F];
                    ecmc_update(site_current, mu_current, theta_update, epsilon_current, R);
                    theta_parcouru_sample += theta_update;
                    theta_parcouru_refresh += theta_update;
                    //On lifte
                    event_counter++;
                    auto l = lift_improved(site_current, mu_current, j, R, lambda_3, rng, set);
                    lift_counter++;
                    site_current = l.first.first;
                    mu_current = l.first.second;
                    epsilon_current = l.second;
                }
                if (F==1) {
                    //On update jusqu'à theta_refresh
                    theta_update = theta_refresh - theta_parcouru_refresh;
                    ecmc_update(site_current, mu_current, theta_update, epsilon_current, R);
                    theta_parcouru_sample += theta_update;
                    theta_parcouru_refresh = 0;
                    if (poisson) theta_refresh = random_theta_refresh(rng); //On retire un nouveau theta_refresh
                    //On refresh
                    event_counter++;
                    site_current = index(random_coord(rng), random_coord(rng),random_coord(rng),random_coord(rng));
                    mu_current = random_dir(rng);
                    epsilon_current = 2* random_eps(rng) -1;
                    R = random_su3(rng);
                }
            }
            else {
                //On update
                theta_update = reject_angles[j];
                ecmc_update(site_current, mu_current, theta_update, epsilon_current, R);
                theta_parcouru_sample += theta_update;
                theta_parcouru_refresh += theta_update;
                //On lift
                event_counter++;
                auto l = lift_improved(site_current, mu_current, j, R, lambda_3, rng, set);
                lift_counter++;
                site_current = l.first.first;
                mu_current = l.first.second;
                epsilon_current = l.second;
            }
        }
        return meas_plaquette;
    }

};

void compute_plaquette_loc_glob(int rank, int size, lat4d &lat, MPI_Comm comm) {
    MPI_Barrier(comm);
    int x0, xL;
    MPI_Cart_shift(comm, 0, 1, &x0, &xL);
    int y0, yL;
    MPI_Cart_shift(comm, 1, 1, &y0, &yL);
    int z0, zL;
    MPI_Cart_shift(comm, 2, 1, &z0, &zL);
    int t0, tL;
    MPI_Cart_shift(comm, 3, 1, &t0, &tL);

    lat.fill_halo_obs(x0, xL, y0, yL, z0, zL, t0, tL, comm);
    double l_plaquette = lat.local_mean_plaquette();
    double g_plaquette;
    MPI_Reduce(&l_plaquette, &g_plaquette, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    if (rank == 0) {
        g_plaquette /= size;
        std::cout << "Plaquette moyenne locale : " << lat.local_mean_plaquette() << std::endl;
        std::cout << "Plaquette moyenne globale : " << g_plaquette << std::endl;
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    //Création de la topologie 4d
    int dims[4] = {2, 2, 2, 2};
    int period[4] = {1, 1, 1, 1};
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
    int L = 6;
    lat4d lat(L);
    //lat.hot_start(gen);
    lat.cold_start();
    if (rank == 0) {
        std::cout << "Lattices créées, cold start\n";
    }

    //ECMC params
    double beta = 6.0;
    int N_samples = 3000;
    int param_theta_sample = 100;
    int param_theta_refresh = 20;
    bool poisson = 0;
    double epsilon_set = 0.15;

    //Initial plaquette
    compute_plaquette_loc_glob(rank, size, lat, cart_comm);

    //ECMC
    for (int shift = 0; shift < 3*L; shift++) {
        std::cout << "ECMC...\n";
        lat.ecmc_samples_improved(beta, N_samples, param_theta_sample, param_theta_refresh,gen, poisson, epsilon_set);
        //Calcul plaquette locale et globale
        compute_plaquette_loc_glob(rank, size, lat, cart_comm);
        //Shift
        if (rank == 0) {
            std::cout << "Shifting...\n";
        }
        lat.full_shift_n_all_dirs(1, x0, xL, y0, yL, z0, zL, t0, tL, cart_comm);
    }

    MPI_Finalize();
    return 0;
}