//
// Created by ozdalkiran-l on 11/28/25.
//

#include <iostream>
#include <vector>
#include <span>
#include <array>
#include <mpi.h>
#include <optional>
#include <string>
#include <format>

struct lat2d_mpi {
    int L;
    int V_active;
    std::vector<int> links;
    std::span<int> active;
    std::span<int> frozen_left;
    std::span<int> frozen_right;
    std::span<int> frozen_bottom;
    std::span<int> frozen_top;
    std::span<int> left_bottom;
    std::span<int> left_top;
    std::span<int> right_bottom;
    std::span<int> right_top;
    std::span<int> halo;
    std::vector<std::array<std::array<std::optional<size_t>, 2>, 2> > neighbors;
    //neighbors[site][mu][dir] = site voisin en +mu si dir =0, -mu si dir =1

    explicit lat2d_mpi(int L_) {
        L = L_; //Taille de la lattice avec zones frozen
        V_active = (L - 2) * (L - 2); //Volume de la lattice de liens actifs
        links.resize(2 * V_active + 4 * 2 * (L - 2) + 4 * 2 + 2 * L);
        //Liens actifs + 4 frozen de taille 2*(L-2) + 4 coins de taille 2 + 1 halo de taille L
        size_t offset = 0;
        active = std::span(links.data() + offset, 2 * V_active);
        offset += 2 * V_active;
        frozen_left = std::span(links.data() + offset, 2 * (L - 2));
        offset += 2 * (L - 2);
        frozen_right = std::span(links.data() + offset, 2 * (L - 2));
        offset += 2 * (L - 2);
        frozen_bottom = std::span(links.data() + offset, 2 * (L - 2));
        offset += 2 * (L - 2);
        frozen_top = std::span(links.data() + offset, 2 * (L - 2));
        offset += 2 * (L - 2);
        left_bottom = std::span(links.data() + offset, 2);
        offset += 2;
        left_top = std::span(links.data() + offset, 2);
        offset += 2;
        right_bottom = std::span(links.data() + offset, 2);
        offset += 2;
        right_top = std::span(links.data() + offset, 2);
        offset += 2;
        halo = std::span(links.data() + offset, 2 * L);

        neighbors.resize(L * L);

        for (int x = 1; x < L - 1; x++) {
            for (int y = 1; y < L - 1; y++) {
                //On initialise tous les voisins possibles
                size_t site = index(x, y);
                if (x + 1 <= L - 1) neighbors[site][0][0] = index(x + 1, y);
                if (x - 1 >= 0) neighbors[site][0][1] = index(x - 1, y);
                if (y + 1 <= L - 1) neighbors[site][1][0] = index(x, y + 1);
                if (y - 1 >= 0) neighbors[site][1][1] = index(x, y - 1);
            }
        }
    }

    [[nodiscard]] int get_link(size_t site, int mu) const {
        return links[2 * site + mu];
    }

    int get_halo(int c, int mu) const {
        return halo[2 * c + mu];
    }

    void set_link(size_t site, int mu, int value) {
        links[2 * site + mu] = value;
    }

    [[nodiscard]] size_t index(int x, int y) const {
        if (x >= 1 && y >= 1 && x <= L - 2 && y <= L - 2) {
            //Le site est dans les liens actifs
            int xi = x - 1;
            int yi = y - 1;
            return xi + (L - 2) * yi;
        }
        if (x == 0 && y >= 1 && y <= L - 2) {
            //Le site est dans frozen_left
            return V_active + y - 1;
        }
        if (x == L - 1 && y >= 1 && y <= L - 2) {
            //Le site est dans frozen_right
            return V_active + L - 2 + y - 1;
        }
        if (y == 0 && x >= 1 && x <= L - 2) {
            //Le site est dans frozen_bottom
            return V_active + 2 * (L - 2) + x - 1;
        }
        if (y == L - 1 && x >= 1 && x <= L - 2) {
            //Le site est dans frozen_up
            return V_active + 3 * (L - 2) + x - 1;
        }
        if (x == 0 && y == 0) {
            return V_active + 4 * (L - 2);
        }
        if (x == 0 && y == L - 1) {
            return V_active + 4 * (L - 2) + 1;
        }
        if (x == L - 1 && y == 0) {
            return V_active + 4 * (L - 2) + 2;
        }
        if (x == L - 1 && y == L - 1) {
            return V_active + 4 * (L - 2) + 3;
        }
    }

    std::string print_links() {
        std::string s;
        for (int y = L - 1; y >= 0; --y) {
            for (int x = 0; x < L; ++x) {
                size_t site = index(x, y);
                s += "(" + std::format("{:02}", get_link(site, 0)) + "," + std::format("{:02}", get_link(site, 1)) +
                        ") ";
            }
            s += '\n';
        }
        return s;
    }

    void fill_unif(int value) {
        //Remplir tout le vector links (sauf halo) avec la valeur value
        for (int x = 0; x < L; x++) {
            for (int y = 0; y < L; y++) {
                size_t site = index(x, y);
                set_link(site, 0, value);
                set_link(site, 1, value);
            }
        }
    }

    void hfill_shift_right(int left, int right, MPI_Comm comm) {
        //On remplit le halo dans le cas d'un shift vers la droite
        MPI_Sendrecv(frozen_right.data(), 2 * (L - 2), MPI_INT, right, 0, halo.data() + 2, 2 * (L - 2), MPI_INT, left,
                     0, comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(right_bottom.data(), 2, MPI_INT, right, 1, halo.data(), 2, MPI_INT, left, 1, comm,
                     MPI_STATUS_IGNORE);
        MPI_Sendrecv(right_top.data(), 2, MPI_INT, right, 2, halo.data() + 2 * (L - 1), 2, MPI_INT, left, 2, comm,
                     MPI_STATUS_IGNORE);
    }

    void shift_right() {
        //Le halo est rempli, on shift
        for (int x = L - 1; x > 0; x--) {
            for (int y = 0; y < L; y++) {
                size_t site = index(x, y);
                size_t site_copy = index(x - 1, y);
                set_link(site, 0, get_link(site_copy, 0));
                set_link(site, 1, get_link(site_copy, 1));
            }
        }
        //Il ne manque plus que la colonne de gauche
        for (int y = 0; y < L; y++) {
            size_t site = index(0, y);
            set_link(site, 0, get_halo(y, 0));
            set_link(site, 1, get_halo(y, 1));
        }
    }
};

int main(int argc, char *argv[]) {
    //Initialisation MPI
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    //Création communicateur MPI 2D
    int dims[2] = {3, 3}; // Dimensions de la grille
    int periods[2] = {1, 1}; // Périodicité en x et y (0 = non périodique, 1 = périodique)
    int reorder = 1; // Autorise MPI à réorganiser les rangs pour optimiser
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD,
                    2, // dimension de la topologie
                    dims,
                    periods,
                    reorder,
                    &cart_comm);
    int cart_rank;
    MPI_Comm_rank(cart_comm, &cart_rank);
    int left, right, up, down;
    MPI_Cart_shift(cart_comm, 0, 1, &left, &right); // déplacement selon l’axe 0
    MPI_Cart_shift(cart_comm, 1, 1, &down, &up); // déplacement selon l’axe 1
    int coords[2];
    MPI_Cart_coords(cart_comm, rank, 2, coords);
    int coords_left[2];
    MPI_Cart_coords(cart_comm, left, 2, coords_left);

    //Création lattice et initialisation au rank
    lat2d_mpi lat(5);
    lat.fill_unif(rank);
    if (rank == 1) {
        std::cout << "Rank : " << rank << ", Coords : " << coords[0] << coords[1] << ", Rank_left : " << left << ", Coords_left : " << coords_left[0] << coords_left[1] << std::endl;
        std::cout << lat.print_links() << std::endl;
    }
    lat.hfill_shift_right(left, right, cart_comm);
    lat.shift_right();
    if (rank == 1) {
        std::cout << "Shift vers la droite :" << std::endl;
        std::cout << lat.print_links() << std::endl;
    }


    //Fermeture MPI
    MPI_Finalize();
    return 0;
}
