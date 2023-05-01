#include <iostream>
#include <vector>
#include <limits>
#include <cassert>
#include <algorithm>
#include <array>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <map>

#include <ultimaille/all.h>
#pragma omp declare reduction(vec_double_plus : std::vector<double> : \
        std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<double>())) \
        initializer(omp_priv = std::vector<double>(omp_orig.size(), 0))

using namespace UM;

#define EPS_FROM_THE_THEOREM 0

template <typename T> T square(T& number) { return number * number; }

using Ref = std::vector<vec3>;

void get_bbox(const PointSet& pts, vec3& min, vec3& max) {
    min = max = pts[0];
    for (auto const& p : pts) {
        for (int d : range(3)) {
            min[d] = std::min(min[d], p[d]);
            max[d] = std::max(max[d], p[d]);
        }
    }
}


double tet_volume(const vec3& A, const vec3& B, const vec3& C, const vec3& D) {
    return ((A - D) * cross(B - D, C - D)) / 6.;
}

double tet_volume(const Tetrahedra& m, const int t) {
    return tet_volume(
        m.points[m.vert(t, 0)],
        m.points[m.vert(t, 1)],
        m.points[m.vert(t, 2)],
        m.points[m.vert(t, 3)]
    );
}

void create_tetrahedra(Tetrahedra& m, const std::vector<vec3>& verts, const std::vector<int>& tets)
{
    m = Tetrahedra();
    m.points.create_points(verts.size());
    for (int v : range(verts.size())) m.points[v] = verts[v];
    m.create_cells(tets.size() / 4);
    for (int t : range(m.ncells())) for (int tv : range(4)) m.vert(t, tv) = tets[4 * t + tv];
}

inline double chi(double eps, double det) {
    if (det > 0)
        return (det + std::sqrt(eps * eps + det * det)) * .5;
    return .5 * eps * eps / (std::sqrt(eps * eps + det * det) - det);
}

inline double chi_deriv(double eps, double det) {
    return .5 + det / (2. * std::sqrt(eps * eps + det * det));
}


struct Untangle58 {

    Untangle58(Tetrahedra& mesh, Tetrahedra& ref, PointAttribute<bool>& l)
        : m(mesh), X(m.nverts() * 3), X_o(m.nverts() * 3), lock(l), J(m), K(m), det(m)
    {
        setup_x();
        { // prepare the reference tetrahedron

            ref_tets.resize(ref.ncells());
            for (auto& ref_tet : ref_tets)ref_tet.resize(4);

            for (int c : cell_iter(ref)) {
                Ref& ref_tet = ref_tets[c];
                double v = tet_volume(ref, c);
                for (int lf : range(4)) { // prepare the data for gradient processing: compute the normal vectors
                    vec3 e0 = ref.points[ref.facet_vert(c, lf, 1)] - ref.points[ref.facet_vert(c, lf, 0)];
                    vec3 e1 = ref.points[ref.facet_vert(c, lf, 2)] - ref.points[ref.facet_vert(c, lf, 0)];
                    ref_tet[lf] = -(cross(e0, e1) / 2.) / (3. * v);
                }
            }

        }
        set_fast_eps(false);

    }
    void set_fast_eps(bool b) {
        fast_update_eps = b;
        VolumeConnectivity vec(m);
        for (int c : cell_iter(m))
            for (int lf : range(4))
                if (vec.adjacent[m.facet(c, lf)] < 0)
                    for (int lv : range(3))
                        lock[m.facet_vert(c, lf, lv)] = b;
    }


    void setup_x() {
        for (int v : vert_iter(m))
            for (int d : range(3))
                X_o[v * 3 + d] = X[v * 3 + d] = m.points[v][d];
    }

    void extract_x() {
        for (int v : vert_iter(m)) {
            vec3 p = { X[v * 3 + 0], X[v * 3 + 1], X[v * 3 + 2] };
            m.points[v] = p;
        }
    }


    void evaluate_jacobian(const std::vector<double>& X) {
        if (debug > 3) std::cerr << "evaluate the jacobian...";
        detmin = std::numeric_limits<double>::max();
        ninverted = 0;
#pragma omp parallel for reduction(min:detmin) reduction(+:ninverted)
        for (int c = 0; c < m.ncells(); c++) {
            Ref& ref_tet = ref_tets[c % ref_tets.size()];
            mat<3, 3>& J = this->J[c];
            J = {};
            for (int i = 0; i < 4; i++)
                for (int d : range(3))
                    J[d] += ref_tet[i] * X[3 * m.vert(c, i) + d];
            det[c] = J.det();
            detmin = std::min(detmin, det[c]);
            ninverted += (det[c] <= 0);

            mat<3, 3>& K = this->K[c];
            K = { // dual basis
                {{
                     J[1].y * J[2].z - J[1].z * J[2].y,
                     J[1].z * J[2].x - J[1].x * J[2].z,
                     J[1].x * J[2].y - J[1].y * J[2].x
                 },
                {
                    J[0].z * J[2].y - J[0].y * J[2].z,
                    J[0].x * J[2].z - J[0].z * J[2].x,
                    J[0].y * J[2].x - J[0].x * J[2].y
                },
                {
                    J[0].y * J[1].z - J[0].z * J[1].y,
                    J[0].z * J[1].x - J[0].x * J[1].z,
                    J[0].x * J[1].y - J[0].y * J[1].x
                }}
            };
        }
        if (debug > 3) std::cerr << "ok" << std::endl;
    }

    double evaluate_energy(const std::vector<double>& X) {
        evaluate_jacobian(X);
        double E = 0;
#pragma omp parallel for reduction(+:E)
        for (int c = 0; c < m.ncells(); c++) {
            double chi_ = chi(eps, det[c]);
            double f = (J[c][0] * J[c][0] + J[c][1] * J[c][1] + J[c][2] * J[c][2]) / pow(chi_, 2. / 3.);
            E += f;

        }
        if (!fast_update_eps) {
            std::vector<double> v(X.size(), 0);
            for (int i : range(v.size())) v[i] = X[i] - X_o[i];
            for (auto i : v) E += 1e6 * (i * i);
        }
        return E;
    }

    bool go() {
        evaluate_jacobian(X);

        eps = 1.;

        for (int iter = 0; iter < maxiter; iter++) {
            if (debug > 0) {
                std::cerr << "iteration #" << iter << std::endl;
                std::cerr << "E: " << evaluate_energy(X) << " eps: " << eps << " detmin: " << detmin << " ninv: " << ninverted << std::endl;
            }

            const hlbfgs_optimizer::simplified_func_grad_eval func = [&](const std::vector<double>& X, double& F, std::vector<double>& G) {
                std::fill(G.begin(), G.end(), 0);
                F = evaluate_energy(X);
#pragma omp parallel for reduction(vec_double_plus:G)
                for (int t = 0; t < m.ncells(); t++) {
                    Ref& ref_tet = ref_tets[t % ref_tets.size()];
                    mat<3, 3>& a = this->J[t]; // tangent basis
                    mat<3, 3>& b = this->K[t]; // dual basis
                    double c1 = chi(eps, det[t]);
                    double c2 = pow(c1, 2. / 3.);
                    double c3 = chi_deriv(eps, det[t]);

                    double f = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]) / c2;

                    for (int dim : range(3)) {
                        vec3 dfda = a[dim] * (2. / c2) - b[dim] * ((2. * f * c3) / (3. * c1));

                        for (int i = 0; i < 4; i++) {
                            int v = m.vert(t, i);
                            if (!lock[v])
                                G[v * 3 + dim] += dfda * ref_tet[i];

                        }
                    }
                }
                if (!fast_update_eps) {
                    std::vector<double> v(X.size());
                    for (int i : range(v.size())) v[i] = X[i] - X_o[i];
                    for (int i : range(G.size())) G[i] += 2e6 * v[i];
                }
            };

            double E_prev = evaluate_energy(X);
            double detmin_pre = detmin;

            hlbfgs_optimizer opt(func);
            opt.set_epsg(bfgs_threshold);
            opt.set_max_iter(bfgs_maxiter);
            opt.set_verbose(true);
            opt.optimize(X);

            double E = evaluate_energy(X);

            if ((detmin < 0) && (eps > std::fabs(detmin)) && ((detmin - detmin_pre) / fabs(detmin_pre) > 1e-4)) {
                if (!fast_update_eps) set_fast_eps(true);
                eps = -0.2 * detmin;
            }
            else {
                if (fast_update_eps) set_fast_eps(false);
                double sigma = std::max(1. - E / E_prev, 1e-1);
                double u = (1 - sigma) * chi(eps, detmin);
                if (detmin >= u) eps = 0;
                else eps = 2 * sqrt(u * (u - detmin));
            }

            if (detmin > 0 && std::abs(E_prev - E) / E < 1e-5) break;
        }

        if (debug > 0) std::cerr << "E: " << evaluate_energy(X) << " detmin: " << detmin << " ninv: " << ninverted << std::endl;
        extract_x();
        return !ninverted;
    }


    ////////////////////////////////
    // Untangle3D state variables //
    ////////////////////////////////

    // optimization input parameters
    Tetrahedra& m;          // the mesh to optimize
    int maxiter = 1000;    // max number of outer iterations
    int bfgs_maxiter = 100; // max number of inner iterations
    double bfgs_threshold = .1;

    int debug = 1;          // verbose level
    std::vector<Ref> ref_tets;

    // optimization state variables

    std::vector<double> X, X_o;     // current geometry
    PointAttribute<bool>& lock; // currently lock = boundary vertices
    CellAttribute<mat<3, 3>> J; // per-tet Jacobian matrix = [[JX.x JX.y, JX.z], [JY.x, JY.y, JY.z], [JZ.x, JZ.y, JZ.z]]
    CellAttribute<mat<3, 3>> K; // per-tet dual basis: det J = dot J[i] * K[i]
    CellAttribute<double> det; // per-tet determinant of the Jacobian matrix
    double eps;       // regularization parameter, depends on min(jacobian)
    bool fast_update_eps = false;

    double detmin;    // min(jacobian) over all tetrahedra
    int ninverted; // number of inverted tetrahedra

};


struct BreakHex
{

    virtual void break_hex_mesh(const Hexahedra& hex_m, Tetrahedra& m) = 0;

    virtual void break_hex_mesh(const Hexahedra& hex_m, int c, std::vector<int>& tets)=0;

    virtual void get_ref_tet_mesh(const Tetrahedra& m, Tetrahedra& ref) = 0;
};

struct BreakHex8 :BreakHex
{
    virtual void break_hex_mesh(const Hexahedra& hex_m, Tetrahedra& m) override
    {
        std::vector<vec3> verts;
        std::vector<int> tets;

        for (int v : range(hex_m.nverts())) verts.push_back(hex_m.points[v]);

        for (int c : cell_iter(hex_m)) {
            break_hex_mesh(hex_m, c, tets);
        }

        create_tetrahedra(m, verts, tets);
        write_by_extension("../mesh/break.vtk", m, empty_attr(m));
    }

    virtual void break_hex_mesh(const Hexahedra& hex_m, int c, std::vector<int>& tets) override
    {
        std::vector<int> diff;
        constexpr int bottom = 4, top = 5;
        for (int lv : range(4)) {
            diff.push_back(hex_m.facet_vert(c, bottom, lv) - hex_m.facet_vert(c, top, 3 - lv));
        }
        for (int lv : range(4)) {
            int v0 = hex_m.facet_vert(c, bottom, lv);
            int v1 = hex_m.facet_vert(c, bottom, (lv + 1) % 4);
            int v2 = hex_m.facet_vert(c, bottom, (lv + 2) % 4);
            tets.insert(tets.end(), { v0, v2 , v1, v1 - diff[(lv + 1) % 4] });
        }
        for (int lv : range(4)) {
            int v0 = hex_m.facet_vert(c, top, lv);
            int v1 = hex_m.facet_vert(c, top, (lv + 1) % 4);
            int v2 = hex_m.facet_vert(c, top, (lv + 2) % 4);
            tets.insert(tets.end(), { v0, v2 , v1, v1 + diff[3 - ((lv + 1) % 4)] });
        }
    }

    virtual void get_ref_tet_mesh(const Tetrahedra& m, Tetrahedra& ref) override
    {
        double volume = 0;
        for (int c : cell_iter(m)) {
            volume += tet_volume(m, c);
        }

        volume /= m.ncells();
        double a = std::cbrt(volume * 6.);

        *ref.points.data = {
            { 0, 0, 0},
            { a, 0, 0},
            { 0, a, 0},
            { 0, 0, a}
        };
        ref.cells = { 0,2,1,3 };
    }

};


struct BreakHex58 :BreakHex
{
    void gen_nonzero_tet_index(std::vector<int>& tets) {
        for (int i = 0; i < 8; ++i) {
            for (int j = i + 1; j < 8; ++j) {
                for (int k = j + 1; k < 8; ++k) {
                    for (int l = k + 1; l < 8; ++l) {
                        double volume = tet_volume(ref_hex[i], ref_hex[j], ref_hex[k], ref_hex[l]);
                        if (fabs(volume) < 1e-9) continue;
                        if (volume < 0) tets.insert(tets.end(), { l ,j ,k ,i });
                        else tets.insert(tets.end(), { i,j,k,l });
                    }
                }
            }
        }
    }
    virtual void break_hex_mesh(const Hexahedra& hex_m, int c, std::vector<int>& tets) override
    {
        static std::vector<int> tets_li;
        if (tets_li.empty()) {
            gen_nonzero_tet_index(tets_li);
        }

        for (int i : tets_li) {
            tets.push_back(hex_m.vert(c, vtk2geo[i]));
        }
        
    }
    virtual void break_hex_mesh(const Hexahedra& hex_m, Tetrahedra& m) override
    {

        std::vector<vec3> verts;
        std::vector<int> tets;

        for (int v : range(hex_m.nverts()))
            verts.push_back(hex_m.points[v]);


        for (int c : cell_iter(hex_m)) {
            break_hex_mesh(hex_m, c, tets);
        }

        create_tetrahedra(m, verts, tets);
        write_by_extension("../mesh/break.vtk", m, empty_attr(m));
    }

    virtual void get_ref_tet_mesh(const Tetrahedra& m, Tetrahedra& ref) override
    {
        Hexahedra temp;
        *temp.points.data = ref_hex;
        temp.cells = vtk2geo;
        break_hex_mesh(temp, ref);

        double volume = 0;
        for (int c : cell_iter(m)) {
            volume += tet_volume(m, c);
        }

        volume /= (m.ncells() / ref.ncells());

        double a = std::cbrt(volume / 10.);
        for (vec3& p : ref.points) // scale the tet
            p = p * a;

    }

    const std::vector<int> vtk2geo = { 0,1,3,2,4,5,7,6 };
    const std::vector<vec3> ref_hex = { {0,0,0},{1,0,0},{1,1,0},{0,1,0},{0,0,1},{1,0,1},{1,1,1}, {0,1,1} };

};


struct UntangleHex {

    UntangleHex(Hexahedra& hex_m, BreakHex* break_hex = new BreakHex58())
        :_hex_m(hex_m), _break(break_hex), lock(hex_m.points), lock_cell(hex_m)
    {
        scale();

        VolumeConnectivity connect(hex_m);
        for (int v : range(hex_m.nverts())) lock[v] = true;
        for (int c : range(hex_m.ncells())) lock_cell[c] = true;

        for (int c : cell_iter(hex_m)) {
            if (evaluate_cell_validity(_hex_m, c))continue;
            n_invalid++;
            //construct 2 layers blob
            for (int lv : range(_hex_m.nverts_per_cell())) {
                lock[_hex_m.vert(c, lv)] = false;
            }
            lock_cell[c] = false;
            for (int lf : range(_hex_m.nfacets_per_cell())) {
                int adj_f = connect.adjacent[_hex_m.facet(c, lf)];
                if (adj_f < 0) continue;
                int adj_c = _hex_m.cell_from_facet(adj_f);

                lock_cell[adj_c] = false;

                for (int adj_lv : range(_hex_m.nverts_per_cell())) {
                    lock[_hex_m.vert(adj_c, adj_lv)] = false;
                }
                for (int adj_lf : range(_hex_m.nfacets_per_cell())) {
                    int adj_f_2 = connect.adjacent[_hex_m.facet(adj_c, adj_lf)];
                    int adj_c_2 = _hex_m.cell_from_facet(adj_f_2);
                    lock_cell[adj_c_2] = false;
                }
            }
        }
        std::cerr << "n_invalid: " << n_invalid  << std::endl;

    }

    bool go() {

        Tetrahedra m;
        std::vector<vec3> verts;
        std::vector<int> tets;
        for (int v : range(_hex_m.nverts())) verts.push_back(_hex_m.points[v]);

        for (int c : cell_iter(_hex_m)) {
            if (lock_cell[c])continue;
            _break->break_hex_mesh(_hex_m, c, tets);
        }
        create_tetrahedra(m, verts, tets);

        { // ascertain the mesh requirements
            double volume = 0;
            for (int c : cell_iter(m))
                volume += tet_volume(m, c);
            volume /= m.ncells();
            if (volume <= 0) {
                std::cerr << "Error: the input mesh must have positive volume" << std::endl;
                return false;
            }
        }

        Tetrahedra ref_m;
        _break->get_ref_tet_mesh(m, ref_m);

        Untangle58 opt(m, ref_m,lock);
        bool success = opt.go();
        if (!success) return false;
        //update hex mesh
        for (int i : range(_hex_m.nverts())) {
            _hex_m.points[i] = m.points[i];
        }
        restore_scale();
        return true;
    }

    bool evaluate_cell_validity(const Hexahedra& hex_m, int c)
    {
        Tetrahedra m;
        std::vector<vec3> verts;
        std::vector<int> tets;

        for (int v : range(hex_m.nverts()))
            verts.push_back(hex_m.points[v]);

        _break->break_hex_mesh(hex_m, c, tets);
        create_tetrahedra(m, verts, tets);

        Tetrahedra ref_m;
        _break->get_ref_tet_mesh(m, ref_m);

        std::vector<Ref> ref_tets;
        for (int c : cell_iter(ref_m)) {
            Ref ref_tet(4);
            double v = tet_volume(ref_m, c);
            for (int lf : range(4)) { // prepare the data for gradient processing: compute the normal vectors
                vec3 e0 = ref_m.points[ref_m.facet_vert(c, lf, 1)] - ref_m.points[ref_m.facet_vert(c, lf, 0)];
                vec3 e1 = ref_m.points[ref_m.facet_vert(c, lf, 2)] - ref_m.points[ref_m.facet_vert(c, lf, 0)];
                ref_tet[lf] = -(cross(e0, e1) / 2.) / (3. * v);
            }
            ref_tets.push_back(ref_tet);
        }

        double detmin_58 = std::numeric_limits<double>::max();
        for (int c = 0; c < m.ncells(); c++) {
            Ref& ref_tet = ref_tets[c];
            mat<3, 3> J = {};
            for (int i = 0; i < 4; i++)
                for (int d : range(3))
                    J[d] += ref_tet[i] * hex_m.points[tets[4 * c + i]][d];
            detmin_58 = std::min(detmin_58, J.det());

        }
        //std::cout << detmin_58 << std::endl;
        return detmin_58 > 0;
    }


    // normalize the mesh, place it well inside the [0,boxside]^2 square (max size will be boxside/shrink)
    void scale() {
        get_bbox(_hex_m.points, bbmin, bbmax);
        double maxside = std::max(bbmax.x - bbmin.x, bbmax.y - bbmin.y);
        for (vec3& p : _hex_m.points)
            p = (p - (bbmax + bbmin) / 2.) * boxsize / (shrink * maxside) + vec3(1, 1, 1) * boxsize / 2;
    }

    void restore_scale() {
        double maxside = std::max(bbmax.x - bbmin.x, bbmax.y - bbmin.y);
        for (int v : vert_iter(_hex_m)) {
            vec3 p = _hex_m.points[v];
            _hex_m.points[v] = (p - vec3(1, 1, 1) * boxsize / 2) * shrink / boxsize * maxside + (bbmax + bbmin) / 2.;
        }
    }

    Hexahedra& _hex_m;
    PointAttribute<bool> lock;
    CellAttribute<bool> lock_cell;
    BreakHex* _break;

    vec3 bbmin, bbmax; // these are used to undo the scaling we apply to the model
    const double boxsize = 10.;
    const double shrink = 1.3;

    int n_invalid = 0;
};


int main(int argc, char** argv) {
    if (2 > argc) {
        std::cerr << "Usage: " << argv[0] << " model.mesh [result.mesh]" << std::endl;
        return 1;
    }
    std::string res_filename = "result.mesh";
    if (3 <= argc) {
        res_filename = std::string(argv[2]);
    }

    Hexahedra hex_m;
    read_by_extension(argv[1], hex_m);
    std::cerr << "Untangling " << argv[1] << "," << hex_m.nverts() << "," << std::endl;

    BreakHex58 break_hex;
    UntangleHex opt_hex(hex_m, &break_hex);

    auto t1 = std::chrono::high_resolution_clock::now();
    bool success = opt_hex.go();
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time = t2 - t1;

    if (success)
        std::cerr << "SUCCESS; running time: " << time.count() << std::endl;
    else
        std::cerr << "FAIL TO UNTANGLE!" << std::endl;
    //write_by_extension(res_filename, hex_m, VolumeAttributes{ { {"selection", opt.lock.ptr} }, { {"det", opt.det.ptr} }, {}, {} });
    write_by_extension(res_filename, hex_m, empty_attr(hex_m));
    return 0;

}



