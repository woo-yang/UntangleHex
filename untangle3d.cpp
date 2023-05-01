#include <iostream>
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

template <typename T> T square(T &number) { return number * number; }

void get_bbox(const PointSet &pts, vec3 &min, vec3 &max) {
    min = max = pts[0];
    for (auto const &p : pts) {
        for (int d : range(3)) {
            min[d] = std::min(min[d], p[d]);
            max[d] = std::max(max[d], p[d]);
        }
    }
}

double get_cell_average_edge_size(Volume &m) {
    double sum = 0;
    int nb = 0;
    for (int c : cell_iter(m))
        for (int lf : range(m.nfacets_per_cell()))
            for (int lv : range(m.facet_size(c, lf))) {
                int i = m.facet_vert(c, lf, lv);
                int j = m.facet_vert(c, lf, (lv+1)%m.facet_size(c, lf));
                sum += (m.points[i] - m.points[j]).norm();
                nb++;
            }
    assert(nb > 0);
    return sum / double(nb);
}

double tet_volume(const vec3 &A, const vec3 &B, const vec3 &C, const vec3 &D) {
    return ((A-D)*cross(B-D, C-D))/6.;
}

double tet_volume(const Tetrahedra &m, const int t) {
    return tet_volume(
            m.points[m.vert(t, 0)],
            m.points[m.vert(t, 1)],
            m.points[m.vert(t, 2)],
            m.points[m.vert(t, 3)]
            );
}

inline double chi(double eps, double det) {
    if (det>0)
        return (det + std::sqrt(eps*eps + det*det))*.5;
    return .5*eps*eps / (std::sqrt(eps*eps + det*det) - det);
}

inline double chi_deriv(double eps, double det) {
    return .5+det/(2.*std::sqrt(eps*eps + det*det));
}


struct Untangle3D {

    Untangle3D(Tetrahedra &mesh): m(mesh), X(m.nverts()*3), lock(m.points), J(m), K(m), det(m) 
    {
        setup_x();
        { // prepare the reference tetrahedron
            double volume = 0;
            for (int c : cell_iter(m)) {
                volume += tet_volume(m, c);
            }
            volume /= m.ncells();
            if (debug > 0) std::cerr << "avg volume: " << volume << std::endl;

            Tetrahedra R; // regular tetrahedron with unit edge length, centered at the origin (sqrt(2)/12 volume)
            *R.points.data = {
                { .5,   0, -1. / (2. * std::sqrt(2.))},
                {-.5,   0, -1. / (2. * std::sqrt(2.))},
                {  0,  .5,  1. / (2. * std::sqrt(2.))},
                {  0, -.5,  1. / (2. * std::sqrt(2.))}
            };
            R.cells = { 0,1,2,3 };

            double a = std::cbrt(volume * 6. * std::sqrt(2.));
            for (vec3& p : R.points) // scale the tet
                p = p * a;
           
            ref_tets.resize(1);
            ref_tets[0].resize(4);
            Ref& ref_tet = ref_tets[0];
            for (int lf : range(4)) { // prepare the data for gradient processing: compute the normal vectors
                vec3 e0 = R.points[R.facet_vert(0, lf, 1)] - R.points[R.facet_vert(0, lf, 0)];
                vec3 e1 = R.points[R.facet_vert(0, lf, 2)] - R.points[R.facet_vert(0, lf, 0)];
                ref_tet[lf] = -(cross(e0, e1)/2.)/(3.*volume);
            }
        }

        VolumeConnectivity vec(m);
        for (int c : cell_iter(m))
            for (int lf : range(4))
                if (vec.adjacent[m.facet(c, lf)] < 0)
                    for (int lv : range(3))
                        lock[m.facet_vert(c, lf, lv)] = true;
    }

    Untangle3D(Tetrahedra& mesh, Tetrahedra& ref) 
        : m(mesh), X(m.nverts() * 3), lock(m.points), J(m), K(m), det(m)
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

        for (int i : range(m.nverts())) lock[i] = true;

        VolumeConnectivity vec(m);
        for (int c : cell_iter(m)) {
            double volume = tet_volume(m, c);
            if (tet_volume(m, c) < 0) {
                for (int lf : range(4)) {
                    for (int lv : range(3)) {
                        lock[m.facet_vert(c, lf, lv)] = false;
                    }
                }
            }
        }

    }

    void setup_x() {
        for (int v : vert_iter(m))
            for (int d : range(3))
                X[v*3+d] = m.points[v][d];
    }

    void extract_x() {
        for (int v : vert_iter(m)) {
            vec3 p = { X[v*3+0], X[v*3+1], X[v*3+2] };
            m.points[v] = p;
        }
    }

    void evaluate_jacobian(const std::vector<double> &X) {
        if (debug>3) std::cerr << "evaluate the jacobian...";
        detmin = std::numeric_limits<double>::max();
        ninverted = 0;
#pragma omp parallel for reduction(min:detmin) reduction(+:ninverted)
        for (int c=0; c<m.ncells(); c++) {
            Ref& ref_tet = ref_tets[c% ref_tets.size()];
            mat<3,3> &J = this->J[c];
            J = {};
            for (int i=0; i<4; i++)
                for (int d : range(3))
                    J[d] += ref_tet[i]*X[3*m.vert(c,i) + d];
            det[c] = J.det();
            detmin = std::min(detmin, det[c]);
            ninverted += (det[c]<=0);

            mat<3,3> &K = this->K[c];
            K = { // dual basis
                    cross(J[1],J[2]),
                    cross(J[2],J[0]),
                    cross(J[0],J[1])
            };
        }
        if (debug>3) std::cerr << "ok" << std::endl;
    }

    double evaluate_energy(const std::vector<double> &X) {
        evaluate_jacobian(X);
        double E = 0;
#pragma omp parallel for reduction(+:E)
        for (int c=0; c<m.ncells(); c++) {
            double chi_ = chi(eps, det[c]);
            double f = (J[c][0]*J[c][0] + J[c][1]*J[c][1] + J[c][2]*J[c][2])/pow(chi_, 2./3.);
            double g = (1+square(det[c]))/chi_;
            E += (1-theta)*f + theta*g;
        }
        return E;
    }

    bool go() {
        evaluate_jacobian(X);
#if EPS_FROM_THE_THEOREM
        eps = 1.;
#else
        double e0 = 1e-3;
#endif

        for (int iter=0; iter<maxiter; iter++) {
            if (debug>0) std::cerr << "iteration #" << iter << std::endl;
#if !EPS_FROM_THE_THEOREM
            if (iter && iter%10==0 && e0>1e-8) e0 /= 2.;
            eps = detmin>0 ? e0 : std::sqrt(square(e0) + 0.04*square(detmin)); //(7)
#endif
            if (debug>0) std::cerr << "E: " << evaluate_energy(X) << " eps: " << eps << " detmin: " << detmin << " ninv: " << ninverted << std::endl;

            const hlbfgs_optimizer::simplified_func_grad_eval func = [&](const std::vector<double>& X, double& F, std::vector<double>& G) {
                std::fill(G.begin(), G.end(), 0);
                F = evaluate_energy(X);
#pragma omp parallel for reduction(vec_double_plus:G)
                for (int t=0; t<m.ncells(); t++) {
                    Ref& ref_tet = ref_tets[t % ref_tets.size()];
                    mat<3,3> &a = this->J[t]; // tangent basis
                    mat<3,3> &b = this->K[t]; // dual basis
                    double c1 = chi(eps, det[t]);
                    double c2 = pow(c1, 2./3.);
                    double c3 = chi_deriv(eps, det[t]);

                    double f = (a[0]*a[0] + a[1]*a[1] + a[2]*a[2])/c2;
                    double g = (1+square(det[t]))/c1;

                    for (int dim : range(3)) {
                        vec3 dfda = a[dim]*(2./c2) - b[dim]*((2.*f*c3)/(3.*c1));
                        vec3 dgda = b[dim]*((2*det[t]-g*c3)/c1);

                        for (int i=0; i<4; i++) {
                            int v = m.vert(t,i);
                            if (!lock[v])
                                G[v*3+dim] += (dfda*(1.-theta) + dgda*theta)*ref_tet[i];
                        }
                    }
                }
            };

            double E_prev = evaluate_energy(X);

            hlbfgs_optimizer opt(func);
            opt.set_epsg(bfgs_threshold);
            opt.set_max_iter(bfgs_maxiter);
            opt.set_verbose(true);
            opt.optimize(X);

            double E = evaluate_energy(X);
#if EPS_FROM_THE_THEOREM
            double sigma = std::max(1.-E/E_prev, 1e-1);
            if (detmin>=0)
                eps *= (1-sigma);
            else
                eps *= 1 - (sigma*std::sqrt(square(detmin) + square(eps)))/(std::abs(detmin) + std::sqrt(square(detmin) + square(eps)));
#endif
            if  (detmin>0 && std::abs(E_prev - E)/E<1e-5) break;
        }

        if (debug>0) std::cerr << "E: " << evaluate_energy(X) << " detmin: " << detmin << " ninv: " << ninverted << std::endl;
        extract_x();
        return !ninverted;
    }


    ////////////////////////////////
    // Untangle3D state variables //
    ////////////////////////////////

    // optimization input parameters
    Tetrahedra &m;          // the mesh to optimize
    double theta = 1./2.;   // the energy is (1-theta)*(shape energy) + theta*(area energy)
    int maxiter = 10000;    // max number of outer iterations
    int bfgs_maxiter = 300; // max number of inner iterations
    double bfgs_threshold = .1;

    int debug = 1;          // verbose level
    //vec3 ref_tet[4] = {};   // reference tetrahedron: array of 4 normal vectors to compute the gradients
    using Ref = std::vector<vec3>;
    std::vector<Ref> ref_tets;

    // optimization state variables

    std::vector<double> X;     // current geometry
    PointAttribute<bool> lock; // currently lock = boundary vertices
    CellAttribute<mat<3,3>> J; // per-tet Jacobian matrix = [[JX.x JX.y, JX.z], [JY.x, JY.y, JY.z], [JZ.x, JZ.y, JZ.z]]
    CellAttribute<mat<3,3>> K; // per-tet dual basis: det J = dot J[i] * K[i]
    CellAttribute<double> det; // per-tet determinant of the Jacobian matrix
    double eps;       // regularization parameter, depends on min(jacobian)

    double detmin;    // min(jacobian) over all tetrahedra
    int ninverted; // number of inverted tetrahedra

};


template <size_t N>
class ud_element 
{
public:
    std::vector<int> _ele_id;
    ud_element(const int (&arr) [N]) {
        for (auto it = std::begin(arr); it != std::end(arr); ++it)
            _ele_id.push_back(*it);
        std::sort(_ele_id.begin(), _ele_id.end());
    }
    bool operator<(const ud_element& rhs) const
    {
        return _ele_id < rhs._ele_id;
    }
};

struct BreakHex
{
    void create_tetrahedra(Tetrahedra& m, const std::vector<vec3>& verts, const std::vector<int>& tets)
    {
        m = Tetrahedra();
        m.points.create_points(verts.size());
        for (int v : range(verts.size())) m.points[v] = verts[v];
        m.create_cells(tets.size() / 4);
        for (int t : range(m.ncells())) for (int tv : range(4)) m.vert(t, tv) = tets[4 * t + tv];
    }

    virtual void break_hex_mesh(const Hexahedra& hex_m, Tetrahedra& m) = 0;

    virtual void get_ref_tet_mesh(const Tetrahedra& m,Tetrahedra& ref) = 0;
};

struct BreakHex48 :BreakHex
{
    virtual void break_hex_mesh(const Hexahedra& hex_m, Tetrahedra& m) override
    {
        std::vector<vec3> verts;
        std::vector<int> tets;

        for (int v : range(hex_m.nverts())) verts.push_back(hex_m.points[v]);
        for (int c : cell_iter(hex_m)) {
            vec3 cp(0, 0, 0);
            for (int lv : range(hex_m.nverts_per_cell())) {
                int v = hex_m.vert(c, lv);
                cp += hex_m.points[v];
            }
            cp = cp / hex_m.nverts_per_cell();
            verts.push_back(cp);//body center  begin from hex_m.nverts()
            //cell-n's body center id is hex_m.nverts()+n
        }

        std::map<ud_element<4>, int> fc_map;
        std::map<ud_element<2>, int> ec_map;

        for (int c : cell_iter(hex_m)) {
            int cp_id = hex_m.nverts() + c;
            for (int lf : range(hex_m.nfacets_per_cell())) {
                vec3 fp(0, 0, 0);
                int face_ele[4];
                for (int lv : range(hex_m.facet_size(c, lf))) {
                    int v = hex_m.facet_vert(c, lf, lv);
                    fp += hex_m.points[v];
                    face_ele[lv] = v;
                }
                fp = fp / 4;
                ud_element ud_face{ face_ele };
                auto iter = fc_map.find(ud_face);
                if (iter == fc_map.end()) {
                    verts.push_back(fp);//face center 
                    iter = fc_map.insert({ ud_face, verts.size() - 1 }).first;
                }
                int fp_id = iter->second;

                for (int lv : range(hex_m.facet_size(c, lf))) {
                    int v1 = hex_m.facet_vert(c, lf, lv);
                    int v2 = hex_m.facet_vert(c, lf, (lv + 1) % hex_m.facet_size(c, lf));
                    vec3 ep = (hex_m.points[v1] + hex_m.points[v2]) / 2;
                    int edge_ele[2] = { v1,v2 };
                    ud_element ud_edge{ edge_ele };
                    auto iter = ec_map.find(ud_edge);
                    if (iter == ec_map.end()) {
                        verts.push_back(ep);
                        iter = ec_map.insert({ ud_edge, verts.size() - 1 }).first;
                    }
                    int ep_id = iter->second;
                    tets.insert(tets.end(), { fp_id , ep_id, v1, cp_id });
                    tets.insert(tets.end(), { fp_id , v2, ep_id,cp_id });
                    
                }
            }
        }

        create_tetrahedra(m, verts, tets);
        write_by_extension("../mesh/break.vtk", m, empty_attr(m));
    }

    virtual void get_ref_tet_mesh(const Tetrahedra& m , Tetrahedra& ref) override
    {

        double volume = 0;
        for (int c : cell_iter(m)) {
            volume += tet_volume(m, c);
        }

        volume /= m.ncells();
        double a = std::cbrt(volume * 6.);

        *ref.points.data = {
            { 0, 0, 0},
            { 0, 0, a},
            { a,-a, 0},
            { a, 0, 0},
            { a, a, 0}
        };
        ref.cells = { 0,3,2,1,0,4,3,1 };

    }

};


struct UntangleHex {

    UntangleHex(Hexahedra& hex_m, BreakHex* break_hex = new BreakHex48()) 
        :_hex_m(hex_m), _break(break_hex) 
    {
        scale();  
    }

    bool go(){

        Tetrahedra m;
        _break->break_hex_mesh(_hex_m, m);
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
        _break->get_ref_tet_mesh(m,ref_m);

        Untangle3D opt(m,ref_m);
        bool success = opt.go();
        if (!success) return false;
        //update hex mesh
        for (int i : range(_hex_m.nverts())) {
            _hex_m.points[i] = m.points[i];
        }
        restore_scale();
        return true; 
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
    BreakHex* _break;

    vec3 bbmin, bbmax; // these are used to undo the scaling we apply to the model
    const double boxsize = 10.;
    const double shrink = 1.3;
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

    BreakHex48 break_hex;
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



