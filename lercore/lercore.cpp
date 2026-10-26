#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <new>
#include <random>
#include <utility>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(_OPENMP) && _OPENMP >= 201307
#define SIMD _Pragma("omp simd")
#else
#define SIMD
#endif

namespace py = pybind11;
using arr = py::array_t<double, py::array::c_style | py::array::forcecast>;
using real = float;

constexpr size_t LINE = 64;
constexpr int LANE = LINE / sizeof(real);

struct Buf {
    real* p = nullptr;
    size_t n = 0;
    ~Buf() { release(); }
    void release() {
        if (p) ::operator delete[](p, std::align_val_t(LINE));
        p = nullptr;
        n = 0;
    }
    real* get(size_t m) {
        if (m > n) {
            release();
            p = static_cast<real*>(::operator new[](m * sizeof(real), std::align_val_t(LINE)));
            n = m;
        }
        return p;
    }
};

struct Work {
    Buf a, b, cQ, act, ring, pad;
    void release() { a.release(); b.release(); cQ.release(); act.release(); ring.release(); pad.release(); }
};
static thread_local Work work;

struct P {
    int nx, ny, ld, n_steps;
    double dx, dy, V, c_pag, c_q, kd, t_dev, R_min, R_max, a_dev, n_dev, pitch;
    real dt;
};

struct Kernel {
    std::vector<real> x, y;
    int rx, ry;
    Kernel(const arr& kx, const arr& ky)
        : x(kx.data(), kx.data() + kx.size()), y(ky.data(), ky.data() + ky.size()),
          rx((int)kx.size() / 2), ry((int)ky.size() / 2) {}
};

#ifdef _OPENMP
static int thread_id() { return omp_get_thread_num(); }
static int thread_count() { return omp_get_num_threads(); }
#else
static int thread_id() { return 0; }
static int thread_count() { return 1; }
#endif

struct Rng {
    uint64_t s[4];
    using result_type = uint64_t;
    static constexpr uint64_t min() { return 0; }
    static constexpr uint64_t max() { return ~0ULL; }
    static uint64_t splitmix(uint64_t& x) {
        uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    Rng(uint64_t seed, uint64_t a, uint64_t b) {
        uint64_t x = seed ^ (a * 0xD1B54A32D192ED03ULL) ^ (b * 0xABC98388FB8FAC03ULL);
        for (auto& v : s) v = splitmix(x);
    }
    static uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    uint64_t operator()() {
        const uint64_t r = rotl(s[1] * 5, 7) * 9, t = s[1] << 17;
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = rotl(s[3], 45);
        return r;
    }
    double u() { return ((*this)() >> 11) * 0x1.0p-53; }
};

struct Poisson {
    double mu, e;
    std::poisson_distribution<int> large;
    explicit Poisson(double m) : mu(m), e(std::exp(-m)), large(m > 0 ? m : 1) {}
    int operator()(Rng& g) {
        if (mu <= 0) return 0;
        if (mu > 30) return large(g);
        double u = g.u(), p = e, c = p;
        int k = 0;
        while (u > c && k < 200) { ++k; p *= mu / k; c += p; }
        return k;
    }
};

static int binom(int n, double p, Rng& g) {
    if (n <= 0 || p <= 0) return 0;
    if (n > 64) return std::binomial_distribution<int>(n, p)(g);
    int k = 0;
    for (int i = 0; i < n; ++i) k += g.u() < p;
    return k;
}

static void conv_x(int nx, const real* a, real* b, real* pad, const real* k, int r) {
    for (int i = 0; i < r; ++i) {
        pad[i] = a[((nx - r + i) % nx + nx) % nx];
        pad[nx + r + i] = a[i % nx];
    }
    std::memcpy(pad + r, a, nx * sizeof(real));
    std::fill(b, b + nx, real(0));
    for (int j = -r; j <= r; ++j) {
        const real w = k[j + r];
        const real* s = pad + r - j;
SIMD
        for (int x = 0; x < nx; ++x) b[x] += w * s[x];
    }
}

static void peb_step(const P& p, const real* in, real* out, real* cQ, real* act, const Kernel& K, real w, bool par) {
    const int rx = K.rx, ry = K.ry;
#pragma omp parallel if (par)
    {
        const int nt = thread_count(), id = thread_id();
        const int y0 = (int)((long long)p.ny * id / nt), y1 = (int)((long long)p.ny * (id + 1) / nt);
        const int R = 2 * ry + 1;
        real* ring = work.ring.get((size_t)R * p.ld);
        real* pad = work.pad.get(p.nx + 2 * rx + LANE);
        auto row = [&](int t) { return ((t % p.ny) + p.ny) % p.ny; };
        auto slot = [&](int t) { return ring + (size_t)(((t % R) + R) % R) * p.ld; };
        for (int t = y0 - ry; t < y0 + ry && y0 < y1; ++t)
            conv_x(p.nx, in + (size_t)row(t) * p.ld, slot(t), pad, K.x.data(), rx);
        for (int y = y0; y < y1; ++y) {
            conv_x(p.nx, in + (size_t)row(y + ry) * p.ld, slot(y + ry), pad, K.x.data(), rx);
            real* b = out + (size_t)y * p.ld;
            std::fill(b, b + p.nx, real(0));
            for (int j = -ry; j <= ry; ++j) {
                const real k = K.y[j + ry];
                const real* a = slot(y - j);
SIMD
                for (int x = 0; x < p.nx; ++x) b[x] += k * a[x];
            }
            if (w == 0) {
SIMD
                for (int x = 0; x < p.nx; ++x) b[x] = std::max(b[x], real(0));
                continue;
            }
            real* q = cQ + (size_t)y * p.ld;
            real* s = act + (size_t)y * p.ld;
            const real wdt = w * p.dt;
SIMD
            for (int x = 0; x < p.nx; ++x) {
                const real v = std::max(b[x], real(0)), m = std::min(v, q[x]);
                b[x] = v - m;
                q[x] -= m;
                s[x] += b[x] * wdt;
            }
        }
    }
}

static double rate(double d, const P& p) {
    d = std::clamp(d, 0.0, 1.0);
    const double dn = d > 0 ? std::exp(p.n_dev * std::log(d)) : 0.0;
    return p.R_min + (p.R_max - p.R_min) * (p.a_dev + 1) * dn / (p.a_dev + dn);
}

static double godunov(double a, double b, double ha, double hb) {
    const double t1 = std::min(a + ha, b + hb);
    if (t1 <= std::max(a, b)) return t1;
    const double wa = 1 / (ha * ha), wb = 1 / (hb * hb), s = wa + wb, m = (wa * a + wb * b) / s;
    const double disc = m * m - (wa * a * a + wb * b * b - 1) / s;
    return disc < 0 ? t1 : std::min(t1, m + std::sqrt(disc));
}

static void develop(const P& p, const real* d, double* xl, double* xr, uint8_t* lost, bool par) {
    const int nx = p.nx, ny = p.ny, NB = std::min(16, ny);
    static thread_local std::vector<double> Tb, fb, Hb;
    Tb.resize((size_t)nx * ny);
    fb.resize((size_t)nx * ny);
    Hb.resize((size_t)2 * NB * nx);
    double *T = Tb.data(), *f = fb.data(), *H = Hb.data();
    const double cap = p.t_dev, tol = 1e-9 * cap, hx = 0.5 * p.dx, hy = 0.5 * p.dy;

#pragma omp parallel for if (par) schedule(static)
    for (int y = 0; y < ny; ++y) {
        double* g = f + (size_t)y * nx;
        double* t = T + (size_t)y * nx;
        for (int x = 0; x < nx; ++x) g[x] = 1.0 / rate(d[(size_t)y * p.ld + x], p);
        t[0] = hx * g[0];
        for (int x = 1; x < nx / 2; ++x) t[x] = t[x - 1] + hx * (g[x - 1] + g[x]);
        t[nx - 1] = hx * g[nx - 1];
        for (int x = nx - 2; x >= nx / 2; --x) t[x] = t[x + 1] + hx * (g[x + 1] + g[x]);
    }

    auto band = [&](int b) { return (int)((long long)ny * b / NB); };
    for (int it = 0; it < 500; ++it) {
        for (int b = 0; b < NB; ++b) {
            const int up = (band(b) + ny - 1) % ny, dn = band(b + 1) % ny;
            std::memcpy(H + (size_t)(2 * b) * nx, T + (size_t)up * nx, nx * sizeof(double));
            std::memcpy(H + (size_t)(2 * b + 1) * nx, T + (size_t)dn * nx, nx * sizeof(double));
        }
        double moved[16] = {0};
#pragma omp parallel for if (par) schedule(static)
        for (int b = 0; b < NB; ++b) {
            const int ya = band(b), yb = band(b + 1);
            const double *hu = H + (size_t)(2 * b) * nx, *hd = H + (size_t)(2 * b + 1) * nx;
            const double *fu = f + (size_t)((ya + ny - 1) % ny) * nx, *fd = f + (size_t)(yb % ny) * nx;
            double mv = 0;
            auto update = [&](int y, int x) {
                const size_t k = (size_t)y * nx + x;
                const double* Tu = y > ya ? T + k - nx - x : hu;
                const double* Td = y + 1 < yb ? T + k + nx - x : hd;
                const double* gu = y > ya ? f + k - nx - x : fu;
                const double* gd = y + 1 < yb ? f + k + nx - x : fd;
                const double l = T[k - 1], r = T[k + 1], tu = Tu[x], td = Td[x];
                const double a = std::min(l, r), b = std::min(tu, td);
                if (std::min(a, b) >= cap) return;
                const double fx = l <= r ? f[k - 1] : f[k + 1], fy = tu <= td ? gu[x] : gd[x];
                const double v = godunov(a, b, hx * (f[k] + fx), hy * (f[k] + fy));
                if (v >= T[k]) return;
                mv = std::max(mv, std::min(T[k], cap) - std::min(v, cap));
                T[k] = v;
            };
            for (int y = ya; y < yb; ++y)
                for (int x = 1; x < nx - 1; ++x) update(y, x);
            for (int y = ya; y < yb; ++y)
                for (int x = nx - 2; x > 0; --x) update(y, x);
            for (int y = yb - 1; y >= ya; --y)
                for (int x = nx - 2; x > 0; --x) update(y, x);
            for (int y = yb - 1; y >= ya; --y)
                for (int x = 1; x < nx - 1; ++x) update(y, x);
            moved[b] = mv;
        }
        if (*std::max_element(moved, moved + NB) <= tol) break;
    }

    const int c = nx / 2;
#pragma omp parallel for if (par) schedule(static)
    for (int y = 0; y < ny; ++y) {
        const double* t = T + (size_t)y * nx;
        const double* g = f + (size_t)y * nx;
        if (t[c - 1] <= p.t_dev || t[c] <= p.t_dev) {
            xl[y] = xr[y] = 0;
            lost[y] = 1;
            continue;
        }
        auto place = [&](int i, int k, double sgn) {
            const double e = p.t_dev - t[i];
            const double s = e < hx * g[i] ? e / g[i] : hx + (e - hx * g[i]) / g[k];
            return (-p.pitch / 2 + (i + 0.5) * p.dx) + sgn * s;
        };
        int i = c - 1, k = c;
        while (i >= 0 && t[i] > p.t_dev) --i;
        while (k < nx && t[k] > p.t_dev) ++k;
        i = std::max(i, 0);
        k = std::min(k, nx - 1);
        xl[y] = place(i, i + 1, 1.0);
        xr[y] = place(k, k - 1, -1.0);
        lost[y] = 0;
    }
}

static void realization(const P& p, const real* pG, const Kernel& K, const Kernel& K0, uint64_t seed, uint64_t idx,
                        bool par, double* xl, double* xr, uint8_t* lost, float* d_out, double* acid_out) {
    const size_t n = (size_t)p.ld * p.ny;
    real *cH = work.a.get(n), *nxt = work.b.get(n), *cQ = work.cQ.get(n), *act = work.act.get(n);
    const double iV = 1.0 / p.V;

#pragma omp parallel for if (par) schedule(static)
    for (int y = 0; y < p.ny; ++y) {
        Rng g(seed, idx, (uint64_t)y);
        Poisson pag(p.c_pag * p.V), quencher(p.c_q * p.V);
        real *h = cH + (size_t)y * p.ld, *q = cQ + (size_t)y * p.ld;
        for (int x = 0; x < p.nx; ++x) {
            h[x] = real(binom(pag(g), pG[x], g) * iV);
            q[x] = real(quencher(g) * iV);
        }
    }

    if (K0.rx > 0 || K0.ry > 0) {
        peb_step(p, cH, nxt, cQ, act, K0, 0, par);
        std::swap(cH, nxt);
        peb_step(p, cQ, nxt, nullptr, nullptr, K0, 0, par);
        std::swap(cQ, nxt);
    }

    if (acid_out) {
        std::fill(acid_out, acid_out + p.nx, 0.0);
        for (int y = 0; y < p.ny; ++y)
            for (int x = 0; x < p.nx; ++x) acid_out[x] += cH[(size_t)y * p.ld + x];
        for (int x = 0; x < p.nx; ++x) acid_out[x] /= p.ny;
    }

#pragma omp parallel for if (par) schedule(static)
    for (int y = 0; y < p.ny; ++y) {
        real *h = cH + (size_t)y * p.ld, *q = cQ + (size_t)y * p.ld, *s = act + (size_t)y * p.ld;
        for (int x = 0; x < p.nx; ++x) {
            const real m = std::min(h[x], q[x]);
            h[x] -= m;
            q[x] -= m;
            s[x] = p.n_steps > 0 ? real(0.5) * h[x] * p.dt : real(0);
        }
    }
    for (int s = 0; s < p.n_steps; ++s) {
        peb_step(p, cH, nxt, cQ, act, K, s + 1 < p.n_steps ? real(1) : real(0.5), par);
        std::swap(cH, nxt);
    }

#pragma omp parallel for if (par) schedule(static)
    for (int y = 0; y < p.ny; ++y) {
        real* a = act + (size_t)y * p.ld;
        for (int x = 0; x < p.nx; ++x) a[x] = real(-std::expm1(-p.kd * a[x]));
        if (d_out) std::memcpy(d_out + (size_t)y * p.nx, a, p.nx * sizeof(real));
    }
    develop(p, act, xl, xr, lost, par);
}

static std::atomic<int> g_threads{0};

void set_threads(int n) { g_threads = std::max(1, n); }

int threads() {
#ifdef _OPENMP
    const int n = g_threads.load();
    return n > 0 ? n : omp_get_max_threads();
#else
    return 1;
#endif
}

py::tuple run(arr pG_, arr kx, arr ky, arr kx0, arr ky0, int ny, int n_real, uint64_t seed, py::dict q, bool keep_map) {
    auto num = [&](const char* k) { return q[k].cast<double>(); };
    P p{};
    p.nx = (int)pG_.shape(pG_.ndim() - 1);
    p.ny = ny;
    p.ld = (p.nx + LANE - 1) / LANE * LANE;
    p.n_steps = q["n_steps"].cast<int>();
    p.dx = num("dx");
    p.dy = num("dy");
    p.V = num("V");
    p.c_pag = num("c_pag");
    p.c_q = num("c_q");
    p.dt = (real)num("dt");
    p.kd = num("kd");
    p.t_dev = num("t_dev");
    p.R_min = num("R_min");
    p.R_max = num("R_max");
    p.a_dev = num("a_dev");
    p.n_dev = num("n_dev");
    p.pitch = num("pitch");

    const int modes = (int)(pG_.size() / p.nx), tasks = modes * n_real;
    const std::vector<real> pG(pG_.data(), pG_.data() + pG_.size());
    const Kernel K(kx, ky), K0(kx0, ky0);

    py::array_t<double> XL({tasks, ny}), XR({tasks, ny}), A(p.nx);
    py::array_t<float> D(keep_map ? std::vector<py::ssize_t>{ny, p.nx} : std::vector<py::ssize_t>{0, 0});
    py::array_t<uint8_t> LOST({tasks, ny});
    double *xl = XL.mutable_data(), *xr = XR.mutable_data(), *acid = A.mutable_data();
    float* d = keep_map ? D.mutable_data() : nullptr;
    uint8_t* lost = LOST.mutable_data();

    auto go = [&](int t, bool par, float* dm, double* am) {
        const int j = t / n_real, i = t % n_real;
        realization(p, pG.data() + (size_t)j * p.nx, K, K0, seed, (uint64_t)i, par,
                    xl + (size_t)t * ny, xr + (size_t)t * ny, lost + (size_t)t * ny, dm, am);
    };
    {
        py::gil_scoped_release nogil;
#ifdef _OPENMP
        if (g_threads.load() > 0) omp_set_num_threads(g_threads.load());
#endif
        go(0, true, d, acid);
#pragma omp parallel for schedule(dynamic)
        for (int t = 1; t < tasks; ++t) go(t, false, nullptr, nullptr);
    }
    return py::make_tuple(XL, XR, LOST, D, A);
}

void release() {
#pragma omp parallel
    work.release();
    work.release();
}

PYBIND11_MODULE(_lercore, m) {
    m.def("run", &run, py::arg("pG"), py::arg("kx"), py::arg("ky"), py::arg("kx0"), py::arg("ky0"), py::arg("ny"),
          py::arg("n_real"), py::arg("seed"), py::arg("q"), py::arg("keep_map") = true);
    m.def("release", &release);
    m.def("threads", &threads);
    m.def("set_threads", &set_threads, py::arg("n"));
}
