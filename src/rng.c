/* rng.c — PCG32 (M.E. O'Neill, "PCG: A Family of Simple Fast Space-Efficient
 * Statistically Good Algorithms for Random Number Generation", 2014).
 * Same seed ⇒ same sequence on every platform, unlike rand(). */
#include "internal.h"

#include <math.h>

uint32_t nmz_rng_u32(nmz_rng *r) {
    uint64_t old = r->state;
    uint32_t xorshifted, rot;
    r->state = old * 6364136223846793005ULL + r->inc;
    xorshifted = (uint32_t)(((old >> 18u) ^ old) >> 27u);
    rot = (uint32_t)(old >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((0u - rot) & 31u));
}

void nmz_rng_seed(nmz_rng *r, uint64_t seed, uint64_t stream) {
    r->state = 0u;
    r->inc = (stream << 1u) | 1u;
    (void)nmz_rng_u32(r);
    r->state += seed;
    (void)nmz_rng_u32(r);
}

/* Lemire's nearly divisionless method: uniform in [0, bound) without the
 * modulo bias of `rand() % n`. */
uint32_t nmz_rng_below(nmz_rng *r, uint32_t bound) {
    uint64_t m;
    uint32_t l;
    if (bound <= 1) return 0;
    m = (uint64_t)nmz_rng_u32(r) * bound;
    l = (uint32_t)m;
    if (l < bound) {
        uint32_t t = (0u - bound) % bound;
        while (l < t) {
            m = (uint64_t)nmz_rng_u32(r) * bound;
            l = (uint32_t)m;
        }
    }
    return (uint32_t)(m >> 32);
}

/* 53 random bits → double in [0, 1). */
double nmz_rng_uniform(nmz_rng *r) {
    uint32_t a = nmz_rng_u32(r) >> 5, b = nmz_rng_u32(r) >> 6;
    return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
}

/* Box–Muller transform. */
double nmz_rng_normal(nmz_rng *r) {
    const double two_pi = 6.283185307179586476925286766559;
    double u1 = 1.0 - nmz_rng_uniform(r); /* (0, 1] so log() is finite */
    double u2 = nmz_rng_uniform(r);
    return sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
}
