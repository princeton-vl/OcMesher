// Copyright (c) Princeton University.
// This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

// Authors: Zeyu Ma

#include <cstdio>
#include <vector>
#include <set>
#include <map>
#include <queue>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#define mp std::make_pair
#define pair std::pair
#define max std::max
#define min std::min
#define INT_MAX 2147483647
#define cubex(x) (x) * (x) * (x)
#define cube_index(x, y, z, s) (x)*(s)*(s)+(y)*(s)+(z)
#define xpp(p) p.first
#define ypp(p) p.second.first
#define zpp(p) p.second.second
#define vector std::vector
#define set std::set
#define map std::map
#define queue std::queue
#define cube_to_key(cube) mp(cube.coords[0], mp(cube.coords[1], mp(cube.coords[2], cube.L)))
#define key_to_cube(c, key) {c.coords[0] = (key).first; c.coords[1] = (key).second.first; c.coords[2] = (key).second.second.first; c.L = (key).second.second.second; }
#define assign(x, y, a, b) {assert((y) < (INT_MAX-(max(0,b)))/(a)); x=(y)*(a)+(b);}
#define leaf_node(node) (node.nxts[0] <= -2)
#define mark_leaf_node(node) node.nxts[0] = -2
#define mark_grid_node(node, gl) node.nxts[0] = -2-gl
#define grid_node_level(node) (-node.nxts[0] - 2)
#define is_regular(cube) (cube.L >= 0)
#define is_boundary(cube) (cube.L == -1)
#define is_exterior(cube) (cube.L == -2)
#define mark_boundary(cube) cube.L = -1
#define mark_exterior(cube) cube.L = -2
#define add_faces(faces, v1, v2, v3) {assert(v1 != v2 && v2 != v3 && v3 != v1); faces.push_back(v1); faces.push_back(v2); faces.push_back(v3);}
#define first_digit(j) (j&1)
#define second_digit(j) ((j>>1)&1)
#define make_int3(x,y,z) mp(x, mp(y, z))
typedef double T;
typedef float sdfT;

struct computed_vertex {
    T c[3], l, r;
};

struct cube {
    int coords[3], L;
};

typedef cube vertex;
typedef pair<int, pair<int, int> > int3;
typedef pair<int, pair<int, pair<int, int> > > key_cube;
typedef pair<int, key_cube> key_edge;


struct node {
    cube c;
    int nxts[8];
};

int int_log(T x) {
    return int(max(T(0), (T)ceil(log2(x))));
}

namespace params {
    int n_cams, memory_limit_mb, coarse_count, n_elements;
    T *center, *cams;
    T size, pixels_per_cube, occ_scale, min_dist;
}

void enumerate_vertices(vertex *v, node n) {
    assert(leaf_node(n));
    int s=grid_node_level(n), ss = 1<<s;
    for (int i = 0; i <= ss; i++)
    for (int j = 0; j <= ss; j++)
    for (int k = 0; k <= ss; k++) {
        int vid = i+(ss+1)*j+(ss+1)*(ss+1)*k;
        for (int p = 0; p < 3; p++) v[vid].coords[p] = n.c.coords[p]*ss + (p==0?i:(p==1?j:k));
        v[vid].L = n.c.L + s;
        for (;;) {
            if (v[vid].L == 0) break;
            bool flag = 1;
            for (int p = 0; p < 3; p++)
                if (v[vid].coords[p]&1 != 0) {
                    flag = 0;
                    break;
                }
            if (!flag) break;
            for (int p = 0; p < 3; p++) v[vid].coords[p] >>= 1;
            v[vid].L--;
        }
    }
}

void compute_coords(T *coords, int *icoords, int L) {
    for (int j = 0; j < 3; j++)
        coords[j] = params::center[j] - params::size / 2 + params::size * icoords[j] / (1 << L);
}

void compute_center(T *coords, cube v) {
    for (int j = 0; j < 3; j++)
        coords[j] = params::center[j] - params::size / 2 + params::size * (v.coords[j] + 0.5) / (1 << v.L);
}

void projected_coords(cube c, int k, T *icoords, T *r) {
    using namespace params;
    T Pw[3], Pc[3];
    T *current_cam = cams + k * (12 + 9 + 2);
    compute_center(Pw, c);
    for (int i = 0; i < 3; i++) {
        Pc[i] = current_cam[i * 4 + 3];
        for (int j = 0; j < 3; j++) {
            Pc[i] += Pw[j] * current_cam[i * 4 + j];
        }
    }
    if (r != NULL) {
        *r = sqrt(Pc[0] * Pc[0] + Pc[1] * Pc[1] + Pc[2] * Pc[2]);
        *r = max(*r, min_dist);
    }
    if (icoords != NULL) {
        for (int i = 0; i < 3; i++) {
            icoords[i] = 0;
            for (int j = 0; j < 3; j++) {
                icoords[i] += Pc[j] * current_cam[12 + i * 3 + j];
            }
        }
        icoords[0] /= icoords[2];
        icoords[1] /= icoords[2];
    }
}

T projected_size(cube c, int k) {
    using namespace params;
    T *current_cam = cams + k * (12 + 9 + 2);
    T r;
    projected_coords(c, k, NULL, &r);
    T W = current_cam[22];
    T pix_ang = atan(W / 2 / current_cam[12]) * 2 / W;
    T ang = pix_ang * pixels_per_cube;
    return size / (1 << c.L) / r / ang;
}

T projected_size(cube c) {
    T max_size = 0;
    for (int k = 0; k < params::n_cams; k++) {
        T size_k = projected_size(c, k);
        max_size = max(max_size, size_k);
    }
    return max_size;
}

void expand_octree(vector<node> &nodes, int index) {
    assert(leaf_node(nodes[index]));
    int base_size = nodes.size();
    for (int i = 0; i < 8; i++) {
        node *current = &nodes[index];
        current->nxts[i] = nodes.size();
        node node0;
        mark_leaf_node(node0);
        node0.c.L = current->c.L + 1;
        for (int k = 0; k < 3; k++) {
            int offset = (i>>k) & 1;
            assign(node0.c.coords[k], current->c.coords[k], 2, offset);
        }
        nodes.push_back(node0);
    }
}

void partial_expand_octree(vector<node> &nodes, int index, int instance) {
    int base_size = nodes.size();
    assert(!leaf_node(nodes[index]));
    for (int i = 0; i < 8; i++) {
        node *current = &nodes[index];
        if ((instance>>i) & 1) {
            if (current->nxts[i] == -1) {
                current->nxts[i] = nodes.size();
                node node0;
                memset(node0.nxts, -1, 8 * sizeof(int));
                node0.c.L = current->c.L + 1;
                for (int k = 0; k < 3; k++) {
                    int offset = (i>>k) & 1;
                    assign(node0.c.coords[k], current->c.coords[k], 2, offset);
                }
                nodes.push_back(node0);
            }
        }
    }
}

cube search(node *nodes, int *coords, int L, bool include_exterior=0) {
    assert(L > 0);
    cube res;
    for (int i = 0; i < 3; i++)
        if (!(coords[i] > 0 && coords[i] < 1<<L)) {
            mark_boundary(res);
            return res;
        }
    node current = nodes[0];
    int current_id = 0;
    for (int l = 0; l < L - 1; l++) {
        if (leaf_node(current)) break;
        for (int i = 0; i < 3; i++)
            if (coords[i] & ((1<<(L-current.c.L-1))-1) == 0) {
                mark_boundary(res);
                return res;
            }
        int nxt = 0;
        for (int p = 0; p < 3; p++)
            if (2 * current.c.coords[p] + 1 <= (coords[p] >> (L - current.c.L - 1))) nxt += (1 << p);
        current_id = current.nxts[nxt];
        if (current_id == -1) {
            if (!include_exterior) {
                mark_exterior(res);
            }
            else {
                for (int k = 0; k < 3; k++) {
                    int offset = (nxt>>k) & 1;
                    assign(res.coords[k], current.c.coords[k], 2, offset);
                }
                res.L = current.c.L + 1;
            }
            return res;
        }
        current = nodes[current_id];
    }
    if (!leaf_node(current)) {
        mark_boundary(res);
        return res;
    }
    int gl = grid_node_level(current);
    if (L <= current.c.L + gl) {
        mark_boundary(res);
        return res;
    }
    for (int i = 0; i < 3; i++)
        if (coords[i] & ((1<<(L-current.c.L-gl))-1) == 0) {
            mark_boundary(res);
            return res;
        }
    res.L = current.c.L + gl;
    for (int p = 0; p < 3; p++) res.coords[p] = coords[p] >> (L - res.L);
    return res;
}

int divide_to_cube(vector<node> &nodes, cube c) {
    node current = nodes[0];
    int current_id = 0;
    for (;;) {
        assert(!leaf_node(current));
        bool flag = 1;
        for (int p = 0; p < 3; p++) flag &= current.c.coords[p] == c.coords[p];
        if (flag && current.c.L == c.L) {
            mark_leaf_node(nodes[current_id]);
            return current_id;
        }
        int nxt = 0;
        for (int p = 0; p < 3; p++)
            if (2 * current.c.coords[p] < c.coords[p] >> (c.L - current.c.L - 1)) nxt += (1 << p);
        partial_expand_octree(nodes, current_id, 1<<nxt);
        current = nodes[current_id];
        current_id = current.nxts[nxt];
        current = nodes[current_id];
    }
}

void find_edges(node n, map<key_cube, int> &vertices, sdfT *sdf, vector<vector<key_edge> > &bipolar_edges) {
    int s = grid_node_level(n), ss=1<<s;
    vertex v[cubex(ss+1)];
    sdfT *sdf_v[cubex(ss+1)];
    enumerate_vertices(v, n);
    for (int i = 0; i < cubex(ss+1); i++) {
        sdf_v[i] = sdf + vertices[cube_to_key(v[i])] * params::n_elements;
    }
    for (int edir = 0; edir < 3; edir++)
    for (int i = 0; i < ss; i++)
    for (int j = 0; j <= ss; j++)
    for (int k = 0; k <= ss; k++) {
        int coords[3];
        coords[edir] = i + 1;
        coords[(edir + 1)%3] = j;
        coords[(edir + 2)%3] = k;
        int vid = coords[0] + coords[1]*(ss+1) + coords[2]*(ss+1)*(ss+1);
        sdfT *sdf1 = sdf_v[vid];
        coords[edir]--;
        vid = coords[0] + coords[1]*(ss+1) + coords[2]*(ss+1)*(ss+1);
        sdfT *sdf2 = sdf_v[vid];
        sdfT sdf1_min=std::numeric_limits<sdfT>::infinity(), sdf2_min=std::numeric_limits<sdfT>::infinity();
        for (int e = 0; e < params::n_elements; e++) {
            sdf1_min = min(sdf1[e], sdf1_min);
            sdf2_min = min(sdf2[e], sdf2_min);
            if ((sdf1[e] >= 0) != (sdf2[e] >= 0)) {
                cube c0;
                for (int p = 0; p < 3; p++)
                    assign(c0.coords[p], n.c.coords[p], ss, coords[p]);
                c0.L = n.c.L + s;
                int dir = edir + 1;
                if (sdf1[e] < 0) dir *= -1;
                bipolar_edges[e].push_back(mp(dir, cube_to_key(c0)));
            }
        }
        if ((sdf1_min >= 0) != (sdf2_min >= 0)) {
            cube c0;
            for (int p = 0; p < 3; p++)
                assign(c0.coords[p], n.c.coords[p], ss, coords[p]);
            c0.L = n.c.L + s;
            int dir = edir + 1;
            if (sdf1_min < 0) dir *= -1;
            bipolar_edges[params::n_elements].push_back(mp(dir, cube_to_key(c0)));
        }
    }
}

int compute_boundary(cube c, cube bound) {
    int b[3][2];
    for (int i = 0; i < 3; i++)
    for (int p = 0; p < 2; p++) {
        b[i][p] = c.coords[i] + p == (bound.coords[i]+p) << (c.L-bound.L);
    }
    int instance=0;
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 3; j++)
            if (b[j][(i>>j)&1]) {
                instance |= 1 << i;
                break;
            }
    }
    return instance;
}

inline T det(T matrix[3][3]) {
    T a = matrix[0][0];
    T b = matrix[0][1];
    T c = matrix[0][2];
    T d = matrix[1][0];
    T e = matrix[1][1];
    T f = matrix[1][2];
    T g = matrix[2][0];
    T h = matrix[2][1];
    T i = matrix[2][2];
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
}

bool tri_seg_intersect(T *t1, T *t2, T *t3, T *s1, T *s2) {
    // note (t1,t3) of the tri is allowed to intersect
    T m[3][3];
    for (int i = 0; i < 3; i++) {
        m[0][i] = t1[i] - s1[i];
        m[1][i] = t2[i] - s1[i];
        m[2][i] = t3[i] - s1[i];
    }
    T det1 = det(m);
    for (int i = 0; i < 3; i++) {
        m[0][i] = t1[i] - s2[i];
        m[1][i] = t2[i] - s2[i];
        m[2][i] = t3[i] - s2[i];
    }
    T det2 = det(m);
    if (!(det1 > 0 && det2 < 0 || det1 < 0 && det2 > 0)) return 0;

    for (int i = 0; i < 3; i++) {
        m[0][i] = t1[i] - s1[i];
        m[1][i] = t2[i] - s1[i];
        m[2][i] = s2[i] - s1[i];
    }
    det1 = det(m);
    for (int i = 0; i < 3; i++) {
        m[0][i] = t2[i] - s1[i];
        m[1][i] = t3[i] - s1[i];
    }
    det2 = det(m);
    if (!(det1 > 0 && det2 > 0 || det1 < 0 && det2 < 0)) return 0;
    for (int i = 0; i < 3; i++) {
        m[0][i] = t3[i] - s1[i];
        m[1][i] = t1[i] - s1[i];
    }
    det1 = det(m);
    if (!(det1 >= 0 && det2 > 0 || det1 <= 0 && det2 < 0)) return 0;
    return 1;
}
