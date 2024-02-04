// Copyright (c) Princeton University.
// This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

// Authors: Zeyu Ma

#include "core.h"

namespace coarse {
    vector<node> nodes;
    std::priority_queue<pair<T, int> > nodes_heap;
    vector<int> nodes_vector;
}

namespace fine {
    int start_node, end_node;
    vector<pair<int, int3> > cubes_queue;
    vector<bool> cubes;
    vector<int> cubes_index;
    vector<int> vertices;
    vector<int> vertices_index;
    vector<vertex> output_vertices;
    vector<int> output_vertices_index;
}

namespace solid {
    vector<cube> cubes;
    set<key_cube> cubes_set;
    set<key_cube> visible_set, occluded_set;
}

namespace final {
    queue<int> new_nodes;
    vector<vertex> v;
    vector<cube> visible_nodes_cube;
    vector<int> occluded_nodes_id;
    int gl, start_node, end_node, size0;
    map<key_cube, int> vertices;
    vector<int> bipolar_edges_s;
    vector<vector<key_edge> > bipolar_edges;
    vector<vector<int> > bipolar_edges_vindices;
    map<pair<int, key_cube>, int> bipolar_edges_vertices;
    vector<int> vertices_cnt;
    vector<vector<key_cube> > bipolar_edges_vertices_vector;
    vector<vector<computed_vertex> > bipolar_edges_computed_vertices;
    vector<node> nodes;
    vector<vector<bool> > in_view_tag;
    vector<cube> searched;
}

namespace computing {
    vector<int> faces;
    vector<pair<int, computed_vertex> > edge_vertices;
    vector<bool> edge_vertices_in_view_tag;
    vector<pair<int, computed_vertex> > face_vertices;
    vector<bool> face_vertices_in_view_tag;
    map<pair<int, int>, int> face_vertices_map;
}

extern "C" {
    int run_coarse(
        T *center, T size,
        int n_cams, T *cams,
        T pixels_per_cube,
        T occ_scale,
        T min_dist,
        int coarse_count,
        int memory_limit_mb,
        int n_elements
    ) {
        using namespace coarse;
        params::center = center;
        params::size = size;
        params::n_cams = n_cams;
        params::cams = cams;
        params::pixels_per_cube = pixels_per_cube;
        params::occ_scale = occ_scale;
        params::min_dist = min_dist;
        params::coarse_count = coarse_count;
        params::memory_limit_mb = memory_limit_mb;
        params::n_elements = n_elements;
        node root;
        mark_leaf_node(root);
        memset(root.c.coords, 0, 3 * sizeof(int));
        root.c.L = 0;
        nodes.clear();
        nodes.push_back(root);
        nodes_heap = std::priority_queue<pair<T, int> >();
        nodes_heap.push(mp(projected_size(root.c), 0));

        int t = 0;
        while (!nodes_heap.empty() && nodes.size() < coarse_count) {
            pair<T, int> top = nodes_heap.top();
            if (top.first < occ_scale) break;
            nodes_heap.pop();
            int i0 = nodes.size();
            expand_octree(nodes, top.second);
            for (int i = 0; i < 8; i++) nodes_heap.push(mp(projected_size(nodes[i0 + i].c), i0 + i));
        }
        fine::end_node = 0;
        solid::cubes.clear();
        solid::cubes_set.clear();
        nodes_vector.clear();
        while (!nodes_heap.empty()) {
            pair<T, int> top = nodes_heap.top();
            mark_grid_node(nodes[top.second], max(0, int_log(top.first / params::occ_scale)));
            nodes_vector.push_back(top.second);
            nodes_heap.pop();
        }
        return nodes_vector.size();
    }

    int fine_group() {
        using namespace coarse;
        using namespace fine;
        cubes_queue.clear();
        cubes_index.clear();
        vertices_index.clear();
        if (end_node == nodes_vector.size()) {
            cubes.clear();
            vertices.clear();
            nodes_vector.clear();
            return 0;
        }
        start_node = end_node;
        int cubes_size = 0, vertices_size = 0;
        for (;;) {
            int top = nodes_vector[end_node++];
            int s = grid_node_level(nodes[top]);
            cubes_index.push_back(cubes_size);
            vertices_index.push_back(vertices_size);
            cubes_size += cubex(1<<s);
            vertices_size += cubex((1<<s)+1);
            if ((vertices_size>>20) * (sizeof(bool)+6*sizeof(int)+sizeof(vertex)+(3+params::n_elements)*sizeof(T)) > params::memory_limit_mb) break;
            if (end_node == nodes_vector.size()) break;
        }
        cubes = vector<bool>(cubes_size, 0);
        vertices = vector<int>(vertices_size, -1);

        for (int i = 0; i < end_node - start_node; i++) {
            int ss = 1 << grid_node_level(nodes[nodes_vector[start_node + i]]);
            for (int j = 0; j < ss; j++)
            for (int k = 0; k < ss; k++)
            for (int f = 0; f < 6; f++) {
                int coords[3];
                coords[f / 2] = (f & 1) * (ss - 1);
                coords[(f/2+1) % 3] = j;
                coords[(f/2+2) % 3] = k;
                int ci = cube_index(coords[0], coords[1], coords[2], ss), ici = cubes_index[i] + ci;
                if (cubes[ici]) continue;
                cubes[ici] = true;
                cubes_queue.push_back(mp(i, make_int3(coords[0], coords[1], coords[2])));
            }
        }
        return end_node - start_node;
    }

    int fine_iteration(sdfT *sdf) {
        using namespace coarse;
        using namespace fine;
        if (sdf != NULL) {
            for (int i = 0; i < output_vertices.size(); i++) {
                assert(!std::isnan(sdf[i]));
                vertices[output_vertices_index[i]] = sdf[i]>=0? 1: 2;
            }
            int cqs = cubes_queue.size();
            for (int j = 0; j < cqs; j++) {
                int i = cubes_queue[j].first;
                int s = grid_node_level(nodes[nodes_vector[start_node + i]]), ss = 1<<s;
                cube cubei = nodes[nodes_vector[start_node + i]].c;
                int3 cqj = cubes_queue[j].second;
                int coords[3] = {xpp(cqj), ypp(cqj), zpp(cqj)};
                bool flag = false;
                for (int e = 0; e < 12; e++) {
                    int vcoords[3];
                    vcoords[e/4] = coords[e/4] + 1;
                    vcoords[(e/4+1) % 3] = coords[(e/4+1) % 3] + (e&1);
                    vcoords[(e/4+2) % 3] = coords[(e/4+2) % 3] + ((e>>1)&1);
                    int sign1 = vertices[vertices_index[i] + cube_index(vcoords[0], vcoords[1], vcoords[2], ss+1)];
                    bool border1 = 0, border2 = 0;
                    for (int p = 0; p < 3; p++) border1 |= vcoords[p]==0 || vcoords[p]==ss;
                    vcoords[e/4] = coords[e/4];
                    int sign2 = vertices[vertices_index[i] + cube_index(vcoords[0], vcoords[1], vcoords[2], ss+1)];
                    for (int p = 0; p < 3; p++) border2 |= vcoords[p]==0 || vcoords[p]==ss;
                    if (sign1 != sign2) {
                        flag = true;
                        if (border1 && border2) {
                            for (int k = 0; k < 4; k++) {
                                int inter_coords[3];
                                for (int p = 0; p < 3; p++) assign(inter_coords[p], cubei.coords[p], 2<<s, 2*vcoords[p]);
                                assign(inter_coords[e/4], inter_coords[e/4], 1, 1);
                                assign(inter_coords[(e/4+1) % 3], inter_coords[(e/4+1) % 3], 1, -1+2*(k&1));
                                assign(inter_coords[(e/4+2) % 3], inter_coords[(e/4+2) % 3], 1, -1+2*((k>>1)&1));
                                cube new_cube = search(&nodes[0], inter_coords, cubei.L + s + 1);
                                if (!is_boundary(new_cube)) solid::cubes_set.insert(cube_to_key(new_cube));
                            }
                        }
                        else {
                            for (int c = 0; c < 4; c++) {
                                int ccoords[3];
                                ccoords[e/4] = vcoords[e/4];
                                ccoords[(e/4+1) % 3] = vcoords[(e/4+1) % 3] + (c&1) - 1;
                                ccoords[(e/4+2) % 3] = vcoords[(e/4+2) % 3] + ((c>>1)&1) - 1;
                                int ci = cube_index(ccoords[0], ccoords[1], ccoords[2], ss), ici = cubes_index[i] + ci;
                                if (cubes[ici]) continue;
                                cubes[ici] = true;
                                cubes_queue.push_back(mp(i, make_int3(ccoords[0], ccoords[1], ccoords[2])));
                                cube c0;
                                for (int p = 0; p < 3; p++) assign(c0.coords[p], cubei.coords[p], 1<<s, ccoords[p]);
                                c0.L = cubei.L + s;
                                solid::cubes_set.insert(cube_to_key(c0));
                            }
                        }
                    }
                }
                if (flag) {
                    cube c0;
                    for (int p = 0; p < 3; p++) assign(c0.coords[p], cubei.coords[p], 1<<s, coords[p]);
                    c0.L = cubei.L + s;
                    solid::cubes_set.insert(cube_to_key(c0));
                }
            }
            cubes_queue.erase(cubes_queue.begin(), cubes_queue.begin() + cqs);
        }
        output_vertices.clear();
        output_vertices_index.clear();
        int cqs = cubes_queue.size();
        for (int j = 0; j < cqs; j++) {
            int i = cubes_queue[j].first;
            int s = grid_node_level(nodes[nodes_vector[start_node + i]]), ss = 1<<s;
            int3 cqj = cubes_queue[j].second;
            for (int dx = 0; dx < 2; dx++)
            for (int dy = 0; dy < 2; dy++)
            for (int dz = 0; dz < 2; dz++) {
                int coords[3] = {xpp(cqj)+dx, ypp(cqj)+dy, zpp(cqj)+dz};
                int ci = cube_index(coords[0], coords[1], coords[2], ss+1), ici = vertices_index[i] + ci;
                if (vertices[ici] == -1) {
                    vertex v;
                    cube c = nodes[nodes_vector[start_node + i]].c;
                    for (int p = 0; p < 3; p++) assign(v.coords[p], c.coords[p], 1<<s, coords[p]);
                    v.L = c.L + s;
                    vertices[ici] = 0;
                    output_vertices.push_back(v);
                    output_vertices_index.push_back(ici);
                }
            }
        }
        return output_vertices.size();
    }

    void fine_iteration_output(T *xyz) {
        using namespace params;
        using namespace fine;
        for (int i = 0; i < output_vertices.size(); i++) {
            compute_coords(xyz + i*3, output_vertices[i].coords, output_vertices[i].L);
        }
    }

    int vis_filter(bool simplify_occluded, int relax_iters) {
        using namespace params;
        using namespace solid;
        for (set<key_cube>::iterator iter = cubes_set.begin(); iter != cubes_set.end(); iter++) {
            cube c;
            key_to_cube(c, *iter);
            cubes.push_back(c);
        }
        cubes_set.clear();

        vector<bool> visible(cubes.size(), false);
        T factor = 10;
        vector<T> canvas;
        for (int k = 0; k < n_cams; k++) {
            T *current_cam = cams + k * (12 + 9 + 2);
            int H = int(current_cam[21] / factor), W = int(current_cam[22] / factor);
            if (simplify_occluded) {
                canvas = vector<T>(H * W, std::numeric_limits<T>::infinity());
                #pragma omp parallel for
                for (int i = 0; i < cubes.size(); i++) {
                    T image_coords[3];
                    projected_coords(cubes[i], k, image_coords, NULL);
                    if (image_coords[2] >= 0) {
                        int x = floor(image_coords[0] / factor);
                        int y = floor(image_coords[1] / factor);
                        if (x >= 0 && y >= 0 && x < W && y < H) {
                            #pragma omp critical
                            {
                                if (image_coords[2] <= canvas[x * H + y]) {
                                    canvas[x * H + y] = image_coords[2];
                                }
                            }
                        }
                    }
                }
            }
            #pragma omp parallel for
            for (int i = 0; i < cubes.size(); i++) {
                T image_coords[3];
                projected_coords(cubes[i], k, image_coords, NULL);
                if (image_coords[2] >= 0) {
                    int x = floor(image_coords[0] / factor);
                    int y = floor(image_coords[1] / factor);
                    if (x >= -relax_iters && y >= -relax_iters && x < W + relax_iters && y < H + relax_iters) {
                        if (simplify_occluded) {
                            for (int dx = -relax_iters; dx <= relax_iters; dx++)
                            for (int dy = -relax_iters; dy <= relax_iters; dy++) {
                                int nx = x + dx, ny = y + dy;
                                if (nx >= 0 && ny >= 0 && nx < W && ny < H) {
                                    if (image_coords[2] <= canvas[nx * H + ny]) {
                                        visible[i] = true;
                                    }
                                }
                            }
                        }
                        else visible[i] = true;
                    }
                }
            }
        }
        visible_set.clear();
        occluded_set.clear();
        set<key_cube> new_visible_set, old_visible_set;
        for (int i = 0; i < visible.size(); i++) {
            if (visible[i]) visible_set.insert(cube_to_key(cubes[i]));
            else occluded_set.insert(cube_to_key(cubes[i]));
        }
        visible.clear();
        cubes.clear();
        for (int i = 0; i < relax_iters; i++) {
            for (set<key_cube>::iterator iter = visible_set.begin(); iter != visible_set.end(); iter++) {
                cube c;
                key_to_cube(c, *iter);
                int coords[3];
                for (int f = 0; f < 6; f++) {
                    assign(coords[f/3], c.coords[f/3], 2, -1 + 4*(f&1));
                    assign(coords[(f/3+1)%3], c.coords[(f/3+1)%3], 2, 1);
                    assign(coords[(f/3+2)%3], c.coords[(f/3+2)%3], 2, 1);
                    cube new_cube = search(&coarse::nodes[0], coords, c.L + 1);
                    if (!is_boundary(new_cube)) {
                        if (occluded_set.count(cube_to_key(new_cube))) {
                            occluded_set.erase(cube_to_key(new_cube));
                            new_visible_set.insert(cube_to_key(new_cube));
                        }
                    }
                }
            }
            old_visible_set.insert(visible_set.begin(), visible_set.end());
            visible_set = new_visible_set;
            new_visible_set.clear();
        }
        visible_set.insert(old_visible_set.begin(), old_visible_set.end());
        old_visible_set.clear();
        new_visible_set.clear();

        node root;
        memset(root.nxts, -1, 8 * sizeof(int));
        memset(root.c.coords, 0, 3 * sizeof(int));
        root.c.L = 0;
        final::nodes.clear();
        final::nodes.push_back(root);
        final::visible_nodes_cube.clear();
        final::occluded_nodes_id.clear();
        for (set<key_cube>::iterator iter = visible_set.begin(); iter != visible_set.end(); iter++) {
            cube c;
            key_to_cube(c, *iter);
            final::visible_nodes_cube.push_back(c);
        }
        for (set<key_cube>::iterator iter = occluded_set.begin(); iter != occluded_set.end(); iter++) {
            cube c;
            key_to_cube(c, *iter);
            final::occluded_nodes_id.push_back(divide_to_cube(final::nodes, c));
        }
        final::bipolar_edges.clear();
        final::bipolar_edges_s.clear();
        final::bipolar_edges_vertices.clear();
        final::in_view_tag.clear();
        final::bipolar_edges_vindices.clear();
        for (int i = 0; i < params::n_elements; i++) {
            final::bipolar_edges.push_back(vector<key_edge>());
            final::bipolar_edges_s.push_back(0);
            final::bipolar_edges_vindices.push_back(vector<int>());
            final::in_view_tag.push_back(vector<bool>());
        }
        final::vertices_cnt = vector<int>(5, 0);
        final::bipolar_edges.push_back(vector<key_edge>());
        final::start_node = 0;
        final::gl = 0;
        for (;;) {
            int total_nodes = 0;
            for (int i = 0; i < final::visible_nodes_cube.size(); i++) {
                int is = max(0, int_log(projected_size(final::visible_nodes_cube[i])) - final::gl);
                total_nodes += max(0, cubex(1<<is) - cubex((1<<is)-2));
            }
            if (((final::nodes.size()+total_nodes)>>20) * sizeof(node) < params::memory_limit_mb * 0.6) break;
            final::gl++;
        }
        return final::visible_nodes_cube.size();
    }

    int final_iteration() {
        using namespace final;
        assert((nodes.size() >> 20) * sizeof(node) < params::memory_limit_mb * 0.8);
        if (start_node == visible_nodes_cube.size()) {
            coarse::nodes.clear();
            solid::visible_set.clear();
            visible_nodes_cube.clear();
            return 0;
        }
        int total_nodes = 0, total_verts = 0;
        for (end_node = start_node; end_node < visible_nodes_cube.size(); end_node++) {
            int is = int_log(projected_size(visible_nodes_cube[end_node]));
            total_nodes += (1 << (3*(is-gl))) * 8 / 7;
            total_verts += cubex((1<<is)+1);
            if (((nodes.size()+total_nodes)>>20) * sizeof(node) + (total_verts>>20) * (sizeof(vertex)+sizeof(T*)+sizeof(key_cube)+sizeof(int)+2*sizeof(key_cube*)+(3+params::n_elements)*sizeof(T)) > params::memory_limit_mb) break;
        }
        assert(end_node != start_node);

        vertices.clear();
        size0 = nodes.size();
        assert(new_nodes.empty());
        for (int i = start_node; i < end_node; i++) {
            new_nodes.push(divide_to_cube(nodes, visible_nodes_cube[i]));
            while (!new_nodes.empty()) {
                int ind = new_nodes.front();
                new_nodes.pop();
                T s = projected_size(nodes[ind].c);
                if (s > 1<<gl) {
                    int base_size = nodes.size();
                    expand_octree(nodes, ind);
                    for (int j = 0; j < 8; j++) new_nodes.push(base_size + j);
                }
                else {
                    int is = int_log(s);
                    mark_grid_node(nodes[ind], is);
                    v.resize(cubex((1<<is)+1));
                    enumerate_vertices(&v[0], nodes[ind]);
                    for (int j = 0; j < cubex((1<<is)+1); j++) if (!vertices.count(cube_to_key(v[j]))) vertices[cube_to_key(v[j])] = vertices.size();
                }
            }
        }
        assert(vertices.size() > 0);
        return vertices.size();
    }

    int final_iteration_occluded() {
        using namespace final;
        for (int i = 0; i < occluded_nodes_id.size(); i++) {
            v.resize(8);
            enumerate_vertices(&v[0], nodes[occluded_nodes_id[i]]);
            for (int j = 0; j < 8; j++) if (!vertices.count(cube_to_key(v[j]))) vertices[cube_to_key(v[j])] = vertices.size();
        }
        return vertices.size();
    }

    void final_iteration2(T *xyz) {
        using namespace params;
        using namespace final;
        for (map<key_cube, int>::iterator iter = vertices.begin(); iter != vertices.end(); iter++) {
            vertex v;
            key_to_cube(v, iter->first);
            compute_coords(xyz + iter->second*3, v.coords, v.L);
        }
    }

    int final_iteration3(sdfT *sdf) {
        using namespace final;
        using namespace solid;
        int start = visible_nodes_cube.size() - start_node;
        for (int i = 0; i < vertices.size() * params::n_elements; i++) assert(!std::isnan(sdf[i]));
        for (int i = size0; i < nodes.size(); i++)
            if (leaf_node(nodes[i])) find_edges(nodes[i], vertices, sdf, bipolar_edges);
        vertices.clear();
        for (int i = 0; i < params::n_elements + 1; i++) {
            vector<key_edge> &bei = bipolar_edges[i];
            int s = i==params::n_elements?0:bipolar_edges_s[i];
            std::sort(bei.begin() + s, bei.end());
            bei.erase(std::unique(bei.begin() + s, bei.end()), bei.end());
            int e = bei.size();
            searched.resize((e-s)*4);
            #pragma omp parallel for
            for (int j = s; j < e; j++) {
                cube e0;
                key_to_cube(e0, bei[j].second);
                int dir = bei[j].first;
                int coords[3], L = e0.L + 1;
                int ks[4] = {0, 1, 3, 2};
                bool flag = true, flag2 = true;
                for (int ik = 0; ik < 4; ik++) {
                    int k = dir > 0? ks[ik]: ks[3 - ik];
                    int c0 = abs(dir) - 1, c1 = (c0 + 1) % 3, c2 = (c1 + 1) % 3;
                    assign(coords[c0], e0.coords[c0], 2, 1);
                    assign(coords[c1], e0.coords[c1], 2, -1 + 2*(k&1));
                    assign(coords[c2], e0.coords[c2], 2, -1 + 2*((k>>1)&1));
                    cube &c = searched[(j-s)*4 + ik];
                    c = search(&nodes[0], coords, L);
                    if (is_boundary(c)) {
                        flag = false;
                        break;
                    }
                    else if (is_exterior(c)) {
                        flag2 = false;
                    }
                }
                if (!flag) {
                    for (int k = 0; k < 4; k++)
                        mark_boundary(searched[(j-s)*4 + k]);
                }
                else if (!flag2) {
                    for (int k = 0; k < 4; k++)
                        mark_exterior(searched[(j-s)*4 + k]);
                }
            }
            if (i == params::n_elements) continue;
            int j1 = s, j2 = e - 1;
            while (j1 < j2) {
                while (j1 < e && is_regular(searched[4*(j1-s)])) j1++;
                while (j2 >= s && !is_regular(searched[4*(j2-s)])) j2--;
                if (j1 < j2) {
                    std::swap(bei[j1], bei[j2]);
                    for (int k = 0; k < 4; k++)
                        std::swap(searched[4*(j1-s) + k], searched[4*(j2-s) + k]);
                    j1++;
                    j2--;
                }
            }
            vector<bool> &ivt = in_view_tag[i];
            for (int j = s; j < j1; j++) {
                for (int k = 0; k < 4; k++) {
                    cube c = searched[(j-s)*4 + k];
                    int vid;
                    key_cube key = cube_to_key(c);
                    if (bipolar_edges_vertices.count(mp(i, key))) vid = bipolar_edges_vertices[mp(i, key)];
                    else {
                        vid = bipolar_edges_vertices[mp(i, key)] = vertices_cnt[i]++;
                        ivt.push_back(occluded_set.count(key) == 0);
                    }
                    bipolar_edges_vindices[i].push_back(vid);
                }
            }
            bipolar_edges_s[i] = j1;
            j2 = e - 1;
            while (j1 < j2) {
                while (j1 < e && is_exterior(searched[4*(j1-s)])) j1++;
                while (j2 >= bipolar_edges_s[i] && is_boundary(searched[4*(j2-s)])) j2--;
                if (j1 < j2) {
                    std::swap(bei[j1], bei[j2]);
                    for (int k = 0; k < 4; k++)
                        std::swap(searched[4*(j1-s) + k], searched[4*(j2-s) + k]);
                    j1++;
                    j2--;
                }
            }
            bei.erase(bei.begin() + j1, bei.end());
        }

        for (int i = 0; i < size0; i++)
            for (int j = 0; j < 8; j++)
                if (nodes[i].nxts[j] >= size0) nodes[i].nxts[j] = -1;
        nodes.erase(nodes.begin() + size0, nodes.end());

        for (int i = start_node; i < end_node; i++) {
            assert(new_nodes.empty());
            int nodes_id = divide_to_cube(nodes, visible_nodes_cube[i]);
            memset(nodes[nodes_id].nxts, -1, 3 * sizeof(int));
            new_nodes.push(nodes_id);
            while (!new_nodes.empty()) {
                int ind = new_nodes.front();
                new_nodes.pop();
                T s = projected_size(nodes[ind].c);
                if (s > 1<<gl) {
                    int instance = compute_boundary(nodes[ind].c, nodes[nodes_id].c);
                    int size0 = nodes.size();
                    partial_expand_octree(nodes, ind, instance);
                    for (int j = size0; j < nodes.size(); j++) new_nodes.push(j);
                }
                else {
                    mark_grid_node(nodes[ind], int_log(s));
                }
            }
        }

        vector<key_edge> &ben = bipolar_edges[params::n_elements];
        for (int i = 0; i < ben.size(); i++) {
            int dir = ben[i].first;
            cube e0;
            key_to_cube(e0, ben[i].second);
            int coords[3], L = e0.L + 1;
            for (int k = 0; k < 4; k++) {
                int c0 = abs(dir) - 1, c1 = (c0 + 1) % 3, c2 = (c1 + 1) % 3;
                assign(coords[c0], e0.coords[c0], 2, 1);
                assign(coords[c1], e0.coords[c1], 2, -1 + 2*(k&1));
                assign(coords[c2], e0.coords[c2], 2, -1 + 2*((k>>1)&1));
                cube c = search(&coarse::nodes[0], coords, L);
                if (!is_boundary(c)) {
                    key_cube key = cube_to_key(c);
                    if (occluded_set.count(key) || visible_set.count(key)) continue;
                    visible_set.insert(key);
                    visible_nodes_cube.push_back(c);
                }
            }
        }
        ben.clear();

        start_node = end_node;
        return start - (visible_nodes_cube.size() - start_node);
    }

    void final_iteration3_occluded(sdfT *sdf) {
        using namespace final;
        using namespace solid;
        for (int i = 0; i < vertices.size() * params::n_elements; i++) assert(!std::isnan(sdf[i]));
        for (int i = 0; i < occluded_nodes_id.size(); i++)
            find_edges(nodes[occluded_nodes_id[i]], vertices, sdf, bipolar_edges);
        vertices.clear();
        occluded_nodes_id.clear();
    }

    void final_remaining(int *nv) {
        using namespace final;
        bipolar_edges_computed_vertices.clear();
        bipolar_edges_vertices_vector.clear();
        for (int i = 0; i < params::n_elements; i++) {
            vector<key_edge> &bei = bipolar_edges[i];
            int s = bipolar_edges_s[i];
            std::sort(bei.begin() + s, bei.end());
            bei.erase(std::unique(bei.begin() + s, bei.end()), bei.end());
            int e = bei.size();
            searched.resize((e-s)*4);
            vector<bool> tags((e-s)*4);
            #pragma omp parallel for
            for (int j = s; j < e; j++) {
                cube e0;
                key_to_cube(e0, bei[j].second);
                int dir = bei[j].first;
                int coords[3], L = e0.L + 1;
                int ks[4] = {0, 1, 3, 2};
                bool flag = false;
                for (int ik = 0; ik < 4; ik++) {
                    int k = dir > 0? ks[ik]: ks[3 - ik];
                    int c0 = abs(dir) - 1, c1 = (c0 + 1) % 3, c2 = (c1 + 1) % 3;
                    assign(coords[c0], e0.coords[c0], 2, 1);
                    assign(coords[c1], e0.coords[c1], 2, -1 + 2*(k&1));
                    assign(coords[c2], e0.coords[c2], 2, -1 + 2*((k>>1)&1));
                    cube &c = searched[(j-s)*4 + ik];
                    c = search(&nodes[0], coords, L, 1);
                    if (is_boundary(c)) {
                        flag = true;
                        break;
                    }
                    else {
                        tags[(j-s)*4 + ik] = solid::occluded_set.count(cube_to_key(c)) == 0 && !is_exterior(search(&nodes[0], coords, L));
                    }
                }
                if (flag) {
                    for (int k = 0; k < 4; k++)
                        mark_boundary(searched[(j-s)*4 + k]);
                }
            }
            int j1 = s, j2 = e - 1;
            while (j1 < j2) {
                while (j1 < e && is_regular(searched[4*(j1-s)])) j1++;
                while (j2 >= s && !is_regular(searched[4*(j2-s)])) j2--;
                if (j1 < j2) {
                    std::swap(bei[j1], bei[j2]);
                    for (int k = 0; k < 4; k++) {
                        std::swap(searched[4*(j1-s) + k], searched[4*(j2-s) + k]);
                        bool tmp = tags[4*(j1-s) + k];
                        tags[4*(j1-s)+k] = tags[4*(j2-s) + k];
                        tags[4*(j2-s)+k] = tmp;
                    }
                    j1++;
                    j2--;
                }
            }
            vector<bool> &ivt = in_view_tag[i];
            for (int j = s; j < j1; j++) {
                for (int k = 0; k < 4; k++) {
                    cube c = searched[(j-s)*4 + k];
                    int vid;
                    key_cube key = cube_to_key(c);
                    if (bipolar_edges_vertices.count(mp(i, key))) vid = bipolar_edges_vertices[mp(i, key)];
                    else {
                        vid = bipolar_edges_vertices[mp(i, key)] = vertices_cnt[i]++;
                        ivt.push_back(tags[(j-s)*4 + k]);
                    }
                    bipolar_edges_vindices[i].push_back(vid);
                }
            }
            bei.erase(bei.begin() + j1, bei.end());
            bipolar_edges_computed_vertices.push_back(vector<computed_vertex>(vertices_cnt[i]));
            bipolar_edges_vertices_vector.push_back(vector<key_cube>(vertices_cnt[i]));
            nv[i] = vertices_cnt[i];
        }
        for (map<pair<int, key_cube>, int>::iterator iter = bipolar_edges_vertices.begin(); iter != bipolar_edges_vertices.end(); iter++) {
            int i = iter->first.first;
            vector<computed_vertex> &becvi = bipolar_edges_computed_vertices[i];
            vector<key_cube> &bevvi = bipolar_edges_vertices_vector[i];
            cube c;
            key_to_cube(c, iter->first.second);
            compute_center(becvi[iter->second].c, c);
            becvi[iter->second].l = 0;
            becvi[iter->second].r = 0.5 * params::size / (1<<c.L);
            bevvi[iter->second] = iter->first.second;
        }
        bipolar_edges_vertices.clear();
        nodes.clear();
        solid::occluded_set.clear();
    }

    void get_verts_center(int e, T *positions) {
        using namespace final;
        vector<computed_vertex> &becv = bipolar_edges_computed_vertices[e];
        for (int i = 0; i < becv.size(); i++) {
            memcpy(positions + 3 * i, becv[i].c, 3 * sizeof(T));
        }
    }

    void get_extra_verts_center(T *epositions, T *fpositions) {
        using namespace computing;
        for (int i = 0; i < edge_vertices.size(); i++) {
            memcpy(epositions + 3 * i, edge_vertices[i].second.c, 3 * sizeof(T));
        }
        for (int i = 0; i < face_vertices.size(); i++) {
            memcpy(fpositions + 3 * i, face_vertices[i].second.c, 3 * sizeof(T));
        }
    }

    void update_verts(int e, sdfT *sdf, sdfT *center_sdf, T *positions) {
        using namespace final;
        vector<computed_vertex> &becv = bipolar_edges_computed_vertices[e];
        #pragma omp parallel for
        for (int i = 0; i < becv.size(); i++) {
            if (sdf != NULL) {
                T mid = (becv[i].l + becv[i].r) / 2;
                bool bipolar = 0;
                for (int j = 0; j < 8; j++)
                    if ((sdf[i * 8 + j] >= 0) != (center_sdf[i] >= 0)) {
                        bipolar = 1;
                        break;
                    }
                if (bipolar) becv[i].r = mid;
                else becv[i].l = mid;
            }
            T mid = (becv[i].l + becv[i].r) / 2;
            for (int j = 0; j < 8; j++) {
                int cid = i * 8 + j;
                for (int k = 0; k < 3; k++)
                    positions[cid*3 + k] = becv[i].c[k] + (((j>>k)&1)*2-1) * mid;
            }
        }
    }

    void update_extra_verts(sdfT *esdf, sdfT *fsdf, sdfT *ecenter_sdf, sdfT *fcenter_sdf, T *epositions, T*fpositions) {
        using namespace computing;
        int edge_n = 2;
        #pragma omp parallel for
        for (int i = 0; i < edge_vertices.size(); i++) {
            if (esdf != NULL) {
                T mid = (edge_vertices[i].second.l + edge_vertices[i].second.r) / 2;
                bool bipolar = 0;
                for (int j = 0; j < edge_n; j++)
                    if ((esdf[i * edge_n + j] >= 0) != (ecenter_sdf[i] >= 0)) {
                        bipolar = 1;
                        break;
                    }
                if (bipolar) edge_vertices[i].second.r = mid;
                else edge_vertices[i].second.l = mid;
            }
            T mid = (edge_vertices[i].second.l + edge_vertices[i].second.r) / 2;
            for (int j = 0; j < edge_n; j++) {
                int cid = i * edge_n + j;
                for (int k = 0; k < 3; k++) {
                    epositions[cid*3 + k] = edge_vertices[i].second.c[k];
                    if (k == edge_vertices[i].first) epositions[cid*3 + k] += (j*2-1) * mid;
                }
            }
        }
        int face_n = 4;
        #pragma omp parallel for
        for (int i = 0; i < face_vertices.size(); i++) {
            if (fsdf != NULL) {
                T mid = (face_vertices[i].second.l + face_vertices[i].second.r) / 2;
                bool bipolar = 0;
                for (int j = 0; j < face_n; j++)
                    if ((fsdf[i * face_n + j] >= 0) != (fcenter_sdf[i] >= 0)) {
                        bipolar = 1;
                        break;
                    }
                if (bipolar) face_vertices[i].second.r = mid;
                else face_vertices[i].second.l = mid;
            }
            T mid = (face_vertices[i].second.l + face_vertices[i].second.r) / 2;
            for (int j = 0; j < face_n; j++) {
                int cid = i * face_n + j;
                for (int k = 0; k < 3; k++) {
                    fpositions[cid*3 + k] = face_vertices[i].second.c[k];
                    if (k == (face_vertices[i].first+1)%3) fpositions[cid*3 + k] += (first_digit(j)*2-1) * mid;
                    else if (k == (face_vertices[i].first+2)%3) fpositions[cid*3 + k] += (second_digit(j)*2-1) * mid;
                }
            }
        }
    }

    void get_lr_verts(int e, T *cube_l, T *cube_r) {
        using namespace final;
        vector<computed_vertex> &becv = bipolar_edges_computed_vertices[e];
        for (int i = 0; i < becv.size(); i++) {
            for (int j = 0; j < 8; j++) {
                int cid = i * 8 + j;
                for (int k = 0; k < 3; k++) {
                    cube_l[cid*3 + k] = becv[i].c[k] + (((j>>k)&1)*2-1) * becv[i].l;
                    cube_r[cid*3 + k] = becv[i].c[k] + (((j>>k)&1)*2-1) * becv[i].r;
                }
            }
        }
    }

    void get_lr_extra_verts(T *epos_l, T *epos_r, T *fpos_l, T *fpos_r) {
        using namespace computing;
        for (int i = 0; i < edge_vertices.size(); i++) {
            for (int j = 0; j < 2; j++) {
                int cid = i * 2 + j;
                for (int k = 0; k < 3; k++) {
                    epos_l[cid*3 + k] = edge_vertices[i].second.c[k];
                    if (k == edge_vertices[i].first) epos_l[cid*3 + k] += (j*2-1) * edge_vertices[i].second.l;
                    epos_r[cid*3 + k] = edge_vertices[i].second.c[k];
                    if (k == edge_vertices[i].first) epos_r[cid*3 + k] += (j*2-1) * edge_vertices[i].second.r;
                }
            }
        }
        for (int i = 0; i < face_vertices.size(); i++) {
            for (int j = 0; j < 4; j++) {
                int cid = i * 4 + j;
                for (int k = 0; k < 3; k++) {
                    fpos_l[cid*3 + k] = face_vertices[i].second.c[k];
                    if (k == (face_vertices[i].first+1)%3) fpos_l[cid*3 + k] += (first_digit(j)*2-1) * face_vertices[i].second.l;
                    else if (k == (face_vertices[i].first+2)%3) fpos_l[cid*3 + k] += (second_digit(j)*2-1) * face_vertices[i].second.l;
                    fpos_r[cid*3 + k] = face_vertices[i].second.c[k];
                    if (k == (face_vertices[i].first+1)%3) fpos_r[cid*3 + k] += (first_digit(j)*2-1) * face_vertices[i].second.r;
                    else if (k == (face_vertices[i].first+2)%3) fpos_r[cid*3 + k] += (second_digit(j)*2-1) * face_vertices[i].second.r;
                }
            }
        }
    }

    // todo consider more than corners when a vetex cube has complex side face
    void finalize_verts(int e, sdfT *sdf_l, sdfT *sdf_r, T *verts) {
        using namespace final;
        vector<computed_vertex> &becv = bipolar_edges_computed_vertices[e];
        for (int i = 0; i < becv.size(); i++) {
            T v[3]={0};
            int w=0;
            for (int j = 0; j < 8; j++)
                if ((sdf_l[i * 8 + j] >= 0) != (sdf_r[i * 8 + j] >= 0)) {
                    w++;
                    for (int k = 0; k < 3; k++) {
                        v[k] += becv[i].c[k] + (((j>>k)&1)*2-1) * (becv[i].l + becv[i].r) / 2;
                    }
                }
            if (w == 0) {
                for (int k = 0; k < 3; k++) verts[i * 3 + k] = becv[i].c[k];
            }
            else {
                for (int k = 0; k < 3; k++) verts[i * 3 + k] = v[k] / w;
            }
        }
        bipolar_edges_computed_vertices[e].clear();
    }

    void finalize_extra_verts(sdfT *esdf_l, sdfT *esdf_r, T *everts, sdfT *fsdf_l, sdfT *fsdf_r, T *fverts) {
        using namespace computing;
        for (int i = 0; i < edge_vertices.size(); i++) {
            T v[3]={0};
            int w=0;
            for (int j = 0; j < 2; j++)
                if ((esdf_l[i * 2 + j] >= 0) != (esdf_r[i * 2 + j] >= 0)) {
                    w++;
                    T mid = (edge_vertices[i].second.l + edge_vertices[i].second.r) / 2;
                    for (int k = 0; k < 3; k++) {
                        v[k] += edge_vertices[i].second.c[k];
                        if (k == edge_vertices[i].first) v[k] += (j*2-1) * mid;
                    }
                }
            assert(w != 0);
            for (int k = 0; k < 3; k++) everts[i * 3 + k] = v[k] / w;
        }
        edge_vertices.clear();
        for (int i = 0; i < face_vertices.size(); i++) {
            T v[3]={0};
            int w=0;
            for (int j = 0; j < 4; j++)
                if ((fsdf_l[i * 4 + j] >= 0) != (fsdf_r[i * 4 + j] >= 0)) {
                    w++;
                    T mid = (face_vertices[i].second.l + face_vertices[i].second.r) / 2;
                    for (int k = 0; k < 3; k++) {
                        v[k] += face_vertices[i].second.c[k];
                        if (k == (face_vertices[i].first+1)%3) v[k] += (first_digit(j)*2-1) * mid;
                        else if (k == (face_vertices[i].first+2)%3) v[k] += (second_digit(j)*2-1) * mid;
                    }
                }
            if (w == 0) {
                for (int k = 0; k < 3; k++) fverts[i * 3 + k] = face_vertices[i].second.c[k];
            }
            else {
                for (int k = 0; k < 3; k++) fverts[i * 3 + k] = v[k] / w;
            }
        }
        face_vertices.clear();
    }

    void get_in_view_tag(int e, bool *output) {
        using namespace final;
        using namespace computing;
        int cnt = 0;
        vector<bool> &ivt = in_view_tag[e];
        for (int i = 0; i < ivt.size(); i++) {
            output[cnt++] = ivt[i];
        }
        ivt.clear();
        for (int i = 0; i < edge_vertices_in_view_tag.size(); i++) {
            output[cnt++] = edge_vertices_in_view_tag[i];
        }
        edge_vertices_in_view_tag.clear();
        for (int i = 0; i < face_vertices_in_view_tag.size(); i++) {
            output[cnt++] = face_vertices_in_view_tag[i];
        }
        face_vertices_in_view_tag.clear();
    }

    void construct_faces(int e, T *final_vertices, int *cnt) {
        using namespace final;
        using namespace computing;
        vector<key_edge> &edge_e = bipolar_edges[e];
        vector<int> &vertices_ids = bipolar_edges_vindices[e];
        vector<key_cube> &unique_vertices = bipolar_edges_vertices_vector[e];
        faces.clear();
        edge_vertices.clear();
        edge_vertices_in_view_tag.clear();
        face_vertices.clear();
        face_vertices_in_view_tag.clear();
        face_vertices_map.clear();
        int nv = unique_vertices.size();
        for (int i = 0; i < edge_e.size(); i++) {
            T computed_edge[6];
            cube c;
            key_to_cube(c, edge_e[i].second);
            compute_coords(computed_edge, c.coords, c.L);
            int dir = abs(edge_e[i].first) - 1;
            c.coords[dir]++;
            compute_coords(computed_edge + 3, c.coords, c.L);
            T computed_faces[12 * 4];
            int computed_faces_L[4], computed_faces_dir[4];
            bool intersect[4]={0};
            bool condition1 = 1;
            for (int j = 0; j < 4; j++) {
                int v1 = vertices_ids[i * 4 + j], v2 = vertices_ids[i * 4 + (j+1) % 4];
                if (v1 == v2) continue;
                cube c1, c2;
                key_to_cube(c1, unique_vertices[v1]);
                key_to_cube(c2, unique_vertices[v2]);
                if (c1.L < c2.L) {
                    cube t = c1;
                    c1 = c2;
                    c2 = t;
                }
                bool found = 0;
                for (int ax = 0; ax < 3; ax++) {
                    for (int dir = 0; dir <= 1; dir++) {
                        if (c1.coords[ax] + dir == (c2.coords[ax] + 1 - dir) << (c1.L - c2.L)) {
                            for (int d1 = 0; d1 <= 1; d1++)
                            for (int d2 = 0; d2 <= 1; d2++) {
                                cube c1_ = c1;
                                c1_.coords[ax] += dir;
                                c1_.coords[(ax+1)%3] += d1;
                                c1_.coords[(ax+2)%3] += d2;
                                compute_coords(computed_faces + 12 * j + 3 * (d1 * 2 + d2), c1_.coords, c1_.L);
                            }
                            computed_faces_L[j] = c1.L;
                            computed_faces_dir[j] = ax;
                            found = 1;
                            break;
                        }
                    }
                    if (found) break;
                }
                assert(found);
                if (c1.L == c2.L) {
                    intersect[j] = 1;
                    continue;
                }
                // tri_seg_intersect should be strict
                bool i1 = tri_seg_intersect(computed_faces + 12 * j + 6, computed_faces + 12 * j, computed_faces + 12 * j + 3, final_vertices + v1 * 3, final_vertices + v2 * 3);
                bool i2 = tri_seg_intersect(computed_faces + 12 * j + 3, computed_faces + 12 * j + 9, computed_faces + 12 * j + 6, final_vertices + v1 * 3, final_vertices + v2 * 3);
                intersect[j] = i1 || i2;
                condition1 &= intersect[j];
            }
            if (!condition1) {
                computed_vertex v;
                for (int j = 0; j < 3; j++)
                    v.c[j] = (computed_edge[j] + computed_edge[j + 3]) / 2;
                v.l = 0;
                v.r = 0.5 * params::size / (1<<c.L);
                edge_vertices.push_back(mp(dir, v));
                bool edge_in_view = false;
                for (int j = 0; j < 4; j++) edge_in_view |= in_view_tag[e][vertices_ids[i * 4 + j]];
                edge_vertices_in_view_tag.push_back(edge_in_view);
                for (int j = 0; j < 4; j++) {
                    int v1 = vertices_ids[i * 4 + j], v2 = vertices_ids[i * 4 + (j+1) % 4];
                    if (v1 == v2) continue;
                    if (!intersect[j]) {
                        int vf;
                        if (face_vertices_map.count(mp(v1, v2))) {
                            vf = face_vertices_map[mp(v1, v2)];
                        }
                        else if (face_vertices_map.count(mp(v2, v1))) {
                            vf = face_vertices_map[mp(v2, v1)];
                        }
                        else {
                            vf = face_vertices.size();
                            face_vertices_map[mp(v1, v2)] = vf;
                            computed_vertex v;
                            for (int k = 0; k < 3; k++)
                                v.c[k] = (computed_faces[12 * j + k + 3] + computed_faces[12 * j + k + 6]) / 2;
                            v.l = 0;
                            v.r = 0.5 * params::size / (1<<computed_faces_L[j]);
                            face_vertices.push_back(mp(computed_faces_dir[j], v));
                            bool face_in_view = in_view_tag[e][v1] || in_view_tag[e][v2];
                            face_vertices_in_view_tag.push_back(face_in_view);
                        }
                        add_faces(faces, v1, -vf - 1, nv + edge_vertices.size() - 1);
                        add_faces(faces, -vf - 1, v2, nv + edge_vertices.size() - 1);
                    }
                    else {
                        add_faces(faces, v1, v2, nv + edge_vertices.size() - 1);
                    }
                }
            }
            else {
                bool condition2 = 0;
                int start_j = -1;
                for (int j = 0; j < 4; j++) {
                    int v1 = vertices_ids[i * 4 + j], v2 = vertices_ids[i * 4 + (j+1) % 4], v3 = vertices_ids[i * 4 + (j+2) % 4];
                    if (v1 == v2 || v2 == v3 || v1 == v3) continue;
                    if (tri_seg_intersect(final_vertices + v1 * 3, final_vertices + v2 * 3, final_vertices + v3 * 3, computed_edge, computed_edge + 3)) {
                        start_j = j;
                    }
                }
                if (start_j != -1) {
                    int j = start_j;
                    int v1 = vertices_ids[i * 4 + j], v2 = vertices_ids[i * 4 + (j+1) % 4], v3 = vertices_ids[i * 4 + (j+2) % 4];
                    int v4 = vertices_ids[i * 4 + (j+3) % 4];
                    if (v4 == v3 || v4 == v1 || tri_seg_intersect(computed_edge, final_vertices + v4 * 3, computed_edge + 3, final_vertices + v1 * 3, final_vertices + v3 * 3)) {
                        condition2 = true;
                    }
                }
                if (!condition2) {
                    computed_vertex v;
                    for (int j = 0; j < 3; j++)
                        v.c[j] = (computed_edge[j] + computed_edge[j + 3]) / 2;
                    v.l = 0;
                    v.r = 0.5 * params::size / (1<<c.L);
                    edge_vertices.push_back(mp(dir, v));
                    bool edge_in_view = false;
                    for (int j = 0; j < 4; j++) edge_in_view |= in_view_tag[e][vertices_ids[i * 4 + j]];
                    edge_vertices_in_view_tag.push_back(edge_in_view);
                    for (int j = 0; j < 4; j++) {
                        int v1 = vertices_ids[i * 4 + j], v2 = vertices_ids[i * 4 + (j+1) % 4];
                        if (v1 == v2) continue;
                        add_faces(faces, v1, v2, nv + edge_vertices.size() - 1);
                    }
                }
                else {
                    int j = start_j;
                    int v1 = vertices_ids[i * 4 + j], v2 = vertices_ids[i * 4 + (j+1) % 4], v3 = vertices_ids[i * 4 + (j+2) % 4];
                    add_faces(faces, v1, v2, v3);
                    int v4 = vertices_ids[i * 4 + (j+3) % 4];
                    if (v4 == v1 || v4 == v3) continue;
                    add_faces(faces, v1, v3, v4);
                }
            }
        }
        face_vertices_map.clear();
        for (int i = 0; i < faces.size(); i++) {
            if (faces[i] < 0) faces[i] = nv + edge_vertices.size() + (-faces[i] - 1);
        }
        vertices_ids.clear();
        unique_vertices.clear();
        edge_e.clear();
        cnt[0] = edge_vertices.size();
        cnt[1] = face_vertices.size();
        cnt[2] = faces.size() / 3;
    }

    void get_faces(int *faces_output) {
        using namespace computing;
        memcpy(faces_output, &faces[0], sizeof(int) * faces.size());
        faces.clear();
    }
}
