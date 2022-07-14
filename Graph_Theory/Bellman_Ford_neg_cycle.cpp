struct edge {
    int a, b, w;
};

// prints negative cycle, if it exists
void print_neg_cycle(vector<edge> &edges, int n, int m, int src) {
    vector<int> dist(n + 1, 1e9);
    vector<int> parent(n + 1, -1);
    dist[src] = 0;

    // all shortest distances are found in n - 1 iterations when there is no neg cycle
    // if there is relaxation on n_th iteration, there is a neg cycle

    int is_cycle = 0, last;
    for (int it = 0; it < n; it ++) {
        int relaxed = 0;
        for (int i = 0; i < m; i ++) {
            if (dist[edges[i].b] > dist[edges[i].a] + edges[i].w) {
                dist[edges[i].b] = dist[edges[i].a] + edges[i].w;
                parent[edges[i].b] = edges[i].a;
                last = edges[i].b;
                relaxed = 1;
            }
        }

        if (!relaxed) {
            break;
        }
        if (it == n - 1 && relaxed) {
            is_cycle = 1;
        }
    }

    if (!is_cycle) {
        cout << "NO" << endl;
    }
    else {
        // maximum cycle length is n, so after n iterations, it is guaranteed that
        // current node is a part of a cycle

        // why does parent[last] != -1?
        // ==> It's know that there exists a negative cycle and that last is connected to it
        // so in < n iterations (here, parent[last] != -1) we enter the negative cycle and in the cycle
        // it is guaranteed that every node has a parent, since every node has been relaxed atleast once

        for (int i = 0; i < n; i ++) {
            last = parent[last];
        }

        int cur = parent[last];
        vector<int> cycle;
        cycle.push_back(last);
        while (cur != last) {
            cycle.push_back(cur);
            cur = parent[cur];
        }
        cycle.push_back(cur);

        reverse(cycle.begin(), cycle.end());
        cout << "YES" << endl;
        for (auto e : cycle) {
            cout << e << ' ';
        }
        cout << endl;
    }
}
