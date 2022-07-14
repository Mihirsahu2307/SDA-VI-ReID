vector<int> dijkstra(int n, vector<pair<int, int>> G[]) {

	set<pair<int, int>> q;
    vector<int> dist(n + 1, 1e15), parent(n + 1, -1);
    dist[1] = 0;
    q.insert({0, 1});

    while (!q.empty()) {
        auto cur = *q.begin();
        q.erase(cur);

        vis[cur.second] = 1;
        for (auto c : G[cur.second]) {
            if (dist[c.first] > dist[cur.second] + c.second) {
                q.erase({dist[c.first], c.first});
                dist[c.first] = dist[cur.second] + c.second;
                parent[c.first] = cur.second;
                q.insert({dist[c.first], c.first});
            }
        }
    }

    if (dist[n] == 1e15) {
        cout << -1 << endl;
        continue;
    }

    vector<int> path;
    int cur = n;
    path.push_back(n);
    while (cur != 1) {
        cur = parent[cur];
        path.push_back(cur);
    }

    reverse(path.begin(), path.end());

    return path;
}


// priority queue (faster)

vector<int> dijkstra(vector<pair<int,int>> adj[], int n, int src) {
    priority_queue<pair<int, int>, vector <pair<int, int>> , greater<pair<int, int>>> pq;
    vector<int> dist(n + 1, INF);
    pq.push({0, src});
    dist[src] = 0;

    while (!pq.empty()) {
        auto u = pq.top();
        pq.pop();

        if(u.first != dist[u.second]) {
            continue; // This value in priority queue is an old value, don't work with it
        }
 
        for (auto x : adj[u]) {
            int v = x.first;
            int weight = x.second;
 
            if (dist[v] > dist[u.first] + weight) {
                dist[v] = dist[u.first] + weight;
                pq.push({dist[v], v});
            }
        }
    }

    return dist;
}

// k_th shortest distances
// space: O((n + m) * k)
// Time: O(k * (n + m*log(nk)))

vector<int> k_distance(vector<vector<pair<int, int>>> &G, int src, int n, int k, int dest) {
    vector<int> dist[n + 1];

    for (int i = 1; i <= n; i ++) {
        for (int j = 0; j < k; j ++) {
            dist[i].push_back(1e15);
        }
    }

    dist[src][0] = 0;

    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> q;
    q.push({0, src});

    while (!q.empty()) {
        auto cur = q.top();
        q.pop();

        // check if the current distance is worse than the k_th shortest dist
        // of the node, if yes, continue
        if (cur.first > dist[cur.second].back()) continue;

        for (auto c : G[cur.second]) {
            if (dist[c.first].back() > cur.first + c.second) {
                int in = lower_bound(dist[c.first].begin(), dist[c.first].end(), cur.first + c.second) - dist[c.first].begin();
                dist[c.first].insert(dist[c.first].begin() + in, cur.first + c.second);
                dist[c.first].pop_back();
                q.push({cur.first + c.second, c.first});
            }
        }
    }

    return dist[dest];
}