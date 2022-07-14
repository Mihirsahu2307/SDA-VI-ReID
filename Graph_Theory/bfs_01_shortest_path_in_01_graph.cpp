vector<int> bfs_01(vector<vector<pair<int, int>>> &G, int src, int n) {
    deque<int> q;
    vector<int> dist(n + 1, 1e9);
    q.push_back(src);
    dist[src] = 0;
    while (!q.empty()) {
        auto cur = q.front();
        q.pop_front();
        for (auto child : G[cur]) {
            if (dist[child.first] > dist[cur] + child.second) {
                dist[child.first] = dist[cur] + child.second;
                // if path needs to be printed, use parent array
                if (child.second == 1) {
                    q.push_back(child.first);
                }
                else {
                    q.push_front(child.first);
                }
            }
        }
    }

    return dist;
}