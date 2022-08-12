#include <bits/stdc++.h>
#define int long long
#define endl "\n"

using namespace std;

// Following code is an idea, not a working code:

signed main() {
    ios_base::sync_with_stdio(0);cin.tie(0);
    #ifndef ONLINE_JUDGE
        freopen("input.txt", "r", stdin);
        freopen("output.txt", "w", stdout);
    #endif

    // Graph G
    stack<int> st = getTopoSort(G); // if G is disconnected, call dfs for each unvisited node

    int dist[n] = {};
    // IMP: initialize dist with 0, not -INF
    // here we longest path in entire DAG, not just from source
    while(!st.empty()) {
        int cur = st.top(); st.pop();
        for(auto& c : G[cur]) {
            dist[c.first] = max(dist[c.first], dist[cur] + c.second);
            // relax even when dist[cur] = 0
        }
    }

    return 0; 
}