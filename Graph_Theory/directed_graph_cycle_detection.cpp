class Solution {
public:

    // copy this function
    void dfs(int u, vector<int> G[], int vis[], int cur_vis[], int &cycle) {
        if(cur_vis[u] && vis[u]) {
            cycle = 1;
            return;
        }   
        
        if(vis[u]) {
            return;
        }
        
        vis[u] = 1;
        cur_vis[u] = 1;
        
        for(auto c : G[u]) {
            dfs(c, G, vis, cur_vis, cycle);
        }
        cur_vis[u] = 0;
        // during backtracking, unmark the node in current_visited array
        return;
    }
    
    bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
        vector<int> G[numCourses];
        for(auto v : prerequisites) {
            G[v[0]].push_back(v[1]);
        }
        
        int vis[numCourses], cur_vis[numCourses], cycle = 0;
        memset(vis, 0, sizeof vis);
        memset(cur_vis, 0, sizeof cur_vis);
        for(int i = 0; i < numCourses; i ++) {
            if(!vis[i]) {
                dfs(i, G, vis, cur_vis, cycle);
            }
            if(cycle) break;
        }
        
        if(cycle) return false;
        return true;
    }
};