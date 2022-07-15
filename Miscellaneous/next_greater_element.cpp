// computes the index of the next greater element
// works even when a is a vector of integers instead of pairs

vector<int> next_greater(vector<pair<int, int>> a) {
    int n = a.size();
    vector<int> ans(n, -1ll);
    stack<pair<int, int>> s;
    for(auto e : a) {
        while(!s.empty() && s.top().first < e.first && ans[s.top().second] == -1) {
            // checking for ans[s.top().second] == -1 is unnecessary in general
            // it was for some LC problem
            ans[s.top().second] = e.second;
            s.pop();
        }
        s.push({e.first, e.second});
    }
    return ans;
}