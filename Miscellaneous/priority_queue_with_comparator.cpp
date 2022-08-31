class Solution {
private:
    // just remember like this:
    // for priority queue, greater comparator corresponds to min heap (counterintuitive)
    // so even the custom comparator should have reversed comparision when used for pq

    // here we are overloading function call operator
    struct compare {
        bool operator() (const pair<int, string>& a, const pair<int, string>& b) { 
            if(a.first == b.first) {
                return a.second < b.second; 
                // actually, we want b to come before a
                // so that when top is popped, a lexicographically larger string is popped from the pq
                // but for pq, we do the reverse
            }
            else {
                return a.first > b.first;
            }
        }
    };
    
public:
    vector<string> topKFrequent(vector<string>& words, int k) {
        // words 
        unordered_map<string, int> mp;
        priority_queue<pair<int, string>, vector<pair<int, string>>, compare> pq;
        
        for(auto& s : words) {
            mp[s] ++;
        }
        for(auto e : mp) {
            pq.push({e.second, e.first});
            if(pq.size() > k) {
                pq.pop();
            }
        }
        
        vector<string> ans;
        while(!pq.empty()) {
            ans.push_back(pq.top().second);
            pq.pop();
        }
        
        reverse(ans.begin(), ans.end());
        return ans;
    }
};