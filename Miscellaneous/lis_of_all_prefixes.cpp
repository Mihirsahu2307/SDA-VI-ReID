#include <bits/stdc++.h>
using namespace std;

const int inf = 1e9;

vector<int> LIS(int a[], int n) {  // use upper_bound if repetition is allowed
    vector<int> ans(n, 0), dp(n, 1e9);

    for (int i = 0; i < n; i ++) {
        int in = lower_bound(dp.begin(), dp.end(), a[i]) - dp.begin();
        dp[in] = a[i];
        ans[i] = in + 1;
        if (i) ans[i] = max(ans[i], ans[i - 1]);
        // don't forget to take max, because lis using the current element
        // as last element may not be the LIS of the prefix
    }
    return ans;
}

signed main() {
    int n = 7;
    int a[n] = {2, 4, 3, 6, 9, 7, 5};
#ifndef ONLINE_JUDGE
    freopen("input.txt", "r", stdin);
    freopen("output.txt", "w", stdout);
#endif

    for (auto i : LIS(a, n)) {
        cout << i << ' ';
    }
    cout << endl;
    return 0;
}