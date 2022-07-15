// common applications:
/*
    1) longest prefix that is also a suffix ==> z array of s + "$" + rev(s)
    2) string compression (shortest repeating string) ==> i + z[i] == n
    3) distinct number of substrings: O(n^2). 
       Can be used to find distinct substrings
       that also satisfy some additional constraint by finding z array for every prefix
       then check the condtion for every suffix ending at i (0 to n)
       **Storing every substring naively will result in MLE**
*/

vector<int> Z(string s) {
    int n = s.size();
    vector<int> z(n, 0ll);
    int l = 0, r = 0;
    for(int i = 1; i < n; i ++) {
        if(i <= r) {
            z[i] = min(r - i + 1, z[i - l]);
        }

        while(i + z[i] < n && s[i + z[i]] == s[z[i]]) {
            z[i] ++;
        }

        if(i + z[i] - 1 > r) {
            r = i + z[i] - 1;
            l = i;
        }
    }

    return z;
}