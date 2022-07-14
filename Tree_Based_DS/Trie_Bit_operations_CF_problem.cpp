#include <bits/stdc++.h>
// #define int long long
#define endl "\n"
#define fo(i, a, b) for(int i = a; i < b; i ++)
using namespace std;

// Problem: https://codeforces.com/problemset/problem/706/D

vector<int> binary(int x) {
    vector<int> b(32, 0);
    for (int i = 0 ; i < 32; i ++) {
        b[i] = x % 2;
        x /= 2;
    }
    reverse(b.begin(), b.end());
    return b;
}

class Node {
public:
    Node() {}
    map<int, Node*> children;
    int cnt = 0;
};

class Trie {
private:
    Node* root;
public:
    Trie() {
        root = new Node();
    }

    ~Trie() {
        for (auto node : root->children) {
            delete node.second;
        }
        delete root;
    }

    void insert(int x) {
        vector<int> a = binary(x);
        Node* cur = root;
        for (int c : a) {
            if (cur->children.find(c) == cur->children.end()) {
                cur->children[c] = new Node();
            }
            cur->cnt ++;
            cur = cur->children[c];
        }
        cur->cnt ++;
    }

    void remove(int x) {
        vector<int> a = binary(x);
        Node* cur = root;

        // it is guaranteed that x exists in the set for this prob. Otherwise, first check
        // if it does by calling search(x)
        for (int c : a) {
            // if (cur->children.find(c) == cur->children.end()) {
            //     return;
            // }
            cur->cnt = max(cur->cnt - 1, 0);
            cur = cur->children[c];
        }
        cur->cnt = max(cur->cnt - 1, 0);
    }

    int query(int x) {
        vector<int> a = binary(x);
        Node* cur = root;

        int good = 0, pos = 31, ans = 0;
        for (int c : a) {
            bool found = (cur->children.find(1 - c) != cur->children.end());
            if (found) {
                found &= ((cur->children[1 - c] != nullptr) && (cur->children[1 - c]->cnt > 0));
            }
            if (!found) {
                if (c == 1) {
                    if (!good) {
                        return x;
                    }
                }
                cur = cur->children[c];
            }
            else {
                if (c == 0)
                    good = 1;
                ans += (1 << pos);
                cur = cur->children[1 - c];
            }
            pos --;
        }
        return ans;
    }
};

signed main() {
    ios_base::sync_with_stdio(0); cin.tie(0);
#ifndef ONLINE_JUDGE
    freopen("input.txt", "r", stdin);
    freopen("output.txt", "w", stdout);
#endif

    int T = 1;
    // cin >> T;
    for (int t = 1; t < T + 1; t ++) {
        int q, x;
        char c;
        cin >> q;
        Trie* trie = new Trie();
        trie->insert(0);
        for (int i = 0; i < q; i ++) {
            cin >> c >> x;
            if (c == '+') {
                trie->insert(x);
            }
            else if (c == '-') {
                trie->remove(x);
            }
            else if (c == '?') {
                cout << trie->query(x) << endl;
            }
        }
    }

    return 0;
}