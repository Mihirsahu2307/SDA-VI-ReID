// remember to not declare any vairable as size (STL issues)

const int N = 1e3 + 5;
vector<int> parent(N), sz(N, 0ll);
multiset<int> sizes; // can remove if not wanted, edit functions likewise

int find_set(int a) {
    if (parent[a] == a) return a;
    return parent[a] = find_set(parent[a]); // path compression
}

void make_set(int a) {
    parent[a] = a;
    sz[a] = 1;
    sizes.insert(1);
}

void union_sets(int a, int b) {
    a = find_set(a);
    b = find_set(b);
    if (a == b) return; // don't forget

    if (sz[b] > sz[a]) {
        swap(a, b);
    }

    sizes.erase(sizes.find(sz[b]));
    sizes.erase(sizes.find(sz[a]));
    sz[a] += sz[b];
    sizes.insert(sz[a]);
    parent[b] = a;
}